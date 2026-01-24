"""
Example 9: Selection-Based MLP Training and Benchmarking

This example demonstrates training a selection-based Multi-Layer Perceptron (MLP)
that learns to directly select optimal moves from game states. Unlike evaluation-based
approaches that score individual placements, this MLP takes the current game state
(board + queue) as input and outputs a probability distribution over all possible
(piece, position) combinations.

The process involves:
1. Generating training data using a modified NRSearch that provides ranked move lists
2. Training an MLP to predict move probabilities from game states
3. Benchmarking by comparing MLP selections with NRSearch rankings
4. Demonstrating RL potential with policy gradient training

This approach enables end-to-end learning for move selection and is well-suited
for reinforcement learning applications.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import pickle
import os
from typing import List, Tuple

from hpyhex import Hex, Piece, HexEngine, Game, PieceFactory


def nrsearch_ranked(engine: HexEngine, queue: List[Piece]) -> List[Tuple[int, Hex, float]]:
    """
    Modified NRSearch that returns a ranked list of all possible moves
    sorted by their NRSearch score in descending order.

    Parameters:
        engine (HexEngine): The game engine.
        queue (list[Piece]): The queue of pieces available for placement.

    Returns:
        ranked_moves: List of (piece_index, position, score) tuples, sorted by score desc.
    """
    options = []
    seen_pieces = {}

    # Iterate through all pieces in the queue
    for piece_index, piece in enumerate(queue):
        key = int(piece)

        # Skip duplicate pieces (same state value)
        if key in seen_pieces:
            continue
        seen_pieces[key] = piece_index

        # Check all valid positions for this piece
        for coord in engine.check_positions(piece):
            # Compute base score: dense index + piece length
            score = engine.compute_dense_index(coord, piece) + len(piece)

            # Simulate the move to compute elimination benefit
            copy_engine = engine.__copy__()
            copy_engine.add_piece(coord, piece)
            elimination_score = len(copy_engine.eliminate()) / engine.radius
            score += elimination_score

            options.append((piece_index, coord, score))

    # Sort by score in descending order
    options.sort(key=lambda x: x[2], reverse=True)
    return options


def flatten_move_to_index(piece_index: int, pos: Hex, queue_length: int, engine: HexEngine) -> int:
    """
    Convert (piece_index, position) to a flat index for the MLP output.
    """
    pos_index = engine.index_block(pos)
    return piece_index * len(engine) + pos_index


def index_to_move(flat_index: int, queue_length: int, engine: HexEngine) -> Tuple[int, Hex]:
    """
    Convert flat index back to (piece_index, position).
    """
    n_blocks = len(engine)
    piece_index = flat_index // n_blocks
    pos_index = flat_index % n_blocks

    pos = engine.coordinate_block(pos_index)

    return piece_index, pos


def generate_selection_samples(n_samples=10000, radius=5, queue_length=3, save_path='selection_samples.pkl'):
    """
    Generate training samples for selection-based MLP.

    Each sample consists of:
    - Game state (board + queue serialized)
    - Target: probability distribution over all possible moves
    """
    print(f"Generating {n_samples} selection-based samples...")
    print(f"Board radius: {radius}, Queue length: {queue_length}")

    samples = []
    games_played = 0

    while len(samples) < n_samples:
        games_played += 1
        game = Game(radius, queue_length)

        while not game.end and len(samples) < n_samples:
            # Get ranked moves from NRSearch
            ranked_moves = nrsearch_ranked(game.engine, game.queue)

            if not ranked_moves:
                break

            # Create target distribution (softmax over scores)
            scores = np.array([score for _, _, score in ranked_moves])
            # Add small temperature for smoothness
            temperatures_scores = scores / 0.1
            max_score = np.max(temperatures_scores)
            exp_scores = np.exp(temperatures_scores - max_score)  # numerical stability
            probabilities = exp_scores / np.sum(exp_scores)

            # Serialize state
            board_np = game.engine.to_numpy_uint32()
            queue_np = np.concatenate([p.to_numpy_uint8() for p in game.queue])

            # Create flat target array
            n_moves = queue_length * len(game.engine)
            target = np.zeros(n_moves, dtype=np.float32)

            for (piece_idx, pos, _), prob in zip(ranked_moves, probabilities):
                flat_idx = flatten_move_to_index(piece_idx, pos, queue_length, game.engine)
                if flat_idx < n_moves:
                    target[flat_idx] = prob

            samples.append((board_np, queue_np, target))

            # Make the best move to progress
            best_piece_idx, best_pos = ranked_moves[0][0], ranked_moves[0][1]
            game.add_piece(best_piece_idx, best_pos)

        if games_played % 10 == 0:
            print(f"  Generated {len(samples)} samples from {games_played} games...")

    print(f"Generated {len(samples)} samples from {games_played} games")

    # Save to binary file
    print(f"Saving samples to {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump(samples, f)
    print(f"Samples saved. File size: {os.path.getsize(save_path)} bytes")

    return samples


class SelectionDataset(Dataset):
    """Dataset for selection-based training samples."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        board_np, queue_np, target = self.samples[idx]

        # Concatenate state
        state = np.concatenate([board_np.astype(np.float32), queue_np.astype(np.float32)])

        return torch.FloatTensor(state), torch.FloatTensor(target)


class SelectionMLP(nn.Module):
    """MLP for move selection from game states."""

    def __init__(self, input_size, output_size, hidden_sizes=[512, 256, 256, 128]):
        super(SelectionMLP, self).__init__()

        layers = []
        prev_size = input_size

        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_size = hidden_size

        # Output layer (logits for move selection)
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_selection_mlp(model, train_loader, val_loader, epochs=40, lr=0.001, device='cpu'):
    """
    Train the selection MLP using cross-entropy loss.
    """
    print("\n" + "=" * 60)
    print("Training Selection MLP")
    print("=" * 60)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_samples = 0

        for states, targets in train_loader:
            states, targets = states.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * states.size(0)
            train_samples += states.size(0)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for states, targets in val_loader:
                states, targets = states.to(device), targets.to(device)
                outputs = model(states)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * states.size(0)
                val_samples += states.size(0)

        # Print statistics
        avg_train_loss = train_loss / train_samples
        avg_val_loss = val_loss / val_samples

        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}")

    print("\nTraining completed!")


def benchmark_selection_mlp(model, n_games=8, radius=5, queue_length=3, device='cpu', k=5):
    """
    Benchmark the selection MLP against NRSearch by comparing top-k move selections.
    """
    print("\n" + "=" * 60)
    print(f"Benchmarking Selection MLP vs NRSearch (Top-{k} Comparison)")
    print("=" * 60)

    model.to(device)
    model.eval()

    overlaps = []

    for game_num in range(n_games):
        print(f"\nGame {game_num + 1}/{n_games}")

        game = Game(radius, queue_length)
        game_moves = 0
        game_overlaps = []

        while not game.end and game_moves < 50:  # Limit moves for demo
            # Get NRSearch ranked moves
            nrsearch_moves = nrsearch_ranked(game.engine, game.queue)
            if not nrsearch_moves:
                break

            top_k_nrsearch = nrsearch_moves[:k]
            nrsearch_positions = {(piece_idx, pos) for piece_idx, pos, _ in top_k_nrsearch}

            # Get MLP predictions
            board_np = game.engine.to_numpy_float32()
            queue_np = np.concatenate([p.to_numpy_float32() for p in game.queue])
            state = np.concatenate([board_np, queue_np])
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(state_tensor).squeeze(0).cpu().numpy()

            # Create mask for valid moves
            mask = np.zeros_like(logits, dtype=bool)
            for piece_idx in range(len(game.queue)):
                valid_positions = game.engine.check_positions(game.queue[piece_idx])
                for pos in valid_positions:
                    flat_idx = flatten_move_to_index(piece_idx, pos, queue_length, game.engine)
                    if flat_idx < len(mask):
                        mask[flat_idx] = True

            # Apply mask to logits (set invalid moves to -inf)
            logits[~mask] = -np.inf

            # Get top-k MLP moves
            top_k_indices = np.argsort(logits)[-k:][::-1]  # descending
            mlp_positions = set()

            for idx in top_k_indices:
                if logits[idx] > -np.inf:  # valid move
                    piece_idx, pos = index_to_move(idx, queue_length, game.engine)
                    mlp_positions.add((piece_idx, pos))

            # Calculate overlap
            overlap = len(nrsearch_positions & mlp_positions)
            overlap_ratio = overlap / k
            game_overlaps.append(overlap_ratio)

            # Make NRSearch move to progress
            best_piece_idx, best_pos = nrsearch_moves[0][0], nrsearch_moves[0][1]
            game.add_piece(best_piece_idx, best_pos)
            game_moves += 1

        avg_overlap = np.mean(game_overlaps) if game_overlaps else 0
        overlaps.append(avg_overlap)
        print(f"  Game average overlap: {avg_overlap:.3f}")

    overall_avg_overlap = np.mean(overlaps)
    print(f"\nOverall average top-{k} overlap: {overall_avg_overlap:.3f}")
    print(f"This indicates the MLP selects {overall_avg_overlap:.1%} of NRSearch's top moves")


def benchmark_mlp_vs_nrsearch(model, n_games=8, radius=5, queue_length=3, device='cpu'):
    """
    Benchmark the trained MLP against NRSearch by playing complete games.

    Args:
        model: Trained MLP model
        n_games: Number of games to play
        radius: Board radius
        queue_length: Queue length
        device: Device for model inference
    """
    print("\n" + "=" * 60)
    print(f"Benchmarking MLP vs NRSearch ({n_games} games)")
    print("=" * 60)

    model.to(device)
    model.eval()

    mlp_scores = []
    mlp_times = []
    nrsearch_scores = []
    nrsearch_times = []

    for game_num in range(n_games):
        print(f"\nGame {game_num + 1}/{n_games}")

        # Play with NRSearch
        print("  Playing with NRSearch...")
        start_time = time.time()
        game = Game(radius, queue_length)
        moves = 0
        while not game.end:
            try:
                best_piece_idx, best_pos = nrsearch_ranked(game.engine, game.queue)[0][:2]
                game.add_piece(best_piece_idx, best_pos)
                moves += 1
            except ValueError:
                break

        nrsearch_time = time.time() - start_time
        nrsearch_scores.append(game.score)
        nrsearch_times.append(nrsearch_time)
        print(f"    NRSearch: Score {game.score}, Moves {moves}, Time {nrsearch_time:.2f}s")

        # Play with MLP
        print("  Playing with MLP...")
        start_time = time.time()
        game = Game(radius, queue_length)
        moves = 0
        while not game.end:
            # Get MLP predictions
            board_np = game.engine.to_numpy_float32()
            queue_np = np.concatenate([p.to_numpy_float32() for p in game.queue])
            state = np.concatenate([board_np, queue_np])
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(state_tensor).squeeze(0).cpu().numpy()

            # Create mask for valid moves
            mask = np.zeros_like(logits, dtype=bool)
            for piece_idx in range(len(game.queue)):
                valid_positions = game.engine.check_positions(game.queue[piece_idx])
                for pos in valid_positions:
                    flat_idx = flatten_move_to_index(piece_idx, pos, queue_length, game.engine)
                    if flat_idx < len(mask):
                        mask[flat_idx] = True

            # Apply mask to logits (set invalid moves to -inf)
            logits[~mask] = -np.inf

            # Select best move
            best_idx = np.argmax(logits)
            piece_idx, pos = index_to_move(best_idx, queue_length, game.engine)

            # Make the move
            game.add_piece(piece_idx, pos)
            moves += 1

        mlp_time = time.time() - start_time
        mlp_scores.append(game.score)
        mlp_times.append(mlp_time)
        print(f"    MLP: Score {game.score}, Moves {moves}, Time {mlp_time:.2f}s")

    # Summary
    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)

    mlp_scores = np.array(mlp_scores)
    nrsearch_scores = np.array(nrsearch_scores)
    mlp_times = np.array(mlp_times)
    nrsearch_times = np.array(nrsearch_times)

    print("MLP Performance:")
    print(f"  Scores: {mlp_scores.mean():.1f} ± {mlp_scores.std():.1f} "
          f"(min: {mlp_scores.min():.0f}, max: {mlp_scores.max():.0f})")
    print(f"  Avg time per game: {mlp_times.mean():.2f}s")

    print("\nNRSearch Performance:")
    print(f"  Scores: {nrsearch_scores.mean():.1f} ± {nrsearch_scores.std():.1f} "
          f"(min: {nrsearch_scores.min():.0f}, max: {nrsearch_scores.max():.0f})")
    print(f"  Avg time per game: {nrsearch_times.mean():.2f}s")

    improvement = (nrsearch_scores.mean() - mlp_scores.mean()) / nrsearch_scores.mean() * 100
    print(f"\nNRSearch outperforms MLP by {improvement:.1f}%")


def demonstrate_rl_potential():
    """
    Demonstrate how this selection-based MLP can be used in reinforcement learning.
    """
    print("\n" + "=" * 60)
    print("Reinforcement Learning Potential")
    print("=" * 60)

    print("""
This selection-based MLP architecture is ideal for reinforcement learning because:

1. **Policy Network**: The MLP directly outputs action probabilities, making it a
   natural policy network for policy gradient methods like REINFORCE or PPO.

2. **End-to-End Learning**: Unlike evaluation-based approaches that require separate
   move enumeration, this model learns to select moves directly from states.

3. **Exploration**: The probabilistic output enables natural exploration through
   sampling from the policy distribution.

4. **Training with Expert Data**: The model can be pre-trained on NRSearch rankings
   (as demonstrated), then fine-tuned with RL for superhuman performance.

Example RL Training Loop:

```python
def rl_training_step(model, optimizer, game_state, reward):
    # Get action probabilities from current state
    state_tensor = torch.FloatTensor(game_state).unsqueeze(0)
    action_logits = model(state_tensor)
    action_probs = F.softmax(action_logits, dim=-1)

    # Sample action from policy
    action_dist = torch.distributions.Categorical(action_probs)
    action = action_dist.sample()

    # Execute action, get next_state and reward
    next_state, reward = execute_action(action)

    # Compute policy gradient loss
    log_prob = action_dist.log_prob(action)
    loss = -log_prob * reward  # REINFORCE

    # Update model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return next_state
```

5. **Curriculum Learning**: Start with expert (NRSearch) supervision, then gradually
   shift to self-play RL as the model improves.

6. **Multi-Task Learning**: The same architecture can be used for both supervised
   pre-training (imitating NRSearch) and RL fine-tuning.

This approach has proven successful in games like AlphaGo and other complex domains.
""")


def main():
    # Note: the function may run for a long time due to training
    """Run the selection-based MLP example."""
    print("\n" + "=" * 60)
    print("HpyHex-RS Selection-Based MLP Example")
    print("=" * 60)
    print("Training an MLP for direct move selection from game states\n")

    # Configuration
    RADIUS = 5
    QUEUE_LENGTH = 3
    N_SAMPLES = 1000000
    TRAIN_SPLIT = 0.8
    BATCH_SIZE = 32
    EPOCHS = 40
    LEARNING_RATE = 0.0004
    SAMPLE_FILE = 'selection_samples.pkl'
    TOP_K = 5

    # Check if model file exists
    if os.path.exists('selection_mlp.pth'):
        print("Trained model 'selection_mlp.pth' already exists. Skipping training.")
        # Load model
        sample_state = np.zeros((7 * QUEUE_LENGTH + HexEngine.solve_length(RADIUS),), dtype=np.float32)
        sample_target = np.zeros((QUEUE_LENGTH * HexEngine.solve_length(RADIUS),), dtype=np.float32)
        input_size = sample_state.shape[0]
        output_size = sample_target.shape[0]
        model = SelectionMLP(input_size=input_size, output_size=output_size)
        model.load_state_dict(torch.load('selection_mlp.pth'))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\nUsing device: {device}")
        benchmark_selection_mlp(model, n_games=8, radius=RADIUS, queue_length=QUEUE_LENGTH, device=device, k=TOP_K)
        benchmark_mlp_vs_nrsearch(model, n_games=8, radius=RADIUS, queue_length=QUEUE_LENGTH, device=device)

        # Demonstrate RL potential
        demonstrate_rl_potential()

        return

    # Check if samples already exist
    if os.path.exists(SAMPLE_FILE):
        print(f"Found existing sample file: {SAMPLE_FILE}")
        samples = load_samples(SAMPLE_FILE)
    else:
        # Generate samples
        samples = generate_selection_samples(
            n_samples=N_SAMPLES,
            radius=RADIUS,
            queue_length=QUEUE_LENGTH,
            save_path=SAMPLE_FILE
        )

    # Split into train/val
    np.random.shuffle(samples)
    split_idx = int(len(samples) * TRAIN_SPLIT)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    print(f"\nDataset split: {len(train_samples)} train, {len(val_samples)} validation")

    # Create datasets and loaders
    train_dataset = SelectionDataset(train_samples)
    val_dataset = SelectionDataset(val_samples)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Calculate input/output sizes
    sample_state, sample_target = train_dataset[0]
    input_size = sample_state.shape[0]
    output_size = sample_target.shape[0]

    print(f"\nModel configuration:")
    print(f"  Input size: {input_size} (board + queue)")
    print(f"  Output size: {output_size} (move logits)")
    print(f"  Hidden layers: [512, 256, 128]")

    # Create model
    model = SelectionMLP(input_size=input_size, output_size=output_size)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Train the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    train_selection_mlp(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, device=device)

    # Save the trained model
    torch.save(model.state_dict(), 'selection_mlp.pth')
    print(f"\nModel saved to selection_mlp.pth")

    # Benchmark
    benchmark_selection_mlp(model, n_games=8, radius=RADIUS, queue_length=QUEUE_LENGTH, device=device, k=TOP_K)
    benchmark_mlp_vs_nrsearch(model, n_games=8, radius=RADIUS, queue_length=QUEUE_LENGTH, device=device)

    # Demonstrate RL potential
    demonstrate_rl_potential()

    print("\n" + "=" * 60)
    print("Selection-based MLP example completed!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("- Selection-based MLPs directly output move probabilities from states")
    print("- Can be trained with supervised learning on expert rankings")
    print("- Natural fit for reinforcement learning with policy gradients")
    print("- Enables end-to-end learning without move enumeration")
    print("- Combines well with curriculum learning and self-play")
    print("=" * 60)
    print("\nPractical Notes:")
    print("- MLP is not the best architecture; consider transformers or CNNs for better performance")
    print("- MLP show significant overfitting; regularization and more data may help")
    print("- Training time can be long; consider using GPUs for acceleration")
    print("=" * 60)


def load_samples(save_path='selection_samples.pkl'):
    """Load serialized samples from file."""
    print(f"Loading samples from {save_path}...")
    with open(save_path, 'rb') as f:
        samples = pickle.load(f)
    print(f"Loaded {len(samples)} samples")
    return samples


if __name__ == "__main__":
    main()