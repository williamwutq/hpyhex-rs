"""
Example 4: PyTorch Integration - Simple MLP for Move Prediction

This example demonstrates how to use PyTorch with hpyhex-rs to train a simple
Multi-Layer Perceptron (MLP) that learns to predict good moves in the game.
This showcases:
- Converting game states to PyTorch tensors
- Training a neural network for game AI
- Using the model to make predictions
- Batch processing for efficient training
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from hpyhex import Hex, Piece, HexEngine, Game, PieceFactory


class GameStateDataset(Dataset):
    """Dataset for game states and their corresponding move scores."""
    
    def __init__(self, n_samples=1000, radius=5):
        """
        Generate a dataset of game states with labeled move quality.
        
        Args:
            n_samples: Number of samples to generate
            radius: Board radius for the games
        """
        self.states = []
        self.pieces = []
        self.scores = []
        
        print(f"Generating {n_samples} training samples...")
        
        for i in range(n_samples):
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{n_samples} samples")
            
            # Create a game with random state
            game = Game(radius, 5)
            
            # Play some random moves to get diverse states
            n_moves = np.random.randint(0, 10)
            for _ in range(n_moves):
                if game.end:
                    break
                positions = game.engine.check_positions(game.queue[0])
                if positions:
                    pos = positions[np.random.randint(len(positions))]
                    game.add_piece(0, pos)
            
            # Get current state and next piece
            board_state = np.array(list(game.engine.states), dtype=np.float32)
            if game.queue:
                piece_state = game.queue[0].to_numpy_float32()
                
                # Compute score for this state (density index as quality metric)
                positions = game.engine.check_positions(game.queue[0])
                if positions:
                    # Get the best density score available
                    densities = [game.engine.compute_dense_index(p, game.queue[0]) 
                                for p in positions]
                    score = max(densities)
                else:
                    score = 0.0
                
                self.states.append(board_state)
                self.pieces.append(piece_state)
                self.scores.append(score)
        
        print(f"Dataset created with {len(self.states)} samples")
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        # Concatenate board state and piece state as input features
        features = np.concatenate([self.states[idx], self.pieces[idx]])
        return torch.FloatTensor(features), torch.FloatTensor([self.scores[idx]])


class MovePredictor(nn.Module):
    """Simple MLP for predicting move quality scores."""
    
    def __init__(self, input_size, hidden_sizes=[128, 64, 32]):
        """
        Initialize the MLP.
        
        Args:
            input_size: Size of input features (board + piece)
            hidden_sizes: List of hidden layer sizes
        """
        super(MovePredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        # Output layer (single value: move quality score)
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    """
    Train the MLP model.
    
    Args:
        model: The neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
    """
    print("\n" + "=" * 60)
    print("Training Model")
    print("=" * 60)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for features, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, targets in val_loader:
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        # Print statistics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}")
    
    print("\nTraining completed!")


def use_model_for_prediction(model, radius=5):
    """
    Demonstrate using the trained model to make move predictions.
    
    Args:
        model: Trained neural network model
        radius: Board radius
    """
    print("\n" + "=" * 60)
    print("Using Model for Move Prediction")
    print("=" * 60)
    
    model.eval()
    
    # Create a test game
    game = Game(radius, 5)
    
    # Play a few moves
    for move_num in range(5):
        if game.end:
            break
        
        # Get current board state and next piece
        board_state = np.array(list(game.engine.states), dtype=np.float32)
        piece_state = game.queue[0].to_numpy_float32()
        
        # Get all valid positions
        positions = game.engine.check_positions(game.queue[0])
        
        if not positions:
            print(f"Move {move_num + 1}: No valid positions")
            break
        
        print(f"\nMove {move_num + 1}:")
        print(f"  Valid positions: {len(positions)}")
        
        # Predict score for each position
        # Note: In a real implementation, we'd need to simulate placing
        # the piece to get the resulting board state. For simplicity,
        # we'll use the current state.
        best_score = -float('inf')
        best_pos = None
        
        with torch.no_grad():
            features = torch.FloatTensor(np.concatenate([board_state, piece_state]))
            features = features.unsqueeze(0)  # Add batch dimension
            predicted_score = model(features).item()
        
        print(f"  Model predicted score: {predicted_score:.4f}")
        
        # Pick first position for demonstration
        chosen_pos = positions[0]
        actual_density = game.engine.compute_dense_index(chosen_pos, game.queue[0])
        
        print(f"  Chosen position: {chosen_pos}")
        print(f"  Actual density: {actual_density:.4f}")
        
        # Make the move
        game.add_piece(0, chosen_pos)
        print(f"  Score after move: {game.score}")


def compare_model_with_heuristic(model, n_games=10, radius=5):
    """
    Compare model-based strategy with density heuristic.
    
    Args:
        model: Trained neural network model
        n_games: Number of games to play for comparison
        radius: Board radius
    """
    print("\n" + "=" * 60)
    print(f"Comparing Model with Heuristic ({n_games} games)")
    print("=" * 60)
    
    model.eval()
    
    heuristic_scores = []
    
    # Test heuristic approach
    for _ in range(n_games):
        game = Game(radius, 5)
        while not game.end:
            positions = game.engine.check_positions(game.queue[0])
            if not positions:
                break
            
            # Use density heuristic
            densities = [game.engine.compute_dense_index(p, game.queue[0]) 
                        for p in positions]
            best_pos = positions[np.argmax(densities)]
            game.add_piece(0, best_pos)
        
        heuristic_scores.append(game.score)
    
    avg_heuristic = np.mean(heuristic_scores)
    std_heuristic = np.std(heuristic_scores)
    
    print(f"\nHeuristic (Density-based) Strategy:")
    print(f"  Average score: {avg_heuristic:.1f} ± {std_heuristic:.1f}")
    print(f"  Min: {min(heuristic_scores):.0f}, Max: {max(heuristic_scores):.0f}")
    
    # Note: A fully trained model would compete here, but for this example
    # we're demonstrating the framework rather than achieving optimal performance
    print(f"\nNote: The model would need more training data and epochs")
    print(f"to compete with the density heuristic. This example demonstrates")
    print(f"the integration framework for ML-based game AI.")


def main():
    """Run the PyTorch integration example."""
    print("\n" + "=" * 60)
    print("HpyHex-RS PyTorch Integration Example")
    print("=" * 60)
    print("Training a simple MLP to predict move quality\n")
    
    # Configuration
    RADIUS = 4  # Smaller for faster training
    TRAIN_SAMPLES = 500  # Reduced for demonstration
    VAL_SAMPLES = 100
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 0.001
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = GameStateDataset(n_samples=TRAIN_SAMPLES, radius=RADIUS)
    val_dataset = GameStateDataset(n_samples=VAL_SAMPLES, radius=RADIUS)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Calculate input size (board blocks + piece blocks)
    # For radius 4: 37 blocks, plus 7 piece blocks = 44 total
    n_board_blocks = len(HexEngine(RADIUS).states)
    n_piece_blocks = 7  # Standard piece size
    input_size = n_board_blocks + n_piece_blocks
    
    print(f"\nModel configuration:")
    print(f"  Input size: {input_size} ({n_board_blocks} board + {n_piece_blocks} piece)")
    print(f"  Hidden layers: [128, 64, 32]")
    print(f"  Output size: 1 (move quality score)")
    
    # Create model
    model = MovePredictor(input_size=input_size, hidden_sizes=[128, 64, 32])
    print(f"\nModel architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train the model
    train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE)
    
    # Use the model
    use_model_for_prediction(model, radius=RADIUS)
    
    # Compare with heuristic
    compare_model_with_heuristic(model, n_games=10, radius=RADIUS)
    
    print("\n" + "=" * 60)
    print("PyTorch integration example completed!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("- HpyHex-RS integrates seamlessly with PyTorch")
    print("- NumPy conversion enables efficient tensor creation")
    print("- The framework supports building game AI with deep learning")
    print("- With more data and training, models can learn complex strategies")


if __name__ == "__main__":
    main()
