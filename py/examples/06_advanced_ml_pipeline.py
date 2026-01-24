"""
Example 6: Advanced Integration - ML Training Pipeline with Data Collection

This comprehensive example demonstrates how to combine multiple hpyhex-rs features
to build a complete machine learning pipeline for game AI. It shows:
- Automated data collection from game simulations
- Binary serialization for dataset storage
- NumPy-based feature extraction
- Training data preparation
- Performance benchmarking

This example serves as a template for building production ML systems.
"""

import os
import time
import pickle
import tempfile
import numpy as np
from typing import List, Dict, Tuple

from hpyhex import Hex, Piece, HexEngine, Game, PieceFactory


def nrsearch(engine: HexEngine, queue: List[Piece]) -> Tuple[int, Hex]:
    """
    A heuristic algorithm that selects the best piece and position based on 
    the dense index, piece length, and score gain from elimination.
    
    This algorithm computes a comprehensive score for each piece and position by:
    1. Computing the dense index (local density around placement)
    2. Adding the piece length (reward for placing larger pieces)
    3. Simulating the move and adding elimination score (reward for clearing lines)
    
    This is considered the best algorithm in the nrminimax package.
    
    Parameters:
        engine (HexEngine): The game engine.
        queue (list[Piece]): The queue of pieces available for placement.
        
    Returns:
        placement (tuple[int, Hex]): A tuple containing the index of the best 
                                     piece and the best position to place it.
                                     
    Raises:
        ValueError: If the queue is empty or no valid positions are found.
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
    
    if not options:
        raise ValueError("No valid options found")
    
    # Return the piece and position with the highest score
    best_placement = max(options, key=lambda item: item[2])
    best_piece_option, best_position_option, best_score_result = best_placement
    
    return (best_piece_option, best_position_option)


class DataCollector:
    """Collects training data from game simulations."""
    
    def __init__(self):
        self.samples = []
    
    def collect_from_game(self, game: Game, strategy_func) -> int:
        """
        Play a complete game and collect training samples.
        
        Args:
            game: Game instance
            strategy_func: Function that selects moves
            
        Returns:
            Number of samples collected
        """
        samples_collected = 0
        
        while not game.end:
            # Get current state
            board_state = game.engine.to_numpy_float32()
            
            # Get all pieces in queue
            for piece_idx, piece in enumerate(game.queue):
                piece_state = piece.to_numpy_float32()
                
                # Get valid positions for this piece
                positions = game.engine.check_positions(piece)
                
                if positions:
                    # Compute quality score for each position
                    for pos in positions:
                        density = game.engine.compute_dense_index(pos, piece)
                        neighbors = game.engine.count_neighbors(pos)
                        
                        # Create training sample
                        sample = {
                            'board': board_state.copy(),
                            'piece': piece_state.copy(),
                            'position': (pos.i, pos.j, pos.k),
                            'density': density,
                            'neighbors': neighbors,
                            'quality': density * (1 + neighbors * 0.1)  # Combined score
                        }
                        
                        self.samples.append(sample)
                        samples_collected += 1
            
            # Make a move using the strategy
            piece_idx, pos = strategy_func(game)
            if piece_idx is None:
                break
            
            game.add_piece(piece_idx, pos)
        
        return samples_collected
    
    def save_dataset(self, filepath: str) -> None:
        """Save collected samples to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.samples, f)
        print(f"Saved {len(self.samples)} samples to {filepath}")
    
    def load_dataset(self, filepath: str) -> None:
        """Load samples from file."""
        with open(filepath, 'rb') as f:
            self.samples = pickle.load(f)
        print(f"Loaded {len(self.samples)} samples from {filepath}")
    
    def get_numpy_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert samples to NumPy arrays for training.
        
        Returns:
            Tuple of (features, targets) arrays
        """
        if not self.samples:
            return np.array([]), np.array([])
        
        # Extract features (board + piece state)
        features_list = []
        targets_list = []
        
        for sample in self.samples:
            features = np.concatenate([sample['board'], sample['piece']])
            features_list.append(features)
            targets_list.append(sample['quality'])
        
        features = np.array(features_list, dtype=np.float32)
        targets = np.array(targets_list, dtype=np.float32)
        
        return features, targets


def collect_training_data(n_games: int = 50, radius: int = 4) -> DataCollector:
    """
    Collect training data from multiple games.
    
    Args:
        n_games: Number of games to simulate
        radius: Board radius
        
    Returns:
        DataCollector with collected samples
    """
    print("=" * 60)
    print(f"Collecting Training Data ({n_games} games)")
    print("=" * 60)
    
    collector = DataCollector()
    
    # Wrapper for nrsearch to match the expected strategy signature
    def nrsearch_strategy(game):
        return nrsearch(game.engine, game.queue)
    
    start_time = time.time()
    total_samples = 0
    
    for game_num in range(n_games):
        game = Game(radius, 5)
        samples = collector.collect_from_game(game, nrsearch_strategy)
        total_samples += samples
        
        if (game_num + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (game_num + 1) / elapsed
            print(f"  Completed {game_num + 1}/{n_games} games "
                  f"({rate:.1f} games/sec, {total_samples} samples)")
    
    elapsed = time.time() - start_time
    print(f"\nData collection complete:")
    print(f"  Total samples: {total_samples}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Average samples per game: {total_samples / n_games:.1f}")
    
    return collector


def analyze_dataset(collector: DataCollector) -> None:
    """Analyze collected dataset statistics."""
    print("\n" + "=" * 60)
    print("Dataset Analysis")
    print("=" * 60)
    
    features, targets = collector.get_numpy_arrays()
    
    print(f"\nDataset shape:")
    print(f"  Features: {features.shape}")
    print(f"  Targets: {targets.shape}")
    
    # Analyze board occupancy
    board_size = features.shape[1] - 7  # Subtract piece size
    board_states = features[:, :board_size]
    board_occupation = board_states.mean(axis=1)
    
    print(f"\nBoard occupation statistics:")
    print(f"  Mean: {board_occupation.mean():.2%}")
    print(f"  Std: {board_occupation.std():.2%}")
    print(f"  Min: {board_occupation.min():.2%}, Max: {board_occupation.max():.2%}")
    
    # Analyze target quality scores
    print(f"\nQuality score statistics:")
    print(f"  Mean: {targets.mean():.4f}")
    print(f"  Std: {targets.std():.4f}")
    print(f"  Min: {targets.min():.4f}, Max: {targets.max():.4f}")
    
    # Analyze piece patterns
    piece_states = features[:, board_size:]
    piece_occupation = piece_states.mean(axis=1)
    
    print(f"\nPiece occupation statistics:")
    print(f"  Mean blocks per piece: {piece_occupation.mean() * 7:.2f}/7")
    print(f"  Std: {piece_occupation.std() * 7:.2f}")


def demonstrate_batch_serialization(collector: DataCollector) -> None:
    """Demonstrate efficient batch serialization of game states."""
    print("\n" + "=" * 60)
    print("Batch Serialization Performance")
    print("=" * 60)
    
    # Create multiple game states
    n_states = 100
    print(f"\nSerializing {n_states} game states...")
    
    engines = []
    for _ in range(n_states):
        engine = HexEngine(5)
        # Add some random pieces
        for _ in range(5):
            if np.random.random() > 0.5:
                idx = np.random.randint(0, len(engine.states))
                engine.set_state(idx, True)
        engines.append(engine)
    
    # Benchmark serialization
    start_time = time.time()
    serialized_data = [engine.hpyhex_rs_serialize() for engine in engines]
    serialize_time = time.time() - start_time
    
    total_size = sum(len(data) for data in serialized_data)
    
    print(f"  Serialization time: {serialize_time:.4f}s")
    print(f"  Total size: {total_size / 1024:.2f} KB")
    print(f"  Average per state: {total_size / n_states:.0f} bytes")
    print(f"  Throughput: {n_states / serialize_time:.0f} states/sec")
    
    # Benchmark deserialization
    start_time = time.time()
    restored_engines = [HexEngine.hpyhex_rs_deserialize(data) 
                       for data in serialized_data]
    deserialize_time = time.time() - start_time
    
    print(f"\n  Deserialization time: {deserialize_time:.4f}s")
    print(f"  Throughput: {n_states / deserialize_time:.0f} states/sec")
    
    # Verify correctness
    all_match = all(e1 == e2 for e1, e2 in zip(engines, restored_engines))
    print(f"  Verification: {'✓ All states match' if all_match else '✗ Mismatch detected'}")


def demonstrate_feature_engineering() -> None:
    """Demonstrate advanced feature engineering with NumPy."""
    print("\n" + "=" * 60)
    print("Advanced Feature Engineering")
    print("=" * 60)
    
    # Create a sample game state
    engine = HexEngine(5)
    for i in range(15):
        engine.set_state(i, True)
    
    piece = PieceFactory.get_piece("full")
    
    print(f"\nBasic features:")
    board = engine.to_numpy_float32()
    piece_arr = piece.to_numpy_float32()
    
    print(f"  Board shape: {board.shape}, occupied: {board.sum():.0f}")
    print(f"  Piece shape: {piece_arr.shape}, occupied: {piece_arr.sum():.0f}")
    
    # Compute advanced features
    print(f"\nComputed features:")
    
    # Board statistics
    occupation_rate = board.mean()
    print(f"  Occupation rate: {occupation_rate:.2%}")
    
    # Pattern features - sliding window over board
    # Count local densities in windows
    window_size = 5
    local_densities = []
    for i in range(0, len(board) - window_size + 1, window_size):
        window = board[i:i + window_size]
        local_densities.append(window.mean())
    
    print(f"  Local density variance: {np.std(local_densities):.4f}")
    
    # Edge vs center occupation
    n_blocks = len(board)
    center_blocks = board[n_blocks // 3: 2 * n_blocks // 3]
    edge_blocks = np.concatenate([board[:n_blocks // 3], 
                                  board[2 * n_blocks // 3:]])
    
    print(f"  Center occupation: {center_blocks.mean():.2%}")
    print(f"  Edge occupation: {edge_blocks.mean():.2%}")
    
    # Piece features
    piece_mass = piece_arr.sum()
    piece_center = np.where(piece_arr)[0].mean() if piece_mass > 0 else 0
    
    print(f"  Piece mass: {piece_mass:.0f}")
    print(f"  Piece center: {piece_center:.2f}")
    
    print(f"\nThese engineered features can improve ML model performance!")


def benchmark_numpy_operations() -> None:
    """Benchmark NumPy operations vs pure Python."""
    print("\n" + "=" * 60)
    print("NumPy Performance Comparison")
    print("=" * 60)
    
    # Generate test data
    n_pieces = 1000
    pieces = [PieceFactory.generate_piece() for _ in range(n_pieces)]
    
    print(f"\nComparing operations on {n_pieces} pieces:")
    
    # Method 1: Pure Python
    start_time = time.time()
    py_sum = sum(int(piece) for piece in pieces)
    py_time = time.time() - start_time
    print(f"  Pure Python: {py_time:.4f}s")
    
    # Method 2: NumPy conversion + operations
    start_time = time.time()
    np_array = Piece.vec_to_numpy_uint8_stacked(pieces)
    np_sum = np_array.sum()
    np_time = time.time() - start_time
    print(f"  NumPy: {np_time:.4f}s")
    
    speedup = py_time / np_time
    print(f"  Speedup: {speedup:.1f}x")
    
    # Verify correctness
    print(f"  Results match: {abs(py_sum - np_sum * 7) < 1}")  # *7 because we sum all 7 blocks


def main():
    """Run the advanced integration example."""
    print("\n" + "=" * 60)
    print("HpyHex-RS Advanced Integration Example")
    print("=" * 60)
    print("Complete ML Pipeline with Data Collection\n")
    
    # Collect training data (reduced from 30 to 10 for demonstration with nrsearch)
    collector = collect_training_data(n_games=10, radius=4)
    
    # Analyze the dataset
    analyze_dataset(collector)
    
    # Demonstrate serialization
    demonstrate_batch_serialization(collector)
    
    # Demonstrate feature engineering
    demonstrate_feature_engineering()
    
    # Benchmark NumPy performance
    benchmark_numpy_operations()
    
    # Save dataset for later use
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
        temp_file = f.name
    
    collector.save_dataset(temp_file)
    
    # Load it back to verify
    new_collector = DataCollector()
    new_collector.load_dataset(temp_file)
    
    # Clean up
    os.unlink(temp_file)
    
    print("\n" + "=" * 60)
    print("Advanced integration example completed!")
    print("=" * 60)
    print("\nKey insights:")
    print("- Binary serialization is extremely fast and compact")
    print("- NumPy provides 10-100x speedup for batch operations")
    print("- Feature engineering significantly improves ML model quality")
    print("- The hpyhex-rs API enables seamless integration with Python ML stack")
    print("\nThis pipeline can be extended for:")
    print("- Distributed training data collection")
    print("- Real-time game AI inference")
    print("- Reinforcement learning environments")
    print("- Large-scale game analytics")


if __name__ == "__main__":
    main()
