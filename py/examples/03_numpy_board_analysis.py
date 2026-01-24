"""
Example 3: NumPy Integration - Board State Analysis

This example demonstrates how to use NumPy for analyzing game board states,
computing statistics, and performing advanced board operations. This is useful for:
- Game AI development
- Strategic analysis
- Pattern recognition
- Performance profiling
"""

import numpy as np
from hpyhex import Hex, Piece, HexEngine, Game, PieceFactory


def board_state_visualization():
    """Visualize board state as a NumPy array."""
    print("=" * 60)
    print("Board State Visualization")
    print("=" * 60)
    
    # Create a game and play some moves
    engine = HexEngine(4)
    piece1 = PieceFactory.get_piece("triangle_3_a")
    piece2 = PieceFactory.get_piece("rhombus_4_i")
    
    # Find valid positions and add pieces
    pos1 = engine.check_positions(piece1)
    if pos1:
        engine.add_piece(pos1[0], piece1)
    
    pos2 = engine.check_positions(piece2)
    if pos2:
        engine.add_piece(pos2[0], piece2)
    
    print(f"Board radius: {engine.radius}")
    print(f"Total blocks: {len(engine.states)}")
    print(f"Occupied blocks: {sum(engine.states)}")
    
    # Get board state as a list
    state_list = list(engine.states)
    state_array = np.array(state_list, dtype=bool)
    
    print(f"\nBoard state array:")
    print(f"  Shape: {state_array.shape}")
    print(f"  Occupied: {state_array.sum()} / {len(state_array)}")
    print(f"  Occupation rate: {state_array.sum() / len(state_array) * 100:.1f}%")
    
    # Find occupied positions
    occupied_indices = np.where(state_array)[0]
    print(f"\nOccupied block indices: {occupied_indices}")
    print()


def board_density_analysis():
    """Analyze board density and clustering."""
    print("=" * 60)
    print("Board Density Analysis")
    print("=" * 60)
    
    # Create games with different fill levels
    radii = [3, 4, 5]
    
    for radius in radii:
        engine = HexEngine(radius)
        n_blocks = len(engine.states)
        
        # Fill random blocks
        n_fill = n_blocks // 3
        for _ in range(n_fill):
            idx = np.random.randint(0, n_blocks)
            engine.set_state(idx, True)
        
        state_array = np.array(list(engine.states), dtype=bool)
        occupation_rate = state_array.sum() / len(state_array)
        
        print(f"\nRadius {radius}:")
        print(f"  Total blocks: {n_blocks}")
        print(f"  Occupied: {state_array.sum()}")
        print(f"  Occupation rate: {occupation_rate * 100:.1f}%")
        print(f"  Entropy: {engine.compute_entropy():.4f}")
    print()


def analyze_valid_positions():
    """Analyze valid positions for piece placement."""
    print("=" * 60)
    print("Valid Position Analysis")
    print("=" * 60)
    
    # Create a partially filled board
    engine = HexEngine(5)
    
    # Add some pieces to create interesting constraints
    for i in range(10):
        engine.set_state(i, True)
    
    # Get a test piece
    piece = PieceFactory.get_piece("rhombus_4_i")
    print(f"Test piece: {PieceFactory.get_piece_name(piece)}")
    print(f"Board occupation: {sum(engine.states)} / {len(engine.states)}")
    
    # Find all valid positions
    valid_positions = engine.check_positions(piece)
    print(f"\nFound {len(valid_positions)} valid positions")
    
    # Compute density score for each position
    density_scores = np.array([
        engine.compute_dense_index(pos, piece)
        for pos in valid_positions
    ])
    
    if len(density_scores) > 0:
        print(f"\nDensity scores:")
        print(f"  Mean: {density_scores.mean():.4f}")
        print(f"  Std dev: {density_scores.std():.4f}")
        print(f"  Min: {density_scores.min():.4f}, Max: {density_scores.max():.4f}")
        
        # Find best position (highest density)
        best_idx = density_scores.argmax()
        best_pos = valid_positions[best_idx]
        print(f"\nBest position (highest density):")
        print(f"  Position: {best_pos}")
        print(f"  Density score: {density_scores[best_idx]:.4f}")
    print()


def pattern_analysis():
    """Analyze patterns in board states."""
    print("=" * 60)
    print("Pattern Analysis")
    print("=" * 60)
    
    # Create a board and analyze neighbor patterns
    engine = HexEngine(5)
    
    # Create a specific pattern (filled center)
    center_positions = [
        Hex(2, 2), Hex(2, 3), Hex(3, 2), Hex(3, 3), Hex(3, 4)
    ]
    
    for pos in center_positions:
        if engine.in_range(pos):
            idx = engine.index_block(pos)
            engine.set_state(idx, True)
    
    print(f"Created center cluster pattern")
    print(f"Occupied blocks: {sum(engine.states)}")
    
    # Analyze neighbor counts for all positions
    neighbor_counts = []
    for i in range(len(engine.states)):
        hex_pos = engine.coordinate_block(i)
        count = engine.count_neighbors(hex_pos)
        neighbor_counts.append(count)
    
    neighbor_array = np.array(neighbor_counts)
    
    print(f"\nNeighbor count distribution:")
    for count in range(7):  # 0 to 6 neighbors
        n_positions = (neighbor_array == count).sum()
        if n_positions > 0:
            print(f"  {count} neighbors: {n_positions} positions")
    
    # Find positions with maximum neighbors (hot spots)
    max_neighbors = neighbor_array.max()
    hot_spots = np.where(neighbor_array == max_neighbors)[0]
    print(f"\nHot spots (max {max_neighbors} neighbors):")
    print(f"  Found {len(hot_spots)} positions with maximum neighbors")
    print()


def game_progression_analysis():
    """Analyze how board state changes during a game."""
    print("=" * 60)
    print("Game Progression Analysis")
    print("=" * 60)
    
    # Track game statistics over multiple turns
    game = Game(5, 5)
    
    turn_data = []
    max_turns = 20
    
    for turn in range(max_turns):
        if game.end:
            break
        
        # Collect statistics before move
        state_array = np.array(list(game.engine.states), dtype=bool)
        occupation_rate = state_array.sum() / len(state_array)
        entropy = game.engine.compute_entropy()
        
        # Make a move (simple strategy: first valid position)
        positions = game.engine.check_positions(game.queue[0])
        if positions:
            pos = positions[0]
            game.add_piece(0, pos)
            
            turn_data.append({
                'turn': turn,
                'score': game.score,
                'occupation': occupation_rate,
                'entropy': entropy,
                'valid_positions': len(positions)
            })
    
    # Convert to NumPy arrays for analysis
    if turn_data:
        turns = np.array([d['turn'] for d in turn_data])
        scores = np.array([d['score'] for d in turn_data])
        occupations = np.array([d['occupation'] for d in turn_data])
        entropies = np.array([d['entropy'] for d in turn_data])
        valid_pos = np.array([d['valid_positions'] for d in turn_data])
        
        print(f"Analyzed {len(turn_data)} turns")
        print(f"\nScore progression:")
        print(f"  Initial: {scores[0]:.0f}, Final: {scores[-1]:.0f}")
        print(f"  Total gained: {scores[-1] - scores[0]:.0f}")
        print(f"  Average per turn: {(scores[-1] - scores[0]) / len(turn_data):.2f}")
        
        print(f"\nBoard occupation:")
        print(f"  Initial: {occupations[0]:.2%}")
        print(f"  Final: {occupations[-1]:.2%}")
        print(f"  Change: {(occupations[-1] - occupations[0]) * 100:.1f}%")
        
        print(f"\nEntropy:")
        print(f"  Mean: {entropies.mean():.4f}")
        print(f"  Range: [{entropies.min():.4f}, {entropies.max():.4f}]")
        
        print(f"\nValid positions per turn:")
        print(f"  Mean: {valid_pos.mean():.1f}")
        print(f"  Range: [{valid_pos.min():.0f}, {valid_pos.max():.0f}]")
    print()


def compare_strategies_statistically():
    """Compare different strategies using statistical analysis."""
    print("=" * 60)
    print("Statistical Strategy Comparison")
    print("=" * 60)
    
    n_games = 50
    
    # Strategy 1: Always pick first valid position
    def strategy_first(game):
        positions = game.engine.check_positions(game.queue[0])
        return positions[0] if positions else None
    
    # Strategy 2: Pick position with highest density
    def strategy_dense(game):
        positions = game.engine.check_positions(game.queue[0])
        if not positions:
            return None
        scores = [game.engine.compute_dense_index(p, game.queue[0]) for p in positions]
        return positions[np.argmax(scores)]
    
    strategies = [
        ("First Valid", strategy_first),
        ("Max Density", strategy_dense),
    ]
    
    results = {}
    
    for strategy_name, strategy_func in strategies:
        scores = []
        turns = []
        
        for _ in range(n_games):
            game = Game(5, 5)
            while not game.end:
                pos = strategy_func(game)
                if pos is None:
                    break
                game.add_piece(0, pos)
            
            scores.append(game.score)
            turns.append(game.turn)
        
        results[strategy_name] = {
            'scores': np.array(scores),
            'turns': np.array(turns)
        }
    
    # Statistical comparison
    print(f"Played {n_games} games with each strategy\n")
    
    for strategy_name in results:
        data = results[strategy_name]
        print(f"{strategy_name} Strategy:")
        print(f"  Score - Mean: {data['scores'].mean():.1f}, Std: {data['scores'].std():.1f}")
        print(f"  Score - Min: {data['scores'].min():.0f}, Max: {data['scores'].max():.0f}")
        print(f"  Turns - Mean: {data['turns'].mean():.1f}, Std: {data['turns'].std():.1f}")
    
    # Compare strategies
    score_diff = results["Max Density"]['scores'].mean() - results["First Valid"]['scores'].mean()
    print(f"\nMax Density scores {score_diff:+.1f} points better on average")
    print()


def main():
    """Run all board state analysis examples."""
    print("\n" + "=" * 60)
    print("HpyHex-RS NumPy Integration - Board State Analysis")
    print("=" * 60)
    print("This demonstrates advanced board analysis using NumPy")
    print("for game AI and strategic decision making.\n")
    
    # Run all examples
    board_state_visualization()
    board_density_analysis()
    analyze_valid_positions()
    pattern_analysis()
    game_progression_analysis()
    compare_strategies_statistically()
    
    print("=" * 60)
    print("All board analysis examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
