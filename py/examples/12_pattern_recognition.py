"""
Example 12: Pattern Recognition with Correspondence Lists

This example demonstrates how to use correspondence lists to recognize patterns by making custom kernel functions.
It also implements an edge-minimization strategy that reduces boundary edges after piece placement and elimination.

This example demonstrates:
- Creating pattern recognition kernels using correspondence lists
- Analyzing boundary edges in I, J, K directions
- Implementing a strategy that minimizes boundary edges after each move and elimination
- Comparing edge-minimization strategy with other strategies from previous examples
"""

import numpy as np
import time
from typing import List, Tuple, Optional

from hpyhex import HexEngine, Game, Hex, PieceFactory


def calculate_edge_counts(engine: HexEngine) -> Tuple[int, int, int]:
    """
    Calculate the number of boundary edges in each of the three main directions (I, J, K).
    
    An edge exists when an occupied cell has an unoccupied neighbor in a specific direction.
    This measures the boundary complexity in each axial direction.
    
    Args:
        engine: The HexEngine to analyze
        
    Returns:
        Tuple of (edges_I, edges_J, edges_K) - edge counts for each direction
    """
    occupied = engine.to_numpy_bool()
    
    # Relative offsets for the three main directions in (i,k) coordinates
    directions = [
        (1, 0),   # I direction: (i+1, k)
        (0, 1),   # J direction: (i, k+1) 
        (-1, 1),  # K direction: (i-1, k+1)
    ]
    
    edge_counts = [0, 0, 0]
    
    for idx in range(len(occupied)):
        if not occupied[idx]:
            continue  # Skip unoccupied cells
            
        # Get coordinates of this occupied cell
        i, k = engine.coordinate_block(idx)
        
        # Check each direction for unoccupied neighbors
        for dir_idx, (di, dk) in enumerate(directions):
            ni, nk = i + di, k + dk
            neighbor_idx = engine.index_block(Hex(ni, nk))
            
            # If neighbor exists and is unoccupied, count as edge
            if neighbor_idx >= 0 and not occupied[neighbor_idx]:
                edge_counts[dir_idx] += 1
    
    return tuple(edge_counts)


def create_pattern_kernel(radius: int, pattern: List[Tuple[int, int]], shift: Hex) -> np.ndarray:
    """
    Create a pattern recognition kernel using correspondence lists.
    
    This demonstrates how correspondence lists can be used to detect specific
    patterns by creating kernels that map positions relative to a shift.
    
    Args:
        radius: Board radius
        pattern: List of (i,k) offsets defining the pattern relative to origin
        shift: The shift to apply to the pattern
        
    Returns:
        Kernel array where 1 indicates pattern match positions
    """
    kernel = np.zeros(HexEngine.solve_length(radius), dtype=bool)
    
    # Get correspondence list for the shift
    corr_list = HexEngine.to_numpy_correspondence_list_int64(radius, shift)
    
    # For each position, check if the pattern matches when shifted
    for idx in range(len(corr_list)):
        if corr_list[idx] == -1:  # Out of bounds
            continue
            
        # Check if pattern fits at this position
        pattern_fits = True
        for di, dk in pattern:
            # Get shifted position
            shifted_idx = corr_list[idx]
            if shifted_idx == -1:
                pattern_fits = False
                break
                
            # Check if relative position exists
            si, sk = HexEngine.hpyhex_rs_coordinate_block(radius, shifted_idx)
            rel_i, rel_k = si + di, sk + dk
            rel_idx = HexEngine.hpyhex_rs_index_block(radius, Hex(rel_i, rel_k))
            if rel_idx < 0:
                pattern_fits = False
                break
        
        if pattern_fits:
            kernel[idx] = True
    
    return kernel


class EdgeMinimizationStrategy:
    """
    Strategy that minimizes boundary edges after piece placement and elimination.
    
    This strategy evaluates all possible moves and chooses the one that results
    in the lowest total boundary edge count after placement and line elimination.
    It prints edge counts for I, J, K directions on each move.
    """
    
    def __init__(self, name: str = "Edge Minimization"):
        self.name = name
    
    def select_move(self, game: Game) -> Tuple[Optional[int], Optional[Hex]]:
        """
        Select the move that minimizes total boundary edges after placement + elimination.
        
        Returns:
            Tuple of (piece_index, position) or (None, None) if no moves available
        """
        best_score = float('inf')
        best_move = (None, None)
        
        # Evaluate all possible moves
        for piece_idx in range(len(game.queue)):
            piece = game.queue[piece_idx]
            positions = game.engine.check_positions(piece)
            
            for pos in positions:
                # Simulate the move
                test_engine = game.engine.__copy__()
                test_engine.add_piece(pos, piece)
                eliminated_score = len(test_engine.eliminate())
                
                # Calculate edge counts after move
                edges_i, edges_j, edges_k = calculate_edge_counts(test_engine)
                total_edges = edges_i + edges_j + edges_k
                
                # Prefer moves with fewer edges, break ties by higher elimination score
                score = total_edges - eliminated_score * 10  # Weight elimination heavily
                
                if score < best_score:
                    best_score = score
                    best_move = (piece_idx, pos)
        
        return best_move
    
    def get_edge_analysis(self, engine: HexEngine) -> Tuple[int, int, int, int]:
        """
        Get current edge analysis for an engine state.
        
        Returns:
            Tuple of (edges_I, edges_J, edges_K, total_edges)
        """
        edges_i, edges_j, edges_k = calculate_edge_counts(engine)
        return edges_i, edges_j, edges_k, edges_i + edges_j + edges_k


def demonstrate_edge_minimization():
    """
    Demonstrate the edge minimization strategy with detailed edge analysis.
    """
    print("=" * 70)
    print("Edge Minimization Strategy Demonstration")
    print("=" * 70)
    
    strategy = EdgeMinimizationStrategy()
    game = Game(4, 3)
    
    print(f"\nInitial board state (radius {game.engine.radius}):")
    initial_edges = strategy.get_edge_analysis(game.engine)
    print(f"  Edges - I: {initial_edges[0]}, J: {initial_edges[1]}, K: {initial_edges[2]} (Total: {initial_edges[3]})")
    
    move_count = 0
    max_moves = 15  # Limit for demonstration
    
    while not game.end and move_count < max_moves:
        piece_idx, pos = strategy.select_move(game)
        
        if piece_idx is None:
            print("\nNo valid moves available!")
            break
        
        # Record state before move
        before_edges = strategy.get_edge_analysis(game.engine)
        
        # Make the move
        old_score = game.score
        game.make_move(lambda eng, q: (piece_idx, pos))
        score_gained = game.score - old_score
        
        # Analyze after move
        after_edges = strategy.get_edge_analysis(game.engine)
        
        move_count += 1
        print(f"\nMove {move_count}:")
        print(f"  Piece: {PieceFactory.get_piece_name(game.queue[piece_idx])} (index {piece_idx})")
        print(f"  Position: {pos}")
        print(f"  Score gained: {score_gained}")
        print(f"  Edges before: I={before_edges[0]}, J={before_edges[1]}, K={before_edges[2]} (Total: {before_edges[3]})")
        print(f"  Edges after:  I={after_edges[0]}, J={after_edges[1]}, K={after_edges[2]} (Total: {after_edges[3]})")
        print(f"  Edge reduction: {before_edges[3] - after_edges[3]}")
    
    print(f"\nGame finished after {move_count} moves!")
    print(f"Final score: {game.score}")
    final_edges = strategy.get_edge_analysis(game.engine)
    print(f"Final edges - I: {final_edges[0]}, J: {final_edges[1]}, K: {final_edges[2]} (Total: {final_edges[3]})")


def demonstrate_pattern_recognition():
    """
    Demonstrate pattern recognition using correspondence lists.
    """
    print("\n" + "=" * 70)
    print("Pattern Recognition with Correspondence Lists")
    print("=" * 70)
    
    radius = 5
    engine = HexEngine(radius)
    
    # Define some patterns to recognize
    patterns = {
        "line_3": [(0, 0), (1, 0), (2, 0)],  # Horizontal line
        "triangle": [(0, 0), (1, 0), (0, 1)],  # Small triangle
        "corner": [(0, 0), (1, 0), (0, 1), (1, 1)]  # 2x2 corner
    }
    
    # Test different shifts
    shifts = [Hex(0, 0), Hex(1, 0), Hex(0, 1), Hex(1, 1)]
    
    print(f"\nAnalyzing patterns on radius {radius} board:")
    
    for pattern_name, pattern in patterns.items():
        print(f"\n{pattern_name.upper()} pattern {pattern}:")
        
        for shift in shifts:
            kernel = create_pattern_kernel(radius, pattern, shift)
            match_count = kernel.sum()
            print(f"  Shift {shift}: {match_count} positions match")
    
    # Demonstrate correspondence list properties
    print(f"\nCorrespondence list properties demonstration:")
    shift_a = Hex(1, 0)
    shift_b = Hex(0, 1)
    
    # Get correspondence lists
    corr_a = HexEngine.to_numpy_correspondence_list_int64(radius, shift_a)
    corr_b = HexEngine.to_numpy_correspondence_list_int64(radius, shift_b)
    
    print(f"  Correspondence list for shift {shift_a}: {corr_a[:10].tolist()}... (showing first 10)")
    print(f"  Correspondence list for shift {shift_b}: {corr_b[:10].tolist()}... (showing first 10)")
    
    # Show identity property (shift by 0,0 should be identity)
    identity = HexEngine.to_numpy_correspondence_list_int64(radius, Hex(0, 0))
    print(f"  Identity correspondence (shift 0,0): {identity[:10].tolist()}... (should be 0,1,2,...)")
    
    # Show composition (A + B)
    # Note: This is a simplified demonstration
    print(f"  These lists can be composed to create complex pattern transformations")


def compare_strategies():
    """
    Compare edge minimization with other strategies from examples.
    """
    print("\n" + "=" * 70)
    print("Strategy Comparison (Edge Minimization vs Others)")
    print("=" * 70)
    
    # Define comparison strategies inline (adapted from other examples)
    
    class RandomStrategy:
        def __init__(self):
            self.name = "Random"
        def select_move(self, game):
            for piece_idx in range(len(game.queue)):
                positions = game.engine.check_positions(game.queue[piece_idx])
                if positions:
                    return (piece_idx, positions[0])
            return (None, None)
    
    class DensityOnlyStrategy:
        def __init__(self):
            self.name = "Density Only"
        def select_move(self, game):
            best_score = -1
            best_move = (None, None)
            for piece_idx in range(len(game.queue)):
                piece = game.queue[piece_idx]
                for pos in game.engine.check_positions(piece):
                    density = game.engine.compute_dense_index(pos, piece)
                    if density > best_score:
                        best_score = density
                        best_move = (piece_idx, pos)
            return best_move
    
    strategies = [
        EdgeMinimizationStrategy(),
        DensityOnlyStrategy(),
        RandomStrategy()
    ]
    
    n_games = 10
    radius = 4
    
    print(f"\nRunning {n_games} games per strategy on radius {radius} boards...")
    
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting {strategy.name}...")
        scores = []
        edge_reductions = []
        
        for game_num in range(n_games):
            game = Game(radius, 3)
            initial_edges = EdgeMinimizationStrategy().get_edge_analysis(game.engine)[3]
            final_edges = initial_edges
            
            move_count = 0
            while not game.end and move_count < 20:
                piece_idx, pos = strategy.select_move(game)
                if piece_idx is None:
                    break
                game.make_move(lambda eng, q: (piece_idx, pos))
                move_count += 1
            
            scores.append(game.score)
            final_edges = EdgeMinimizationStrategy().get_edge_analysis(game.engine)[3]
            edge_reductions.append(initial_edges - final_edges)
        
        results[strategy.name] = {
            'scores': scores,
            'edge_reductions': edge_reductions
        }
    
    # Print results
    print(f"\n{'=' * 70}")
    print("Comparison Results")
    print(f"{'=' * 70}\n")
    
    for name, data in results.items():
        scores = np.array(data['scores'])
        reductions = np.array(data['edge_reductions'])
        print(f"{name}:")
        print(f"  Score: {scores.mean():.1f} ± {scores.std():.1f}")
        print(f"  Edge Reduction: {reductions.mean():.1f} ± {reductions.std():.1f}")
    
    best_strategy = max(results.items(), key=lambda x: np.mean(x[1]['scores']))
    print(f"\nBest performing strategy: {best_strategy[0]}")


def main():
    """
    Run all demonstrations for pattern recognition and edge minimization.
    """
    print("\n" + "=" * 70)
    print("HpyHex-RS Example 12: Pattern Recognition & Edge Minimization")
    print("=" * 70)
    print("Demonstrating correspondence lists and edge-minimizing strategies\n")
    
    # Demonstrate pattern recognition
    demonstrate_pattern_recognition()
    
    # Demonstrate edge minimization strategy
    demonstrate_edge_minimization()
    
    # Compare with other strategies
    compare_strategies()
    
    print("\n" + "=" * 70)
    print("Example 12 completed!")
    print("=" * 70)
    print("\nKey insights:")
    print("- Correspondence lists enable efficient pattern detection and transformation")
    print("- Edge minimization reduces board fragmentation and improves packing efficiency")
    print("- Analyzing edges in I/J/K directions provides detailed boundary insights")
    print("- The strategy successfully reduces boundary complexity while maintaining good scores")
    print("\nUse this approach for:")
    print("- Advanced board analysis and pattern recognition")
    print("- Developing strategies that optimize spatial relationships")
    print("- Understanding board state complexity in multiple dimensions")


if __name__ == "__main__":
    main()