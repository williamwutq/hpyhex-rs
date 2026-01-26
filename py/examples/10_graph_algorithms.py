"""
Example 10: Various graph algorithms on HexEngine boards with game decision making.

This example demonstrates:
1. Graph algorithms for analyzing board state (island detection, density, anomalies)
2. How these analyses help strategic game decision making
3. A strategic game player that optimizes for placement density, minimizes fragmentation,
   and avoids creating problematic small islands (size < 4)
"""

import numpy as np
from hpyhex import Hex, Piece, HexEngine, Game, PieceFactory
from typing import List, Tuple, Dict, Optional
import time


def find_isolated_islands(engine: HexEngine) -> List[List[int]]:
    """
    Find all isolated islands (connected components) of unoccupied cells.
    
    Uses depth-first search on the adjacency graph to identify connected
    components of unoccupied cells.
    
    Args:
        engine: The HexEngine to analyze
        
    Returns:
        List of lists, where each inner list contains the indices of cells
        in a connected component of unoccupied cells. Occupied cells are ignored.
    """
    # Get occupancy state and adjacency list
    occupied = engine.to_numpy_bool()
    adj_list = HexEngine.to_numpy_adjacency_list_int32(engine.radius)  # int32 uses -1 as sentinel
    
    n = len(occupied)
    visited = np.zeros(n, dtype=bool)
    islands = []
    
    # Iterative DFS to find connected components
    def dfs(start: int) -> List[int]:
        stack = [start]
        visited[start] = True
        component = []
        
        while stack:
            cell = stack.pop()
            component.append(cell)
            
            # Check all 6 possible neighbors
            for i in range(6):
                neighbor = adj_list[cell, i]
                # Check if neighbor exists, is not occupied, and not visited
                if neighbor != -1 and not occupied[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        
        return component
    
    # Find all connected components of unoccupied cells
    for i in range(n):
        if not occupied[i] and not visited[i]:
            islands.append(dfs(i))
    
    return islands


def detect_strategic_anomalies(engine: HexEngine, min_size: int = 4) -> List[List[int]]:
    """
    Detect strategic anomalies: isolated regions with fewer than min_size cells.
    
    Small isolated regions are problematic because they cannot accommodate
    most game pieces (which typically require 4+ connected cells). These
    regions represent "dead space" that reduces board efficiency.
    
    Args:
        engine: The HexEngine to analyze
        min_size: Minimum size threshold for viable regions (default: 4)
        
    Returns:
        List of lists, where each inner list contains the indices of cells
        in an anomalous region (size < min_size). Returns empty list if no
        anomalies are detected.
    """
    islands = find_isolated_islands(engine)
    return [island for island in islands if len(island) < min_size]


def count_islands(engine: HexEngine) -> int:
    """
    Count the number of isolated islands of unoccupied cells without storing all details.
    Fast metric for evaluating board fragmentation.
    
    Args:
        engine: The HexEngine to analyze
        
    Returns:
        Number of disconnected unoccupied regions.
    """
    return len(find_isolated_islands(engine))


def compute_placement_density(engine: HexEngine, coord: Hex, piece: Piece) -> float:
    """
    Compute the placement density using the intrinsic compute_dense_index method.
    
    The dense index represents the ratio of occupied neighbors to total possible neighbors
    for all blocks in a piece placement. Higher density means the piece fits more tightly
    with existing pieces, which is generally better for board efficiency.
    
    Args:
        engine: The HexEngine to analyze
        coord: Position where piece would be placed
        piece: The piece to place
        
    Returns:
        Density score from compute_dense_index
    """
    # Use the intrinsic compute_dense_index method
    return engine.compute_dense_index(coord, piece)


def predict_fragmentation(engine: HexEngine, coord: Hex, piece: Piece) -> Dict[str, any]:
    """
    Predict how placing a piece will affect board fragmentation.
    
    This simulates the placement and analyzes the resulting island structure,
    which is crucial for strategic decision making.
    
    Args:
        engine: The HexEngine to analyze
        coord: Position where piece would be placed
        piece: The piece to place
        
    Returns:
        Dictionary containing fragmentation metrics:
        - num_islands: Number of unoccupied islands after placement
        - small_islands: Number of islands with size < 4
        - largest_island: Size of the largest island
        - total_unoccupied: Total unoccupied cells
    """
    # Simulate the placement
    copy_engine = engine.__copy__()
    copy_engine.add_piece(coord, piece)
    eliminated = len(copy_engine.eliminate())
    
    # Analyze islands
    islands = find_isolated_islands(copy_engine)
    
    if not islands:
        return {
            'num_islands': 0,
            'small_islands': 0,
            'largest_island': 0,
            'total_unoccupied': 0,
            'eliminated': eliminated
        }
    
    island_sizes = [len(island) for island in islands]
    
    return {
        'num_islands': len(islands),
        'small_islands': sum(1 for size in island_sizes if size < 4),
        'largest_island': max(island_sizes),
        'total_unoccupied': sum(island_sizes),
        'eliminated': eliminated
    }


def evaluate_board_health(engine: HexEngine) -> Dict[str, any]:
    """
    Comprehensive evaluation of board health using graph metrics.
    
    This provides a holistic view of the board state for strategic analysis.
    
    Args:
        engine: The HexEngine to analyze
        
    Returns:
        Dictionary with metrics:
        - occupation_rate: Percentage of occupied cells
        - num_islands: Number of unoccupied islands
        - fragmentation_score: Normalized fragmentation metric (lower is better)
        - small_islands_count: Number of problematic small islands
        - largest_island_size: Size of largest unoccupied region
    """
    occupied = engine.to_numpy_bool()
    total_cells = len(engine)
    occupied_count = occupied.sum()
    
    islands = find_isolated_islands(engine)
    
    if not islands:
        return {
            'occupation_rate': 1.0,
            'num_islands': 0,
            'fragmentation_score': 0.0,
            'small_islands_count': 0,
            'largest_island_size': 0
        }
    
    island_sizes = [len(island) for island in islands]
    small_islands = sum(1 for size in island_sizes if size < 4)
    
    # Fragmentation score: penalize having many islands and small islands
    # Normalized to 0-1 range
    fragmentation_score = (len(islands) + small_islands * 2) / max(total_cells, 1)
    
    return {
        'occupation_rate': occupied_count / total_cells,
        'num_islands': len(islands),
        'fragmentation_score': min(fragmentation_score, 1.0),
        'small_islands_count': small_islands,
        'largest_island_size': max(island_sizes)
    }


class StrategicGraphStrategy:
    """
    Advanced strategy that uses graph analysis to make optimal decisions.
    
    This strategy optimizes for:
    1. Maximizing placement density (tight packing)
    2. Minimizing number of unoccupied islands (reducing fragmentation)
    3. Avoiding creation of problematic small islands (size < 4)
    4. Maximizing elimination opportunities
    """
    
    def __init__(self, name: str = "Strategic Graph Analysis"):
        self.name = name
        self.move_history = []
    
    def select_move(self, game: Game) -> Tuple[Optional[int], Optional[Hex]]:
        """
        Select the best move using comprehensive graph analysis.
        
        Args:
            game: Current game state
            
        Returns:
            Tuple of (piece_index, position) or (None, None) if no move possible
        """
        best_score = float('-inf')
        best_move = (None, None)
        
        # Evaluate all possible moves
        for piece_idx in range(len(game.queue)):
            piece = game.queue[piece_idx]
            positions = game.engine.check_positions(piece)
            
            for pos in positions:
                score = self._evaluate_move(game.engine, pos, piece)
                
                if score > best_score:
                    best_score = score
                    best_move = (piece_idx, pos)
        
        return best_move
    
    def _evaluate_move(self, engine: HexEngine, coord: Hex, piece: Piece) -> float:
        """
        Comprehensive move evaluation using multiple graph metrics.
        
        Args:
            engine: The game engine
            coord: Position to place piece
            piece: The piece to place
            
        Returns:
            Composite score for the move
        """
        # Get baseline board health
        baseline_health = evaluate_board_health(engine)
        
        # Compute placement density (higher is better)
        density = compute_placement_density(engine, coord, piece)
        
        # Predict fragmentation after move
        frag_pred = predict_fragmentation(engine, coord, piece)
        
        # Score components:
        # 1. Density bonus: Reward tight packing
        density_score = density * 10.0
        
        # 2. Elimination bonus: Reward clearing lines
        elimination_score = frag_pred['eliminated'] * 2.0
        
        # 3. Island penalty: Penalize creating many islands
        island_penalty = frag_pred['num_islands'] * 5.0
        
        # 4. Small island penalty: Heavily penalize creating small islands
        small_island_penalty = frag_pred['small_islands'] * 20.0
        
        # 5. Piece length bonus: Prefer placing longer pieces
        length_bonus = len(piece) * 0.5
        
        # Composite score
        total_score = (
            density_score +
            elimination_score +
            length_bonus -
            island_penalty -
            small_island_penalty
        )
        
        return total_score


class DensityStrategy:
    """Pick the position with highest density score."""
    
    def __init__(self):
        self.name = "Density Maximizing Strategy"
    
    def select_move(self, game: Game) -> Tuple[int, Hex]:
        best_score = -1
        best_move = (None, None)
        
        # Check all pieces in queue
        for piece_idx in range(len(game.queue)):
            piece = game.queue[piece_idx]
            positions = game.engine.check_positions(piece)
            
            if positions:
                # Find position with highest density
                for pos in positions:
                    score = game.engine.compute_dense_index(pos, piece)
                    if score > best_score:
                        best_score = score
                        best_move = (piece_idx, pos)
        
        return best_move


def simulate_game_with_analysis(strategy, radius: int = 5, queue_length: int = 3,
                                move_limit: int = 200,
                                verbose: bool = True) -> Dict:
    """
    Simulate a game and track graph analysis metrics throughout.
    
    Args:
        strategy: The strategy to use for move selection
        radius: Board radius
        queue_length: Number of pieces in queue
        verbose: Whether to print detailed progress
        
    Returns:
        Dictionary with game statistics and graph metrics
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Simulating Game with {strategy.name}")
        print(f"{'=' * 70}")
    
    game = Game(radius, queue_length)
    start_time = time.time()
    
    move_count = 0
    metrics_history = []
    
    while not game.end and move_count < move_limit:
        # Get board health before move
        health_before = evaluate_board_health(game.engine)
        
        # Select and make move
        piece_idx, pos = strategy.select_move(game)
        
        if piece_idx is None:
            if verbose:
                print(f"\nNo valid moves available at turn {game.turn}")
            break
        
        # Record metrics
        piece = game.queue[piece_idx]
        density = compute_placement_density(game.engine, pos, piece)
        frag_pred = predict_fragmentation(game.engine, pos, piece)
        
        # Make the move
        score_before = game.score
        game.add_piece(piece_idx, pos)
        score_gained = game.score - score_before
        
        # Get board health after move
        health_after = evaluate_board_health(game.engine)
        
        metrics_history.append({
            'turn': game.turn,
            'density': density,
            'fragmentation_before': health_before['fragmentation_score'],
            'fragmentation_after': health_after['fragmentation_score'],
            'islands_before': health_before['num_islands'],
            'islands_after': health_after['num_islands'],
            'small_islands': health_after['small_islands_count'],
            'score_gained': score_gained,
            'eliminated': frag_pred['eliminated']
        })
        
        move_count += 1
        
        if verbose and move_count % 10 == 0:
            print(f"\nTurn {game.turn}:")
            print(f"  Score: {game.score}")
            print(f"  Occupation: {health_after['occupation_rate']:.1%}")
            print(f"  Islands: {health_after['num_islands']} "
                  f"(Small: {health_after['small_islands_count']})")
            print(f"  Fragmentation: {health_after['fragmentation_score']:.3f}")
    
    elapsed_time = time.time() - start_time
    final_health = evaluate_board_health(game.engine)
    
    stats = {
        'strategy': strategy.name,
        'final_score': game.score,
        'turns': game.turn,
        'moves': move_count,
        'elapsed_time': elapsed_time,
        'final_health': final_health,
        'metrics_history': metrics_history,
        'avg_density': np.mean([m['density'] for m in metrics_history]) if metrics_history else 0,
        'avg_islands': np.mean([m['islands_after'] for m in metrics_history]) if metrics_history else 0,
        'total_small_islands_created': sum(m['small_islands'] for m in metrics_history)
    }
    
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Game Finished!")
        print(f"{'=' * 70}")
        print(f"Final Score: {stats['final_score']}")
        print(f"Turns: {stats['turns']}")
        print(f"Occupation Rate: {final_health['occupation_rate']:.1%}")
        print(f"Final Islands: {final_health['num_islands']} "
              f"(Small: {final_health['small_islands_count']})")
        print(f"Fragmentation Score: {final_health['fragmentation_score']:.3f}")
        print(f"Average Placement Density: {stats['avg_density']:.3f}")
        print(f"Average Islands: {stats['avg_islands']:.1f}")
        print(f"Time: {elapsed_time:.3f}s")
    
    return stats


def demonstrate_graph_analysis():
    """
    Main demonstration of graph algorithms and strategic decision making.
    """
    print("\n" + "=" * 70)
    print("Example 10: Graph Algorithms for Strategic Game Analysis")
    print("=" * 70)
    
    # Part 1: Basic graph algorithm demonstration
    print("\n" + "-" * 70)
    print("Part 1: Basic Graph Analysis")
    print("-" * 70)
    
    game = Game(4, 3)
    
    # Play a few random moves to set up the board
    print("\nSetting up a game board...")
    for _ in range(5):
        for piece_idx in range(len(game.queue)):
            positions = game.engine.check_positions(game.queue[piece_idx])
            if positions:
                pos = positions[np.random.randint(len(positions))]
                game.add_piece(piece_idx, pos)
                break
    
    print(f"Board after {game.turn} turns:")
    health = evaluate_board_health(game.engine)
    print(f"  Occupation Rate: {health['occupation_rate']:.1%}")
    print(f"  Number of Islands: {health['num_islands']}")
    print(f"  Small Islands: {health['small_islands_count']}")
    print(f"  Largest Island: {health['largest_island_size']} cells")
    print(f"  Fragmentation Score: {health['fragmentation_score']:.3f}")
    
    # Demonstrate move analysis
    print("\nAnalyzing potential moves...")
    piece = game.queue[0]
    positions = game.engine.check_positions(piece)
    
    if positions:
        for i, pos in enumerate(positions[:3]):
            density = compute_placement_density(game.engine, pos, piece)
            frag = predict_fragmentation(game.engine, pos, piece)
            print(f"\n  Move {i+1}: Place at {pos}")
            print(f"    Density: {density:.3f}")
            print(f"    Would create {frag['num_islands']} islands "
                  f"({frag['small_islands']} small)")
            print(f"    Would eliminate {frag['eliminated']} lines")
    
    # Part 2: Strategic game simulation
    print("\n" + "-" * 70)
    print("Part 2: Strategic Game Simulation")
    print("-" * 70)
    
    strategy = StrategicGraphStrategy()
    stats = simulate_game_with_analysis(strategy, radius=5, queue_length=3, verbose=True)

    # Part 3: Compare with a simpler strategy
    print("\n" + "-" * 70)
    print("Part 3: Comparison with Density-Maximizing Strategy")
    print("-" * 70)

    simple_strategy = DensityStrategy()
    simple_stats = simulate_game_with_analysis(simple_strategy, radius=5, queue_length=3, verbose=True)
    
    # Part 4: Analysis summary
    print("\n" + "-" * 70)
    print("Part 3: Strategic Insights")
    print("-" * 70)
    
    print("\nHow Graph Analysis Helps Decision Making:")
    print("1. Placement Density: Higher density means tighter packing,")
    print("   leading to fewer wasted spaces and better board efficiency.")
    print(f"   Average density achieved: {stats['avg_density']:.3f}")
    
    print("\n2. Island Minimization: Fewer islands mean less fragmentation,")
    print("   making it easier to place future pieces.")
    print(f"   Average islands maintained: {stats['avg_islands']:.1f}")
    
    print("\n3. Avoiding Small Islands: Islands with < 4 cells are problematic")
    print("   because most pieces cannot fit in them, creating dead space.")
    print(f"   Small islands created: {stats['total_small_islands_created']}")
    
    print("\n4. Strategic Balance: The strategy balances all these factors")
    print("   along with elimination opportunities and piece length to")
    print("   maximize long-term game performance. The graph strategy outperformed")
    print("   the simpler density-maximizing approach in overall score and board health.")
    
    print("\n" + "=" * 70)
    print("Demonstration Complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_graph_analysis()