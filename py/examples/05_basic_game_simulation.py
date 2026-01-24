"""
Example 5: Automated Game Simulation with Different Strategies

This example demonstrates how to create automated game players with various
strategies and simulate complete games. This is useful for:
- Testing game AI algorithms
- Benchmarking different strategies
- Generating training data for machine learning
- Game balancing and analysis
"""

import time
import numpy as np
from typing import List, Tuple, Callable

from hpyhex import Hex, Piece, HexEngine, Game, PieceFactory


class GameStrategy:
    """Base class for game playing strategies."""
    
    def __init__(self, name: str):
        self.name = name
    
    def select_move(self, game: Game) -> Tuple[int, Hex]:
        """
        Select a move (piece index and position) for the current game state.
        
        Args:
            game: Current game state
            
        Returns:
            Tuple of (piece_index, position) or (None, None) if no move possible
        """
        raise NotImplementedError


class RandomStrategy(GameStrategy):
    """Randomly select a valid move."""
    
    def __init__(self):
        super().__init__("Random")
    
    def select_move(self, game: Game) -> Tuple[int, Hex]:
        # Try each piece in the queue
        for piece_idx in range(len(game.queue)):
            positions = game.engine.check_positions(game.queue[piece_idx])
            if positions:
                # Pick a random valid position
                pos = positions[np.random.randint(len(positions))]
                return piece_idx, pos
        return None, None


class GreedyStrategy(GameStrategy):
    """Always pick the first valid position for the first piece."""
    
    def __init__(self):
        super().__init__("Greedy (First Valid)")
    
    def select_move(self, game: Game) -> Tuple[int, Hex]:
        positions = game.engine.check_positions(game.queue[0])
        if positions:
            return 0, positions[0]
        return None, None


class DensityStrategy(GameStrategy):
    """Pick the position with highest density score."""
    
    def __init__(self):
        super().__init__("Density-Maximizing")
    
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


class CenterBiasStrategy(GameStrategy):
    """Prefer positions closer to the center of the board."""
    
    def __init__(self):
        super().__init__("Center-Biased")
    
    def select_move(self, game: Game) -> Tuple[int, Hex]:
        best_distance = float('inf')
        best_move = (None, None)
        
        # Check all pieces
        for piece_idx in range(len(game.queue)):
            piece = game.queue[piece_idx]
            positions = game.engine.check_positions(piece)
            
            if positions:
                for pos in positions:
                    # Distance from center (0, 0)
                    distance = abs(pos.i) + abs(pos.j) + abs(pos.k)
                    if distance < best_distance:
                        best_distance = distance
                        best_move = (piece_idx, pos)
        
        return best_move


class AdaptiveStrategy(GameStrategy):
    """Adapt strategy based on board state (early vs late game)."""
    
    def __init__(self):
        super().__init__("Adaptive")
    
    def select_move(self, game: Game) -> Tuple[int, Hex]:
        # Get board occupation rate
        state_array = game.engine.to_numpy_bool()
        occupation_rate = state_array.sum() / len(state_array)
        
        # Early game (< 30% full): maximize density
        # Late game (>= 30% full): try any valid move
        
        if occupation_rate < 0.3:
            # Use density strategy
            best_score = -1
            best_move = (None, None)
            
            for piece_idx in range(len(game.queue)):
                piece = game.queue[piece_idx]
                positions = game.engine.check_positions(piece)
                
                if positions:
                    for pos in positions:
                        score = game.engine.compute_dense_index(pos, piece)
                        if score > best_score:
                            best_score = score
                            best_move = (piece_idx, pos)
            
            return best_move
        else:
            # Use greedy strategy (first valid)
            for piece_idx in range(len(game.queue)):
                positions = game.engine.check_positions(game.queue[piece_idx])
                if positions:
                    return piece_idx, positions[0]
            return None, None


def simulate_single_game(strategy: GameStrategy, radius: int = 5, 
                        queue_size: int = 5, verbose: bool = False) -> dict:
    """
    Simulate a complete game with the given strategy.
    
    Args:
        strategy: The strategy to use
        radius: Board radius
        queue_size: Number of pieces in queue
        verbose: Whether to print game progress
        
    Returns:
        Dictionary with game statistics
    """
    game = Game(radius, queue_size)
    moves = []
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Simulating game with {strategy.name} strategy")
        print(f"{'=' * 60}")
    
    start_time = time.time()
    
    while not game.end:
        piece_idx, pos = strategy.select_move(game)
        
        if piece_idx is None:
            if verbose:
                print(f"Turn {game.turn}: No valid moves available")
            break
        
        # Record move
        old_score = game.score
        game.add_piece(piece_idx, pos)
        score_gained = game.score - old_score
        
        moves.append({
            'turn': game.turn - 1,  # -1 because turn was incremented
            'piece_idx': piece_idx,
            'position': pos,
            'score_gained': score_gained,
            'total_score': game.score
        })
        
        if verbose and game.turn % 5 == 0:
            print(f"Turn {game.turn}: Score = {game.score}")
    
    elapsed_time = time.time() - start_time
    
    # Get final board state
    final_occupation = game.engine.to_numpy_bool().sum()
    occupation_rate = final_occupation / len(game.engine)
    
    stats = {
        'strategy': strategy.name,
        'final_score': game.score,
        'turns': game.turn,
        'moves': len(moves),
        'occupation_rate': occupation_rate,
        'elapsed_time': elapsed_time,
        'end_reason': 'natural' if game.end else 'no_moves',
        'avg_score_per_turn': game.score / max(game.turn, 1)
    }
    
    if verbose:
        print(f"\nGame finished!")
        print(f"  Final score: {stats['final_score']}")
        print(f"  Turns: {stats['turns']}")
        print(f"  Occupation: {occupation_rate:.1%}")
        print(f"  Time: {elapsed_time:.3f}s")
    
    return stats


def compare_strategies(strategies: List[GameStrategy], n_games: int = 20, 
                       radius: int = 5) -> None:
    """
    Compare multiple strategies by simulating many games.
    
    Args:
        strategies: List of strategies to compare
        n_games: Number of games per strategy
        radius: Board radius
    """
    print(f"\n{'=' * 60}")
    print(f"Strategy Comparison ({n_games} games per strategy)")
    print(f"{'=' * 60}")
    
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting {strategy.name}...")
        scores = []
        turns = []
        times = []
        occupations = []
        
        for i in range(n_games):
            stats = simulate_single_game(strategy, radius=radius, verbose=False)
            scores.append(stats['final_score'])
            turns.append(stats['turns'])
            times.append(stats['elapsed_time'])
            occupations.append(stats['occupation_rate'])
        
        results[strategy.name] = {
            'scores': np.array(scores),
            'turns': np.array(turns),
            'times': np.array(times),
            'occupations': np.array(occupations)
        }
    
    # Print comparison
    print(f"\n{'=' * 60}")
    print("Results Summary")
    print(f"{'=' * 60}\n")
    
    # Print detailed statistics for each strategy
    for strategy_name in results:
        data = results[strategy_name]
        print(f"{strategy_name}:")
        print(f"  Score:       {data['scores'].mean():7.1f} ± {data['scores'].std():6.1f} "
              f"(min: {data['scores'].min():5.0f}, max: {data['scores'].max():5.0f})")
        print(f"  Turns:       {data['turns'].mean():7.1f} ± {data['turns'].std():6.1f}")
        print(f"  Occupation:  {data['occupations'].mean():6.1%} ± {data['occupations'].std():5.1%}")
        print(f"  Time/game:   {data['times'].mean():7.4f}s")
        print()
    
    # Find best strategy by average score
    best_strategy = max(results.items(), key=lambda x: x[1]['scores'].mean())
    print(f"Best strategy by average score: {best_strategy[0]}")
    print(f"  Average score: {best_strategy[1]['scores'].mean():.1f}")


def demonstrate_game_replay(strategy: GameStrategy = None, radius: int = 4) -> None:
    """
    Demonstrate game replay by recording and displaying moves.
    
    Args:
        strategy: Strategy to use (default: DensityStrategy)
        radius: Board radius
    """
    if strategy is None:
        strategy = DensityStrategy()
    
    print(f"\n{'=' * 60}")
    print(f"Game Replay Demonstration")
    print(f"{'=' * 60}")
    
    game = Game(radius, 5)
    move_history = []
    
    print(f"\nPlaying game with {strategy.name} strategy...")
    
    while not game.end and len(move_history) < 10:  # Limit to 10 moves for demo
        piece_idx, pos = strategy.select_move(game)
        
        if piece_idx is None:
            break
        
        # Record state before move
        piece = game.queue[piece_idx]
        
        # Make move
        old_score = game.score
        game.add_piece(piece_idx, pos)
        
        # Record move
        move_history.append({
            'turn': game.turn - 1,
            'piece': PieceFactory.get_piece_name(piece),
            'position': pos,
            'score_gained': game.score - old_score,
            'total_score': game.score
        })
    
    # Display replay
    print(f"\nGame Replay ({len(move_history)} moves):")
    print(f"\n{'Turn':<6} {'Piece':<20} {'Position':<20} {'Points':<8} {'Total'}")
    print("-" * 60)
    
    for move in move_history:
        print(f"{move['turn']:<6} {move['piece']:<20} {str(move['position']):<20} "
              f"{move['score_gained']:<8} {move['total_score']}")
    
    print(f"\nFinal score: {game.score}")


def analyze_score_progression(strategy: GameStrategy = None, 
                               radius: int = 5) -> None:
    """
    Analyze how score progresses during a game.
    
    Args:
        strategy: Strategy to use
        radius: Board radius
    """
    if strategy is None:
        strategy = DensityStrategy()
    
    print(f"\n{'=' * 60}")
    print(f"Score Progression Analysis")
    print(f"{'=' * 60}")
    
    game = Game(radius, 5)
    turn_scores = [0]  # Start with 0 score
    
    while not game.end:
        piece_idx, pos = strategy.select_move(game)
        
        if piece_idx is None:
            break
        
        game.add_piece(piece_idx, pos)
        turn_scores.append(game.score)
    
    turn_scores = np.array(turn_scores)
    score_deltas = np.diff(turn_scores)
    
    print(f"\nUsing {strategy.name} strategy:")
    print(f"  Total turns: {len(turn_scores) - 1}")
    print(f"  Final score: {turn_scores[-1]}")
    print(f"  Average score per turn: {turn_scores[-1] / (len(turn_scores) - 1):.2f}")
    print(f"\nScore increments:")
    print(f"  Mean: {score_deltas.mean():.2f}")
    print(f"  Std dev: {score_deltas.std():.2f}")
    print(f"  Min: {score_deltas.min():.0f}, Max: {score_deltas.max():.0f}")
    
    # Show first few and last few scores
    print(f"\nFirst 5 turns: {turn_scores[:6].tolist()}")
    print(f"Last 5 turns:  {turn_scores[-6:].tolist()}")


def main():
    """Run all game simulation examples."""
    print("\n" + "=" * 60)
    print("HpyHex-RS Automated Game Simulation Examples")
    print("=" * 60)
    print("Demonstrating various game-playing strategies\n")
    
    # Create strategies
    strategies = [
        RandomStrategy(),
        GreedyStrategy(),
        DensityStrategy(),
        CenterBiasStrategy(),
        AdaptiveStrategy()
    ]
    
    # Run a detailed game with one strategy
    print("\nDetailed game simulation:")
    simulate_single_game(DensityStrategy(), radius=5, verbose=True)
    
    # Compare all strategies
    compare_strategies(strategies, n_games=20, radius=5)
    
    # Demonstrate game replay
    demonstrate_game_replay(DensityStrategy(), radius=4)
    
    # Analyze score progression
    analyze_score_progression(DensityStrategy(), radius=5)
    
    print("\n" + "=" * 60)
    print("All game simulation examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
