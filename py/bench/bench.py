import time
import statistics
from typing import Callable, Any
try:
    from hpyhex.hex import Hex, Piece, HexEngine
    from hpyhex.game import random_engine, PieceFactory, Game
    version = "Native Python"
except ImportError:
    from hpyhex import Hex, Piece, HexEngine, random_engine, PieceFactory, Game
    version = "Rust"

class Benchmark:
    """Simple benchmark runner with timing and statistics."""
    
    def __init__(self, name: str, warmup: int = 3, iterations: int = 100):
        self.name = name
        self.warmup = warmup
        self.iterations = iterations
        self.results = []
    
    def run(self, func: Callable[[], Any]) -> dict:
        """Run benchmark with warmup and multiple iterations."""
        # Warmup
        for _ in range(self.warmup):
            func()
        
        # Actual benchmark
        times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            func()
            end = time.perf_counter()
            times.append(end - start)
        
        return {
            'name': self.name,
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'stdev': statistics.stdev(times) if len(times) > 1 else 0,
            'min': min(times),
            'max': max(times),
            'iterations': self.iterations
        }


def format_time(seconds: float) -> str:
    """Format time in appropriate units."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.2f} µs"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds:.2f} s"


def get_machine_info() -> str:
    """Get basic machine info for benchmark context."""
    import platform
    import sys
    return f"{platform.system()} {platform.release()}, Python {sys.version.split()[0]}"


def get_local_datetime() -> str:
    """Get current local date and time as string."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_result(result: dict):
    """Pretty print benchmark result."""
    print(f"\n{result['name']}")
    print(f"  Mean:   {format_time(result['mean'])}")
    print(f"  Median: {format_time(result['median'])}")
    print(f"  StdDev: {format_time(result['stdev'])}")
    print(f"  Min:    {format_time(result['min'])}")
    print(f"  Max:    {format_time(result['max'])}")
    print(f"  Iterations: {result['iterations']}")


def benchmark_hex_creation():
    """Benchmark Hex creation (cached and uncached)."""
    print("\n" + "="*60)
    print("HEX CREATION BENCHMARKS")
    print("="*60)
    
    # Cached creation (within -64 to 64 range)
    bench = Benchmark("Hex Creation (Cached)", iterations=10000)
    result = bench.run(lambda: [Hex(i, i) for i in range(-10, 10)])
    print_result(result)
    
    # Uncached creation (outside cache range)
    bench = Benchmark("Hex Creation (Uncached)", iterations=10000)
    result = bench.run(lambda: [Hex(i, i) for i in range(100, 120)])
    print_result(result)
    
    # # Creation from tuple
    # bench = Benchmark("Hex Creation (From Tuple)", iterations=10000)
    # result = bench.run(lambda: [Hex((i, i)) for i in range(-10, 10)])
    # print_result(result)


def benchmark_hex_arithmetic():
    """Benchmark Hex arithmetic operations."""
    print("\n" + "="*60)
    print("HEX ARITHMETIC BENCHMARKS")
    print("="*60)
    
    h1 = Hex(5, 10)
    h2 = Hex(3, 7)
    
    # Addition
    bench = Benchmark("Hex Addition", iterations=100000)
    result = bench.run(lambda: h1 + h2)
    print_result(result)
    
    # Subtraction
    bench = Benchmark("Hex Subtraction", iterations=100000)
    result = bench.run(lambda: h1 - h2)
    print_result(result)
    
    # Addition with tuple
    bench = Benchmark("Hex Addition (with tuple)", iterations=100000)
    result = bench.run(lambda: h1 + (2, 3))
    print_result(result)


def benchmark_hex_methods():
    """Benchmark Hex method calls."""
    print("\n" + "="*60)
    print("HEX METHOD BENCHMARKS")
    print("="*60)
    
    h = Hex(5, 10)
    
    # Property access
    bench = Benchmark("Hex Property Access (i, j, k)", iterations=100000)
    result = bench.run(lambda: (h.i, h.j, h.k))
    print_result(result)
    
    # Shift operations
    bench = Benchmark("Hex shift_i", iterations=100000)
    result = bench.run(lambda: h.shift_i(3))
    print_result(result)
    
    bench = Benchmark("Hex shift_j", iterations=100000)
    result = bench.run(lambda: h.shift_j(3))
    print_result(result)
    
    bench = Benchmark("Hex shift_k", iterations=100000)
    result = bench.run(lambda: h.shift_k(3))
    print_result(result)
    
    # Equality
    h2 = Hex(5, 10)
    bench = Benchmark("Hex Equality", iterations=100000)
    result = bench.run(lambda: h == h2)
    print_result(result)
    
    # Hash
    bench = Benchmark("Hex Hash", iterations=100000)
    result = bench.run(lambda: hash(h))
    print_result(result)


def benchmark_hex_collections():
    """Benchmark Hex in collections."""
    print("\n" + "="*60)
    print("HEX COLLECTION BENCHMARKS")
    print("="*60)
    
    # Set operations
    hexes = [Hex(i, j) for i in range(-20, 20) for j in range(-20, 20)]
    
    bench = Benchmark("Create Set of Hexes", iterations=1000)
    result = bench.run(lambda: set(hexes))
    print_result(result)
    
    hex_set = set(hexes)
    test_hex = Hex(10, 10)
    
    bench = Benchmark("Hex Set Lookup", iterations=100000)
    result = bench.run(lambda: test_hex in hex_set)
    print_result(result)
    
    # Dict operations
    bench = Benchmark("Create Dict with Hex Keys", iterations=1000)
    result = bench.run(lambda: {h: i for i, h in enumerate(hexes)})
    print_result(result)
    
    hex_dict = {h: i for i, h in enumerate(hexes)}
    
    bench = Benchmark("Hex Dict Lookup", iterations=100000)
    result = bench.run(lambda: hex_dict.get(test_hex))
    print_result(result)


def benchmark_piece_creation():
    """Benchmark Piece creation."""
    print("\n" + "="*60)
    print("PIECE CREATION BENCHMARKS")
    print("="*60)
    
    # Creation from int
    bench = Benchmark("Piece Creation (from int)", iterations=10000)
    result = bench.run(lambda: [Piece(i) for i in range(0, 128)])
    print_result(result)
    
    # Creation from list
    states = [True, False, True, True, False, True, False]
    bench = Benchmark("Piece Creation (from list)", iterations=10000)
    result = bench.run(lambda: Piece(states))
    print_result(result)


def benchmark_piece_methods():
    """Benchmark Piece methods."""
    print("\n" + "="*60)
    print("PIECE METHOD BENCHMARKS")
    print("="*60)
    
    piece = Piece(0b1111111)  # All blocks occupied
    
    # Length
    bench = Benchmark("Piece Length", iterations=100000)
    result = bench.run(lambda: len(piece))
    print_result(result)
    
    # Int conversion
    bench = Benchmark("Piece to Int", iterations=100000)
    result = bench.run(lambda: int(piece))
    print_result(result)
    
    # Coordinates
    bench = Benchmark("Piece Coordinates", iterations=100000)
    result = bench.run(lambda: piece.coordinates)
    print_result(result)
    
    # Count neighbors
    test_hex = Hex(0, 0)
    bench = Benchmark("Piece Count Neighbors", iterations=100000)
    result = bench.run(lambda: piece.count_neighbors(test_hex))
    print_result(result)
    
    # Equality
    piece2 = Piece(0b1111111)
    bench = Benchmark("Piece Equality", iterations=100000)
    result = bench.run(lambda: piece == piece2)
    print_result(result)
    
    # Hash
    bench = Benchmark("Piece Hash", iterations=100000)
    result = bench.run(lambda: hash(piece))
    print_result(result)


def benchmark_piece_iteration():
    """Benchmark Piece iteration and filtering."""
    print("\n" + "="*60)
    print("PIECE ITERATION BENCHMARKS")
    print("="*60)
    
    # Iterate all pieces
    bench = Benchmark("Get All Pieces (generator)", iterations=1000)
    result = bench.run(lambda: list(Piece.all_pieces()))
    print_result(result)
    
    # Get contiguous pieces
    bench = Benchmark("Get Contiguous Pieces", iterations=100)
    result = bench.run(lambda: Piece.contigous_pieces())
    print_result(result)


def benchmark_mixed_operations():
    """Benchmark realistic mixed operations."""
    print("\n" + "="*60)
    print("MIXED OPERATION BENCHMARKS")
    print("="*60)
    
    # Typical workflow: create hexes, do arithmetic, store in dict
    def workflow():
        hexes = {}
        for i in range(-10, 10):
            for j in range(-10, 10):
                h = Hex(i, j)
                shifted = h.shift_i(1).shift_k(1)
                hexes[shifted] = Piece(i % 128)
        return hexes
    
    bench = Benchmark("Mixed Workflow (Hex + Piece operations)", iterations=100)
    result = bench.run(workflow)
    print_result(result)
    
    # Path calculation (common in hex grids)
    def calculate_path():
        start = Hex(0, 0)
        path = [start]
        for _ in range(20):
            current = path[-1]
            next_hex = current.shift_i(1).shift_j(1)
            path.append(next_hex)
        return path
    
    bench = Benchmark("Calculate Hex Path", iterations=10000)
    result = bench.run(calculate_path)
    print_result(result)


def benchmark_hexengine_creation():
    """Benchmark HexEngine creation."""
    print("\n" + "="*60)
    print("HEXENGINE CREATION BENCHMARKS")
    print("="*60)
    
    # Creation from radius
    bench = Benchmark("HexEngine Creation (radius=3)", iterations=10000)
    result = bench.run(lambda: HexEngine(3))
    print_result(result)
    
    bench = Benchmark("HexEngine Creation (radius=5)", iterations=10000)
    result = bench.run(lambda: HexEngine(5))
    print_result(result)
    
    bench = Benchmark("HexEngine Creation (radius=7)", iterations=5000)
    result = bench.run(lambda: HexEngine(7))
    print_result(result)
    
    # Creation from string
    state_str = "0" * 19  # radius=3
    bench = Benchmark("HexEngine Creation (from string, r=3)", iterations=10000)
    result = bench.run(lambda: HexEngine(state_str))
    print_result(result)
    
    # Creation from list
    state_list = [False] * 19  # radius=3
    bench = Benchmark("HexEngine Creation (from list, r=3)", iterations=10000)
    result = bench.run(lambda: HexEngine(state_list))
    print_result(result)


def benchmark_hexengine_coordinate_ops():
    """Benchmark HexEngine coordinate operations."""
    print("\n" + "="*60)
    print("HEXENGINE COORDINATE OPERATION BENCHMARKS")
    print("="*60)
    
    engine = HexEngine(5)
    test_hex = Hex(2, 3)
    
    # in_range check
    bench = Benchmark("HexEngine in_range", iterations=100000)
    result = bench.run(lambda: engine.in_range(test_hex))
    print_result(result)
    
    # index_block (O(1) lookup)
    bench = Benchmark("HexEngine index_block", iterations=100000)
    result = bench.run(lambda: engine.index_block(test_hex))
    print_result(result)
    
    # coordinate_block (reverse lookup)
    bench = Benchmark("HexEngine coordinate_block", iterations=100000)
    result = bench.run(lambda: engine.coordinate_block(15))
    print_result(result)
    
    # Round-trip conversion
    bench = Benchmark("HexEngine round-trip (hex->index->hex)", iterations=50000)
    result = bench.run(lambda: engine.coordinate_block(engine.index_block(test_hex)))
    print_result(result)


def benchmark_hexengine_state_ops():
    """Benchmark HexEngine state operations."""
    print("\n" + "="*60)
    print("HEXENGINE STATE OPERATION BENCHMARKS")
    print("="*60)
    
    engine = HexEngine(5)
    test_hex = Hex(2, 3)
    
    # get_state
    bench = Benchmark("HexEngine get_state (by Hex)", iterations=100000)
    result = bench.run(lambda: engine.get_state(test_hex))
    print_result(result)
    
    bench = Benchmark("HexEngine get_state (by index)", iterations=100000)
    result = bench.run(lambda: engine.get_state(15))
    print_result(result)
    
    # set_state
    bench = Benchmark("HexEngine set_state (by Hex)", iterations=100000)
    result = bench.run(lambda: engine.set_state(test_hex, True))
    print_result(result)
    
    bench = Benchmark("HexEngine set_state (by index)", iterations=100000)
    result = bench.run(lambda: engine.set_state(15, False))
    print_result(result)
    
    # reset
    bench = Benchmark("HexEngine reset", iterations=10000)
    result = bench.run(lambda: engine.reset())
    print_result(result)


def benchmark_hexengine_piece_ops():
    """Benchmark HexEngine piece operations."""
    print("\n" + "="*60)
    print("HEXENGINE PIECE OPERATION BENCHMARKS")
    print("="*60)
    
    engine = HexEngine(5)
    test_hex = Hex(2, 3)
    test_piece = Piece(0b1111000)  # 4 blocks occupied
    
    # check_add
    bench = Benchmark("HexEngine check_add", iterations=50000)
    result = bench.run(lambda: engine.check_add(test_hex, test_piece))
    print_result(result)
    
    # add_piece
    def add_and_reset():
        engine.reset()
        engine.add_piece(test_hex, test_piece)
    
    bench = Benchmark("HexEngine add_piece", iterations=10000)
    result = bench.run(add_and_reset)
    print_result(result)
    
    # check_positions
    bench = Benchmark("HexEngine check_positions (r=3)", iterations=100)
    engine3 = HexEngine(3)
    result = bench.run(lambda: engine3.check_positions(test_piece))
    print_result(result)
    
    bench = Benchmark("HexEngine check_positions (r=5)", iterations=50)
    result = bench.run(lambda: engine.check_positions(test_piece))
    print_result(result)


def benchmark_hexengine_neighbor_ops():
    """Benchmark HexEngine neighbor operations."""
    print("\n" + "="*60)
    print("HEXENGINE NEIGHBOR OPERATION BENCHMARKS")
    print("="*60)
    
    engine = HexEngine(5)
    # Create a pattern with some occupied blocks
    for i in range(10):
        engine.set_state(i, True)
    
    test_hex = Hex(2, 3)
    
    # count_neighbors
    bench = Benchmark("HexEngine count_neighbors", iterations=50000)
    result = bench.run(lambda: engine.count_neighbors(test_hex))
    print_result(result)
    
    # get_pattern
    bench = Benchmark("HexEngine get_pattern", iterations=50000)
    result = bench.run(lambda: engine.get_pattern(test_hex))
    print_result(result)


def benchmark_hexengine_eliminate():
    """Benchmark HexEngine eliminate operations."""
    print("\n" + "="*60)
    print("HEXENGINE ELIMINATE BENCHMARKS")
    print("="*60)
    
    # Create engine with full line for radius=3
    engine = HexEngine(3)
    # Fill first i-line (indices 0-2)
    for i in range(3):
        engine.set_state(i, True)
    
    bench = Benchmark("HexEngine eliminate (r=3, 1 line)", iterations=10000)
    def eliminate_and_reset():
        # Reset to filled line
        engine.reset()
        for i in range(3):
            engine.set_state(i, True)
        return engine.eliminate()
    result = bench.run(eliminate_and_reset)
    print_result(result)
    
    # Larger grid
    engine5 = HexEngine(5)
    # Fill first i-line (indices 0-4)
    for i in range(5):
        engine5.set_state(i, True)
    
    bench = Benchmark("HexEngine eliminate (r=5, 1 line)", iterations=5000)
    def eliminate_and_reset5():
        engine5.reset()
        for i in range(5):
            engine5.set_state(i, True)
        return engine5.eliminate()
    result = bench.run(eliminate_and_reset5)
    print_result(result)


def benchmark_hexengine_analysis():
    """Benchmark HexEngine analysis operations."""
    print("\n" + "="*60)
    print("HEXENGINE ANALYSIS BENCHMARKS")
    print("="*60)
    
    engine = HexEngine(5)
    # Create semi-random pattern
    for i in range(0, 30, 3):
        engine.set_state(i, True)
    
    test_hex = Hex(2, 3)
    test_piece = Piece(0b1111000)
    
    # compute_dense_index
    bench = Benchmark("HexEngine compute_dense_index", iterations=10000)
    result = bench.run(lambda: engine.compute_dense_index(test_hex, test_piece))
    print_result(result)
    
    # compute_entropy (expensive operation)
    bench = Benchmark("HexEngine compute_entropy (r=3)", iterations=1000)
    engine3 = HexEngine(3)
    for i in range(0, 10, 2):
        engine3.set_state(i, True)
    result = bench.run(lambda: engine3.compute_entropy())
    print_result(result)
    
    bench = Benchmark("HexEngine compute_entropy (r=5)", iterations=500)
    result = bench.run(lambda: engine.compute_entropy())
    print_result(result)


def benchmark_hexengine_serialization():
    """Benchmark HexEngine serialization operations."""
    print("\n" + "="*60)
    print("HEXENGINE SERIALIZATION BENCHMARKS")
    print("="*60)
    
    engine = HexEngine(5)
    for i in range(0, 30, 2):
        engine.set_state(i, True)
    
    # String representation
    bench = Benchmark("HexEngine __repr__ (to string)", iterations=10000)
    result = bench.run(lambda: repr(engine))
    print_result(result)
    
    # Deserialization from string
    state_str = repr(engine)
    bench = Benchmark("HexEngine from string (deserialize)", iterations=10000)
    result = bench.run(lambda: HexEngine(state_str))
    print_result(result)
    
    # Copy
    bench = Benchmark("HexEngine __copy__", iterations=10000)
    result = bench.run(lambda: engine.__copy__())
    print_result(result)
    
    # Hash
    bench = Benchmark("HexEngine __hash__", iterations=10000)
    result = bench.run(lambda: hash(engine))
    print_result(result)
    
    # Equality
    engine2 = engine.__copy__()
    bench = Benchmark("HexEngine __eq__", iterations=50000)
    result = bench.run(lambda: engine == engine2)
    print_result(result)


def benchmark_hexengine_collections():
    """Benchmark HexEngine in collections."""
    print("\n" + "="*60)
    print("HEXENGINE COLLECTION BENCHMARKS")
    print("="*60)
    
    # Create multiple engines
    engines = []
    for i in range(20):
        engine = HexEngine(3)
        for j in range(i):
            if j < len(engine.states):
                engine.set_state(j, True)
        engines.append(engine)
    
    # Set operations
    bench = Benchmark("Create Set of HexEngines", iterations=1000)
    result = bench.run(lambda: set(engines))
    print_result(result)
    
    engine_set = set(engines)
    test_engine = engines[10]
    
    bench = Benchmark("HexEngine Set Lookup", iterations=10000)
    result = bench.run(lambda: test_engine in engine_set)
    print_result(result)
    
    # Dict operations
    bench = Benchmark("Create Dict with HexEngine Keys", iterations=1000)
    result = bench.run(lambda: {e: i for i, e in enumerate(engines)})
    print_result(result)
    
    engine_dict = {e: i for i, e in enumerate(engines)}
    
    bench = Benchmark("HexEngine Dict Lookup", iterations=10000)
    result = bench.run(lambda: engine_dict.get(test_engine))
    print_result(result)


def benchmark_hexengine_mixed():
    """Benchmark realistic mixed HexEngine operations."""
    print("\n" + "="*60)
    print("HEXENGINE MIXED OPERATION BENCHMARKS")
    print("="*60)
    
    # Typical game loop: create engine, add pieces, check, eliminate
    def game_simulation():
        engine = HexEngine(5)
        pieces = [Piece(0b1111000), Piece(0b1110001), Piece(0b1010101)]
        positions = [Hex(1, 1), Hex(2, 2), Hex(3, 3)]
        
        for pos, piece in zip(positions, pieces):
            if engine.check_add(pos, piece):
                engine.add_piece(pos, piece)
        
        engine.eliminate()
        return engine
    
    bench = Benchmark("Game Simulation (add 3 pieces + eliminate)", iterations=5000)
    result = bench.run(game_simulation)
    print_result(result)
    
    # AI evaluation workflow
    def ai_evaluation():
        engine = HexEngine(4)
        # Add some pieces
        for i in range(10):
            engine.set_state(i, True)
        
        piece = Piece(0b1111000)
        positions = engine.check_positions(piece)
        
        # Evaluate each position
        best_score = 0.0
        for pos in positions[:5]:  # Limit to first 5 for speed
            score = engine.compute_dense_index(pos, piece)
            if score > best_score:
                best_score = score
        
        return best_score
    
    bench = Benchmark("AI Evaluation (check + score positions)", iterations=100)
    result = bench.run(ai_evaluation)
    print_result(result)
    
    # Full game state analysis
    def state_analysis():
        engine = HexEngine(4)
        for i in range(0, 20, 2):
            engine.set_state(i, True)
        
        entropy = engine.compute_entropy()
        neighbors = engine.count_neighbors(Hex(2, 2))
        pattern = engine.get_pattern(Hex(2, 2))
        
        return entropy, neighbors, pattern
    
    bench = Benchmark("State Analysis (entropy + neighbors + pattern)", iterations=500)
    result = bench.run(state_analysis)
    print_result(result)


def benchmark_random_engine_creation():
    """Benchmark random HexEngine creation."""
    print("\n" + "="*60)
    print("RANDOM HEXENGINE CREATION BENCHMARKS")
    print("="*60)
    
    # Creation with radius=3
    bench = Benchmark("Random HexEngine Creation (r=3)", iterations=10000)
    result = bench.run(lambda: random_engine(3))
    print_result(result)
    
    # Creation with radius=5
    bench = Benchmark("Random HexEngine Creation (r=5)", iterations=10000)
    result = bench.run(lambda: random_engine(5))
    print_result(result)
    
    # Creation with radius=7
    bench = Benchmark("Random HexEngine Creation (r=7)", iterations=2000)
    result = bench.run(lambda: random_engine(7))
    print_result(result)

    # Creation with radius=51
    bench = Benchmark("Random HexEngine Creation (r=51)", iterations=200)
    result = bench.run(lambda: random_engine(51))
    print_result(result)

    # Creation with radius=100
    bench = Benchmark("Random HexEngine Creation (r=100)", iterations=100)
    result = bench.run(lambda: random_engine(100))
    print_result(result)


def benchmark_piecefactory_lookups():
    """Benchmark PieceFactory lookup operations."""
    print("\n" + "="*60)
    print("PIECEFACTORY LOOKUP BENCHMARKS")
    print("="*60)
    
    # get_piece by name
    bench = Benchmark("PieceFactory get_piece (by name)", iterations=100000)
    result = bench.run(lambda: PieceFactory.get_piece("triangle_3_a"))
    print_result(result)
    
    # get_piece_name
    piece = PieceFactory.get_piece("triangle_3_a")
    bench = Benchmark("PieceFactory get_piece_name", iterations=100000)
    result = bench.run(lambda: PieceFactory.get_piece_name(piece))
    print_result(result)
    
    # Dictionary lookup (pieces)
    bench = Benchmark("PieceFactory direct dict access (pieces)", iterations=100000)
    result = bench.run(lambda: PieceFactory.pieces["triangle_3_a"])
    print_result(result)
    
    # Dictionary lookup (reverse_pieces)
    bench = Benchmark("PieceFactory direct dict access (reverse)", iterations=100000)
    result = bench.run(lambda: PieceFactory.reverse_pieces[13])
    print_result(result)


def benchmark_piecefactory_generation():
    """Benchmark PieceFactory piece generation."""
    print("\n" + "="*60)
    print("PIECEFACTORY GENERATION BENCHMARKS")
    print("="*60)
    
    # generate_piece (random)
    bench = Benchmark("PieceFactory generate_piece", iterations=50000)
    result = bench.run(lambda: PieceFactory.generate_piece())
    print_result(result)
    
    # all_pieces
    bench = Benchmark("PieceFactory all_pieces", iterations=10000)
    result = bench.run(lambda: PieceFactory.all_pieces())
    print_result(result)
    
    # Generate batch of pieces
    bench = Benchmark("PieceFactory generate 10 pieces", iterations=10000)
    result = bench.run(lambda: [PieceFactory.generate_piece() for _ in range(10)])
    print_result(result)
    
    bench = Benchmark("PieceFactory generate 100 pieces", iterations=1000)
    result = bench.run(lambda: [PieceFactory.generate_piece() for _ in range(100)])
    print_result(result)


def benchmark_piecefactory_validation():
    """Benchmark PieceFactory validation and error handling."""
    print("\n" + "="*60)
    print("PIECEFACTORY VALIDATION BENCHMARKS")
    print("="*60)
    
    # Valid piece lookup
    bench = Benchmark("PieceFactory get_piece (valid)", iterations=50000)
    result = bench.run(lambda: PieceFactory.get_piece("rhombus_4_i"))
    print_result(result)
    
    # Invalid piece lookup (with exception handling)
    def safe_get_piece():
        try:
            PieceFactory.get_piece("invalid_piece")
        except ValueError:
            pass
    
    bench = Benchmark("PieceFactory get_piece (invalid, caught)", iterations=10000)
    result = bench.run(safe_get_piece)
    print_result(result)
    
    # Valid piece name lookup
    piece = PieceFactory.get_piece("full")
    bench = Benchmark("PieceFactory get_piece_name (valid)", iterations=50000)
    result = bench.run(lambda: PieceFactory.get_piece_name(piece))
    print_result(result)


def benchmark_game_creation():
    """Benchmark Game creation."""
    print("\n" + "="*60)
    print("GAME CREATION BENCHMARKS")
    print("="*60)
    
    # Create game with int parameters
    bench = Benchmark("Game creation (radius=3, queue=3)", iterations=1000)
    result = bench.run(lambda: Game(3, 3))
    print_result(result)
    
    bench = Benchmark("Game creation (radius=5, queue=5)", iterations=1000)
    result = bench.run(lambda: Game(5, 5))
    print_result(result)
    
    bench = Benchmark("Game creation (radius=7, queue=7)", iterations=500)
    result = bench.run(lambda: Game(7, 7))
    print_result(result)
    
    # Create game with HexEngine
    engine = HexEngine(5)
    bench = Benchmark("Game creation (with HexEngine, queue=5)", iterations=1000)
    result = bench.run(lambda: Game(engine, 5))
    print_result(result)
    
    # Create game with piece list
    pieces = [PieceFactory.generate_piece() for _ in range(5)]
    bench = Benchmark("Game creation (with piece list)", iterations=5000)
    result = bench.run(lambda: Game(5, pieces))
    print_result(result)


def benchmark_game_properties():
    """Benchmark Game property access."""
    print("\n" + "="*60)
    print("GAME PROPERTY BENCHMARKS")
    print("="*60)
    
    game = Game(5, 5)
    
    # Property access
    bench = Benchmark("Game .end property", iterations=100000)
    result = bench.run(lambda: game.end)
    print_result(result)
    
    bench = Benchmark("Game .result property", iterations=100000)
    result = bench.run(lambda: game.result)
    print_result(result)
    
    bench = Benchmark("Game .turn property", iterations=100000)
    result = bench.run(lambda: game.turn)
    print_result(result)
    
    bench = Benchmark("Game .score property", iterations=100000)
    result = bench.run(lambda: game.score)
    print_result(result)
    
    bench = Benchmark("Game .engine property", iterations=100000)
    result = bench.run(lambda: game.engine)
    print_result(result)
    
    bench = Benchmark("Game .queue property", iterations=100000)
    result = bench.run(lambda: game.queue)
    print_result(result)


def benchmark_game_add_piece():
    """Benchmark Game add_piece operation."""
    print("\n" + "="*60)
    print("GAME ADD_PIECE BENCHMARKS")
    print("="*60)
    
    # Successful add
    def add_piece_success():
        game = Game(5, 5)
        positions = game.engine.check_positions(game.queue[0])
        if positions:
            return game.add_piece(0, positions[0])
        return False
    
    bench = Benchmark("Game add_piece (successful)", iterations=5000)
    result = bench.run(add_piece_success)
    print_result(result)
    
    # Failed add (overlap)
    def add_piece_fail():
        game = Game(5, 5)
        # Try to add at occupied position
        return game.add_piece(0, Hex(0, 0))
    
    bench = Benchmark("Game add_piece (failed, overlap)", iterations=5000)
    result = bench.run(add_piece_fail)
    print_result(result)
    
    # Multiple sequential adds
    def add_multiple_pieces():
        game = Game(5, 5)
        for i in range(3):
            positions = game.engine.check_positions(game.queue[0])
            if positions:
                game.add_piece(0, positions[0])
        return game
    
    bench = Benchmark("Game add 3 pieces sequentially", iterations=1000)
    result = bench.run(add_multiple_pieces)
    print_result(result)


def benchmark_game_make_move():
    """Benchmark Game make_move with different algorithms."""
    print("\n" + "="*60)
    print("GAME MAKE_MOVE BENCHMARKS")
    print("="*60)
    
    # Simple random algorithm
    def random_algorithm(engine, queue):
        import random
        piece_idx = random.randint(0, len(queue) - 1)
        positions = engine.check_positions(queue[piece_idx])
        if positions:
            return piece_idx, random.choice(positions)
        return 0, Hex(0, 0)
    
    bench = Benchmark("Game make_move (random algorithm)", iterations=5000)
    result = bench.run(lambda: Game(5, 5).make_move(random_algorithm))
    print_result(result)
    
    # First available position algorithm
    def first_available_algorithm(engine, queue):
        for i, piece in enumerate(queue):
            positions = engine.check_positions(piece)
            if positions:
                return i, positions[0]
        return 0, Hex(0, 0)
    
    bench = Benchmark("Game make_move (first available)", iterations=5000)
    result = bench.run(lambda: Game(5, 5).make_move(first_available_algorithm))
    print_result(result)
    
    # Density-based algorithm
    def density_algorithm(engine, queue):
        best_score = -1
        best_move = (0, Hex(0, 0))
        for i, piece in enumerate(queue):
            positions = engine.check_positions(piece)
            for pos in positions[:5]:  # Check first 5 positions
                score = engine.compute_dense_index(pos, piece)
                if score > best_score:
                    best_score = score
                    best_move = (i, pos)
        return best_move
    
    bench = Benchmark("Game make_move (density-based)", iterations=500)
    result = bench.run(lambda: Game(5, 5).make_move(density_algorithm))
    print_result(result)


def benchmark_game_full_game():
    """Benchmark complete game simulations."""
    print("\n" + "="*60)
    print("GAME FULL SIMULATION BENCHMARKS")
    print("="*60)
    
    # Play until game ends (random)
    def play_random_game():
        import random
        game = Game(4, 3)
        moves = 0
        while not game.end and moves < 50:  # Limit to 50 moves
            piece_idx = random.randint(0, len(game.queue) - 1)
            positions = game.engine.check_positions(game.queue[piece_idx])
            if positions:
                game.add_piece(piece_idx, random.choice(positions))
            moves += 1
        return game.result
    
    bench = Benchmark("Game play random game (max 50 moves)", iterations=100)
    result = bench.run(play_random_game)
    print_result(result)
    
    # Play with first-available strategy
    def play_greedy_game():
        game = Game(4, 3)
        moves = 0
        while not game.end and moves < 50:
            for i, piece in enumerate(game.queue):
                positions = game.engine.check_positions(piece)
                if positions:
                    game.add_piece(i, positions[0])
                    break
            moves += 1
        return game.result
    
    bench = Benchmark("Game play greedy game (max 50 moves)", iterations=100)
    result = bench.run(play_greedy_game)
    print_result(result)
    
    # Play 10 moves
    def play_10_moves():
        import random
        game = Game(5, 5)
        for _ in range(10):
            if game.end:
                break
            piece_idx = random.randint(0, len(game.queue) - 1)
            positions = game.engine.check_positions(game.queue[piece_idx])
            if positions:
                game.add_piece(piece_idx, random.choice(positions))
        return game.result
    
    bench = Benchmark("Game play 10 moves (random)", iterations=500)
    result = bench.run(play_10_moves)
    print_result(result)


def benchmark_game_serialization():
    """Benchmark Game serialization and representation."""
    print("\n" + "="*60)
    print("GAME SERIALIZATION BENCHMARKS")
    print("="*60)
    
    game = Game(5, 5)
    # Add a few pieces
    for i in range(3):
        positions = game.engine.check_positions(game.queue[0])
        if positions:
            game.add_piece(0, positions[0])
    
    # String representation
    bench = Benchmark("Game __str__", iterations=10000)
    result = bench.run(lambda: str(game))
    print_result(result)
    
    # Repr
    bench = Benchmark("Game __repr__", iterations=10000)
    result = bench.run(lambda: repr(game))
    print_result(result)


def benchmark_game_edge_cases():
    """Benchmark Game edge cases and error handling."""
    print("\n" + "="*60)
    print("GAME EDGE CASE BENCHMARKS")
    print("="*60)
    
    # Invalid piece index
    game = Game(5, 5)
    bench = Benchmark("Game add_piece (invalid index)", iterations=50000)
    result = bench.run(lambda: game.add_piece(99, Hex(0, 0)))
    print_result(result)
    
    # Invalid coordinate type
    bench = Benchmark("Game add_piece (invalid coord type)", iterations=50000)
    result = bench.run(lambda: game.add_piece(0, (0, 0)))
    print_result(result)
    
    # Make move on ended game
    ended_game = Game(2, 1)
    # Force game to end
    ended_game._Game__end = True
    bench = Benchmark("Game make_move (on ended game)", iterations=50000)
    def dummy_algo(e, q):
        return 0, Hex(0, 0)
    result = bench.run(lambda: ended_game.make_move(dummy_algo))
    print_result(result)


def benchmark_integration():
    """Benchmark integrated PieceFactory and Game operations."""
    print("\n" + "="*60)
    print("INTEGRATION BENCHMARKS")
    print("="*60)
    
    # Create game and play with generated pieces
    def integrated_workflow():
        game = Game(5, 5)
        import random
        for _ in range(5):
            if game.end:
                break
            piece_idx = random.randint(0, len(game.queue) - 1)
            positions = game.engine.check_positions(game.queue[piece_idx])
            if positions:
                game.add_piece(piece_idx, random.choice(positions))
        return game.score
    
    bench = Benchmark("Integrated: Create game + 5 moves", iterations=500)
    result = bench.run(integrated_workflow)
    print_result(result)
    
    # Piece generation and validation
    def piece_generation_validation():
        pieces = [PieceFactory.generate_piece() for _ in range(10)]
        names = [PieceFactory.get_piece_name(p) for p in pieces]
        return len(set(names))  # Unique pieces
    
    bench = Benchmark("Generate 10 pieces + get names", iterations=5000)
    result = bench.run(piece_generation_validation)
    print_result(result)
    
    # Full game with piece tracking
    def full_game_with_tracking():
        import random
        game = Game(4, 3)
        piece_history = []
        moves = 0
        while not game.end and moves < 30:
            piece_idx = random.randint(0, len(game.queue) - 1)
            piece = game.queue[piece_idx]
            positions = game.engine.check_positions(piece)
            if positions:
                piece_history.append(PieceFactory.get_piece_name(piece))
                game.add_piece(piece_idx, random.choice(positions))
            moves += 1
        return len(piece_history)
    
    bench = Benchmark("Full game with piece name tracking", iterations=50)
    result = bench.run(full_game_with_tracking)
    print_result(result)


def run_all_benchmarks():
    """Run all benchmark suites."""
    print("\n" + "="*60)
    print("PYTHON HPYHEX BENCHMARK SUITE")
    print("="*60)
    
    start_time = time.perf_counter()
    
    benchmark_hex_creation()
    benchmark_hex_arithmetic()
    benchmark_hex_methods()
    benchmark_hex_collections()
    benchmark_piece_creation()
    benchmark_piece_methods()
    benchmark_piece_iteration()
    benchmark_mixed_operations()
    benchmark_hexengine_creation()
    benchmark_hexengine_coordinate_ops()
    benchmark_hexengine_state_ops()
    benchmark_hexengine_piece_ops()
    benchmark_hexengine_neighbor_ops()
    benchmark_hexengine_eliminate()
    benchmark_hexengine_analysis()
    benchmark_hexengine_serialization()
    benchmark_hexengine_collections()
    benchmark_hexengine_mixed()
    benchmark_random_engine_creation()
    benchmark_piecefactory_lookups()
    benchmark_piecefactory_generation()
    benchmark_piecefactory_validation()
    benchmark_game_creation()
    benchmark_game_properties()
    benchmark_game_add_piece()
    benchmark_game_make_move()
    benchmark_game_full_game()
    benchmark_game_serialization()
    benchmark_game_edge_cases()
    benchmark_integration()
    
    total_time = time.perf_counter() - start_time
    
    print("\n" + "="*60)
    print("BENCHMARK SUITE COMPLETE")
    print(get_local_datetime() + " on " + get_machine_info())
    print("="*60)
    print(f"Version: {version}")
    print(f"Total Time: {format_time(total_time)}")
    print("="*60)


if __name__ == "__main__":
    run_all_benchmarks()