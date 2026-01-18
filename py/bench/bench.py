import time
import statistics
from typing import Callable, Any
try:
    from hpyhex.hex import Hex, Piece
except ImportError:
    from hpyhex import Hex, Piece

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


def run_all_benchmarks():
    """Run all benchmark suites."""
    print("\n" + "="*60)
    print("PYTHON HEX & PIECE BENCHMARK SUITE")
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
    
    total_time = time.perf_counter() - start_time
    
    print("\n" + "="*60)
    print("BENCHMARK SUITE COMPLETE")
    print(get_local_datetime() + " on " + get_machine_info())
    print("="*60)
    print(f"Total Time: {format_time(total_time)}")
    print("="*60)


if __name__ == "__main__":
    run_all_benchmarks()