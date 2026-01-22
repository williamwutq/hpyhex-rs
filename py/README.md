# hpyhex-rs
Simplified implementations of the HappyHex game components and hexagonal system in Rust.
This is a drop-in replacement for the original `hpyhex` package, optimized for performance and memory usage. Offers up to 200x speed improvements in critical operations and around 60x speed improvements in essential gameplay workflows.

## Installation
```bash
pip install hpyhex-rs
```

## **Important Notes**
1. **Conflicting with Native Python Package**
   
   `hpyhex-rs` conflicts with the existing `hpyhex` package on PyPI. If you have `hpyhex` installed, please uninstall it first using:
   ```bash
   pip uninstall hpyhex
   ```
   
3. **Difference in Importing Modules**  
   
   In `hpyhex-rs`, all main classes and functions are located directly under the `hpyhex` module. For example, to import the `Hex` class, use:
   ```python
   from hpyhex import Hex, Game
   ```
   In contrast, the original `hpyhex` package requires importing from submodules (`hex` and `game`), such as:
   ```python
   from hpyhex.hex import Hex
   from hpyhex.game import Game
   ```
   
   For the best import compatibility, use the following pattern:
   ```python
   try:
      from hpyhex import Hex, Game  # hpyhex-rs
      hpyhex_version = "hpyhex-rs"
   except ImportError:
      from hpyhex.hex import Hex    # hpyhex
      from hpyhex.game import Game   # hpyhex
      hpyhex_version = "hpyhex"
   ```
   This code attempts to import from `hpyhex-rs` first, and falls back to the original `hpyhex` package if that fails, allowing your code to work with either package seamlessly.

4. **Not Interoperable with Original Package**
   
   Due to differences in the Rust implementation, `hpyhex-rs` objects cannot be mixed with the original `hpyhex` package objects. The `Hex` of `hpyhex-rs` is not compatible and cannot be converted to/from the `Hex` of `hpyhex`, for example.
   
   **This matters primarily in serialization scenarios**, but not in regular usage, as you would typically use either `hpyhex` or `hpyhex-rs` exclusively in a project. 
   
   If you are using built-in APIs in `hpyhex` to serialize data structures (e.g., `int(piece_value)`, `Piece(integer_value)`), you can load them back using `hpyhex-rs`, and vice versa. The byte representation of pieces is compatible between the two packages.
   
   However, if you use a python tool to serialize data structures from `hpyhex` as Python objects (e.g., `pickle`), you cannot load them back using `hpyhex-rs`, and vice versa. `hpyhex-rs` offers `serialize` and `deserialize` functions for its own data structures.

5. **Does Not Contain `benchmark` Module (Yet)**
   
   The original `hpyhex` package contains a `benchmark` module for performance testing of machine learned, heuristic, determinstic, and random algorithms. This module is not yet implemented in `hpyhex-rs`, but may be added in future releases. The source code for the benchmark module is very short and can be found [online](https://raw.githubusercontent.com/williamwutq/hpyhexml/main/hpyhex/hpyhex/benchmark.py). You may copy it into your project if needed.
   
6. **Updates Can Lag Behind Original Package**
   
   This package currently targets the [0.2.0](https://pypi.org/project/hpyhex/0.2.0/) version of `hpyhex`. Features from later versions may not be fully supported yet, but may be added in future releases.

## Features
- Hexagonal grid representation
- Basic game mechanics for HappyHex
- Utility functions for hexagonal calculations

## Author
Developed by William Wu.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Quickstart

1. Install the package:
   ```bash
   pip install hpyhex
   ```
2. Import and use the main classes as shown above.
3. Create custom algorithms to interact with the game environment.

## Main Classes

- **Hex**: Represents a hexagonal grid coordinate using a custom line-based system. Supports arithmetic, hashing, and tuple compatibility.
- **Piece**: Represents a shape made of 7 blocks, optimized for memory and performance. Use `PieceFactory` to create pieces by name or byte value.
- **HexEngine**: Manages the hexagonal grid, supports adding pieces, eliminating lines, and computing entropy.
- **PieceFactory**: Utility for creating pieces by name, byte, or randomly. Provides access to all predefined pieces.
- **Game**: Manages the game state, piece queue, score, and turn. Supports adding pieces and making moves with algorithms.

## Usage
```python
from hpyhex import Hex, Piece, HexEngine
from hpyhex import Game, PieceFactory

# Create a hexagonal coordinate
coo = Hex(0, 1)

# Create a piece by name
piece = PieceFactory.get_piece("triangle_3_a")

# Create a game engine with radius 3
engine = HexEngine(radius=3)

# Add a piece to the engine
engine.add_piece(piece, coo)

# Eliminate lines and get score
score = len(engine.eliminate()) * 5

# Create a game with engine radius and queue size
game = Game(engine=3, queue=5)
print(game)

# Make a move using a custom algorithm
def simple_algorithm(engine, queue):
	# Always place the first piece at the center
	return 0, Hex(0, 0)
game.make_move(simple_algorithm)
```

## Usage Advices

### Use Objects Provided by This Package
When using `hpyhex-rs`, ensure that you create and manipulate objects (like `Hex`, `Piece`, `HexEngine`, etc.) using the classes provided by this package. Although the API, which is defined in the original `hpyhex` package, accepts various types of inputs (like tuples for coordinates), using the native classes from `hpyhex-rs` ensures optimal performance and compatibility.

For example, following the original flyweight pattern in the `Hex` coordinate class, which uses a cache for small coordinates, the `Hex` class in `hpyhex-rs` also has a similar cache in Rust memory, which is not held by the GIL. Effectively, this means small Hex objects do not contain actual data, but just a pointer to a shared object in Rust memory. There are multiple ways to represent a `Hex` coordinate, either as a tuple `(i, k)`, `(i, j, k)`, or a `Hex` object. While all of them are accepted by most functions in the API, only `Hex` participates in the caching mechanism. Therefore, for frequently used coordinates, it is recommended to create and reuse `Hex` objects from `hpyhex-rs` instead of using tuples.

Take another example of `Piece` objects. Like the original optimized `Piece` in `hpyhex`, no pieces are created at all. Since there are only a total of 127 pieces made out of blocks, all pieces are pre-defined and stored in a global registry. When you create a piece using `Piece()`, it simply returns a reference to the corresponding pre-defined piece object. The Rust implementation further optimizes this by storing all piece objects in Rust memory, removing them from the control of the GIL. When expensive piece operations such as `count_neighbors` are performed, the Rust implementation quickly accesses the piece data and performs raw arithmetic and bit operations in Rust, significantly improving performance compared to the original Python implementation. None of those benefits are provided if integers are used instead of `Piece` objects, although they may seem smaller in memory. (Remember all Python objects have overhead in memory, and an integer is a Python object too.)

### Use Optimized Methods Provided by This Package
When using `hpyhex-rs`, prefer using methods provided by this package for better performance. If a function is already provided by the package, don't write your own implementation in Python, as it may be less efficient.

To illustrate this, take the example of `check_positions` of `HexEngine`. The original `hpyhex` package implements `check_positions` in Python as follows:
```python
def check_positions(self, piece: Union[Piece, int]) -> List[Hex]:
   if isinstance(piece, int):
      piece = Piece(piece)
   elif not isinstance(piece, Piece):
      raise TypeError("Piece must be an instance of Piece or an integer representing a Piece state")
   positions = []
   for a in range(self.radius * 2):
      for b in range(self.radius * 2):
         hex = Hex(a, b)
         if self.check_add(hex, piece):
            positions.append(hex)
   return positions
```

Obviously, if the fact that `hpyhex-rs` provides a Rust-backed implementation of `check_positions` is ignored, the above Python implementation can be used as `hpyhex-rs` also provides the `radius` attribute and `check_add` method. However, this implementation is inefficient as it creates various temporary Python objects, which are managed by the GIL, and performs various method calls (such as `range`) in Python, which are slow.

The `hpyhex-rs` package provides a Rust-backed implementation of `check_positions`, which performs all operations in Rust memory, avoiding the overhead of Python object management and method calls. In the entire expensive process of checking all possible positions, the GIL is only acquired once. The radius is not passed as a Python object, but as a direct integer in the Rust struct. The nested loops are performed in Rust, and `Hex` objects are created directly as structs without going through Python constructors. Further, instead of calling the `check_add` method, a special version of `check_add` that takes in raw Rust structs representing `Hex` and `Piece` is used, avoiding the overhead of interacting with Python objects at all. These optimizations mean the Rust-backed `check_positions` is more than **100 times** faster than the native Python implementation, as per [benchmarking](./bench/bench.py) results.

### Don't Reinvent the Wheel
It is tempting to implement your own versions of the various abstractions provided by this package, such as `Game`, which intuitively is just a combination of `HexEngine` and a piece queue, and does not offer too much extra customization. Unless your purpose is different from the original intention of `hpyhex`, it is recommended to use the provided `Game` class directly, as it interacts with the optimized Rust versions of `HexEngine` and `PieceFactory` without the overhead of creating intermediate Python objects. For extra functionality, consider building on top of `Game` instead of re-implementing it completely.

### Not Enough for GUI Applications
If you are building a GUI application for a simple version of HappyHex and deeply hated the original Java codebase, you possibly have pondered upon this package for performance, as it advertises itself as a high-performance implementation of the Python `hpyhex` package, which has a simple and useful API. Unless you already did a lot of work in Python, however, you should not use Python for your GUI applications, as it is not well-suited for GUI development and may lead to performance issues and a poor user experience. The [hpyhex-rs](https://crates.io/crates/hpyhex-rs) Rust crate, which is inspired by the Python API, not only provides similar functionality and abstractions, which make your transition to that package easier, but also provides further abstractions such as thread-safe guards, extended HexEngine with potential attributes for each cell, and an integrated game environment designed specifically for GUI threading needs. Consider using Rust as your main programming language for GUI applications, or integrate with C++ via FFI to use existing C++ GUI frameworks.

## The Statistics

(See [bench directory](./bench/) for full benchmarking code and results.)

All are tested on Apple M2 Pro with 16GB RAM, Python 3.11, Rust 1.92.0, macOS Sonoma 14.5.

### Speed Improvements

The Rust implementation of `hpyhex-rs` delivers dramatic performance improvements over the native Python `hpyhex` package. By leveraging Rust's zero-cost abstractions, efficient memory management, and ability to operate outside Python's Global Interpreter Lock (GIL), `hpyhex-rs` achieves speedups ranging from 2x to over 200x across different operations. These improvements are particularly significant for computationally intensive tasks like position checking, neighbor counting, and game simulations, making `hpyhex-rs` ideal for AI training, Monte Carlo simulations, and other performance-critical applications.

### Benchmark Comparison

The following table summarizes the performance improvements across major operation categories. All measurements represent typical use cases from each category, with speedup calculated as the ratio of Python execution time to Rust execution time.

| Category                | Representative Operation     | Python (µs) | Rust (µs) | Speedup    |
|-------------------------|------------------------------|-------------|-----------|------------|
| Hex Creation            | Cached hex creation          | 4.52        | 2.73      | 1.7x       |
| Hex Arithmetic          | Addition                     | 0.655       | 0.082     | 8.0x       |
| Hex Methods             | shift_i/j/k operations       | 0.272       | 0.068     | 4.0x       |
| Hex Collections         | Create set of hexes          | 108.01      | 58.41     | 1.8x       |
| Piece Creation          | From integer                 | 13.88       | 5.12      | 2.7x       |
| Piece Methods           | Count neighbors              | 3.37        | 0.077     | **43.8x**  |
| Piece Iteration         | Get contiguous pieces        | 47.65       | 0.990     | **48.1x**  |
| Mixed Operations        | Hex + Piece workflow         | 355.70      | 103.08    | 3.5x       |
| HexEngine Creation      | Radius 3 engine              | 0.195       | 0.131     | 1.5x       |
| HexEngine Coordinates   | index_block operation        | 0.412       | 0.087     | 4.7x       |
| HexEngine State         | get_state by hex             | 0.474       | 0.187     | 2.5x       |
| HexEngine Piece Ops     | check_positions (r=3)        | 73.69       | 0.459     | **160.5x** |
| HexEngine Neighbors     | count_neighbors              | 6.56        | 0.101     | **64.9x**  |
| HexEngine Eliminate     | eliminate (r=3, 1 line)      | 6.71        | 0.461     | 14.6x      |
| HexEngine Analysis      | compute_dense_index          | 32.63       | 0.214     | **152.5x** |
| HexEngine Serialization | From string                  | 3.44        | 0.462     | 7.4x       |
| HexEngine Collections   | Create set of engines        | 2.48        | 1.68      | 1.5x       |
| HexEngine Mixed         | AI evaluation                | 282.37      | 2.12      | **133.2x** |
| Random Creation         | Random engine (r=100)        | 11,980      | 302.38    | **39.6x**  |
| PieceFactory Lookup     | get_piece by name            | 0.206       | 0.133     | 1.5x       |
| PieceFactory Generation | Generate 100 pieces          | 54.15       | 10.57     | 5.1x       |
| PieceFactory Validation | get_piece (valid)            | 0.199       | 0.132     | 1.5x       |
| Game Creation           | Radius 3, queue 3            | 2.30        | 0.467     | 4.9x       |
| Game Properties         | Queue property access        | 0.074       | 0.237     | 0.3x*      |
| Game Add Piece          | Successful add               | 463.95      | 3.86      | **120.1x** |
| Game Make Move          | Random algorithm             | 451.23      | 4.40      | **102.6x** |
| Game Full Simulation    | 10 random moves              | 3,730       | 38.32     | **97.3x**  |
| Game Serialization      | __str__ method               | 75.25       | 6.10      | 12.3x      |
| Game Edge Cases         | Invalid index handling       | 0.306       | 0.191     | 1.6x       |
| Integration             | Create game + 5 moves        | 2,050       | 21.29     | **96.3x**  |

*Note: The queue property shows slower performance in Rust due to the overhead of converting Rust data structures to Python objects.*

### Highlights

Several operation categories demonstrate exceptional performance gains:

- **HexEngine check_positions** is **160x** faster. check_positions is a critical operation used by many heuristic algorithms and optimizers to gather valid piece placements. This speedup hugely benefits all downstream algorithms relying on position checking.
- **HexEngine compute_dense_index** is **152x** faster. A few critical algorithms, such as `nrsearch`, depends on Density Index computations. This speedup makes those algorithms significantly faster.
- **HexEngine AI evaluation** (checking and scoring positions) is **133x** faster. This is mainly due to the combined speedups in various critical HexEngine operations used to play the game.
- **Game add_piece** operation are **120x** faster. The core of the game is adding pieces to the engine, and this speedup directly translates to faster game simulations and AI training.
- **Game make_move** operations are **102x** faster, enabling rapid turn-based simulations.
- **Full game simulations** run **97x** faster, reducing a 3.7ms Python game to just 38µs in Rust. This benefits reinforcement learning, Monte Carlo Tree Search, test data generation, and other scenarios requiring many game simulations.
- **Piece count_neighbors** operations are **44x faster**.

These improvements are achieved through Rust's ability to perform raw arithmetic and bit operations in native code, combined with intelligent caching strategies that keep frequently-used data structures in Rust memory outside the GIL's control.
