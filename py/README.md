# hpyhex-rs
Simplified implementations of the HappyHex game components and hexagonal system in Rust.
This is a drop-in replacement for the original `hpyhex` package, optimized for performance and memory usage.

## Installation
```bash
pip install hpyhex-rs
```

## **Important Notes**
1. **Conflicting with Native Python Package**
   &nbsp;
   `hpyhex-rs` conflicts with the existing `hpyhex` package on PyPI. If you have `hpyhex` installed, please uninstall it first using:
   ```bash
   pip uninstall hpyhex
   ```
   &nbsp;
2. **Difference in Importing Modules**
   &nbsp;
   In `hpyhex-rs`, all main classes and functions are located directly under the `hpyhex` module. For example, to import the `Hex` class, use:
   ```python
   from hpyhex import Hex, Game
   ```
   In contrast, the original `hpyhex` package requires importing from submodules (`hex` and `game`), such as:
   ```python
   from hpyhex.hex import Hex
   from hpyhex.game import Game
   ```
3. **Not Interoperable with Original Package**
   &nbsp;
   Due to differences in the Rust implementation, `hpyhex-rs` objects cannot be mixed with the original `hpyhex` package objects. The `Hex` of `hpyhex-rs` is not compatible and cannot be converted to/from the `Hex` of `hpyhex`, for example.
   &nbsp;
   **This matters primarily in serialization scenarios**, but not in regular usage, as you would typically use either `hpyhex` or `hpyhex-rs` exclusively in a project. 
   &nbsp;
   If you are using built-in APIs in `hpyhex` to serialize data structures (e.g., `int(piece_value)`, `Piece(integer_value)`), you can load them back using `hpyhex-rs`, and vice versa. The byte representation of pieces is compatible between the two packages.
   &nbsp;
   However, if you use a python tool to serialize data structures from `hpyhex` as Python objects (e.g., `pickle`), you cannot load them back using `hpyhex-rs`, and vice versa. `hpyhex-rs` offers `serialize` and `deserialize` functions for its own data structures.
   &nbsp;
4. **Updates Can Lag Behind Original Package**
   &nbsp;
   This package currently targets the [0.2.0](https://pypi.org/project/hpyhex/0.2.0/) version of `hpyhex`. Features from later versions may not be fully supported yet, but may be added in future releases.

## Features
- Hexagonal grid representation
- Basic game mechanics for HappyHex
- Utility functions for hexagonal calculations

## Author
Developed by William Wu.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

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

## Main Classes

- **Hex**: Represents a hexagonal grid coordinate using a custom line-based system. Supports arithmetic, hashing, and tuple compatibility.
- **Piece**: Represents a shape made of 7 blocks, optimized for memory and performance. Use `PieceFactory` to create pieces by name or byte value.
- **HexEngine**: Manages the hexagonal grid, supports adding pieces, eliminating lines, and computing entropy.
- **PieceFactory**: Utility for creating pieces by name, byte, or randomly. Provides access to all predefined pieces.
- **Game**: Manages the game state, piece queue, score, and turn. Supports adding pieces and making moves with algorithms.

## Quickstart

1. Install the package:
   ```bash
   pip install hpyhex
   ```
2. Import and use the main classes as shown above.
3. Create custom algorithms to interact with the game environment.

## Benchmarking Algorithms

The package includes benchmarking tools to evaluate and compare algorithms:

```python
from hpyhex.benchmark import benchmark, compare

# Benchmark a single algorithm
avg_score, avg_turn = benchmark(simple_algorithm, engine_radius=3, queue_size=5, eval_times=10)

# Compare two algorithms
similarity = compare(alg1, alg2, engine_radius=3, queue_size=5, eval_times=100)
```
