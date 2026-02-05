For the Rust crate of the same name, see [hpyhex-rs on crates.io](https://crates.io/crates/hpyhex-rs) or [hpyhex-rs Rust Source](https://github.com/williamwutq/hpyhex-rs/tree/master/README.md).

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
- High performance through Rust implementation
- Native serialization and deserialization methods compatible with Rust hpyhex-rs crate
- NumPy integration for machine learning applications

## Author
Developed by William Wu.

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/williamwutq/hpyhex-rs/tree/master/py/LICENSE) file for details. Visit [MIT License](https://opensource.org/licenses/MIT) for more information about the MIT License. The choice of license is to offer compatibility with the original `hpyhex` package and the broader HappyHex project, which also uses the MIT License.

## Quickstart

1. Install the package:
   ```bash
   pip install hpyhex-rs
   ```
2. Import and use the main classes as shown above.
3. Create custom algorithms to interact with the game environment.

## Examples

See the [examples directory](https://github.com/williamwutq/hpyhex-rs/tree/master/py/examples/) for complete example scripts demonstrating various functionalities of the library, including basic usage, game simulations, serialization, and NumPy integration.

Also see the [benchmark directory](https://github.com/williamwutq/hpyhex-rs/tree/master/py/bench) for performance benchmarking code, which are excellent examples of the hpyhex API usage, supplemental to the simple [hpyhex](https://pypi.org/project/hpyhex/) documentation.

## Main Classes

- **Hex**: Represents a hexagonal grid coordinate using a custom line-based system. Supports arithmetic, hashing, and tuple compatibility.
- **Piece**: Represents a shape made of 7 blocks, optimized for memory and performance. Use `PieceFactory` to create pieces by name or byte value.
- **HexEngine**: Manages the hexagonal grid, supports adding pieces, eliminating lines, and computing entropy.
- **PieceFactory**: Utility for creating pieces by name, byte, or randomly. Provides access to all predefined pieces.
- **Game**: Manages the game state, piece queue, score, and turn. Supports adding pieces and making moves with algorithms.

## Hexagonal System

The `Hex` class represents a 2D coordinate in a hexagonal grid system using a specialized integer coordinate model. It supports both raw coordinate access and derived line-based computations across three axes: I, J, and K.

### Coordinate System

This system uses three axes (I, J, K) that run diagonally through the hexagonal grid:

- I+ is 60 degrees from J+, J+ is 60 degrees from K+, and K+ is 60 degrees from I-.
- Coordinates (i, k) correspond to a basis for representing any hexagon.
- **Raw coordinates** (or hex coordinates) refer to the distance of a point along one of the axes multiplied by 2.
- For raw coordinates, the relationships between the axes are defined such that `i - j + k = 0`.
- **Line coordinates** (or line-distance based coordinates) are based on the distance perpendicular to the axes.
- For line coordinates, the relationships between the axes are defined such that `I + J - K = 0`.
- All line coordinates correspond to some raw coordinate, but the inverse is not true. Due to the complexities with dealing with raw coordinates, it is preferable to use line coordinates. **The `hpyhex` API discourages the use of raw coordinates, and all its methods refers to line coordinates only, except those for backward compatibility.**

#### Coordinate System Visualization

Three example points with raw coordinates (2i, 2j, 2k):

```
   I
  / * (5, 4, -1)
 /     * (5, 7, 2)
o - - J
 \ * (0, 3, 3)
  \
   K
```

Three example points with line coordinates (I, J, K):

```
   I
  / * (1, 2, 3)
 /     * (3, 1, 4)
o - - J
 \ * (2, -1, 1)
  \
   K
```

### Grid Structure

- Uses an axial coordinate system (I, K) to represent hexagonal grids, where J = K - I.
- Three axes: I, J, K (not to be confused with 3D coordinates).
- Line-coordinates (I, K) are perpendicular distances to axes, calculated from raw coordinates.

### Grid Size

The total number of blocks in a hexagonal grid of radius `r` is calculated as:

```
Aₖ = 1 + 3*r*(r - 1)
```

This is derived from the recursive pattern:

```
Aₖ = Aₖ₋₁ + 6*(k - 1); A₁ = 1
```

Valid hexagonal grid sizes for common radii:
- Radius 0: 0 cell (Not valid for HexEngine, but can be valid for other purposes)
- Radius 1: 1 cell
- Radius 2: 7 cells
- Radius 3: 19 cells
- Radius 4: 37 cells
- Radius 5: 61 cells
- Radius 6: 91 cells
- Radius 10: 271 cells

### Hex Class Details

Represents a hexagonal grid coordinate using a custom line-based coordinate system.

This class models hexagonal positions with two line coordinates (i, k), implicitly defining the third axis (j) as `j = k - i` to maintain hex grid constraints. It supports standard arithmetic, equality, and hashing operations, as well as compatibility with coordinate tuples.

For small grids, Hex instances are cached for performance, allowing more efficient memory usage and faster access. The caching is limited to a range of -64 to 64 for both i and k coordinates.

Use of Hex over tuples is recommended for clarity and to leverage the singleton feature of small Hexes.

#### Attributes
- `i` (int): The line i coordinate.
- `j` (int): The computed line j coordinate (k - i).
- `k` (int): The line k coordinate.

#### Notes
- This class is immutable and optimized with `__slots__`.
- Raw coordinate methods (`__i__`, `__j__`, `__k__`) are retained for backward compatibility.
- Only basic functionality is implemented; complex adjacency, iteration, and mutability features are omitted for simplicity.

## Usage
```python
from hpyhex import Hex, Piece, HexEngine
from hpyhex import Game, PieceFactory, random_engine

# Create a hexagonal coordinate
coo = Hex(0, 1)

# Create a piece by name
piece = PieceFactory.get_piece("triangle_3_a")

# Create a game engine with radius 3
engine = HexEngine(3)

# Add a piece to the engine
engine.add_piece(piece, coo)

# Eliminate lines and get score
score = len(engine.eliminate()) * 5

# Create a game with engine radius and queue size
# Note that although the parameters are named engine and queue, they refer to the radius of the engine and the size of the piece queue respectively. They are not named engine_radius or queue_size.
game = Game(engine=3, queue=5)
print(game)

# Make a move using a custom algorithm
def simple_algorithm(engine, queue):
	# Always place the first piece at the center
	return 0, Hex(0, 0)
game.make_move(simple_algorithm)

# Serialize and save the game state compatibly with hpyhex-rs crate
serialized_engine = engine.hpyhex_rs_serialize()
serialized_pieces = [p.hpyhex_rs_serialize() for p in game.piece_queue]
with open("my_game_data.bin", "wb") as binary_file:
   binary_file.write(serialized_engine)
   for piece_bytes in serialized_pieces:
      binary_file.write(piece_bytes)

# Interact with NumPy
import numpy as np

# Convert a piece to a NumPy boolean array
piece_array = piece.to_numpy()

# Create a piece from a NumPy uint8 array
arr = np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.uint8)
new_piece = Piece.from_numpy_uint8(arr)

# Convert a random engine to a NumPy array
a_random_engine = random_engine(6)
engine_array = a_random_engine.to_numpy_uint32()

# Create an engine from a NumPy uint32 array
arr_engine = np.random.randint(0, 2, size=91, dtype=np.uint32) # Example for radius 6
new_engine = HexEngine.from_numpy_uint32(arr_engine, radius=6)

# Note that all dtypes listed in the NumPy Integration section are supported, and float16 is also supported if compiled with the "half" feature.
```

## Native Serialization

`hpyhex-rs` provides native serialization and deserialization methods for `HexEngine` and `Piece` classes, compatible with the Rust `hpyhex-rs` crate's `TryFrom<Vec<u8>>` and `Into<Vec<u8>>` implementations.

The serialization methods are named `hpyhex_rs_serialize()` and `hpyhex_rs_deserialize(data: bytes)`, and are available as instance methods for serialization and class methods for deserialization. The naming are prefixed with `hpyhex_rs_` to be future-proof against potential naming conflicts with other serialization methods that might be provided by the target package, `hpyhex`, in the future.

For examples, see [examples 1](./examples/01_binary_serialization.py) for serializing and deserializing `HexEngine`, `Piece`, and entire `Game` states using these methods. For more complex integration with other workflows, see other examples in the [examples directory](./examples/).

### Hex Serialization
- `hpyhex_rs_serialize() -> bytes`: Serializes the `Hex` coordinate into a byte vector.
- `hpyhex_rs_deserialize(data: bytes) -> Hex`: Deserializes a byte vector into a `Hex` instance.

### Piece Serialization
- `hpyhex_rs_serialize() -> bytes`: Serializes the `Piece` into a single byte representing the occupancy state of its blocks.
- `hpyhex_rs_deserialize(data: bytes) -> Piece`: Deserializes a byte vector into a `Piece` instance.

### HexEngine Serialization
- `hpyhex_rs_serialize() -> bytes`: Serializes the `HexEngine` into a byte vector. The format includes the radius as a 4-byte little-endian integer followed by the block states.
- `hpyhex_rs_deserialize(data: bytes) -> HexEngine`: Deserializes a byte vector into a `HexEngine` instance.

### Game Serialization
- `hpyhex_rs_serialize() -> bytes`: Serializes the `Game` into a byte vector. First, the score and turn of the `Game` are serialized into 4-byte little-edian integers, followed by the Vector of `Piece`s, and then the Game's engine with `HexEngine.hpyhex_rs_serialize`.
- `hpyhex_rs_deserialize(data: bytes) -> HexEngine`: Deserializes a byte vector into a `Game` instance, creating its own `HexEngine` instance.

## Native Representation

- `hpyhex_rs_render(prefix: str = "", suffix: str = "", space: str = " ", fill: str = "X", empty: str = "O") -> str`: An instance method on `HexEngine` that renders the current state of the hexagonal grid as a string. The grid is displayed with filled blocks represented by `fill`, empty blocks by `empty`, and spacing controlled by `space`. Optional `prefix` and `suffix` can be added to each line.

```python
>>> from hpyhex import random_engine, HexEngine
>>> print(random_engine(7).hpyhex_rs_render())
       X X O X O O X       
      X O O O X O O O      
     X X X X O O X O X     
    O O X O O X X O X O    
   O O O O O X X O X O O   
  O X O O X X O O X X X O  
 X X O X X X X O O O X O X 
  O O O X O O X O O O O X  
   O X X O X O X X O O X   
    O O O O X O O O X X    
     O X O X O X X O X     
      X X X O X X O X      
       O X X X O O O 
```

- `hpyhex_rs_render_external(values: List[Any], prefix: str = "", suffix: str = "", extra: bool = False) -> str`: A static method that renders a list of values into a string representation of a hexagonal grid. The values are arranged in a hexagonal pattern, with optional prefix and suffix for each line, and an extra line toggle for better alignment. Supports any Python object types as values, converting them to strings for display.

```python
>>> from hpyhex import HexEngine
>>> print(HexEngine.hpyhex_rs_render_external([2, 5, 6.7, 4, 3.65, 4, 5, 3, 0, 0, 45, [6,], 9, 3, 6, 6.7, "He", 4, 9], extra=True))
                                            
            2       5       6.7             
                                            
        4       3.65    4       5           
                                            
    3       0       0       45      [6]     
                                            
        9       3       6       6.7         
                                            
            He      4       9               
                                            
>>> import math
>>> print(HexEngine.hpyhex_rs_render_external(["Number", "PI?", ["r", 5], math.pi, 3, "Math", -5.42]))
                    Number              PI?                           
          ['r', 5]            3.14159265          3                   
                    Math                -5.42                         
```

## Native Methods

- `hpyhex_rs_add_piece_with_index(piece_index: int, position_index: int) -> bool`: A special method in the `Game` class that allows adding a piece using its index in the piece queue and the position index in the engine directly. This method is not part of the original `hpyhex` API but is provided for performance optimization.

- `hpyhex_rs_index_block(radius: int, coo: Hex) -> int`: A static method that retrieves the index of the block at the specified Hex coordinate for a given radius without needing a HexEngine instance. Returns the index or -1 if out of range. This avoids the need to create a HexEngine just to get an index, improving performance for batch operations.

- `hpyhex_rs_coordinate_block(radius: int, index: int) -> Hex`: A static method that retrieves the Hex coordinate of the block at the specified index for a given radius without needing a HexEngine instance. This simplifies coordinate calculation and may improve performance by avoiding unnecessary instance creation.

- `hpyhex_rs_adjacency_list(radius: int) -> List[List[int]]`: A static method that generates the adjacency list for blocks in a hexagonal grid of the specified radius. Each inner list contains the indices of neighboring blocks for the corresponding block. This provides direct access to adjacency information, enabling efficient batch workflows and eliminating redundant calculations across multiple HexEngine instances with the same radius.

- `hpyhex_rs_pair_vec_to_list_any(radius: int, sentinel: Any, values: List[Tuple[Any, Hex]]) -> List[Any]`: Converts a list of (value, Hex) pairs into a list aligned with the hexagonal grid, filling unfilled positions with the specified sentinel value. Supports any Python object types for values and sentinels. NumPy versions also exist for various dtypes with fixed sentinels (see [Pair Vector to List Conversion section](#pair-vector-to-list-conversion) for details).

See the [Adjacency Structure for HexEngine](#adjacency-structure-for-hexEngine) section for more details on how to use the adjacency list. The section describes usage of the adjacency list in the context of NumPy integration, but the same principles apply when using the native method.

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

The `hpyhex-rs` package provides a Rust-backed implementation of `check_positions`, which performs all operations in Rust memory, avoiding the overhead of Python object management and method calls. In the entire expensive process of checking all possible positions, the GIL is only acquired once. The radius is not passed as a Python object, but as a direct integer in the Rust struct. The nested loops are performed in Rust, and `Hex` objects are created directly as structs without going through Python constructors. Further, instead of calling the `check_add` method, a special version of `check_add` that takes in raw Rust structs representing `Hex` and `Piece` is used, avoiding the overhead of interacting with Python objects at all. These optimizations mean the Rust-backed `check_positions` is more than **100 times** faster than the native Python implementation, as per [benchmarking](https://github.com/williamwutq/hpyhex-rs/tree/master/py/bench/bench.py) results.

### Don't Reinvent the Wheel
It is tempting to implement your own versions of the various abstractions provided by this package, such as `Game`, which intuitively is just a combination of `HexEngine` and a piece queue, and does not offer too much extra customization. Unless your purpose is different from the original intention of `hpyhex`, it is recommended to use the provided `Game` class directly, as it interacts with the optimized Rust versions of `HexEngine` and `PieceFactory` without the overhead of creating intermediate Python objects. For extra functionality, consider building on top of `Game` instead of re-implementing it completely.

### Not Enough for GUI Applications
If you are building a GUI application for a simple version of HappyHex and deeply hated the original Java codebase, you possibly have pondered upon this package for performance, as it advertises itself as a high-performance implementation of the Python `hpyhex` package, which has a simple and useful API. Unless you already did a lot of work in Python, however, you should not use Python for your GUI applications, as it is not well-suited for GUI development and may lead to performance issues and a poor user experience. The [hpyhex-rs](https://crates.io/crates/hpyhex-rs) Rust crate, which is inspired by the Python API, not only provides similar functionality and abstractions, which make your transition to that package easier, but also provides further abstractions such as thread-safe guards, extended HexEngine with potential attributes for each cell, and an integrated game environment designed specifically for GUI threading needs. Consider using Rust as your main programming language for GUI applications, or integrate with C++ via FFI to use existing C++ GUI frameworks.

## The Statistics

(See [bench directory](https://github.com/williamwutq/hpyhex-rs/tree/master/py/bench/) for full benchmarking code and results.)

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

## NumPy Integration

`hpyhex-rs` provides NumPy integration for machine learning and development of fast game-playing heuristics agents. This is what makes `hpyhex-rs` stand out from the original `hpyhex` package, which does not provide any NumPy integration.

### Installation

The default pre-built wheels on PyPI include NumPy support. Simply install via pip:
```bash
pip install hpyhex-rs
```

Or if building from source, enable the `numpy` feature in your `Cargo.toml`.

### Experimental Features

Float16 (half precision) support is experimental and requires enabling the `half` feature flag during build. To use float16 serialization methods, ensure you have NumPy installed with float16 support, and compile the library from source with the `half` feature enabled:

```toml
[dependencies.hpyhex-rs]
version = "..."
features = ["numpy", "half"]
```

Note that the feature is experimental and not officially supported nor tested extensively. On machines that does not support float16 or installed with a version of numpy that does not support float16, this function may lead to undefined behavior or crashes. Those unintended behaviors could be subtle and hard to debug, so even if code with this feature seems to work, make sure to check the output as it has known to misintepret memory or lead to silent data corruption in some cases.

### Examples

See the [examples directory](https://github.com/williamwutq/hpyhex-rs/tree/master/py/examples/) for complete example scripts demonstrating NumPy integration. These examples cover converting `HexEngine` and `Piece` objects to and from NumPy arrays, serializing game states, and integrating with machine learning workflows. The demonstrations include both basic usage and advanced scenarios. The example framework is PyTorch for machine learning, but the NumPy integration is framework-agnostic and can be used with any library that supports NumPy arrays (which are most of them).

### No Serialization for Hex

Hex has no need for serialization to numpy arrays, as it is just a coordinate container. Batch serialization of hex coordinates are needed, but an array of hexagonal coordinates only has meaning in the context of a grid, which is either a HexEngine or a Piece. Therefore, serialization from and to NumPy is only implemented for HexEngine or a Piece, but not Hex.

### Serialization for Piece

The `Piece` class provides efficient conversion to and from NumPy arrays representing its 7 block states. All conversions produce or consume 1-dimensional arrays of shape `(7,)`, where each element represents whether the corresponding block is occupied.

#### Converting to NumPy

The `to_numpy()` method returns a boolean array by default:
```python
from hpyhex import PieceFactory
import numpy as np

piece = PieceFactory.get_piece("triangle_3_a")

# Default: boolean array
arr = piece.to_numpy()
# arr.dtype == np.bool_
# arr.shape == (7,)
# arr = [True, True, False, True, False, False, False]
```

For specific numeric types, use the typed conversion methods:
```python
# Integer types
arr_i8 = piece.to_numpy_int8()      # dtype: int8
arr_u8 = piece.to_numpy_uint8()     # dtype: uint8
arr_i16 = piece.to_numpy_int16()    # dtype: int16
arr_u16 = piece.to_numpy_uint16()   # dtype: uint16
arr_i32 = piece.to_numpy_int32()    # dtype: int32
arr_u32 = piece.to_numpy_uint32()   # dtype: uint32
arr_i64 = piece.to_numpy_int64()    # dtype: int64
arr_u64 = piece.to_numpy_uint64()   # dtype: uint64

# Floating point types
arr_f32 = piece.to_numpy_float32()  # dtype: float32
arr_f64 = piece.to_numpy_float64()  # dtype: float64

# Half precision (requires "half" feature, experimental)
arr_f16 = piece.to_numpy_half()     # dtype: float16
```

#### Converting from NumPy

Use the corresponding `from_numpy_*` methods to construct a Piece from a NumPy array. The array must have shape `(7,)` and the appropriate dtype. For unsigned integer types, non-zero values are treated as occupied blocks, for signed integers and floating point types, positive values are treated as occupied blocks and zero or negative values as empty blocks. This design aims to make conversion from a softmax output of a neural network straightforward.
```python
# From boolean array
arr = np.array([True, True, True, False, False, False, False])
piece = Piece.from_numpy_bool(arr)

# From integer arrays
arr_u8 = np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.uint8)
piece = Piece.from_numpy_uint8(arr_u8)

arr_i32 = np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.int32)
piece = Piece.from_numpy_int32(arr_i32)

# From floating point arrays
arr_f64 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
piece = Piece.from_numpy_float64(arr_f64)
```

#### Validation and Error Handling

All `from_numpy_*` methods validate the input array:

- **Shape validation**: Array must have exactly shape `(7,)` 
- **Type validation**: Array dtype must match the method's expected type

If validation fails, a `ValueError` is raised:
```python
# Wrong shape
arr = np.array([1, 1, 1, 0, 0])  # Only 5 elements
try:
    piece = Piece.from_numpy_uint8(arr)
except ValueError as e:
    print(f"Error: {e}")  # Shape mismatch
# Wrong dtype
arr = np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.float32)
try:
    piece = Piece.from_numpy_uint8(arr)
except ValueError as e:
    print(f"Error: {e}")  # Dtype mismatch
```

#### Type Casting Considerations

NumPy arrays cannot be easily cast between types at the Rust/Python boundary. Therefore, **there is no universal `from_numpy()` method**. You must use the specific typed method that matches your array's dtype:
```python
# No automatic type detection
arr = np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.int32)
# piece = Piece.from_numpy(arr)  # This method doesn't exist!

# Use the typed method matching your dtype
piece = Piece.from_numpy_int32(arr)

# If you need to convert between types, do it in NumPy first:
arr_f32 = arr.astype(np.float32) # Note that Numpy does a copy here
piece = Piece.from_numpy_float32(arr_f32)
```

#### Zero Copy

There is no need for zero-copy conversion between NumPy arrays and `Piece` objects, as the data size is only 7 bytes. In addition, since Pieces are optimized with a fixed cache of pre-defined objects, they are already "zero-copy" in a sense that no new memory allocation is needed when creating a Piece from its byte representation. Therefore, all conversions involve "copying" data between the NumPy array and the Piece object.

#### Supported Data Types

The following table summarizes all supported NumPy dtypes for Piece serialization:

| NumPy dtype | `to_numpy_*` method     | `from_numpy_*` method    | Notes                                  |
|-------------|-------------------------|--------------------------|----------------------------------------|
| `bool_`     | `to_numpy()` (default)  | `from_numpy_bool()`      | Most memory efficient                  |
| `int8`      | `to_numpy_int8()`       | `from_numpy_int8()`      | Signed 8-bit integer                   |
| `uint8`     | `to_numpy_uint8()`      | `from_numpy_uint8()`     | Unsigned 8-bit integer                 |
| `int16`     | `to_numpy_int16()`      | `from_numpy_int16()`     | Signed 16-bit integer                  |
| `uint16`    | `to_numpy_uint16()`     | `from_numpy_uint16()`    | Unsigned 16-bit integer                |
| `int32`     | `to_numpy_int32()`      | `from_numpy_int32()`     | Signed 32-bit integer                  |
| `uint32`    | `to_numpy_uint32()`     | `from_numpy_uint32()`    | Unsigned 32-bit integer                |
| `int64`     | `to_numpy_int64()`      | `from_numpy_int64()`     | Signed 64-bit integer                  |
| `uint64`    | `to_numpy_uint64()`     | `from_numpy_uint64()`    | Unsigned 64-bit integer                |
| `float16`   | `to_numpy_half()`       | `from_numpy_half()`      | Requires "half" feature (experimental) |
| `float32`   | `to_numpy_float32()`    | `from_numpy_float32()`   | Common for ML applications             |
| `float64`   | `to_numpy_float64()`    | `from_numpy_float64()`   | Double precision                       |

**Recommended types:**
- Use `bool_` for minimal memory footprint or in machine learning
- Use `uint8` for serialization to compact integer formats
- Use `float32` for general machine learning (PyTorch, TensorFlow default)

### Serialization for Vector of Piece (Piece Queues)

The `Piece` class provides efficient conversion to and from NumPy arrays for collections of pieces, commonly used for piece queues in game states. All conversions work with lists of `Piece` objects.

#### Converting to NumPy

The `vec_to_numpy_flat()` method returns a flattened 1D boolean array by default, concatenating all pieces' block states:
```python
from hpyhex import PieceFactory
import numpy as np

pieces = [
   PieceFactory.get_piece("triangle_3_a"),
   PieceFactory.get_piece("triangle_3_b"),
   PieceFactory.get_piece("corner_3_a")
]

# Default: flattened boolean array
arr = Piece.vec_to_numpy_flat(pieces)
# arr.dtype == np.bool_
# arr.shape == (21,)  # 3 pieces * 7 blocks each
```

For stacked representation, use `vec_to_numpy_stacked()` which returns a 2D array:
```python
# Stacked: 2D boolean array
arr_2d = Piece.vec_to_numpy_stacked(pieces)
# arr_2d.dtype == np.bool_
# arr_2d.shape == (3, 7)  # (num_pieces, 7)
# arr_3d.stride == (8, 1)  # row-major order, padded for alignment
```

For specific numeric types, use the typed conversion methods:
```python
# Integer types (flat)
arr_i8_flat = Piece.vec_to_numpy_int8_flat(pieces)      # dtype: int8
arr_u8_flat = Piece.vec_to_numpy_uint8_flat(pieces)     # dtype: uint8
arr_i16_flat = Piece.vec_to_numpy_int16_flat(pieces)    # dtype: int16
arr_u16_flat = Piece.vec_to_numpy_uint16_flat(pieces)   # dtype: int16
arr_i32_flat = Piece.vec_to_numpy_int32_flat(pieces)    # dtype: int32
arr_u32_flat = Piece.vec_to_numpy_uint32_flat(pieces)   # dtype: uint32
arr_i64_flat = Piece.vec_to_numpy_int64_flat(pieces)    # dtype: int64
arr_u64_flat = Piece.vec_to_numpy_uint64_flat(pieces)   # dtype: uint64

# Integer types (stacked)
arr_i8_stacked = Piece.vec_to_numpy_int8_stacked(pieces)  # shape: (3, 7)
arr_u8_stacked = Piece.vec_to_numpy_uint8_stacked(pieces) # shape: (3, 7)
arr_i16_stacked = Piece.vec_to_numpy_int16_stacked(pieces) # shape: (3, 7)
arr_u16_stacked = Piece.vec_to_numpy_uint16_stacked(pieces) # shape: (3, 7)
arr_i32_stacked = Piece.vec_to_numpy_int32_stacked(pieces) # shape: (3, 7)
arr_u32_stacked = Piece.vec_to_numpy_uint32_stacked(pieces) # shape: (3, 7)
arr_i64_stacked = Piece.vec_to_numpy_int64_stacked(pieces) # shape: (3, 7)
arr_u64_stacked = Piece.vec_to_numpy_uint64_stacked(pieces) # shape: (3, 7)

# Floating point types (flat)
arr_f32_flat = Piece.vec_to_numpy_float32_flat(pieces)  # dtype: float32
arr_f64_flat = Piece.vec_to_numpy_float64_flat(pieces)  # dtype: float64

# Floating point types (stacked)
arr_f32_stacked = Piece.vec_to_numpy_float32_stacked(pieces)  # shape: (3, 7)
arr_f64_stacked = Piece.vec_to_numpy_float64_stacked(pieces)  # shape: (3, 7)

# Half precision (requires "half" feature, experimental)
arr_f16_flat = Piece.vec_to_numpy_float16_flat(pieces)      # dtype: float16
arr_f16_stacked = Piece.vec_to_numpy_float16_stacked(pieces)  # shape: (3, 7)
```

#### Converting from NumPy

Use the corresponding `vec_from_numpy_*` methods to construct a list of `Piece` objects from NumPy arrays.

For flat arrays (1D), the array length must be a multiple of 7:
```python
# From flat boolean array
arr_flat = np.array([True, True, False, True, False, False, False,  # piece 1
                     True, True, False, False, False, False, False,  # piece 2
                     True, True, True, False, False, False, False])  # piece 3
pieces = Piece.vec_from_numpy_bool_flat(arr_flat)
print(len(pieces))  # 3

# From flat integer arrays
arr_u8_flat = np.array([1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], dtype=np.uint8)
pieces = Piece.vec_from_numpy_uint8_flat(arr_u8_flat)

# From flat floating point arrays
arr_f32_flat = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
pieces = Piece.vec_from_numpy_float32_flat(arr_f32_flat)
```

For stacked arrays (2D), the shape must be `(num_pieces, 7)`:
```python
# From stacked boolean array
arr_stacked = np.array([[True, True, False, True, False, False, False],
                        [True, True, False, False, False, False, False],
                        [True, True, True, False, False, False, False]], dtype=bool)
pieces = Piece.vec_from_numpy_bool_stacked(arr_stacked)

# From stacked integer arrays
arr_i32_stacked = np.array([[1, 1, 0, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0, 0]], dtype=np.int32)
pieces = Piece.vec_from_numpy_int32_stacked(arr_i32_stacked)

# From stacked floating point arrays
arr_f64_stacked = np.array([[1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
pieces = Piece.vec_from_numpy_float64_stacked(arr_f64_stacked)
```

#### Validation and Error Handling

All `vec_from_numpy_*` methods validate the input array:

- **Shape validation**: For flat arrays, length must be a multiple of 7. For stacked arrays, shape must be `(n, 7)` where `n >= 1`
- **Type validation**: Array dtype must match the method's expected type

If validation fails, a `ValueError` is raised:
```python
# Wrong length for flat array
arr = np.array([1, 1, 1, 0, 0])  # 5 elements, not multiple of 7
try:
    pieces = Piece.vec_from_numpy_uint8_flat(arr)
except ValueError as e:
    print(f"Error: {e}")  # Invalid array length

# Wrong shape for stacked array
arr = np.array([[1, 1, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0]])  # Second row has only 6 elements
try:
    pieces = Piece.vec_from_numpy_uint8_stacked(arr)
except ValueError as e:
    print(f"Error: {e}")  # Shape mismatch
```

#### Type Casting Considerations

NumPy arrays cannot be easily cast between types at the Rust/Python boundary. Therefore, **there is no universal `vec_from_numpy()` method**. You must use the specific typed method that matches your array's dtype:
```python
# No automatic type detection
arr = np.array([1, 1, 1, 0, 0, 0, 0,
                1, 1, 0, 0, 0, 0, 0], dtype=np.int32)
# pieces = Piece.vec_from_numpy(arr)  # This method doesn't exist!

# Use the typed method matching your dtype
pieces = Piece.vec_from_numpy_int32_flat(arr)

# If you need to convert between types, do it in NumPy first:
arr_f32 = arr.astype(np.float32) # Note that Numpy does a copy here
pieces = Piece.vec_from_numpy_float32_flat(arr_f32)
```

1D (flat) and 2D (stacked) representations are not interchangeable. You must use the appropriate method for the array shape you have. Casting between these two will copy data and may impact performance.
This is because the internal memory layout differs: flat arrays are contiguous 1D arrays, while stacked arrays have row-major order with potential padding for alignment. To convert between flat and stacked representations, do so in NumPy before passing to the Rust methods:
```python
# Convert flat to stacked in NumPy
arr_flat = np.array([...], dtype=np.bool_)  # shape: (n*7,)
num_pieces = arr_flat.shape[0] // 7
arr_stacked = arr_flat.reshape((num_pieces, 7))  # shape: (n, 7)
pieces = Piece.vec_from_numpy_bool_stacked(arr_stacked)

# Convert stacked to flat in NumPy
arr_stacked = np.array([...], dtype=np.bool_)  # shape: (n, 7)
arr_flat = arr_stacked.reshape((-1,))  # shape: (n*7,)
pieces = Piece.vec_from_numpy_bool_flat(arr_flat)
```

#### Zero Copy

For the same reason as single Piece serialization, there is no need for zero-copy conversion between NumPy arrays and lists of `Piece` objects.

#### Supported Data Types

The following table summarizes all supported NumPy dtypes for vector of pieces serialization:

| NumPy dtype | `vec_to_numpy_*_flat`           | `vec_to_numpy_*_stacked`           | `vec_from_numpy_*_flat`         | `vec_from_numpy_*_stacked`         | Notes                                   |
|-------------|---------------------------------|------------------------------------|---------------------------------|------------------------------------|-----------------------------------------|
| `bool_`     | `vec_to_numpy_flat()` (default) | `vec_to_numpy_stacked()` (default) | `vec_from_numpy_bool_flat()`    | `vec_from_numpy_bool_stacked()`    | Most memory efficient                   |
| `int8`      | `vec_to_numpy_int8_flat()`      | `vec_to_numpy_int8_stacked()`      | `vec_from_numpy_int8_flat()`    | `vec_from_numpy_int8_stacked()`    | Signed 8-bit integer                    |
| `uint8`     | `vec_to_numpy_uint8_flat()`     | `vec_to_numpy_uint8_stacked()`     | `vec_from_numpy_uint8_flat()`   | `vec_from_numpy_uint8_stacked()`   | Unsigned 8-bit integer                  |
| `int16`     | `vec_to_numpy_int16_flat()`     | `vec_to_numpy_int16_stacked()`     | `vec_from_numpy_int16_flat()`   | `vec_from_numpy_int16_stacked()`   | Signed 16-bit integer                   |
| `uint16`    | `vec_to_numpy_uint16_flat()`    | `vec_to_numpy_uint16_stacked()`    | `vec_from_numpy_uint16_flat()`  | `vec_from_numpy_uint16_stacked()`  | Unsigned 16-bit integer                 |
| `int32`     | `vec_to_numpy_int32_flat()`     | `vec_to_numpy_int32_stacked()`     | `vec_from_numpy_int32_flat()`   | `vec_from_numpy_int32_stacked()`   | Signed 32-bit integer                   |
| `uint32`    | `vec_to_numpy_uint32_flat()`    | `vec_to_numpy_uint32_stacked()`    | `vec_from_numpy_uint32_flat()`  | `vec_from_numpy_uint32_stacked()`  | Unsigned 32-bit integer                 |
| `int64`     | `vec_to_numpy_int64_flat()`     | `vec_to_numpy_int64_stacked()`     | `vec_from_numpy_int64_flat()`   | `vec_from_numpy_int64_stacked()`   | Signed 64-bit integer                   |
| `uint64`    | `vec_to_numpy_uint64_flat()`    | `vec_to_numpy_uint64_stacked()`    | `vec_from_numpy_uint64_flat()`  | `vec_from_numpy_uint64_stacked()`  | Unsigned 64-bit integer                 |
| `float16`   | `vec_to_numpy_float16_flat()`   | `vec_to_numpy_float16_stacked()`   | `vec_from_numpy_float16_flat()` | `vec_from_numpy_float16_stacked()` | Requires "half" feature (experimental)  |
| `float32`   | `vec_to_numpy_float32_flat()`   | `vec_to_numpy_float32_stacked()`   | `vec_from_numpy_float32_flat()` | `vec_from_numpy_float32_stacked()` | Common for ML applications              |
| `float64`   | `vec_to_numpy_float64_flat()`   | `vec_to_numpy_float64_stacked()`   | `vec_from_numpy_float64_flat()` | `vec_from_numpy_float64_stacked()` | Double precision                        |

**Recommended types:**
- Use `bool_` for minimal memory footprint
- Use `uint8` for compact integer formats
- Use `float32` for machine learning applications

### Serialization for HexEngine

The `HexEngine` class provides comprehensive NumPy integration for converting hexagonal game boards to and from array representations. All conversions produce or consume 1-dimensional arrays where the length corresponds to the total number of cells in the hexagonal grid (for a radius `r`, this is `3r² + 3r + 1` cells). See original `hpyhex` documentation for details on hexagonal grid sizing.

#### Array Shape and Grid Mapping

Unlike rectangular grids, hexagonal grids don't map naturally to 2D arrays. The `HexEngine` uses a **flattened 1D representation** where each index corresponds to a specific hexagonal cell:
```python
from hpyhex import HexEngine

engine = HexEngine(radius=4)
# Array shape will be (37,)

arr = engine.to_numpy()
print(arr.shape)  # (37,)
```

The mapping from array index to hexagonal coordinate is determined by the `index_block()` and `coordinate_block()` methods:
```python
# Get the hex coordinate for array index 10
hex_coord = engine.coordinate_block(10)

# Get the array index for a hex coordinate
index = engine.index_block(hex_coord)
```

#### Converting to NumPy

The `to_numpy()` method returns a boolean array by default:
```python
from hpyhex import HexEngine, Hex, PieceFactory

engine = HexEngine(radius=4)
piece = PieceFactory.get_piece("triangle_3_a")
engine.add_piece(piece, Hex(0, 0))

# Default: boolean array representing occupied/empty cells
arr = engine.to_numpy()
# arr.dtype == np.bool_
# arr.shape == (37,)
# arr[i] = True if cell i is occupied, False otherwise
```

For specific numeric types, use the typed conversion methods:
```python
# Integer types
arr_i8 = engine.to_numpy_int8()      # dtype: int8, values 0 or 1
arr_u8 = engine.to_numpy_uint8()     # dtype: uint8, values 0 or 1
arr_i16 = engine.to_numpy_int16()    # dtype: int16, values 0 or 1
arr_u16 = engine.to_numpy_uint16()   # dtype: uint16, values 0 or 1
arr_i32 = engine.to_numpy_int32()    # dtype: int32, values 0 or 1
arr_u32 = engine.to_numpy_uint32()   # dtype: uint32, values 0 or 1
arr_i64 = engine.to_numpy_int64()    # dtype: int64, values 0 or 1
arr_u64 = engine.to_numpy_uint64()   # dtype: uint64, values 0 or 1

# Floating point types
arr_f32 = engine.to_numpy_float32()  # dtype: float32, values 0.0 or 1.0
arr_f64 = engine.to_numpy_float64()  # dtype: float64, values 0.0 or 1.0

# Half precision (requires "half" feature, experimental)
arr_f16 = engine.to_numpy_float16()  # dtype: float16, values 0.0 or 1.0
```

#### Converting from NumPy

Use the corresponding `from_numpy_*` methods to construct a `HexEngine` from a NumPy array. The array length must correspond to a valid hexagonal grid size, and the dtype must match the method. Internally, non-zero values are treated as occupied cells for integer types, and positive values are treated as occupied cells for floating point types. Values are copied into a new HexEngine instance, which is managed independently of the NumPy array.
```python
import numpy as np
from hpyhex import HexEngine

# From boolean array (radius automatically inferred from length)
arr = np.zeros(37, dtype=bool)  # 37 cells = radius 4
arr[0] = True
arr[5] = True
engine = HexEngine.from_numpy_bool(arr)
print(engine.radius)  # 4

# From integer arrays (non-zero values treated as occupied)
arr_u8 = np.array([1, 0, 1, 0, 1] + [0]*32, dtype=np.uint8)
engine = HexEngine.from_numpy_uint8(arr_u8)

arr_i32 = np.ones(37, dtype=np.int32)
engine = HexEngine.from_numpy_int32(arr_i32)  # Fully occupied board

# From floating point arrays (values > 0.0 treated as occupied)
arr_f64 = np.random.rand(37)  # Random values [0, 1)
engine = HexEngine.from_numpy_float64(arr_f64)
# Cells with values > 0.0 will be occupied
```

#### Validation and Error Handling

All `from_numpy_*` methods perform validation on the input array:

- **Length validation**: Array length must correspond to a valid hexagonal grid (i.e., `length = 3r² + 3r + 1` for some non-negative integer `r`)
- **Type validation**: Array dtype must match the method's expected type

If validation fails, a `ValueError` is raised:
```python
# Wrong length (not a valid hexagonal grid size)
arr = np.zeros(40, dtype=bool)  # 40 is not a valid hex grid size
try:
    engine = HexEngine.from_numpy_bool(arr)
except ValueError as e:
    print(f"Error: {e}")  # Invalid array length for hexagonal grid

# Wrong dtype
arr = np.zeros(37, dtype=np.float32)
try:
    engine = HexEngine.from_numpy_uint8(arr)  # Expects uint8, got float32
except ValueError as e:
    print(f"Error: {e}")  # Type mismatch
```

Valid hexagonal grid sizes for common radii:
- Radius 1: 7 cells
- Radius 2: 19 cells
- Radius 3: 37 cells
- Radius 4: 61 cells
- Radius 5: 91 cells
- Radius 10: 331 cells

#### Unchecked Conversions for Performance

For performance-critical code where you're certain the input is valid, use the `*_unchecked` variants. These skip validation but require the array length to be a valid hexagonal grid size. Note that copying still occurs and these methods are memory safe as long as the input array is valid.
```python
# Unchecked conversion (faster, but unsafe if array is invalid)
arr = np.zeros(37, dtype=bool)
engine = HexEngine.from_numpy_bool_unchecked(arr)  # No validation

# Available for all types:
engine = HexEngine.from_numpy_uint8_unchecked(arr_u8)
engine = HexEngine.from_numpy_int32_unchecked(arr_i32)
engine = HexEngine.from_numpy_float64_unchecked(arr_f64)
# ... and so on
```

**Warning**: Using `*_unchecked` methods with invalid array lengths will cause undefined behavior, potentially leading to runtime errors or panics later in your program.

#### Zero-Copy View (Advanced)

For maximum performance in specialized scenarios, `from_numpy_raw_view` creates a HexEngine that directly references the NumPy array's memory without copying:
```python
arr = np.zeros(37, dtype=bool)
engine = HexEngine.from_numpy_raw_view(arr)  # Zero-copy, extremely fast

# Modifying arr also modifies engine (they share memory!)
arr[10] = True
# engine's state at index 10 is now also True
```

The array must be a 1 dimension boolean NumPy array of valid hexagonal grid length.

**Critical Safety Requirements** for `from_numpy_raw_view`:

1. **Array length must correspond to a valid hexagonal grid size** - The method assumes the provided NumPy array length corresponds to a valid hexagonal grid size and does not perform any checks. If the length is invalid or zero, the behavior is undefined and may cause runtime errors or panics later in your program.
2. **Array must be contiguous** in memory - If the array is not contiguous, the function will panic.
3. **Array must be host (CPU) memory** - The array must be allocated on host (CPU) memory. If allocated on a different device (e.g., GPU), accessing its memory directly from Rust will lead to undefined behavior or mysterious crashes.
4. **Memory layout compatibility** - The array's memory must be allocated in a way that is compatible with Rust's `Vec<bool>` memory layout. This means it must not be padded or aligned in a way that would be incompatible with Rust's expectations.
5. **Array must not be used elsewhere** after calling this method - Since the function takes a view of the data, any further use of the original NumPy array will lead to undefined behavior, including potential crashes or data corruption.
6. **Engine lifetime must not exceed array lifetime** - The lifetime of the HexEngine must not exceed that of the original NumPy array in both Python and NumPy memory management. If this is violated, it is highly likely that garbage data or segmentation faults will occur when accessing the HexEngine's states.
7. **Array must be mutable and not shared** across threads - If the NumPy array is shared across multiple references or threads, modifying it in Rust could lead to data corruption or race conditions.

Similarly, `to_numpy_raw_view` creates a NumPy array that directly references the HexEngine's memory without copying:
```python
from hpyhex import HexEngine

engine = HexEngine(radius=3)
arr = engine.to_numpy_raw_view()  # Zero-copy, extremely fast

# Modifying arr also modifies engine (they share memory!)
arr[10] = True
# engine's state at index 10 is now also True
```

**Critical Safety Requirements** for `to_numpy_raw_view`:

The following conditions must be met for safe usage:

It is assumed that the HexEngine contains a valid hexagonal grid state and does not perform any checks.

The method also assumes that the memory of the HexEngine's states:

- Is compatible with NumPy's memory layout. This means that NumPy must be able to interpret the HexEngine's internal memory representation correctly as a NumPy array of the expected dtype and shape, and must not expect special padding or alignment that is not present.
- Is not used elsewhere after this function is called. Since the function takes a view of the data, any further use of the original HexEngine will lead to undefined behavior, including potential crashes or data corruption.
- Is mutable and not shared. If the HexEngine's states are shared across multiple references or threads, modifying it in NumPy could lead to data corruption or race conditions.
- Has a lifetime that does not exceed that of the HexEngine in both Python and Rust memory management. If this is violated, it is highly likely that garbage data or segmentation faults will occur when accessing the NumPy array's data.

**Double-Free Memory Management Issue**: Under normal conditions, even if all the above conditions are met, these methods will eventually lead to a double-free error when both Rust and Python attempt to free the same memory during their respective deallocation processes. To prevent this, manually increment the reference count of either the NumPy array or the HexEngine instance in Python using methods like `ctypes.pythonapi.Py_IncRef` to ensure that only one of them is responsible for freeing the memory. If this is undesirable, consider holding references to both objects until the end of the program execution so that all double-free errors occur only at program termination.

Violating these requirements leads to undefined behavior including segmentation faults, data corruption, or mysterious crashes. **Use `from_numpy_bool()` and `to_numpy_bool()` instead unless performance is absolutely critical and you understand the risks.**

#### Type Casting Considerations

NumPy arrays cannot be easily cast between types at the Rust/Python boundary. Therefore, **there is no universal `from_numpy()` method**. You must use the specific typed method matching your array's dtype:
```python
# No automatic type detection
arr = np.ones(37, dtype=np.int32)
# engine = HexEngine.from_numpy(arr)  # This method doesn't exist!

# Use the typed method matching your dtype
engine = HexEngine.from_numpy_int32(arr)

# If you need to convert between types, do it in NumPy first:
arr_f32 = arr.astype(np.float32)
engine = HexEngine.from_numpy_float32(arr_f32)
```

#### Supported Data Types

The following table summarizes all supported NumPy dtypes for HexEngine serialization:

| NumPy dtype | `to_numpy_*` method     | `from_numpy_*` method  | `from_numpy_*_unchecked`         | Notes                                  |
|-------------|-------------------------|------------------------|----------------------------------|----------------------------------------|
| `bool_`     | `to_numpy()` (default)  | `from_numpy_bool()`    | `from_numpy_bool_unchecked()`    | Boolean representation                 |
| `int8`      | `to_numpy_int8()`       | `from_numpy_int8()`    | `from_numpy_int8_unchecked()`    | Signed 8-bit integer                   |
| `uint8`     | `to_numpy_uint8()`      | `from_numpy_uint8()`   | `from_numpy_uint8_unchecked()`   | Unsigned 8-bit integer                 |
| `int16`     | `to_numpy_int16()`      | `from_numpy_int16()`   | `from_numpy_int16_unchecked()`   | Signed 16-bit integer                  |
| `uint16`    | `to_numpy_uint16()`     | `from_numpy_uint16()`  | `from_numpy_uint16_unchecked()`  | Unsigned 16-bit integer                |
| `int32`     | `to_numpy_int32()`      | `from_numpy_int32()`   | `from_numpy_int32_unchecked()`   | Signed 32-bit integer                  |
| `uint32`    | `to_numpy_uint32()`     | `from_numpy_uint32()`  | `from_numpy_uint32_unchecked()`  | Unsigned 32-bit integer                |
| `int64`     | `to_numpy_int64()`      | `from_numpy_int64()`   | `from_numpy_int64_unchecked()`   | Signed 64-bit integer                  |
| `uint64`    | `to_numpy_uint64()`     | `from_numpy_uint64()`  | `from_numpy_uint64_unchecked()`  | Unsigned 64-bit integer                |
| `float16`   | `to_numpy_float16()`    | `from_numpy_float16()` | `from_numpy_float16_unchecked()` | Requires "half" feature (experimental) |
| `float32`   | `to_numpy_float32()`    | `from_numpy_float32()` | `from_numpy_float32_unchecked()` | Common for ML applications             |
| `float64`   | `to_numpy_float64()`    | `from_numpy_float64()` | `from_numpy_float64_unchecked()` | Double precision                       |

**Recommended types:**
- Use `bool_` for minimal memory footprint or in machine learning
- Use `uint8` for serialization to compact integer formats
- Use `float32` for general machine learning (PyTorch, TensorFlow default)

**Special note on `from_numpy_raw_view` and `to_numpy_raw_view`:**
Only `from_numpy_raw_view()` is available for zero-copy views, and it only works with `bool_` dtype arrays. This is the only method converting from NumPy that doesn't copy data, but it comes with significant safety requirements as documented above. Similarly, `to_numpy_raw_view()` only produces `bool_` dtype arrays, and requires careful management to avoid double-free errors.

#### Positions Mask

The `HexEngine` provides methods to get NumPy arrays indicating valid positions for adding a specific piece. These masks are 1D arrays where each index corresponds to a hexagonal cell, and the value indicates whether that position is valid for placing the given piece.

```python
from hpyhex import HexEngine, PieceFactory

engine = HexEngine(radius=3)
piece = PieceFactory.get_piece("triangle_3_a")

# Get boolean mask of valid positions
mask = engine.to_numpy_positions_mask(piece)
# mask.shape == (19,)
# mask[i] = True if piece can be placed at cell i

# Available for all numeric types
mask_u8 = engine.to_numpy_positions_mask_uint8(piece)   # uint8, 0 or 1
mask_f32 = engine.to_numpy_positions_mask_float32(piece) # float32, 0.0 or 1.0
# ... and all other types including float16 (requires "half" feature)
```

These methods are useful for game logic, AI decision making, and visualization of possible moves.

### Pair Vector to List Conversion

The `HexEngine` provides methods to convert a list of (value, Hex coordinate) pairs into 1D Numpy arrays aligned with the hexagonal grid.

This functionality is useful for creating custom board representations, feature maps, or any scenario where you need to map values to specific grid positions. It allows sparse specification of values at specific coordinates, automatically filling the rest of the grid with a sentinel value determined by the data type.

The indexing of the resulting data structures is equivalent to `index_block()`, where each index corresponds to a specific Hex coordinate in the grid.

#### Getting NumPy Arrays

```python
# Default int16 with sentinel i16::MAX
arr_i16 = HexEngine.pair_vec_to_numpy(radius=3, values=pairs)

# Specific dtypes
arr_u8 = HexEngine.pair_vec_to_numpy_uint8(radius=3, values=pairs)    # uint8, sentinel u8::MAX
arr_i32 = HexEngine.pair_vec_to_numpy_int32(radius=3, values=pairs)   # int32, sentinel i32::MAX
arr_f32 = HexEngine.pair_vec_to_numpy_float32(radius=3, values=pairs) # float32, sentinel f32::NAN
arr_f64 = HexEngine.pair_vec_to_numpy_float64(radius=3, values=pairs) # float64, sentinel f64::NAN

# Half precision (requires "half" feature)
arr_f16 = HexEngine.pair_vec_to_numpy_float16(radius=3, values=pairs) # float16, sentinel f16::NAN
```

#### Sentinel Values

NumPy methods use fixed sentinel values for type consistency:

- **Signed integer types** (int8, int16, int32, int64): Use -1 as the sentinel value
- **Unsigned integer types** (uint8, uint16, uint32, uint64): Use the type's maximum value as sentinel
- **Floating point types** (float16, float32, float64): Use NaN as the sentinel value

To get the maximum value for unsigned types in various languages:
- In Python, use `np.iinfo(dtype).max` for unsigned types
- In C, use constants like `UINT8_MAX`, `UINT16_MAX`, etc.
- In C++, use `std::numeric_limits<uint8_t>::max()`
- In Rust, use `u8::MAX`, `u16::MAX`, etc.

To get the NaN value for floating point types:
- In Python, use `float('nan')` or `np.nan`
- In C, use `NAN` from `<math.h>`
- In C++, use `std::numeric_limits<float>::quiet_NaN()` and `std::numeric_limits<double>::quiet_NaN()`
- In Rust, use `f32::NAN`, `f64::NAN`

Or ideally, since there are multiple NaN representations, use functions that check for NaN rather than direct equality:
- In Python, use `math.isnan(value)` or `np.isnan(value)`
- In C, use `isnan(value)` from `<math.h>`
- In C++, use `std::isnan(value)` from `<cmath>`
- In Rust, use `value.is_nan()`

#### Grid Indexing

The array/list indices correspond directly to grid positions as determined by `index_block()`:

```python
# Get index for a coordinate
hex_coo = Hex(1, -2)
index = HexEngine(3).index_block(hex_coo)
# Or use the faster HexEngine.hpyhex_rs_index_block(radius=3, hex_coo)

# The value at that index in the converted array/list
# corresponds to the value for that Hex coordinate
```

#### Supported Methods

The following methods are available for converting pair vectors to NumPy arrays, compared to [`hpyhex_rs_pair_vec_to_list_any()`](#native-methods), which provides maximum flexibility with Python objects:

| Method                             | Return Type                | Sentinel Value | Notes                             |
|------------------------------------|----------------------------|----------------|-----------------------------------|
| `hpyhex_rs_pair_vec_to_list_any()` | `List[Any]`                | parameter      | Flexible Python objects           |
| `pair_vec_to_numpy()`              | `numpy.ndarray[int16]`     | `-1`           | Default method                    |
| `pair_vec_to_numpy_int8()`         | `numpy.ndarray[int8]`      | `-1`           | Signed 8-bit                      |
| `pair_vec_to_numpy_uint8()`        | `numpy.ndarray[uint8]`     | `u8::MAX`      | Unsigned 8-bit                    |
| `pair_vec_to_numpy_int16()`        | `numpy.ndarray[int16]`     | `-1`           | Signed 16-bit                     |
| `pair_vec_to_numpy_uint16()`       | `numpy.ndarray[uint16]`    | `u16::MAX`     | Unsigned 16-bit                   |
| `pair_vec_to_numpy_int32()`        | `numpy.ndarray[int32]`     | `-1`           | Signed 32-bit                     |
| `pair_vec_to_numpy_uint32()`       | `numpy.ndarray[uint32]`    | `u32::MAX`     | Unsigned 32-bit                   |
| `pair_vec_to_numpy_int64()`        | `numpy.ndarray[int64]`     | `-1`           | Signed 64-bit                     |
| `pair_vec_to_numpy_uint64()`       | `numpy.ndarray[uint64]`    | `u64::MAX`     | Unsigned 64-bit                   |
| `pair_vec_to_numpy_float32()`      | `numpy.ndarray[float32]`   | `f32::NAN`     | Single precision                  |
| `pair_vec_to_numpy_float64()`      | `numpy.ndarray[float64]`   | `f64::NAN`     | Double precision                  |
| `pair_vec_to_numpy_float16()`      | `numpy.ndarray[float16]`   | `f16::NAN`     | Half precision (experimental)     |

The `pair_vec_to_numpy_*` methods produce 1D NumPy arrays of length equal to the number of cells in the hexagonal grid for the given radius, providing NumPy-compatible representations of sparse value mappings. The `hpyhex_rs_pair_vec_to_list_any` method produces a Python list of the same length and is intended for maximum flexibility with Python objects.

### Adjacency Structure for HexEngine

The `HexEngine` provides methods to obtain adjacency structures representing the connectivity between hexagonal cells. These are essential for graph-based algorithms, convolution operations, and advanced board state evaluation in hexagonal grid games.

For more on graph algorithms and representations, see the [Graph Theory](https://en.wikipedia.org/wiki/Graph_theory) article on Wikipedia.

For more on convolution operations on graphs, see the [Graph Convolutional Network](https://en.wikipedia.org/wiki/Graph_convolutional_network) article on Wikipedia and the [Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network) article on Wikipedia.

For explaination of the hexagonal system used in hpyhex, see the documentation from the `hpyhex` library or the [Hexagonal System](#hexagonal-system) section in this documentation.

**Important Note**: Both the adjacency list, adjacency matrix, and correspondence list methods are static methods that require the radius of the hexagonal grid as an argument. They do not depend on the specific state of a `HexEngine` instance. This design is intentional to allow users to obtain adjacency structures for any hexagonal grid size without needing to create a full `HexEngine` instance, and to reuse these structures across multiple instances or computations.

#### Adjacency List

The adjacency list represents the graph structure where each cell is connected to its neighboring cells. For hexagonal grids, each cell has up to 6 neighbors. This sparse representation is memory-efficient and suitable for most graph algorithms.

##### Getting Adjacency Lists

```python
from hpyhex import HexEngine

engine = HexEngine(radius=3)

# Get adjacency list as 2D array (default: int64 with -1 sentinel)
adj_list = HexEngine.to_numpy_adjacency_list(engine.radius)
# adj_list.shape == (19, 6)  # 19 cells, up to 6 neighbors each
# adj_list[i, j] = neighbor index or -1 if no neighbor

# Typed versions for different integer types
adj_list_int8 = HexEngine.to_numpy_adjacency_list_int8(engine.radius)    # int8, sentinel -1
adj_list_uint8 = HexEngine.to_numpy_adjacency_list_uint8(engine.radius)  # uint8, sentinel 255
adj_list_int16 = HexEngine.to_numpy_adjacency_list_int16(engine.radius)  # int16, sentinel -1
adj_list_uint16 = HexEngine.to_numpy_adjacency_list_uint16(engine.radius) # uint16, sentinel 65535
adj_list_int32 = HexEngine.to_numpy_adjacency_list_int32(engine.radius)  # int32, sentinel -1
adj_list_uint32 = HexEngine.to_numpy_adjacency_list_uint32(engine.radius) # uint32, sentinel 4294967295
adj_list_int64 = HexEngine.to_numpy_adjacency_list_int64(engine.radius)  # int64, sentinel -1
adj_list_uint64 = HexEngine.to_numpy_adjacency_list_uint64(engine.radius) # uint64, sentinel 18446744073709551615
```

##### Sentinel Values

- **Signed integer types** (int8, int16, int32, int64): Use -1 as the sentinel value to indicate missing neighbors
- **Unsigned integer types** (uint8, uint16, uint32, uint64): Use the type's maximum value as sentinel (e.g., 255 for uint8, 65535 for uint16)

To get the maximum value for unsigned types in various languages:
- In Python, you get those sentinel values using `np.iinfo(dtype).max` for unsigned types.
- In C, you can import `<limits.h>` and use constants like `UINT8_MAX`, `UINT16_MAX`, etc. for unsigned types.
- In C++, use `std::numeric_limits<uint8_t>::max()` for unsigned types.
- In Rust, use `u8::MAX`, `u16::MAX`, etc. for unsigned types.

##### Positioning of Neighbors

The adjacency list is structured such that for each cell `i`, the neighbors are ordered consistently based on hexagonal directions, following the following array of hex coordinates:

```
[
    Hex(-1, -1),
    Hex(-1, 0),
    Hex(0, -1),
    Hex(0, 1),
    Hex(1, 0),
    Hex(1, 1)
]
```

If a neighbor does not exist (e.g., edge cells), the corresponding entry will contain the sentinel value. Unlike the native hpyhex-rs `hpyhex_rs_adjacency_list()` method, the adjacency list here is strictly aligned, and the positioning in the array can be trusted for the direction of each neighbor.

For more convenient query of hexagonal positions, the above array can be derived from the following code snippet:

```python
from hpyhex import Piece

neighbors = [p for p in Piece.positions if p != (0, 0)]
```

#### Adjacency Matrix

The adjacency matrix provides a dense representation where `matrix[i,j] = True` if cells i and j are adjacent.

##### Getting Adjacency Matrices

```python
# Boolean matrix (default)
adj_matrix = HexEngine.to_numpy_adjacency_matrix(engine.radius)
# adj_matrix.shape == (37, 37)
# adj_matrix[i, j] = True if cells i and j are adjacent

# Typed versions
adj_matrix_bool = HexEngine.to_numpy_adjacency_matrix_bool(engine.radius)    # bool
adj_matrix_int8 = HexEngine.to_numpy_adjacency_matrix_int8(engine.radius)    # int8, 0 or 1
adj_matrix_uint8 = HexEngine.to_numpy_adjacency_matrix_uint8(engine.radius)  # uint8, 0 or 1
adj_matrix_int16 = HexEngine.to_numpy_adjacency_matrix_int16(engine.radius)  # int16, 0 or 1
adj_matrix_uint16 = HexEngine.to_numpy_adjacency_matrix_uint16(engine.radius) # uint16, 0 or 1
adj_matrix_int32 = HexEngine.to_numpy_adjacency_matrix_int32(engine.radius)  # int32, 0 or 1
adj_matrix_uint32 = HexEngine.to_numpy_adjacency_matrix_uint32(engine.radius) # uint32, 0 or 1
adj_matrix_int64 = HexEngine.to_numpy_adjacency_matrix_int64(engine.radius)  # int64, 0 or 1
adj_matrix_uint64 = HexEngine.to_numpy_adjacency_matrix_uint64(engine.radius) # uint64, 0 or 1
adj_matrix_float32 = HexEngine.to_numpy_adjacency_matrix_float32(engine.radius) # float32, 0.0 or 1.0
adj_matrix_float64 = HexEngine.to_numpy_adjacency_matrix_float64(engine.radius) # float64, 0.0 or 1.0
adj_matrix_float16 = HexEngine.to_numpy_adjacency_matrix_float16(engine.radius) # float16, 0.0 or 1.0 (requires "half" feature)
```

##### Using Adjacency Matrix for Convolution Operations

For convolution operations of radius 1, you can use the adjacency matrix to aggregate values from neighboring cells:

```python
import numpy as np

# Get adjacency matrix (boolean)
adj_matrix = HexEngine.to_numpy_adjacency_matrix_bool(engine.radius)
# adj_matrix.shape == (37, 37)

# Block values (e.g., occupied = 1, empty = 0)
block_values = engine.to_numpy_uint8()  # shape (37,)

# Convolution: sum of neighbor values for each cell
convolved = adj_matrix @ block_values  # shape (37,)
# convolved[i] = sum of block_values[j] for all neighbors j of i
```

This is similar to convolution kernels in image processing, where each cell's new value is computed based on its neighbors.

##### Generating Convolution Kernels for Larger Radii

To find cells within a larger radius, you can combine adjacency matrices:

```python
# Get base adjacency matrix
adj_r1 = HexEngine.to_numpy_adjacency_matrix_bool(engine.radius)

# Radius 2 kernel: cells within 2 steps (OR of matrix and its square)
adj_r2 = adj_r1 | (adj_r1 @ adj_r1)

# Radius 3 kernel: cells within 3 steps
adj_r3 = adj_r2 | (adj_r1 @ adj_r2)

# General approach: OR of matrix powers up to desired radius
def get_radius_kernel(adj_matrix, radius):
    kernel = adj_matrix.copy()
    current = adj_matrix
    for r in range(2, radius + 1):
        current = current @ adj_matrix
        kernel = kernel | current
    return kernel
```

Alternatively, implement breadth-first search (BFS) with limited depth to generate the kernel:

```python
def get_radius_kernel_bfs(engine, max_radius):
    n = len(engine.to_numpy())  # Number of cells
    kernel = np.zeros((n, n), dtype=bool)
    
    adj_list = HexEngine.to_numpy_adjacency_list_int64(engine.radius)  # Using int64 for neighbors
    
    for start in range(n):
        visited = np.zeros(n, dtype=bool)
        queue = [(start, 0)]  # (cell, distance)
        visited[start] = True
        
        while queue:
            cell, dist = queue.pop(0)
            if dist < max_radius:
                # Get neighbors from adjacency list
                for j in range(6):
                    neighbor = adj_list[cell, j]
                    if neighbor != -1 and not visited[neighbor]:
                        visited[neighbor] = True
                        kernel[start, neighbor] = True
                        kernel[neighbor, start] = True  # Undirected graph
                        queue.append((neighbor, dist + 1))
    
    return kernel
```

##### Dynamic Relation Kernels

For more complex relationships, apply topological sort on the graph defined by the adjacency list to create custom kernels based on specific criteria (e.g., distance, connectivity).

Graph algorithms can be used to derive various relational structures beyond simple adjacency, enabling advanced analysis and operations on the hexagonal grid. These algorithms can be accelerated by NumPy or specialized libraries like NetworkX or SciPy.

##### Memory Considerations

Adjacency matrices require O(N²) space, which becomes memory-intensive for large grids (e.g., radius 10 has 331 cells, requiring ~100KB for boolean matrix). Since hexagonal graphs are sparse (each node connects to ~6 neighbors), adjacency lists are preferred for efficiency and scalability. Use adjacency matrices only for small grids or when matrix-based algorithms are specifically required.

#### Correspondence List

The correspondence list provides a mapping from each block index in the hexagonal grid to another index, shifted by a specified Hex offset. This is useful for operations that require translating positions across the grid, such as convolution kernels or spatial transformations.

The indexed mapping is called a correspondence list, where each entry corresponds to a block in the hexagonal grid.

It has the following properties:

- The inverse of a correspondence list can be obtained by applying the negative of the original shift, and applying correspondence list T obtained from shift S with its inverse T^-1 results in the identity mapping, except those within S from the border of the grid.
- The correspondence lists A and B obtained from shifts S_A and S_B respectively can be composed to form a new correspondence list C that represents the combined shift S_C = S_A + S_B. This operation is commutative and associative, and the out of bound indices will be the same. This means S_A + S_B is equivalent to S_B + S_A when applied to the correspondence lists.
- The correspondence list of the origin (0, 0) is the identity list.

If a correspondence matrix M is constructed from a correspondence list L that is a result of shift S, then it has the following properties:

- M[i, j] = 1 if j is the shifted index of i by S, otherwise M[i, j] = 0.
- The correspondence matrix M is sparse, with exactly one non-zero entry per row for valid shifts.
- if L_A is the inverse of L_B, it is not necessary that M_A is the inverse of M_B, since the multiplication of sparse matrices may lead to loss of information due to out of bound indices. This means M_A * M_B is not necessarily the identity matrix, but it will have less non-zero entries than the identity matrix.
- if L_A + L_B = L_C, then M_A * M_B = M_C and M_B * M_A = M_C. This means the multiplication of correspondence matrices is commutative and associative, similar to correspondence lists.
- If |S| = s and grid radius = r, then the number of valid (non-sentinel) entries in the correspondence list is approximately equal to the total number of cells in a hexagonal grid of radius (r - s). This is because the shift S effectively reduces the usable area of the grid by s layers of cells from the border.

##### Getting Correspondence Lists

```python
from hpyhex import HexEngine, Hex

engine = HexEngine(radius=5)
shift = Hex(1, 0)  # Shift by (1, 0)

# Get correspondence list as 1D array (default: int64 with -1 sentinel)
corr_list = HexEngine.to_numpy_correspondence_list(radius=engine.radius, shift=shift)
# corr_list.shape == (61,)  # One entry per cell
# corr_list[i] = shifted index or -1 if out of bounds

# Typed versions for different integer types
corr_list_int64 = HexEngine.to_numpy_correspondence_list_int64(engine.radius, shift)    # int64, sentinel -1
corr_list_uint16 = HexEngine.to_numpy_correspondence_list_uint16(engine.radius, shift)  # uint16, sentinel 65535
corr_list_uint32 = HexEngine.to_numpy_correspondence_list_uint32(engine.radius, shift)  # uint32, sentinel 4294967295
corr_list_uint64 = HexEngine.to_numpy_correspondence_list_uint64(engine.radius, shift)  # uint64, sentinel 18446744073709551615
corr_list_int16 = HexEngine.to_numpy_correspondence_list_int16(engine.radius, shift)    # int16, sentinel -1
corr_list_int32 = HexEngine.to_numpy_correspondence_list_int32(engine.radius, shift)    # int32, sentinel -1
```

##### Sentinel Values

- **Signed integer types** (int16, int32, int64): Use -1 as the sentinel value to indicate out-of-bounds shifts
- **Unsigned integer types** (uint16, uint32, uint64): Use the type's maximum value as sentinel

To get the maximum value for unsigned types in various languages:
- In Python, you get those sentinel values using `np.iinfo(dtype).max` for unsigned types.
- In C, you can import `<limits.h>` and use constants like `UINT16_MAX`, `UINT32_MAX`, etc. for unsigned types.
- In C++, use `std::numeric_limits<uint16_t>::max()` for unsigned types.
- In Rust, use `u16::MAX`, `u32::MAX`, etc. for unsigned types.

##### Generating Custom Hexagonal Kernels from Correspondence Lists

Correspondence lists enable creating arbitrary convolution patterns beyond the standard 6-neighbor kernel. By defining custom shift patterns, you can build specialized kernels for different spatial operations.

A correspondence list maps each cell to its shifted position. Multiple correspondence lists can be combined with learnable weights to create convolution kernels:

```python
import numpy as np
from hpyhex import HexEngine, Hex

radius = 5

# Define kernel as list of hex shifts
kernel_shifts = [
    (0, 0),    # Self
    (-1, -1),  # Neighbor 0
    (-1, 0),   # Neighbor 1
    (0, -1),   # Neighbor 2
    (0, 1),    # Neighbor 3
    (1, 0),    # Neighbor 4
    (1, 1),    # Neighbor 5
    # ... expand with more shifts for larger kernels
]

# Create correspondence matrices for each shift
correspondence_matrices = []
for shift in kernel_shifts:
    corr_list = HexEngine.to_numpy_correspondence_list_int64(radius, Hex(shift[0], shift[1]))
    
    # Convert to sparse matrix
    num_cells = len(corr_list)
    matrix = np.zeros((num_cells, num_cells))
    for i in range(num_cells):
        j = corr_list[i]
        if j != -1:
            matrix[i, j] = 1.0
    
    correspondence_matrices.append(matrix)

# Apply convolution with learned weights
weights = np.array([1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # Example weights
board_state = np.random.rand(num_cells)

result = sum(w * (M @ board_state) for w, M in zip(weights, correspondence_matrices))
```

Correspondence matrices compose via multiplication due to their commutative property:

```python
def create_correspondence_matrix(radius, shift):
    '''
    Create a correspondence matrix for a given shift by first getting the correspondence list
    and then converting it to a dense matrix.
    '''
    corr_list = HexEngine.to_numpy_correspondence_list(radius, shift)
    num_cells = len(corr_list)
    matrix = np.zeros((num_cells, num_cells))
    for i in range(num_cells):
        j = corr_list[i]
        if j != -1:
            matrix[i, j] = 1.0
    return matrix

# Two-hop kernel: applies shift A then shift B
M_A = create_correspondence_matrix(radius, Hex(1, 0))
M_B = create_correspondence_matrix(radius, Hex(0, 1))
M_composed = M_A @ M_B  # Equivalent to shift Hex(1, 1)

# Multi-scale kernel: combine different rings
M_center = create_correspondence_matrix(radius, Hex(0, 0))
M_ring1 = sum(create_correspondence_matrix(radius, Hex(s[0], s[1])) for s in ring1_shifts)
M_ring2 = sum(create_correspondence_matrix(radius, Hex(s[0], s[1])) for s in ring2_shifts)

# Learnable weights for each ring
w0, w1, w2 = 1.0, 0.5, 0.25  # Example: center-weighted
M_kernel = w0 * M_center + w1 * M_ring1 + w2 * M_ring2

result = M_kernel @ board_state
```

**Applications of Custom Kernels**

**Edge Detection** - Ring-1 kernel without center detects local changes:
```python
edge_shifts = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, 0), (1, 1)]
# Weights: negative center, positive neighbors approximates Laplacian
```

**Multi-Scale Features** - Stack kernels with increasing receptive fields with Torch:
```python
model = nn.Sequential(
    HexConvCustom(radius=5, kernel_shifts=ring1_shifts, in_channels=1, out_channels=8),
    nn.ReLU(),
    HexConvCustom(radius=5, kernel_shifts=filled2_shifts, in_channels=8, out_channels=16),
    nn.ReLU(),
    HexConvCustom(radius=5, kernel_shifts=cross_shifts, in_channels=16, out_channels=1)
)
```

**Pattern Recognition** - Asymmetric kernels for specific shapes:
```python
# L-shape pattern
l_shape_shifts = [(0, 0), (1, 0), (2, 0), (0, 1), (0, 2)]
conv_l = HexConvCustom(radius=5, kernel_shifts=l_shape_shifts)
```

**Improved Density Index** - Replace `compute_dense_index` with learned features:
```python
# Trainable density estimation
density_conv = HexConvCustom(radius=5, kernel_shifts=filled2_shifts, 
                            in_channels=1, out_channels=1)
# Train on boards where nrsearch performs well
```

**Performance Considerations for Custom Kernels from Correspondence Matrices**

- **Memory**: Each kernel stores k matrices of size (n×n) where n = number of cells
  - For radius 5 (61 cells): filled-2 kernel (19 shifts) requires ~7.1 MB
  - Use sparse tensors for large grids or many shifts
  
- **Computation**: Time complexity O(k × n) per forward pass
  - Highly parallelizable on GPU via matrix multiplication
  - Batch all shifts together for efficiency
  
- **Parameter Count**: out_channels × in_channels × kernel_size
  - Ring-1: 7 parameters per channel pair
  - Filled-2: 19 parameters per channel pair
  - Cross-3: 19 parameters per channel pair (1 + 6×3)

##### Direct Convolution with Correspondence Lists

Direct convolution with correspondence lists enables efficient computation of convolutional operations on hexagonal grids by leveraging precomputed mappings between grid positions. This approach avoids the need for dense kernel matrices and instead uses sparse indexing to gather relevant input values.

In traditional convolution, a kernel slides over the input feature map, computing weighted sums of neighboring values. For hexagonal grids, correspondence lists provide a way to define which input positions contribute to each output position for a given kernel pattern. Each correspondence list corresponds to a specific shift or kernel position, mapping each output cell to its corresponding input cell under that shift.

To illustrate this, first consider the definition of the convolution operation, where the output at each position is computed as a weighted sum of input values from positions defined by the correspondence lists. This is easily achieved by iterating over the output positions and using the correspondence lists to gather input values.

Consider an input feature map `x` with shape `(batch_size, input_channels, num_cells)`, a weight tensor `w` with shape `(output_channels, input_channels, kernel_size)`, and correspondence lists `corr` as a 2D array of shape `(kernel_size, num_cells)`.

For each output position `i`, the convolution computes a weighted sum of input values from positions specified by the correspondence lists:

```python
for b in range(batch_size):
    for c_out in range(output_channels):
        for i in range(num_cells):
            sum_val = 0.0
            for k in range(kernel_size):
                input_pos = corr[k, i]
                if input_pos != -1:  # valid mapping
                    for c_in in range(input_channels):
                        sum_val += w[c_out, c_in, k] * x[b, c_in, input_pos]
            output[b, c_out, i] = sum_val
```

This shows how each output cell aggregates contributions from multiple input cells, weighted by the kernel parameters, using the correspondence lists to determine which input cells contribute to each output cell.

The above code can then be vectorized with matrix operations provided by NumPy as follows:

```python
def hex_convolution(x, w, corr):
    """
    x: shape (batch_size, input_channels, num_cells)
    w: shape (output_channels, input_channels, kernel_size)
    corr: shape (kernel_size, num_cells), each entry is int (input index or -1 for invalid)
    Returns: output of shape (batch_size, output_channels, num_cells)
    """
    batch_size, input_channels, num_cells = x.shape
    output_channels, _, kernel_size = w.shape

    # Gather input values for each kernel position
    # For invalid indices (-1), fill with zeros
    # We'll use np.take with mode='clip' and mask out invalids
    input_indices = np.clip(corr, 0, num_cells - 1)  # shape: (kernel_size, num_cells)
    valid_mask = (corr != -1)                        # shape: (kernel_size, num_cells)

    # Gathered: (batch_size, input_channels, kernel_size, num_cells)
    gathered = np.take(x, input_indices, axis=2)     # shape: (batch_size, input_channels, kernel_size, num_cells)
    gathered = gathered * valid_mask[None, None, :, :]  # mask out invalids

    # Now, contract input_channels and kernel_size with weights
    # w: (output_channels, input_channels, kernel_size)
    # gathered: (batch_size, input_channels, kernel_size, num_cells)
    # einsum: 'oik,bikn->bon'
    output = np.einsum('oik,bikn->bon', w, gathered)
    return output
```

The computation involves gathering input values using the correspondence lists as indices, applying the weights through element-wise multiplication and summation over channels and kernel positions. This sparse approach is particularly efficient for hexagonal grids where traditional dense convolutions would be wasteful due to the irregular connectivity.

**Performance Considerations**

- **Memory Efficiency**: This method avoids storing large dense kernel matrices, using only the correspondence lists and weight tensors.
- **Computation Efficiency**: The gather operation and subsequent einsum are optimized in NumPy, allowing for efficient execution on both CPU and GPU. The example shows how to leverage NumPy's advanced indexing and broadcasting capabilities to perform the convolution operation efficiently, and similar techniques can be applied in deep learning frameworks like PyTorch or TensorFlow.
- **Scalability**: The approach scales well with larger grids and more complex kernels, as the correspondence lists remain compact. However, on smaller grids, the overhead of gather operations may outweigh the benefits compared to dense convolutions.

#### Advanced Board State Evaluation

Combine adjacency structures with position masks and piece placement for strategic game analysis.

##### Counting Isolated Islands

Isolated islands refer to disconnected groups of occupied cells on the board. Counting these helps evaluate board fragmentation and strategic positioning. A board with many small islands may be harder to play effectively, as pieces placed in one island cannot influence others. This metric is useful for game AI to assess board quality and make strategic decisions about piece placement.

##### Detecting Strategic Anomalies

Strategic anomalies are problematic board configurations that can hinder gameplay. The most common anomaly is isolated regions with fewer than 4 connected cells, which are often impossible to fill with standard pieces. These regions represent "dead space" that cannot be utilized effectively, reducing the overall board efficiency. 

Detection of such anomalies can be performed using the adjacency list to identify connected components and evaluate their sizes using graph traversal algorithms (e.g., DFS or BFS). Regions with fewer than 4 connected occupied cells can be flagged as anomalies.

Detecting such anomalies helps AI systems avoid moves that create unwinnable positions or identify when a board state has become strategically compromised.

##### Position Mask Integration

Position masks indicate where pieces can legally be placed, but combining this with adjacency information provides deeper strategic insights. By analyzing the connectivity of potential placement positions, AI can evaluate not just whether a move is legal, but how well it integrates with the existing board structure. Positions that increase connectivity (bridging islands) or avoid creating anomalies are generally more valuable. This approach enables sophisticated move ordering and position evaluation beyond simple validity checks.

These adjacency-based techniques enable sophisticated game strategies by analyzing board connectivity, identifying problematic isolated regions, and evaluating position quality beyond simple validity checks.

### Serialization for Game

The `Game` class provides comprehensive NumPy integration for converting game states, including both the engine and piece queue, to and from array representations. This enables efficient serialization for machine learning applications, game state analysis, and reinforcement learning.

#### Converting to NumPy

The `to_numpy()` method returns a 1D boolean array representing the entire game state (engine followed by queue):
```python
from hpyhex import Game

game = Game(radius=3, queue=3)
# Add some pieces...
arr = game.to_numpy()
# arr.dtype == np.bool_
# arr.shape == (37 + 3*7,)  # engine cells + queue pieces * 7 blocks
# arr[:37] represents the engine state
# arr[37:] represents the flattened queue
```

For specific numeric types, use the typed conversion methods:
```python
# Integer types
arr_i8 = game.to_numpy_int8()      # dtype: int8
arr_u8 = game.to_numpy_uint8()     # dtype: uint8
arr_i16 = game.to_numpy_int16()    # dtype: int16
arr_u16 = game.to_numpy_uint16()   # dtype: int16
arr_i32 = game.to_numpy_int32()    # dtype: int32
arr_u32 = game.to_numpy_uint32()   # dtype: uint32
arr_i64 = game.to_numpy_int64()    # dtype: int64
arr_u64 = game.to_numpy_uint64()   # dtype: uint64

# Floating point types
arr_f32 = game.to_numpy_float32()  # dtype: float32
arr_f64 = game.to_numpy_float64()  # dtype: float64

# Half precision (requires "half" feature, experimental)
arr_f16 = game.to_numpy_float16()  # dtype: float16
```

#### Converting from NumPy

Use the `from_numpy_with_*` methods to construct a `Game` instance from a NumPy array. You must specify either the radius or queue length to properly interpret the array structure.

For radius-based construction:
```python
import numpy as np
from hpyhex import Game

# Array with engine (37 cells) + queue (3 pieces * 7 = 21 blocks) = 58 elements
arr = np.zeros(58, dtype=bool)
# Set some engine cells and queue pieces...
game = Game.from_numpy_with_radius_bool(radius=3, arr=arr)
print(game.engine.radius)  # 3
print(len(game.queue))     # Inferred from array length
```

For queue length-based construction:
```python
# Same array, but specify queue length instead
game = Game.from_numpy_with_queue_length_bool(length=3, arr=arr)
print(game.engine.radius)  # Inferred from array length
print(len(game.queue))     # 3
```

For specific numeric types:
```python
# Radius-based
game_u8 = Game.from_numpy_with_radius_uint8(radius=3, arr=arr_u8)
game_f32 = Game.from_numpy_with_radius_float32(radius=3, arr=arr_f32)

# Queue length-based
game_u8 = Game.from_numpy_with_queue_length_uint8(length=3, arr=arr_u8)
game_f32 = Game.from_numpy_with_queue_length_float32(length=3, arr=arr_f32)
```

#### Queue-Only Conversion

For converting just the piece queue, use the `queue_to_numpy_*` methods:
```python
# Flat representation (1D array concatenating all pieces)
queue_flat = game.queue_to_numpy_flat()
# queue_flat.shape == (3*7,)  # 21 elements

# Stacked representation (2D array, one row per piece)
queue_stacked = game.queue_to_numpy_stacked()
# queue_stacked.shape == (3, 7)  # 3 pieces, 7 blocks each

# Typed versions
queue_u8_flat = game.queue_to_numpy_uint8_flat()
queue_u8_stacked = game.queue_to_numpy_uint8_stacked()
```

#### Engine-Only Conversion

To convert just the engine, access it directly through the game instance:
```python
engine_arr = game.engine.to_numpy()
# This uses HexEngine's to_numpy method
# See HexEngine serialization documentation for details
```

Since the `engine` is stored as a Python reference within the `Game` instance, no additional copying or Python object creation is needed, making this operation as efficient as if separate methods were provided.

#### Validation and Error Handling

All `from_numpy_with_*` methods validate the input array:

- **Length validation**: Array length must correspond to a valid game state (engine + queue)
- **Type validation**: Array dtype must match the method's expected type
- **Parameter validation**: Specified radius/queue length must be consistent with array structure

If validation fails, a `ValueError` is raised:
```python
# Wrong length
arr = np.zeros(50, dtype=bool)  # Not a valid game state length
try:
    game = Game.from_numpy_with_radius_bool(radius=3, arr=arr)
except ValueError as e:
    print(f"Error: {e}")  # Invalid array length for game state

# Inconsistent parameters
arr = np.zeros(58, dtype=bool)
try:
    game = Game.from_numpy_with_queue_length_bool(length=5, arr=arr)  # Wrong queue length
except ValueError as e:
    print(f"Error: {e}")  # Queue length doesn't match array structure
```

#### Type Casting Considerations

NumPy arrays cannot be easily cast between types at the Rust/Python boundary. Therefore, **there is no universal `from_numpy()` method**. You must use the specific typed method that matches your array's dtype:
```python
# No automatic type detection
arr = np.ones(58, dtype=np.int32)
# game = Game.from_numpy_with_radius(arr, radius=3)  # This method doesn't exist!

# Use the typed method matching your dtype
game = Game.from_numpy_with_radius_int32(radius=3, arr=arr)

# If you need to convert between types, do it in NumPy first:
arr_f32 = arr.astype(np.float32)
game = Game.from_numpy_with_radius_float32(radius=3, arr_f32)
```

#### Zero Copy

Game serialization always involves copying data between NumPy arrays and Game instances, as the internal representations are optimized for different access patterns.

#### Supported Data Types

The following table summarizes all supported NumPy dtypes for Game serialization:

| NumPy dtype | `to_numpy_*`           | `from_numpy_with_radius_*`         | `from_numpy_with_queue_length_*`         | `queue_to_numpy_*_flat`         | `queue_to_numpy_*_stacked`         |
|-------------|------------------------|------------------------------------|------------------------------------------|---------------------------------|------------------------------------|
| `bool_`     | `to_numpy()` (default) | `from_numpy_with_radius_bool()`    | `from_numpy_with_queue_length_bool()`    | `queue_to_numpy_flat()`         | `queue_to_numpy_stacked()`         |
| `int8`      | `to_numpy_int8()`      | `from_numpy_with_radius_int8()`    | `from_numpy_with_queue_length_int8()`    | `queue_to_numpy_int8_flat()`    | `queue_to_numpy_int8_stacked()`    |
| `uint8`     | `to_numpy_uint8()`     | `from_numpy_with_radius_uint8()`   | `from_numpy_with_queue_length_uint8()`   | `queue_to_numpy_uint8_flat()`   | `queue_to_numpy_uint8_stacked()`   |
| `int16`     | `to_numpy_int16()`     | `from_numpy_with_radius_int16()`   | `from_numpy_with_queue_length_int16()`   | `queue_to_numpy_int16_flat()`   | `queue_to_numpy_int16_stacked()`   |
| `uint16`    | `to_numpy_uint16()`    | `from_numpy_with_radius_uint16()`  | `from_numpy_with_queue_length_uint16()`  | `queue_to_numpy_uint16_flat()`  | `queue_to_numpy_uint16_stacked()`  |
| `int32`     | `to_numpy_int32()`     | `from_numpy_with_radius_int32()`   | `from_numpy_with_queue_length_int32()`   | `queue_to_numpy_int32_flat()`   | `queue_to_numpy_int32_stacked()`   |
| `uint32`    | `to_numpy_uint32()`    | `from_numpy_with_radius_uint32()`  | `from_numpy_with_queue_length_uint32()`  | `queue_to_numpy_uint32_flat()`  | `queue_to_numpy_uint32_stacked()`  |
| `int64`     | `to_numpy_int64()`     | `from_numpy_with_radius_int64()`   | `from_numpy_with_queue_length_int64()`   | `queue_to_numpy_int64_flat()`   | `queue_to_numpy_int64_stacked()`   |
| `uint64`    | `to_numpy_uint64()`    | `from_numpy_with_radius_uint64()`  | `from_numpy_with_queue_length_uint64()`  | `queue_to_numpy_uint64_flat()`  | `queue_to_numpy_uint64_stacked()`  |
| `float16`   | `to_numpy_float16()`   | `from_numpy_with_radius_float16()` | `from_numpy_with_queue_length_float16()` | `queue_to_numpy_float16_flat()` | `queue_to_numpy_float16_stacked()` |
| `float32`   | `to_numpy_float32()`   | `from_numpy_with_radius_float32()` | `from_numpy_with_queue_length_float32()` | `queue_to_numpy_float32_flat()` | `queue_to_numpy_float32_stacked()` |
| `float64`   | `to_numpy_float64()`   | `from_numpy_with_radius_float64()` | `from_numpy_with_queue_length_float64()` | `queue_to_numpy_float64_flat()` | `queue_to_numpy_float64_stacked()` |

Call `.engine.to_numpy_*()` on the `engine` attribute of the `Game` instance to convert the engine portion separately.

See [HexEngine serialization documentation](#serialization-for-hexengine) for details on engine array representations.
See [Queue serialization documentation](#serialization-for-vector-of-piece-piece-queues) for details on queue array representations.

**Recommended types:**
- Use `bool_` for minimal memory footprint
- Use `uint8` for compact integer formats
- Use `float32` for machine learning applications

### Making Moves with NumPy Arrays

The `Game` class provides methods to make moves using 2D NumPy arrays representing piece selection and placement positions. These methods are useful for machine learning applications where moves are encoded as arrays.

#### Mask-Based Moves

Use `move_with_numpy_mask_<type>()` methods to make a move by specifying a boolean-like mask where exactly one non-zero value indicates the selected piece and placement position:

```python
import numpy as np
from hpyhex import Game

game = Game(radius=5, queue=3)
# Create a 2D mask: (queue_length, engine_cells)
mask = np.zeros((3, 61), dtype=np.bool_)
mask[1, 10] = True  # Select piece 1, place at engine position 10

success = game.move_with_numpy_mask_bool(mask)
```

Available for all numeric types:
- `move_with_numpy_mask_bool()` - Boolean mask
- `move_with_numpy_mask_int8()`, `move_with_numpy_mask_uint8()`
- `move_with_numpy_mask_int16()`, `move_with_numpy_mask_uint16()`
- `move_with_numpy_mask_int32()`, `move_with_numpy_mask_uint32()`
- `move_with_numpy_mask_float32()`, `move_with_numpy_mask_float64()`
- `move_with_numpy_mask_float16()` (requires "half" feature)

#### Maximum Value Moves

Use `move_with_numpy_max_<type>()` methods to make a move by selecting the position with the maximum value in the array:

```python
# Create a 2D array with move values/scores
move_scores = np.random.rand(3, 37).astype(np.float32)
# The position with the highest score will be selected
success = game.move_with_numpy_max_float32(move_scores)
```

Available for the same types as mask methods.

Both methods return `True` if the move was successful, `False` otherwise. They raise `ValueError` for invalid inputs or impossible moves.
