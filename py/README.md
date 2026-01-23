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

### Supported Data Types

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

### Serialization for HexEngine

The `HexEngine` class provides comprehensive NumPy integration for converting hexagonal game boards to and from array representations. All conversions produce or consume 1-dimensional arrays where the length corresponds to the total number of cells in the hexagonal grid (for a radius `r`, this is `3r² + 3r + 1` cells). See original `hpyhex` documentation for details on hexagonal grid sizing.

#### Array Shape and Grid Mapping

Unlike rectangular grids, hexagonal grids don't map naturally to 2D arrays. The `HexEngine` uses a **flattened 1D representation** where each index corresponds to a specific hexagonal cell:
```python
from hpyhex import HexEngine

engine = HexEngine(radius=3)
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

engine = HexEngine(radius=3)
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
arr = np.zeros(37, dtype=bool)  # 37 cells = radius 3
arr[0] = True
arr[5] = True
engine = HexEngine.from_numpy_bool(arr)
print(engine.radius)  # 3

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

- Is allocated on the host (CPU) memory. If the data is allocated on a different device (e.g., GPU), accessing its memory directly from NumPy will lead to undefined behavior or mysterious crashes.
- Is allocated in a way that is compatible with NumPy's memory layout. This means that it is not padded or aligned in a way that would be incompatible with NumPy's expectations.
- Is contiguous. If it is not contiguous, the function will panic.
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

### Supported Data Types

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

