================
Crate ``hpyhex``
================


.. rust:crate:: hpyhex
   :index: 0


   .. rust:use:: hpyhex
      :used_name: self


   .. rust:use:: hpyhex
      :used_name: crate


   .. rust:use:: pyo3::types::PyAny
      :used_name: PyAny


   .. rust:use:: pyo3::types::PyList
      :used_name: PyList


   .. rust:use:: pyo3::types::PyType
      :used_name: PyType


   .. rust:use:: std::sync::OnceLock
      :used_name: OnceLock


   .. rust:use:: std::hash::Hash
      :used_name: Hash


   .. rust:use:: std::hash::Hasher
      :used_name: Hasher


   .. rust:use:: rand::Rng
      :used_name: Rng


   .. rust:use:: std::collections::HashMap
      :used_name: HashMap


   .. rust:use:: once_cell::sync::Lazy
      :used_name: Lazy


   .. rubric:: Functions


   .. rust:function:: hpyhex::random_engine
      :index: 0
      :vis: pub
      :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"random_engine"},{"type":"punctuation","value":"("},{"type":"name","value":"radius"},{"type":"punctuation","value":": "},{"type":"link","value":"usize","target":"usize"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"HexEngine","target":"HexEngine"},{"type":"punctuation","value":">"}]

      Generate a random HexEngine with a given radius. True randomness or random distribution is not guaranteed
      as elimination is applied to the engine, reducing some instances to other instances.
      
      This is superior than HexEngine.all_engines(radius) because it does not consume significant memory and time.
      
      Arguments:
         - radius (int): The radius of the hexagonal game board.
      Returns:
         - HexEngine: A new randomized HexEngine instance with the specified radius.
      Raises:
         - TypeError: If radius is not an integer or is less than 2.

   .. rubric:: Structs and Unions


   .. rust:struct:: hpyhex::Game
      :index: 1
      :vis: pub
      :toc: struct Game
      :layout: [{"type":"keyword","value":"struct"},{"type":"space"},{"type":"name","value":"Game"}]

      Game is a class that represents the game environment for Hex.
      It manages the game engine, the queue of pieces, and the game state.
      It provides methods to add pieces, make moves, and check the game status.
      Its methods are intended to catch exceptions and handle errors gracefully.
      
      Attributes:
         - engine (HexEngine): The game engine that manages the game state.
         - queue (list[Piece]): The queue of pieces available for placement.
         - result (tuple[int, int]): The current result of the game, including the score and turn number.
         - score (int): The current score of the game.
         - turn (int): The current turn number in the game.
         - end (bool): Whether the game has ended.

      .. rubric:: Implementations


      .. rust:impl:: hpyhex::Game
         :index: -1
         :vis: pub
         :layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"Game","target":"Game"}]
         :toc: impl Game


         .. rubric:: Functions


         .. rust:function:: hpyhex::Game::add_piece
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"add_piece"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"python"},{"type":"punctuation","value":": "},{"type":"link","value":"Python","target":"Python"},{"type":"punctuation","value":", "},{"type":"name","value":"piece_index"},{"type":"punctuation","value":": "},{"type":"link","value":"usize","target":"usize"},{"type":"punctuation","value":", "},{"type":"name","value":"coord"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"Bound","target":"Bound"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"Hex","target":"Hex"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"bool","target":"bool"},{"type":"punctuation","value":">"}]

            Add a piece to the game engine at the specified coordinates.
            
            Parameters:
               - piece_index (int): The index of the piece in the queue to be added.
               - coord (Hex): The coordinates where the piece should be placed.
            Returns:
               - bool: True if the piece was successfully added, False otherwise.

         .. rust:function:: hpyhex::Game::make_move
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"make_move"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"python"},{"type":"punctuation","value":": "},{"type":"link","value":"Python","target":"Python"},{"type":"punctuation","value":", "},{"type":"name","value":"algorithm"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"Bound","target":"Bound"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"PyAny","target":"PyAny"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"bool","target":"bool"},{"type":"punctuation","value":">"}]

            Make a move using the specified algorithm.
            
            Parameters:
               - algorithm (callable): The algorithm to use for making the move.
                 The algorithm should follow the signature: `algorithm(engine: HexEngine, queue: list[Piece]) -> tuple[int, Hex]`.
            Returns:
               - bool: True if the move was successfully made, False otherwise.

         .. rust:function:: hpyhex::Game::new
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"new"},{"type":"punctuation","value":"("},{"type":"name","value":"engine"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"Bound"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"PyAny"},{"type":"punctuation","value":">"},{"type":"punctuation","value":", "},{"type":"name","value":"queue"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"Bound"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"PyAny"},{"type":"punctuation","value":">"},{"type":"punctuation","value":", "},{"type":"name","value":"initial_turn"},{"type":"punctuation","value":": "},{"type":"link","value":"Option","target":"Option"},{"type":"punctuation","value":"<"},{"type":"link","value":"i64","target":"i64"},{"type":"punctuation","value":">"},{"type":"punctuation","value":", "},{"type":"name","value":"initial_score"},{"type":"punctuation","value":": "},{"type":"link","value":"Option","target":"Option"},{"type":"punctuation","value":"<"},{"type":"link","value":"i64","target":"i64"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"Self","target":"Self"},{"type":"punctuation","value":">"}]

            Initialize the game with a game engine of radius r and game queue of length q.
            
            Parameters:
               - engine (HexEngine | int): The game engine to use, either as a HexEngine instance or an integer representing the radius.
               - queue (list[Piece] | int): The queue of pieces to use, either as a list of Piece instances or an integer representing the size of the queue.
               - initial_turn (int): The initial turn number of the game, default is 0.
               - initial_score (int): The initial score of the game, default is 0.
            Returns:
               - None
            Raises:
               - ValueError: If the engine radius is less than 2 or if the queue size is less than 1.
               - TypeError: If the engine is not a HexEngine instance or an integer, or if the queue is not a list of Piece instances or an integer, or if initial_turn or initial_score is not a non-negative integer.


   .. rust:struct:: hpyhex::Hex
      :index: 1
      :vis: pub
      :toc: struct Hex
      :layout: [{"type":"keyword","value":"struct"},{"type":"space"},{"type":"name","value":"Hex"}]

      Represents a hexagonal grid coordinate using a custom line-based coordinate system.
      
      This class models hexagonal positions with two line coordinates (i, k), implicitly
      defining the third axis (j) as `j = k - i` to maintain hex grid constraints.
      It supports standard arithmetic, equality, and hashing operations, as well as
      compatibility with coordinate tuples.
      
      For small grids, Hex instances are cached for performance, allowing more efficient memory usage
      and faster access. The caching is limited to a range of -64 to 64 for both i and k coordinates.
      
      Use of Hex over tuples is recommended for clarity and to leverage the singleton feature of small Hexes.
      
      Coordinate Systems:
         - Raw Coordinates (i, j, k): Three axes satisfying i + j + k = 0, where each axis is diagonal to the others at 60° increments.
         - Line Coordinates (i, k): Derived coordinates representing distances perpendicular to axes, simplifying grid operations.
      
      Note:
         - This class is immutable and optimized with __slots__.
         - Raw coordinate methods (__i__, __j__, __k__) are retained for backward compatibility.
         - Only basic functionality is implemented; complex adjacency, iteration, and mutability features are omitted for simplicity.
      
      Attributes:
         - i (int): The line i coordinate.
         - j (int): The computed line j coordinate (k - i).
         - k (int): The line k coordinate.

      .. rubric:: Implementations


      .. rust:impl:: hpyhex::Hex
         :index: -1
         :vis: pub
         :layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"Hex","target":"Hex"}]
         :toc: impl Hex


         .. rubric:: Functions


         .. rust:function:: hpyhex::Hex::__add__
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__add__"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"other"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"Bound"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"PyAny","target":"PyAny"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"Self","target":"Self"},{"type":"punctuation","value":">"}]

            Add another Hex or a tuple of coordinates to this hex.
            
            Arguments:
               - other (Hex or tuple): The value to add.
            Returns:
               - Hex: A new Hex with the added coordinates.
            Raises:
               - TypeError: If the other operand is not a Hex or a tuple of coordinates.

         .. rust:function:: hpyhex::Hex::__bool__
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__bool__"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"bool","target":"bool"}]

            Check if the Hex is not at the origin (0, 0).
            
            Returns:
               - bool: True if the Hex is not at the origin, False otherwise.

         .. rust:function:: hpyhex::Hex::__copy__
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__copy__"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"Self","target":"Self"}]

            Create a copy of this Hex.
            
            Returns:
               - Hex: A new Hex with the same coordinates.

         .. rust:function:: hpyhex::Hex::__deepcopy__
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__deepcopy__"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"_memo"},{"type":"punctuation","value":": "},{"type":"link","value":"Option","target":"Option"},{"type":"punctuation","value":"<"},{"type":"punctuation","value":"&"},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"Bound"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"PyAny","target":"PyAny"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"Self","target":"Self"}]

            Create a deep copy of this Hex.
            
            Arguments:
               - memo (dict): A dictionary to keep track of copied objects.
            Returns:
               - Hex: A new Hex with the same coordinates.

         .. rust:function:: hpyhex::Hex::__eq__
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__eq__"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"other"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"Bound"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"PyAny","target":"PyAny"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"bool","target":"bool"},{"type":"punctuation","value":">"}]

            Check equality with another Hex or a tuple of coordinates.
            
            Arguments:
               - value (Hex or tuple): The value to compare with.
            Returns:
               - bool: True if the coordinates match, False otherwise.

         .. rust:function:: hpyhex::Hex::__hash__
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__hash__"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"u64","target":"u64"}]

            Return a hash of the hex coordinates.
            
            Returns:
               - int: The hash value of the hex coordinates.

         .. rust:function:: hpyhex::Hex::__i__
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__i__"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"i32","target":"i32"}]

            Return the raw i coordinate of the hex.
            
            Returns:
               - int: The raw i coordinate.

         .. rust:function:: hpyhex::Hex::__iter__
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__iter__"},{"type":"punctuation","value":"("},{"type":"name","value":"slf"},{"type":"punctuation","value":": "},{"type":"link","value":"PyRef","target":"PyRef"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"Self","target":"Self"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"Py","target":"Py"},{"type":"punctuation","value":"<"},{"type":"link","value":"PyAny","target":"PyAny"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"}]

            Return an iterator over the hex coordinates.
            
            Yields:
               - int: The I-line coordinate of the hex.
               - int: The K-line coordinate of the hex.

         .. rust:function:: hpyhex::Hex::__j__
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__j__"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"i32","target":"i32"}]

            Return the raw j coordinate of the hex.
            
            Returns:
               - int: The raw j coordinate.

         .. rust:function:: hpyhex::Hex::__k__
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__k__"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"i32","target":"i32"}]

            Return the raw k coordinate of the hex.
            
            Returns:
               - int: The raw k coordinate.

         .. rust:function:: hpyhex::Hex::__radd__
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__radd__"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"other"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"Bound"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"PyAny","target":"PyAny"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"Self","target":"Self"},{"type":"punctuation","value":">"}]

            Reverse addition of this hex to another Hex or a tuple.
            
            Arguments:
               - other (Hex or tuple): The value to add this hex to.
            Returns:
               - Hex: A new Hex with the added coordinates.
            Raises:
               - TypeError: If the other operand is not a Hex or a tuple of coordinates.

         .. rust:function:: hpyhex::Hex::__repr__
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__repr__"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"String","target":"String"}]

            Return a string representation of the hex coordinates for debugging.
            
            Format: Hex(i, j, k), where i, j, and k are the line coordinates.

            Returns:
               - str: The string representation of the hex.

         .. rust:function:: hpyhex::Hex::__rsub__
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__rsub__"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"other"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"Bound"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"PyAny","target":"PyAny"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"Self","target":"Self"},{"type":"punctuation","value":">"}]

            Reverse subtraction of this hex from another Hex or a tuple.
            
            Arguments:
               - other (Hex or tuple): The value to subtract this hex from.
            Returns:
               - Hex: A new Hex with the subtracted coordinates.
            Raises:
               - TypeError: If the other operand is not a Hex or a tuple of coordinates.

         .. rust:function:: hpyhex::Hex::__str__
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__str__"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"String","target":"String"}]

            Return a string representation of the hex coordinates.
            
            Format: Hex(i, j, k), where i, j, and k are the line coordinates.

            Returns:
               - str: The string representation of the hex.

         .. rust:function:: hpyhex::Hex::__sub__
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__sub__"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"other"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"Bound"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"PyAny","target":"PyAny"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"Self","target":"Self"},{"type":"punctuation","value":">"}]

            Subtract another Hex or a tuple of coordinates from this hex.
            
            Arguments:
               - other (Hex or tuple): The value to subtract.
            Returns:
               - Hex: A new Hex with the subtracted coordinates.
            Raises:
               - TypeError: If the other operand is not a Hex or a tuple of coordinates.

         .. rust:function:: hpyhex::Hex::i
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"i"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"i32","target":"i32"}]

            Get the I-line coordinate of the hex.
            
            Returns:
               - int: The I-line coordinate.

         .. rust:function:: hpyhex::Hex::j
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"j"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"i32","target":"i32"}]

            Get the J-line coordinate of the hex.
            
            Returns:
               - int: The J-line coordinate.

         .. rust:function:: hpyhex::Hex::k
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"k"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"i32","target":"i32"}]

            Get the K-line coordinate of the hex.
            
            Returns:
               - int: The K-line coordinate.

         .. rust:function:: hpyhex::Hex::new
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"new"},{"type":"punctuation","value":"("},{"type":"name","value":"i"},{"type":"punctuation","value":": "},{"type":"link","value":"i32","target":"i32"},{"type":"punctuation","value":", "},{"type":"name","value":"k"},{"type":"punctuation","value":": "},{"type":"link","value":"i32","target":"i32"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"Self","target":"Self"}]

            Initialize a Hex coordinate at (i, k). Defaults to (0, 0).
            
            Arguments:
               - i (int): The I-line coordinate of the hex, or a tuple (i, k) or (i, j, k).
               - k (int): The K-line coordinate of the hex.
            Returns:
               - Hex
            Raises:
               - TypeError: If i or k is not an integer.

         .. rust:function:: hpyhex::Hex::shift_i
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"shift_i"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"units"},{"type":"punctuation","value":": "},{"type":"link","value":"i32","target":"i32"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"Self","target":"Self"}]

            Return a new Hex shifted along the i-axis by units.
            
            Arguments:
               - units (int): The number of units to shift along the i-axis.
            Returns:
               - Hex: A new Hex shifted by the specified units along the i-axis.
            Raises:
               - TypeError: If units is not an integer.

         .. rust:function:: hpyhex::Hex::shift_j
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"shift_j"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"units"},{"type":"punctuation","value":": "},{"type":"link","value":"i32","target":"i32"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"Self","target":"Self"}]

            Return a new Hex shifted along the j-axis by units.
            
            Arguments:
               - units (int): The number of units to shift along the j-axis.
            Returns:
               - Hex: A new Hex shifted by the specified units along the j-axis.
            Raises:
               - TypeError: If units is not an integer.

         .. rust:function:: hpyhex::Hex::shift_k
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"shift_k"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"units"},{"type":"punctuation","value":": "},{"type":"link","value":"i32","target":"i32"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"Self","target":"Self"}]

            Return a new Hex shifted along the k-axis by units.
            
            Arguments:
               - units (int): The number of units to shift along the k-axis.
            Returns:
               - Hex: A new Hex shifted by the specified units along the k-axis.
            Raises:
               - TypeError: If units is not an integer.


   .. rust:struct:: hpyhex::HexEngine
      :index: 1
      :vis: pub
      :toc: struct HexEngine
      :layout: [{"type":"keyword","value":"struct"},{"type":"space"},{"type":"name","value":"HexEngine"}]

      The HexEngine class provides a complete engine for managing a two-dimensional hexagonal
      block grid used for constructing and interacting with hex-based shapes in the game.
      
      The engine does not actually contain any blocks, but instead contains a list of booleans
      representing the occupancy state of each block in the hexagonal grid. The correspondence is achieved
      through optimized indexing and coordinate transformations.
      
      Grid Structure:
         - Uses an axial coordinate system (i, k), where i - j + k = 0, and j is derived as j = i + k.
         - Three axes: I, J, K. I+ is 60° from J+, J+ is 60° from K+, K+ is 60° from I-.
         - Raw coordinates: distance along an axis multiplied by 2.
         - Line-coordinates (I, K) are perpendicular distances to axes, calculated from raw coordinates.
         - Blocks are stored in a sorted array by increasing raw coordinate i, then k.
      
      Grid Size:
         - Total blocks for radius r: Aₖ = 1 + 3*r*(r-1)
         - Derived from: Aₖ = Aₖ₋₁ + 6*(k-1); A₁ = 1
      
      Machine Learning:
         - Supports reward functions for evaluating action quality.
         - check_add discourages invalid moves (e.g., overlaps).
         - compute_dense_index evaluates placement density for rewarding efficient gap-filling.
      
      Attributes:
         - radius (int): The radius of the hexagonal grid, defining the size of the grid.
         - states (list[bool]): A list of booleans representing the occupancy state of each block in the grid.

      .. rubric:: Implementations


      .. rust:impl:: hpyhex::HexEngine
         :index: -1
         :vis: pub
         :layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"HexEngine","target":"HexEngine"}]
         :toc: impl HexEngine


      .. rust:impl:: hpyhex::HexEngine
         :index: -1
         :vis: pub
         :layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"HexEngine","target":"HexEngine"}]
         :toc: impl HexEngine


         .. rubric:: Functions


         .. rust:function:: hpyhex::HexEngine::__copy__
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__copy__"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"Self","target":"Self"}]

            Create a deep copy of the HexEngine.
            
            Returns:
            HexEngine: A new HexEngine with the same radius and states.

         .. rust:function:: hpyhex::HexEngine::__deepcopy__
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__deepcopy__"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"memo"},{"type":"punctuation","value":": "},{"type":"link","value":"Option","target":"Option"},{"type":"punctuation","value":"<"},{"type":"punctuation","value":"&"},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"Bound"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"PyAny","target":"PyAny"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"Self","target":"Self"}]

            Create a deep copy of the HexEngine.
            Arguments:
               - memo (dict): A dictionary to keep track of copied objects.
            Returns:
               - HexEngine: A new HexEngine instance with the same radius and blocks.

         .. rust:function:: hpyhex::HexEngine::__eliminate_i
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__eliminate_i"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"eliminated"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"Bound"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"PyList","target":"PyList"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"punctuation","value":"("},{"type":"punctuation","value":")"},{"type":"punctuation","value":">"}]

            Identify coordinates along I axis that can be eliminated and insert them into the input list
            
            Arguments:
               - eliminate (list[Hex]): Mutable list to append eliminated coordinates

         .. rust:function:: hpyhex::HexEngine::__eliminate_j
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__eliminate_j"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"eliminated"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"Bound"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"PyList","target":"PyList"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"punctuation","value":"("},{"type":"punctuation","value":")"},{"type":"punctuation","value":">"}]

            Identify coordinates along J axis that can be eliminated and insert them into the input list
            
            Arguments:
               - eliminate (list[Hex]): Mutable list to append eliminated coordinates

         .. rust:function:: hpyhex::HexEngine::__eliminate_k
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__eliminate_k"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"eliminated"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"Bound"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"PyList","target":"PyList"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"punctuation","value":"("},{"type":"punctuation","value":")"},{"type":"punctuation","value":">"}]

            Identify coordinates along K axis that can be eliminated and insert them into the input list
            
            Arguments:
               - eliminate (list[Hex]): Mutable list to append eliminated coordinates

         .. rust:function:: hpyhex::HexEngine::__eq__
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__eq__"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"value"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"Bound"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"PyAny","target":"PyAny"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"bool","target":"bool"},{"type":"punctuation","value":">"}]

            Check equality with another HexEngine or a list of booleans.
            Returns True if the states match, False otherwise.
            
            Arguments:
               - value (HexEngine | list[bool]): The HexEngine or list of booleans to compare with.
            Returns:
               - bool: True if the states match, False otherwise.

         .. rust:function:: hpyhex::HexEngine::__hash__
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__hash__"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"u64","target":"u64"}]

            Return a hash of the HexEngine's occupancy states.
            This method uses the tuple representation of the states for hashing.
            
            Returns:
               - int: The hash value of the HexEngine.

         .. rust:function:: hpyhex::HexEngine::__in_range
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__in_range"},{"type":"punctuation","value":"("},{"type":"name","value":"coo"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"Bound"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"PyAny","target":"PyAny"},{"type":"punctuation","value":">"},{"type":"punctuation","value":", "},{"type":"name","value":"radius"},{"type":"punctuation","value":": "},{"type":"link","value":"usize","target":"usize"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"bool","target":"bool"},{"type":"punctuation","value":">"}]

            Check if a Hex coordinate is within the specified radius of the hexagonal grid.
            
            Arguments:
               - coo: Hex coordinate to check.
               - radius: Radius of the hexagonal grid.
            Returns:
               - bool: True if the coordinate is within range, False otherwise.

         .. rust:function:: hpyhex::HexEngine::__iter__
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__iter__"},{"type":"punctuation","value":"("},{"type":"name","value":"slf"},{"type":"punctuation","value":": "},{"type":"link","value":"PyRef","target":"PyRef"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"Self","target":"Self"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"Py","target":"Py"},{"type":"punctuation","value":"<"},{"type":"link","value":"PyAny","target":"PyAny"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"}]

            Return an iterator over the occupancy states of the hexagonal grid blocks.
            
            Yields:
               - bool: The occupancy state of each block in the grid.

         .. rust:function:: hpyhex::HexEngine::__len__
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__len__"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"usize","target":"usize"}]

            Get the number of blocks in the hexagonal grid.
            
            Returns:
               - int: The number of blocks in the grid.

         .. rust:function:: hpyhex::HexEngine::__repr__
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__repr__"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"String","target":"String"}]

            Return a string representation of the grid block states.
            This representation is useful for debugging and serialization.
            Format: "1" for occupied blocks, "0" for unoccupied blocks.
            
            Returns:
               - str: A string representation of the grid block states.

         .. rust:function:: hpyhex::HexEngine::__str__
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__str__"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"String","target":"String"}]

            Return a string representation of the grid block states.
            Format: "HexEngine[blocks = {block1, block2, ...}]",
            where each block is represented by its string representation.
            
            Returns:
               - str: The string representation of the HexEngine.

         .. rust:function:: hpyhex::HexEngine::add_piece
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"add_piece"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"coo"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"Bound"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"PyAny","target":"PyAny"},{"type":"punctuation","value":">"},{"type":"punctuation","value":", "},{"type":"name","value":"piece"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"Bound"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"PyAny","target":"PyAny"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"punctuation","value":"("},{"type":"punctuation","value":")"},{"type":"punctuation","value":">"}]

            Add a Piece to the hexagonal grid at the specified Hex coordinate.
            
            This method places the Piece on the grid, updating the occupancy state of
            the blocks based on the Piece's states. If the Piece cannot be added due to
            overlaps or out-of-range coordinates, it raises a ValueError.
            
            Arguments:
               - coo (Hex | tuple): The Hex coordinate to add the Piece.
               - piece (Piece | int): The Piece to add to the grid.
            Raises:
               - ValueError: If the Piece cannot be added due to overlaps or out-of-range coordinates.
               - TypeError: If piece is not a valid Piece instance.

         .. rust:function:: hpyhex::HexEngine::all_engines
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"all_engines"},{"type":"punctuation","value":"("},{"type":"name","value":"_cls"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"Bound"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"PyType","target":"PyType"},{"type":"punctuation","value":">"},{"type":"punctuation","value":", "},{"type":"name","value":"radius"},{"type":"punctuation","value":": "},{"type":"link","value":"usize","target":"usize"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"Vec","target":"Vec"},{"type":"punctuation","value":"<"},{"type":"link","value":"HexEngine","target":"HexEngine"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"}]

            Generate all possible HexEngine instances representing valid occupancy states for a given radius.
            All generated HexEngines will have eliminations already applied, meaning they will not contain any fully occupied lines.
            
            For large radius values, this method may take a long time and significant resource to compute due to the exponential growth of possible states.
            It is recommended to cache the results for specific radius values to avoid recomputation. HexEngine does not provide a dictionary for caching such data.
            
            Arguments:
               - radius (int): The radius of the hexagonal grid for which to generate all possible HexEngines.
            Returns:
               - list[HexEngine]: A list of HexEngine instances representing all valid occupancy states for the specified radius.
            Raises:
               - TypeError: If radius is not an integer greater than 1. Only empty engine is valid for radius 1.

         .. rust:function:: hpyhex::HexEngine::check_add
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"check_add"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"coo"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"Bound"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"PyAny","target":"PyAny"},{"type":"punctuation","value":">"},{"type":"punctuation","value":", "},{"type":"name","value":"piece"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"Bound"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"PyAny","target":"PyAny"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"bool","target":"bool"},{"type":"punctuation","value":">"}]

            Check if a Piece can be added to the hexagonal grid without overlaps.
            
            This method checks if the Piece can be placed on the grid without overlapping
            any existing occupied blocks. It returns True if the Piece can be added,
            otherwise returns False.
            
            Arguments:
               - coo (Hex | tuple): The Hex coordinate to check for addition.
               - piece (Piece | int): The Piece to check for addition.
            Returns:
               - bool: True if the Piece can be added, False otherwise.
            Raises:
               - TypeError: If piece is not a Piece instance.

         .. rust:function:: hpyhex::HexEngine::check_positions
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"check_positions"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"piece"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"Bound"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"PyAny","target":"PyAny"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"Vec","target":"Vec"},{"type":"punctuation","value":"<"},{"type":"link","value":"Py","target":"Py"},{"type":"punctuation","value":"<"},{"type":"link","value":"Hex","target":"Hex"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"}]

            Return all valid positions where another grid can be added.
            
            This method returns a list of Hex coordinate candidates where the Piece can be added
            without overlaps. It checks each position in the Piece and returns the Hex coordinates
            of the occupied blocks.
            If the Piece is not valid, it raises a ValueError.
            
            Arguments:
               - piece (Piece): The Piece to check for occupied positions.
            Returns:
               - list[Hex]: A list of Hex coordinates for the occupied blocks in the Piece.
            Raises:
               - TypeError: If the piece is not a valid Piece instance.

         .. rust:function:: hpyhex::HexEngine::compute_dense_index
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"compute_dense_index"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"coo"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"Bound"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"PyAny","target":"PyAny"},{"type":"punctuation","value":">"},{"type":"punctuation","value":", "},{"type":"name","value":"piece"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"Bound"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"PyAny","target":"PyAny"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"f64","target":"f64"},{"type":"punctuation","value":">"}]

            Compute a density index score for hypothetically placing another piece.
            
            Returns a value between 0 and 1 representing surrounding density.
            A score of 1 means all surrounding blocks would be filled, 0 means the grid would be alone.
            
            Arguments:
               - coo (Hex): Position for hypothetical placement.
               - piece (Piece): The Piece to evaluate for placement.
            Returns:
               - float: Density index (0 to 1), or 0 if placement is invalid or no neighbors exist.

         .. rust:function:: hpyhex::HexEngine::compute_entropy
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"compute_entropy"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"f64","target":"f64"},{"type":"punctuation","value":">"}]

            Compute the entropy of the hexagonal grid based on the distribution of 7-block patterns.
            
            Entropy is calculated using the Shannon entropy formula, measuring the randomness of block
            arrangements in the grid. Each pattern consists of a central block and its six neighbors,
            forming a 7-block hexagonal box, as defined by the _get_pattern method. The entropy reflects
            the diversity of these patterns: a grid with randomly distributed filled and empty blocks
            has higher entropy than one with structured patterns (e.g., all blocks in a line or cluster).
            A grid with all blocks filled or all empty has zero entropy. Inverting the grid (swapping
            filled and empty states) results in the same entropy, as the pattern distribution is unchanged.
            
            The method iterates over all blocks within the grid's radius (excluding the outermost layer
            to ensure all neighbors are in range), counts the frequency of each possible 7-block pattern
            (2^7 = 128 patterns), and computes the entropy using the Shannon entropy formula:
                H = -Σ (p * log₂(p))
            where p is the probability of each pattern (frequency divided by total patterns counted).
            Blocks on the grid's boundary (beyond radius - 1) are excluded to avoid incomplete patterns.
            
            Returns:
               - entropy (float): The entropy of the grid in bits, a non-negative value representing the randomness
                 of block arrangements. Returns 0.0 for a uniform grid (all filled or all empty) or if no valid patterns are counted.

         .. rust:function:: hpyhex::HexEngine::coordinate_block
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"coordinate_block"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"index"},{"type":"punctuation","value":": "},{"type":"link","value":"usize","target":"usize"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"Py","target":"Py"},{"type":"punctuation","value":"<"},{"type":"link","value":"Hex","target":"Hex"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"}]

            Get the Hex coordinate of the Block at the specified index.
            
            This method retrieves the Hex coordinate based on the index in the hexagonal grid.
            If the index is out of range, raise ValueError.
            
            Arguments:
               - index (int): The index of the Block.
            Returns:
               - Hex: The Hex coordinate of the Block.
            Raises:
               - TypeError: If index is not an integer.
               - ValueError: If the index is out of range.

         .. rust:function:: hpyhex::HexEngine::count_neighbors
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"count_neighbors"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"coo"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"Bound"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"PyAny","target":"PyAny"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"usize","target":"usize"},{"type":"punctuation","value":">"}]

            Count occupied neighboring Blocks around the given Hex position.
            
            Checks up to six adjacent positions to the block at Hex coordinate.
            A neighbor is occupied if the block is null or its state is True 
            
            Arguments:
               - coo (Hex | tuple): The Hex coordinate to check for neighbors.
            Returns:
               - int: The count of occupied neighboring Blocks.
            Raises:
               - TypeError: If coo is not a Hex or a tuple of coordinates.

         .. rust:function:: hpyhex::HexEngine::eliminate
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"eliminate"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"Vec","target":"Vec"},{"type":"punctuation","value":"<"},{"type":"link","value":"Py","target":"Py"},{"type":"punctuation","value":"<"},{"type":"link","value":"Hex","target":"Hex"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"}]

            Eliminate fully occupied lines along I, J, or K axes and return eliminated coordinates.
            
            Modifies the grid permanently.
            
            Returns:
               - list[Hex]: A list of Hex coordinates that were eliminated.

         .. rust:function:: hpyhex::HexEngine::get_pattern
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"get_pattern"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"coo"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"Bound"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"PyAny","target":"PyAny"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"u8","target":"u8"},{"type":"punctuation","value":">"}]

            Determine the pattern of blocks around the given position in the hexagonal grid, including the block itself.
            
            This method checks up to seven positions in a hexagonal box centered at coordinates (i, k).
            It returns a value representing the pattern of occupied/unoccupied blocks, ignoring block colors.
            The pattern is encoded as a 7-bit integer (0 to 127) based on the state of the central block
            and its six neighbors. If a neighboring position is out of range or contains a None block,
            it is treated as occupied or unoccupied based on the include_null flag.
            
            Arguments:
               - coo (Hex | tuple): The hex coordinate of the block at the center of the box.
            Returns:
               - pattern (int): A number in the range [0, 127] representing the pattern of blocks in the hexagonal box.

         .. rust:function:: hpyhex::HexEngine::get_state
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"get_state"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"coo"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"Bound"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"PyAny","target":"PyAny"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"bool","target":"bool"},{"type":"punctuation","value":">"}]

            Get the Block occupancy state at the specified Hex coordinate or index.
            
            This method retrieves the Block state based on either a Hex coordinate or an index.
            If the coordinate or index is out of range, raise ValueError.
            
            Arguments:
               - coo (Hex | tuple | int): The Hex coordinate or index of the Block.
            Returns:
               - bool: The occupancy state of the Block.
            Raises:
               - TypeError: If coo is not a Hex, tuple, or integer.
               - ValueError: If the coordinate or index is out of range.

         .. rust:function:: hpyhex::HexEngine::in_range
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"in_range"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"coo"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"Bound"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"PyAny","target":"PyAny"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"bool","target":"bool"},{"type":"punctuation","value":">"}]

            Check if a Hex coordinate is within the radius of the hexagonal grid.
            
            Arguments:
               - coo: Hex coordinate to check.
            Returns:
               - bool: True if the coordinate is within range, False otherwise.

         .. rust:function:: hpyhex::HexEngine::index_block
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"index_block"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"coo"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"Bound"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"PyAny","target":"PyAny"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"isize","target":"isize"},{"type":"punctuation","value":">"}]

            Get the index of the Block at the specified Hex coordinate.
            
            This method is heavily optimized for performance and achieves O(1) complexity by using direct index formulas
            based on the hexagonal grid's structure. It calculates the index based on the I and K coordinates of the Hex.
            
            Arguments:
               - coo: The Hex coordinate.
            Returns:
               - int: The index of the Block, or -1 if out of range.

         .. rust:function:: hpyhex::HexEngine::new
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"new"},{"type":"punctuation","value":"("},{"type":"name","value":"arg"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"Bound"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"PyAny","target":"PyAny"},{"type":"punctuation","value":">"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"Self","target":"Self"},{"type":"punctuation","value":">"}]

            Construct a HexEngine with the specified radius, states, or string.
            
            This method initializes the hexagonal grid with a given radius,
            creating an array of booleans to represent the grid.
            
            Arguments:
               - arg (int | str | list[bool]):
                  - An integer representing the radius of the hexagonal grid.
                  - A list of booleans representing the occupancy state of each block.
                  - A string representation of the occupancy state, either as 'X'/'O' or '1'/'0'.
            Raises:
               - TypeError: If radius is not an integer greater than 0, or if the list contains non-boolean values.
               - ValueError: If radius is less than 1, or if the length of the list/string does not match a valid hexagonal grid size.

         .. rust:function:: hpyhex::HexEngine::radius
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"radius"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"usize","target":"usize"}]

            Get the radius of the hexagonal grid.
            
            Returns:
               - int: The radius of the hexagonal grid.

         .. rust:function:: hpyhex::HexEngine::reset
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"reset"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"}]

            Reset the HexEngine grid to its initial state, clearing all blocks.
            This method reinitializes the grid, setting all blocks to unoccupied.
            
            Returns:
               - None

         .. rust:function:: hpyhex::HexEngine::set_state
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"set_state"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"mut"},{"type":"space"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"coo"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"pyo3","target":"pyo3"},{"type":"punctuation","value":"::"},{"type":"name","value":"Bound"},{"type":"punctuation","value":"<"},{"type":"lifetime","value":"'_"},{"type":"punctuation","value":", "},{"type":"link","value":"PyAny","target":"PyAny"},{"type":"punctuation","value":">"},{"type":"punctuation","value":", "},{"type":"name","value":"state"},{"type":"punctuation","value":": "},{"type":"link","value":"bool","target":"bool"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"punctuation","value":"("},{"type":"punctuation","value":")"},{"type":"punctuation","value":">"}]

            Set the occupancy state of the Block at the specified Hex coordinate.
            
            This method updates the state of a Block at the given coordinate.
            If the coordinate is out of range, raise ValueError.
            
            Arguments:
               - coo (Hex | tuple | int): The Hex coordinate or index of the block to set.
               - state (bool): The new occupancy state to set for the Block.
            Raises:
               - ValueError: If the coordinate is out of range.
               - TypeError: If the coordinate type is unsupported, or state is not a boolean.

         .. rust:function:: hpyhex::HexEngine::solve_length
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"solve_length"},{"type":"punctuation","value":"("},{"type":"name","value":"radius"},{"type":"punctuation","value":": "},{"type":"link","value":"usize","target":"usize"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"isize","target":"isize"}]

            Solves for the length of a HexEngine based on its radius.
            
            Arguments:
               - radius (int): The radius of the hexagonal grid.
            Returns:
               - int: The length of the hexagonal grid, or -1 if the radius is invalid.

         .. rust:function:: hpyhex::HexEngine::solve_radius
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"solve_radius"},{"type":"punctuation","value":"("},{"type":"name","value":"length"},{"type":"punctuation","value":": "},{"type":"link","value":"usize","target":"usize"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"isize","target":"isize"}]

            Solves for the radius of a HexEngine based on its length.
            
            Arguments:
               - radius (int): The radius of the hexagonal grid.
            Returns:
               - int: The radius of the hexagonal grid, or -1 if the length is invalid.

         .. rust:function:: hpyhex::HexEngine::states
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"states"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"Vec","target":"Vec"},{"type":"punctuation","value":"<"},{"type":"link","value":"bool","target":"bool"},{"type":"punctuation","value":">"}]

            Get the occupancy states of the hexagonal grid blocks.
            
            Returns:
               - list[bool]: The occupancy states of the hexagonal grid blocks.


   .. rust:struct:: hpyhex::Piece
      :index: 1
      :vis: pub
      :toc: struct Piece
      :layout: [{"type":"keyword","value":"struct"},{"type":"space"},{"type":"name","value":"Piece"}]

      Represents a shape or unit made up of 7 Block instances,
      typically forming a logical structure such as a game piece.
      
      This implementation of piece contains no blocks, and instead only contains 
      a single u8 value representing the occupancy state of each block (7 bits used).
      
      This is a singleton class, meaning that each unique Piece state is cached
      and reused to save memory and improve performance.
      
      Attributes:
         - positions (list[Hex]): A list of Hex coordinates representing the positions of the blocks in the piece.
         - state (u8): A byte value representing the occupancy state of each block in the piece.

      .. rubric:: Implementations

      .. rust:impl:: hpyhex::Piece
         :index: -1
         :vis: pub
         :layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"Piece","target":"Piece"}]
         :toc: impl Piece

         .. rubric:: Variables
         
         .. rust:variable:: hpyhex::Piece::positions
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"pub"},{"type":"space"},{"type":"name","value":"positions"},{"type":"punctuation","value":":"},{"type":"space"},{"type":"link","value":"Vec","target":"Vec"},{"type":"punctuation","value":"<"},{"type":"link","value":"Hex","target":"Hex"},{"type":"punctuation","value":">"}]

            The fixed positions of the 7 blocks in a standard Piece.

            Returns:
               - list[Hex]: A list of Hex coordinates representing the positions of the blocks in the piece.

         .. rubric:: Functions


         .. rust:function:: hpyhex::Piece::__bool__
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__bool__"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"bool","target":"bool"}]

            Check if the Piece has any occupied blocks.
            
            Returns:
               - bool: True if any block is occupied, False otherwise.

         .. rust:function:: hpyhex::Piece::__eq__
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__eq__"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"other"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"Piece","target":"Piece"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"bool","target":"bool"}]

            Returns True if the occupancy states match, False otherwise.
            
            Arguments:
               - other (Piece): The Piece to compare with.
            Returns:
               - bool: True if the occupancy states match, False otherwise.

         .. rust:function:: hpyhex::Piece::__hash__
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__hash__"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"u8","target":"u8"}]

            Return a hash of the Piece's occupancy states.
            
            This method directly uses the byte representation of the Piece to generate a hash value.

            Returns:
               - int: The hash value of the Piece.

         .. rust:function:: hpyhex::Piece::__int__
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__int__"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"u8","target":"u8"}]

            Return a byte representation of the blocks in a standard 7-Block piece.
            
            Returns:
               - int: A byte representation of the Piece, where each bit represents the occupancy state of a

         .. rust:function:: hpyhex::Piece::__iter__
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__iter__"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"Py","target":"Py"},{"type":"punctuation","value":"<"},{"type":"link","value":"PyAny","target":"PyAny"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"}]

            Return an iterator over the occupancy states of the Piece.
            
            Yields:
               - bool: The occupancy state of each block in the Piece.

         .. rust:function:: hpyhex::Piece::__len__
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__len__"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"usize","target":"usize"}]

            Return the number of occupied blocks in the Piece.
            
            Returns:
               - int: The number of occupied blocks in the Piece.

         .. rust:function:: hpyhex::Piece::__repr__
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__repr__"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"String","target":"String"}]

            Return a string representation of the Piece in byte format.
            This representation is useful for debugging and serialization.
            
            Returns:
               - str: A string representation of the Piece in byte format.

         .. rust:function:: hpyhex::Piece::__str__
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"__str__"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"String","target":"String"}]

            Return a string representation of the Piece.
            
            Format: Piece{Block(i, j, k, state), ...}, where i, j, and k are the line coordinates of each block,
            and state is the occupancy state, if occupied, else "null".
            
            Returns:
               - str: The string representation of the Piece.

         .. rust:function:: hpyhex::Piece::all_pieces
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"all_pieces"},{"type":"punctuation","value":"("},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"Py","target":"Py"},{"type":"punctuation","value":"<"},{"type":"link","value":"PyAny","target":"PyAny"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"}]

            Get a list of all possible non-empty Piece instances.
            This method returns all cached Piece instances representing different occupancy states.
            
            The return of this method does not guarantee that pieces are spacially contigous.

            Returns:
               - list[Piece]: A list of all possible Piece instances.

         .. rust:function:: hpyhex::Piece::contigous_pieces
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"contigous_pieces"},{"type":"punctuation","value":"("},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"Py","target":"Py"},{"type":"punctuation","value":"<"},{"type":"link","value":"PyAny","target":"PyAny"},{"type":"punctuation","value":">"},{"type":"punctuation","value":">"}]

            Get a list of all possible contigous Piece instances.
            This method returns all cached Piece instances representing different occupancy states
            that are spatially contiguous.
            
            Returns:
               - slist[Piece]: A list of all possible contigous Piece instances.

         .. rust:function:: hpyhex::Piece::coordinates
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"coordinates"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"Vec","target":"Vec"},{"type":"punctuation","value":"<"},{"type":"link","value":"Hex","target":"Hex"},{"type":"punctuation","value":">"}]

            Get the list of Hex coordinates representing the positions of the blocks in the Piece.
            
            Returns:
               - list[Hex]: The list of Hex coordinates for the Piece.

         .. rust:function:: hpyhex::Piece::count_neighbors
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"count_neighbors"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":", "},{"type":"name","value":"coo"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"Hex","target":"Hex"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"usize","target":"usize"}]

            Count occupied neighboring Blocks around the given Hex position.
            
            Checks up to six adjacent positions to the block at Hex coordinate.
            A neighbor is occupied if the block is non-null and its state is True.
            
            Parameters:
               - coo (Hex | tuple): The Hex coordinate to check for neighbors.
            Returns:
               - int: The count of occupied neighboring Blocks.
            Raises:
               - TypeError: If coo is not a Hex or a tuple of coordinates.

         .. rust:function:: hpyhex::Piece::states
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"states"},{"type":"punctuation","value":"("},{"type":"punctuation","value":"&"},{"type":"keyword","value":"self"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"punctuation","value":"["},{"type":"link","value":"bool","target":"bool"},{"type":"punctuation","value":"; "},{"type":"literal","value":"7"},{"type":"punctuation","value":"]"}]

            Get the tuple of boolean values representing the occupancy state of each block in the Piece.
            
            Returns:
               - tuple[bool, ...]: The tuple of boolean values for the Piece.


   .. rust:struct:: hpyhex::PieceFactory
      :index: 1
      :vis: pub
      :toc: struct PieceFactory
      :layout: [{"type":"keyword","value":"struct"},{"type":"space"},{"type":"name","value":"PieceFactory"}]

      PieceFactory is a utility class that provides methods for creating and managing game pieces.
      It includes a predefined set of pieces, their corresponding byte values, and reverse mappings
      to retrieve piece names from byte values. The class also supports generating random pieces
      based on predefined probabilities.
      
      Attributes:
         - pieces (dict): A dictionary mapping piece names (str) to their corresponding byte values (int).
         - reverse_pieces (dict): A reverse mapping of `pieces`, mapping byte values (int) to piece names (str).

      .. rubric:: Implementations


      .. rust:impl:: hpyhex::PieceFactory
         :index: -1
         :vis: pub
         :layout: [{"type":"keyword","value":"impl"},{"type":"space"},{"type":"link","value":"PieceFactory","target":"PieceFactory"}]
         :toc: impl PieceFactory


         .. rubric:: Functions


         .. rust:function:: hpyhex::PieceFactory::all_pieces
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"all_pieces"},{"type":"punctuation","value":"("},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"Vec","target":"Vec"},{"type":"punctuation","value":"<"},{"type":"link","value":"Piece","target":"Piece"},{"type":"punctuation","value":">"}]

            Return all pieces that are defined in this factory.
            
            # Returns
               - list[Piece]: A list of all Piece instances defined in the factory.

         .. rust:function:: hpyhex::PieceFactory::generate_piece
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"generate_piece"},{"type":"punctuation","value":"("},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"Piece","target":"Piece"},{"type":"punctuation","value":">"}]

            Generate a random piece based on frequency distribution.
            
            # Returns
               - Piece: A randomly selected piece object.

         .. rust:function:: hpyhex::PieceFactory::get_piece
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"get_piece"},{"type":"punctuation","value":"("},{"type":"name","value":"name"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"str","target":"str"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"Piece","target":"Piece"},{"type":"punctuation","value":">"}]

            Get a piece by its name.
            
            # Arguments
               - name (str): The name of the piece to retrieve.
            # Returns
               - Piece: The piece object corresponding to the given name.
            # Raises
               - ValueError: If the piece name is not found in the factory.

         .. rust:function:: hpyhex::PieceFactory::get_piece_name
            :index: -1
            :vis: pub
            :layout: [{"type":"keyword","value":"fn"},{"type":"space"},{"type":"name","value":"get_piece_name"},{"type":"punctuation","value":"("},{"type":"name","value":"p"},{"type":"punctuation","value":": "},{"type":"punctuation","value":"&"},{"type":"link","value":"Piece","target":"Piece"},{"type":"punctuation","value":")"},{"type":"space"},{"type":"returns"},{"type":"space"},{"type":"link","value":"PyResult","target":"PyResult"},{"type":"punctuation","value":"<"},{"type":"link","value":"String","target":"String"},{"type":"punctuation","value":">"}]

            Get the name of a piece based on its byte value.
            
            # Arguments
               - p (Piece): The piece object whose name is to be retrieved.
            # Returns
               - String: The name of the piece corresponding to the given byte value.
            # Raises
               - ValueError: If the piece byte value is not found in the factory.
