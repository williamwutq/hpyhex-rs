================
Crate ``hpyhex``
================


.. rust:crate:: hpyhex
   :index: 0


   .. rust:use:: hpyhex
      :used_name: self


   .. rust:use:: hpyhex
      :used_name: crate


   .. rust:use:: std::collections::HashMap
      :used_name: HashMap


   .. rust:use:: std::sync::OnceLock
      :used_name: OnceLock


   .. rust:use:: std::hash::Hash
      :used_name: Hash


   .. rust:use:: std::hash::Hasher
      :used_name: Hasher


   .. rubric:: Structs and Unions


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


