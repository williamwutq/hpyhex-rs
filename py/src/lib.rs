#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;
use pyo3::types::{PyList, PyAny, PyType};

#[pymodule]
fn hpyhex(_py: Python, m: &pyo3::Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Hex>()?;
    m.add_class::<Piece>()?;
    Ok(())
}


use std::sync::OnceLock;
use std::hash::{Hash, Hasher};

/// Represents a hexagonal grid coordinate using a custom line-based coordinate system.
///
/// This class models hexagonal positions with two line coordinates (i, k), implicitly
/// defining the third axis (j) as `j = k - i` to maintain hex grid constraints.
/// It supports standard arithmetic, equality, and hashing operations, as well as
/// compatibility with coordinate tuples.
///
/// For small grids, Hex instances are cached for performance, allowing more efficient memory usage
/// and faster access. The caching is limited to a range of -64 to 64 for both i and k coordinates.
///
/// Use of Hex over tuples is recommended for clarity and to leverage the singleton feature of small Hexes.
///
/// Coordinate Systems:
/// - Raw Coordinates (i, j, k): Three axes satisfying i + j + k = 0, where
///   each axis is diagonal to the others at 60° increments.
/// - Line Coordinates (i, k): Derived coordinates representing distances
///   perpendicular to axes, simplifying grid operations.
///
/// Note:
/// - This class is immutable and optimized with __slots__.
/// - Raw coordinate methods (__i__, __j__, __k__) are retained for backward compatibility.
/// - Only basic functionality is implemented; complex adjacency, iteration,
///   and mutability features are omitted for simplicity.
///
/// Attributes:
/// - i (int): The line i coordinate.
/// - j (int): The computed line j coordinate (k - i).
/// - k (int): The line k coordinate.
#[pyclass(frozen)]
#[derive(Eq, Clone)]
pub struct Hex {
    i: i32,
    k: i32,
}

impl PartialEq for Hex {
    fn eq(&self, other: &Self) -> bool {
        self.i == other.i && self.k == other.k
    }
}

// Caching for small grids: -64..=64 for both i and k
const CACHE_MIN: i32 = -64;
const CACHE_MAX: i32 = 64;
// Singleton cache for all small Hexes in the range -64..=64 for both i and k
static HEX_CACHE: OnceLock<[[Option<Py<Hex>>; (CACHE_MAX - CACHE_MIN + 1) as usize]; (CACHE_MAX - CACHE_MIN + 1) as usize]> = OnceLock::new();

fn initialize_hex_cache(py: Python) -> [[Option<Py<Hex>>; (CACHE_MAX - CACHE_MIN + 1) as usize]; (CACHE_MAX - CACHE_MIN + 1) as usize] {
    use std::mem::MaybeUninit;
    const N: usize = (CACHE_MAX - CACHE_MIN + 1) as usize;
    let mut cache: [[MaybeUninit<Option<Py<Hex>>>; N]; N] = unsafe { MaybeUninit::uninit().assume_init() };
    for i in 0..N {
        for k in 0..N {
            let hex = Hex { i: (i as i32) + CACHE_MIN, k: (k as i32) + CACHE_MIN };
            cache[i][k].write(Some(Py::new(py, hex).unwrap()));
        }
    }
    unsafe { std::mem::transmute::<_, [[Option<Py<Hex>>; N]; N]>(cache) }
}

fn get_hex(i: i32, k: i32) -> Py<Hex> {
    if (CACHE_MIN..=CACHE_MAX).contains(&i) && (CACHE_MIN..=CACHE_MAX).contains(&k) {
        Python::with_gil(|py| {
            let cache = HEX_CACHE.get_or_init(|| initialize_hex_cache(py));
            let idx_i = (i - CACHE_MIN) as usize;
            let idx_k = (k - CACHE_MIN) as usize;
            if let Some(hex) = &cache[idx_i][idx_k] {
                return hex.clone_ref(py);
            }
            // This branch should never be hit, but fallback just in case
            Py::new(py, Hex { i, k }).unwrap()
        })
    } else {
        Python::with_gil(|py| Py::new(py, Hex { i, k }).unwrap())
    }
}

impl Into<(i32, i32)> for Hex {
    fn into(self) -> (i32, i32) {
        (self.i, self.k)
    }
}

impl Hex {
    /// Convert the Hex coordinates to a byte vector representation.
    ///
    /// Arguments:
    /// - minimize (bool): If true, use i16 representation if possible.
    /// Returns:
    /// - Vec<u8>: The byte vector representation of the Hex coordinates.
    pub fn to_bytes(&self, minimize: bool) -> Vec<u8> {
        if minimize {
            // Try i16
            if let (Ok(i16_i), Ok(i16_k)) = (i16::try_from(self.i), i16::try_from(self.k)) {
                let mut bytes = Vec::with_capacity(4);
                bytes.extend_from_slice(&i16_i.to_le_bytes());
                bytes.extend_from_slice(&i16_k.to_le_bytes());
                return bytes;
            }
            // Fallback to i32
        }
        let mut bytes = Vec::with_capacity(8);
        bytes.extend_from_slice(&self.i.to_le_bytes());
        bytes.extend_from_slice(&self.k.to_le_bytes());
        bytes
    }

    /// Create a Hex from a byte vector representation.
    ///
    /// The byte vector can represent the coordinates in three formats:
    /// - 4 bytes: Two i16 values (2 bytes each) for i and k coordinates.
    pub fn try_from(value: Vec<u8>) -> Result<Py<Hex>, pyo3::PyErr> {
        match value.len() {
            4 => {
                let i = i32::from(i16::from_le_bytes([value[0], value[1]]));
                let k = i32::from(i16::from_le_bytes([value[2], value[3]]));
                Ok(get_hex(i, k))
            }
            8 => {
                let i = i32::from_le_bytes([value[0], value[1], value[2], value[3]]);
                let k = i32::from_le_bytes([value[4], value[5], value[6], value[7]]);
                Ok(get_hex(i, k))
            }
            16 => {
                let i_i64 = i64::from_le_bytes([
                    value[0], value[1], value[2], value[3],
                    value[4], value[5], value[6], value[7],
                ]);
                let k_i64 = i64::from_le_bytes([
                    value[8], value[9], value[10], value[11],
                    value[12], value[13], value[14], value[15],
                ]);
                match (i32::try_from(i_i64), i32::try_from(k_i64)) {
                    (Ok(i), Ok(k)) => Ok(get_hex(i, k)),
                    _ => Err(pyo3::exceptions::PyTypeError::new_err("i or k value out of range for i32")),
                }
            }
            _ => Err(pyo3::exceptions::PyTypeError::new_err("Invalid byte length for Hex conversion")),
        }
    }
}

#[pymethods]
impl Hex {
    /// Initialize a Hex coordinate at (i, k). Defaults to (0, 0).
    ///
    /// Arguments:
    /// - i (int): The I-line coordinate of the hex, or a tuple (i, k) or (i, j, k).
    /// - k (int): The K-line coordinate of the hex.
    /// Returns:
    /// - Hex
    /// Raises:
    /// - TypeError: If i or k is not an integer.
    #[new]
    pub fn new(
        i: Option<&pyo3::Bound<'_, PyAny>>,
        k: Option<&pyo3::Bound<'_, PyAny>>,
    ) -> pyo3::PyResult<Py<Hex>> {
        if let Some(i_obj) = i {
            if let Ok(tuple) = i_obj.extract::<(i32, i32)>() {
                return Ok(get_hex(tuple.0, tuple.1));
            } else if let Ok(tuple3) = i_obj.extract::<(i32, i32, i32)>() {
                let (i_val, _j_val, k_val) = tuple3;
                return Ok(get_hex(i_val, k_val));
            } else {
                let i_val: i32 = i_obj.extract()?;
                let k_val: i32 = if let Some(k_obj) = k {
                    k_obj.extract()?
                } else {
                    0
                };
                return Ok(get_hex(i_val, k_val));
            }
        } else {
            let i_val = 0;
            let k_val: i32 = if let Some(k_obj) = k {
                k_obj.extract()?
            } else {
                0
            };
            return Ok(get_hex(i_val, k_val));
        }
    }

    /// Get the I-line coordinate of the hex.
    ///
    /// Returns:
    /// - int: The I-line coordinate.
    #[inline]
    #[getter]
    pub fn i(&self) -> i32 {
        self.i
    }

    /// Get the J-line coordinate of the hex.
    ///
    /// Returns:
    /// - int: The J-line coordinate.
    #[inline]
    #[getter]
    pub fn j(&self) -> i32 {
        self.k - self.i
    }

    /// Get the K-line coordinate of the hex.
    ///
    /// Returns:
    /// - int: The K-line coordinate.
    #[inline]
    #[getter]
    pub fn k(&self) -> i32 {
        self.k
    }

    /// Return an iterator over the hex coordinates.
    ///
    /// Yields:
    /// - int: The I-line coordinate of the hex.
    /// - int: The K-line coordinate of the hex.
    pub fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let tuple = pyo3::types::PyTuple::new_bound(py, &[slf.i.into_py(py), slf.k.into_py(py)]);
            Ok(tuple.into_py(py))
        })
    }

    /// Return the raw i coordinate of the hex.
    ///
    /// Returns:
    /// - int: The raw i coordinate.
    #[inline]
    pub fn __i__(&self) -> i32 {
        self.k * 2 - self.i
    }

    /// Return the raw j coordinate of the hex.
    ///
    /// Returns:
    /// - int: The raw j coordinate.
    #[inline]
    pub fn __j__(&self) -> i32 {
        self.i + self.k
    }

    /// Return the raw k coordinate of the hex.
    ///
    /// Returns:
    /// - int: The raw k coordinate.
    #[inline]
    pub fn __k__(&self) -> i32 {
        self.i * 2 - self.k
    }

    /// Return a string representation of the hex coordinates.
    ///
    /// Format: Hex(i, j, k), where i, j, and k are the line coordinates.
    /// Returns:
    /// - str: The string representation of the hex.
    pub fn __str__(&self) -> String {
        format!("Hex({}, {}, {})", self.i, self.k - self.i, self.k)
    }

    /// Return a string representation of the hex coordinates for debugging.
    ///
    /// Format: Hex(i, j, k), where i, j, and k are the line coordinates.
    /// Returns:
    /// - str: The string representation of the hex.
    pub fn __repr__(&self) -> String {
        format!("({}, {})", self.i, self.k)
    }

    /// Check equality with another Hex or a tuple of coordinates.
    ///
    /// Arguments:
    /// - value (Hex or tuple): The value to compare with.
    /// Returns:
    /// - bool: True if the coordinates match, False otherwise.
    pub fn __eq__(&self, other: &pyo3::Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(other_hex) = other.extract::<PyRef<Hex>>() {
            Ok(self.i == other_hex.i && self.k == other_hex.k)
        } else if let Ok(tuple) = other.extract::<(i32, i32)>() {
            Ok(self.i == tuple.0 && self.k == tuple.1)
        } else {
            Ok(false)
        }
    }

    /// Return a hash of the hex coordinates.
    ///
    /// Returns:
    /// - int: The hash value of the hex coordinates.
    pub fn __hash__(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.i.hash(&mut hasher);
        self.k.hash(&mut hasher);
        hasher.finish()
    }

    /// Add another Hex or a tuple of coordinates to this hex.
    ///
    /// Arguments:
    /// - other (Hex or tuple): The value to add.
    /// Returns:
    /// - Hex: A new Hex with the added coordinates.
    /// Raises:
    /// - TypeError: If the other operand is not a Hex or a tuple of coordinates.
    pub fn __add__(&self, other: &pyo3::Bound<'_, PyAny>) -> PyResult<Py<Hex>> {
        Python::with_gil(|_py| {
            if let Ok(other_hex) = other.extract::<PyRef<Hex>>() {
                Ok(get_hex(self.i + other_hex.i, self.k + other_hex.k))
            } else if let Ok(tuple) = other.extract::<(i32, i32)>() {
                Ok(get_hex(self.i + tuple.0, self.k + tuple.1))
            } else {
                Err(pyo3::exceptions::PyTypeError::new_err("Unsupported type for addition with Hex"))
            }
        })
    }

    /// Reverse addition of this hex to another Hex or a tuple.
    ///
    /// Arguments:
    /// - other (Hex or tuple): The value to add this hex to.
    /// Returns:
    /// - Hex: A new Hex with the added coordinates.
    /// Raises:
    /// - TypeError: If the other operand is not a Hex or a tuple of coordinates.
    pub fn __radd__(&self, other: &pyo3::Bound<'_, PyAny>) -> PyResult<Py<Hex>> {
        Python::with_gil(|_py| {
            if let Ok(other_hex) = other.extract::<PyRef<Hex>>() {
                Ok(get_hex(other_hex.i + self.i, other_hex.k + self.k))
            } else if let Ok(tuple) = other.extract::<(i32, i32)>() {
                Ok(get_hex(tuple.0 + self.i, tuple.1 + self.k))
            } else {
                Err(pyo3::exceptions::PyTypeError::new_err("Unsupported type for reverse addition with Hex"))
            }
        })
    }

    /// Subtract another Hex or a tuple of coordinates from this hex.
    ///
    /// Arguments:
    /// - other (Hex or tuple): The value to subtract.
    /// Returns:
    /// - Hex: A new Hex with the subtracted coordinates.
    /// Raises:
    /// - TypeError: If the other operand is not a Hex or a tuple of coordinates.
    pub fn __sub__(&self, other: &pyo3::Bound<'_, PyAny>) -> PyResult<Py<Hex>> {
        Python::with_gil(|_py| {
            if let Ok(other_hex) = other.extract::<PyRef<Hex>>() {
                Ok(get_hex(self.i - other_hex.i, self.k - other_hex.k))
            } else if let Ok(tuple) = other.extract::<(i32, i32)>() {
                Ok(get_hex(self.i - tuple.0, self.k - tuple.1))
            } else {
                Err(pyo3::exceptions::PyTypeError::new_err("Unsupported type for subtraction with Hex"))
            }
        })
    }

    /// Reverse subtraction of this hex from another Hex or a tuple.
    ///
    /// Arguments:
    /// - other (Hex or tuple): The value to subtract this hex from.
    /// Returns:
    /// - Hex: A new Hex with the subtracted coordinates.
    /// Raises:
    /// - TypeError: If the other operand is not a Hex or a tuple of coordinates.
    pub fn __rsub__(&self, other: &pyo3::Bound<'_, PyAny>) -> PyResult<Py<Hex>> {
        Python::with_gil(|_py| {
            if let Ok(other_hex) = other.extract::<PyRef<Hex>>() {
                Ok(get_hex(other_hex.i - self.i, other_hex.k - self.k))
            } else if let Ok(tuple) = other.extract::<(i32, i32)>() {
                Ok(get_hex(tuple.0 - self.i, tuple.1 - self.k))
            } else {
                Err(pyo3::exceptions::PyTypeError::new_err("Unsupported type for reverse subtraction with Hex"))
            }
        })
    }

    /// Create a copy of this Hex.
    ///
    /// Returns:
    /// - Hex: A new Hex with the same coordinates.
    #[inline]
    pub fn __copy__(&self) -> Py<Hex> {
        get_hex(self.i, self.k)
    }

    /// Create a deep copy of this Hex.
    ///
    /// Arguments:
    /// - memo (dict): A dictionary to keep track of copied objects.
    /// Returns:
    /// - Hex: A new Hex with the same coordinates.
    #[inline]
    pub fn __deepcopy__(&self, _memo: Option<&pyo3::Bound<'_, PyAny>>) -> Py<Hex> {
        get_hex(self.i, self.k)
    }

    /// Check if the Hex is not at the origin (0, 0).
    ///
    /// Returns:
    /// - bool: True if the Hex is not at the origin, False otherwise.
    #[inline]
    pub fn __bool__(&self) -> bool {
        self.i != 0 || self.k != 0
    }

    /// Return a new Hex shifted along the i-axis by units.
    ///
    /// Arguments:
    /// - units (int): The number of units to shift along the i-axis.
    /// Returns:
    /// - Hex: A new Hex shifted by the specified units along the i-axis.
    /// Raises:
    /// - TypeError: If units is not an integer.
    #[inline]
    pub fn shift_i(&self, units: i32) -> Py<Hex> {
        get_hex(self.i + units, self.k)
    }

    /// Return a new Hex shifted along the j-axis by units.
    ///
    /// Arguments:
    /// - units (int): The number of units to shift along the j-axis.
    /// Returns:
    /// - Hex: A new Hex shifted by the specified units along the j-axis.
    /// Raises:
    /// - TypeError: If units is not an integer.
    #[inline]
    pub fn shift_j(&self, units: i32) -> Py<Hex> {
        get_hex(self.i - units, self.k + units)
    }

    /// Return a new Hex shifted along the k-axis by units.
    ///
    /// Arguments:
    /// - units (int): The number of units to shift along the k-axis.
    /// Returns:
    /// - Hex: A new Hex shifted by the specified units along the k-axis.
    /// Raises:
    /// - TypeError: If units is not an integer.
    #[inline]
    pub fn shift_k(&self, units: i32) -> Py<Hex> {
        get_hex(self.i, self.k + units)
    }
}

#[pyclass(frozen)]
#[derive(Eq, Clone)]
/// Represents a shape or unit made up of 7 Block instances,
/// typically forming a logical structure such as a game piece.
///
/// This implementation of piece contains no blocks, and instead only contains 
/// a single u8 value representing the occupancy state of each block (7 bits used).
///
/// This is a singleton class, meaning that each unique Piece state is cached
/// and reused to save memory and improve performance.
///
/// Attributes:
/// - positions (list[Hex]): A list of Hex coordinates representing the positions of the blocks in the piece.
/// - state (u8): A byte value representing the occupancy state of each block in the piece.
pub struct Piece {
    state: u8, // Only 7 bits used
}

// Singleton cache for all 128 possible Piece states
static PIECE_CACHE: OnceLock<[Py<Piece>; 128]> = OnceLock::new();

fn initialize_piece_cache() -> [Py<Piece>; 128] {
    Python::with_gil(|py| {
        use std::mem::MaybeUninit;
        // Create an uninitialized array of MaybeUninit<Py<Piece>>
        let mut cache: [MaybeUninit<Py<Piece>>; 128] = unsafe { MaybeUninit::uninit().assume_init() };
        for state in 0u8..=127u8 {
            let piece = Piece { state };
            cache[state as usize].write(Py::new(py, piece).unwrap());
        }
        // Transmute to initialized array
        unsafe { std::mem::transmute::<_, [Py<Piece>; 128]>(cache) }
    })
}

impl PartialEq for Piece {
    fn eq(&self, other: &Self) -> bool {
        self.state == other.state
    }
}

#[pymethods]
impl Piece {
    /// The fixed positions of the 7 blocks in a standard Piece.
    /// 
    /// Returns:
    /// - list[Hex]: The list of Hex coordinates for the blocks in the Piece.
    #[classattr]
    #[allow(non_upper_case_globals)] // To allow the name 'positions' in python
    pub const positions: [Hex; 7] = [
        Hex { i: -1, k: -1 },
        Hex { i: -1, k: 0 },
        Hex { i: 0, k: -1 },
        Hex { i: 0, k: 0 },
        Hex { i: 0, k: 1 },
        Hex { i: 1, k: 0 },
        Hex { i: 1, k: 1 },
    ];
    /// Initialize a Piece with the given occupancy states.
    /// 
    /// Arguments:
    /// - states (int | list[bool] | None): The occupancy states of the blocks in the Piece.
    ///   - If an integer is provided, it is treated as a byte representation of the states.
    ///   - If a list of booleans is provided, it should have exactly 7 elements, each representing
    #[new]
    fn new(py: Python, states: Option<&Bound<'_, PyAny>>) -> PyResult<Py<Piece>> {
        let key: u8 = if let Some(s) = states {
            if let Ok(int_val) = s.extract::<u8>() {
                int_val
            } else if let Ok(list) = s.extract::<Vec<bool>>() {
                if list.len() != 7 {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "List must have exactly 7 boolean elements"
                    ));
                }
                list.iter()
                    .enumerate()
                    .fold(0u8, |acc, (i, &b)| {
                        if b { acc | (1 << (6 - i)) } else { acc }
                    })
            } else if s.is_none() {
                0
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Invalid type for states"
                ));
            }
        } else {
            0
        };
        let cache = PIECE_CACHE.get_or_init(|| initialize_piece_cache());
        Ok(cache[key as usize].clone_ref(py))
    }


    /// Return a string representation of the Piece in byte format.
    /// This representation is useful for debugging and serialization.
    /// 
    /// Returns:
    /// - str: A string representation of the Piece in byte format.
    #[inline]
    pub fn __repr__(&self) -> String {
        format!("{}", self.state)
    }

    /// Return a string representation of the Piece.
    /// 
    /// Format: Piece{Block(i, j, k, state), ...}, where i, j, and k are the line coordinates of each block,
    /// and state is the occupancy state, if occupied, else "null".
    /// 
    /// Returns:
    /// - str: The string representation of the Piece.
    pub fn __str__(&self) -> String {
        let mut s = String::from("Piece{");
        for i in 0..7 {
            if i > 0 {
                s.push_str(", ");
            }
            let hex = &Piece::positions[i];
            let occupied = (self.state & (1 << (6 - i))) != 0;
            s.push_str(&format!("({}, {}, {})", hex.i(), hex.k(), occupied));
        }
        s.push('}');
        s
    }

    /// Return an iterator over the occupancy states of the Piece.
    /// 
    /// Yields:
    /// - bool: The occupancy state of each block in the Piece.
    pub fn __iter__(&self) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let tuple = pyo3::types::PyTuple::new_bound(py, &self.states());
            Ok(tuple.into_py(py))
        })
    }

    /// Return the number of occupied blocks in the Piece.
    /// 
    /// Returns:
    /// - int: The number of occupied blocks in the Piece.
    #[inline]
    pub fn __len__(&self) -> usize {
        (0..7).filter(|&i| (self.state & (1 << (6 - i))) != 0).count()
    }

    /// Check if the Piece has any occupied blocks.
    /// 
    /// Returns:
    /// - bool: True if any block is occupied, False otherwise.
    #[inline]
    pub fn __bool__(&self) -> bool {
        self.state != 0
    }

    /// Return a byte representation of the blocks in a standard 7-Block piece.
    /// 
    /// Returns:
    /// - int: A byte representation of the Piece, where each bit represents the occupancy state of a
    #[inline]
    pub fn __int__(&self) -> u8 {
        self.state
    }

    /// Get the tuple of boolean values representing the occupancy state of each block in the Piece.
    /// 
    /// Returns:
    /// - tuple[bool, ...]: The tuple of boolean values for the Piece.
    #[getter]
    #[inline]
    pub fn states(&self) -> [bool; 7] {
        let mut arr = [false; 7];
        for i in 0..7 {
            arr[i] = (self.state & (1 << (6 - i))) != 0;
        }
        arr
    }

    /// Get the list of Hex coordinates representing the positions of the blocks in the Piece.
    /// 
    /// Returns:
    /// - list[Hex]: The list of Hex coordinates for the Piece.
    #[getter]
    #[inline]
    pub fn coordinates(&self) -> Vec<Hex> {
        let mut coords = Vec::new();
        for i in 0..7 {
            if (self.state & (1 << (6 - i))) != 0 {
                coords.push(Piece::positions[i].clone());
            }
        }
        coords
    }

    /// Returns True if the occupancy states match, False otherwise.
    /// 
    /// Arguments:
    /// - other (Piece): The Piece to compare with.
    /// Returns:
    /// - bool: True if the occupancy states match, False otherwise.
    #[inline]
    pub fn __eq__(&self, other: &Piece) -> bool {
        self.state == other.state
    }

    /// Return a hash of the Piece's occupancy states.
    /// 
    /// This method directly uses the byte representation of the Piece to generate a hash value.
    /// Returns:
    /// - int: The hash value of the Piece.
    #[inline]
    pub fn __hash__(&self) -> u8 {
        self.state
    }

    /// Count occupied neighboring Blocks around the given Hex position.
    ///
    /// Checks up to six adjacent positions to the block at Hex coordinate.
    /// A neighbor is occupied if the block is non-null and its state is True.
    ///
    /// Parameters:
    /// - coo (Hex | tuple): The Hex coordinate to check for neighbors.
    /// Returns:
    /// - int: The count of occupied neighboring Blocks.
    /// Raises:
    /// - TypeError: If coo is not a Hex or a tuple of coordinates.
    pub fn count_neighbors(&self, coo: &Hex) -> usize {
        let mut count = 0;
        let idx = Piece::positions.iter().position(|h| h == coo);
        if let Some(idx) = idx {
            if (self.state & (1 << (6 - idx))) != 0 {
                for (__i, pos) in Piece::positions.iter().enumerate() {
                    let neighbor = Hex { i: pos.i + coo.i, k: pos.k + coo.k };
                    if let Some(nidx) = Piece::positions.iter().position(|h| h == &neighbor) {
                        if (self.state & (1 << (6 - nidx))) != 0 {
                            count += 1;
                        }
                    }
                }
            }
        }
        count
    }

    /// Get a list of all possible non-empty Piece instances.
    /// This method returns all cached Piece instances representing different occupancy states.
    ///
    /// The return of this method does not guarantee that pieces are spacially contigous.
    /// Returns:
    /// - list[Piece]: A list of all possible Piece instances.
    #[staticmethod]
    pub fn all_pieces() -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let cache = PIECE_CACHE.get_or_init(initialize_piece_cache);
            let pylist = pyo3::types::PyList::new_bound(
                py,
                cache
                    .iter()
                    .filter_map(|p| {
                        let piece = p.borrow(py);
                        if piece.__bool__() {
                            Some(p.clone_ref(py).into_py(py))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>(),
            );
            Ok(pylist.into_py(py))
        })
    }

    /// Get a list of all possible contigous Piece instances.
    /// This method returns all cached Piece instances representing different occupancy states
    /// that are spatially contiguous.
    ///
    /// Returns:
    /// - slist[Piece]: A list of all possible contigous Piece instances.
    #[staticmethod]
    pub fn contigous_pieces() -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let cache = PIECE_CACHE.get_or_init(initialize_piece_cache);
            let mut result = Vec::new();
            for p in cache.iter() {
                let piece = p.borrow(py);
                let s = piece.state;
                let bit = |i| (s & (1u8 << (6 - i))) != 0u8;
                if !piece.__bool__() {
                    continue;
                }
                if bit(3) {
                    result.push(p.clone_ref(py).into_py(py));
                } else if piece.__len__() == 1 || piece.__len__() == 5 || piece.__len__() == 6 {
                    result.push(p.clone_ref(py).into_py(py));
                } else if piece.__len__() == 2 {
                    if (bit(0) && (bit(1) || bit(2))) || (bit(4) && (bit(1) || bit(6))) || (bit(5) && (bit(2) || bit(6))) {
                        result.push(p.clone_ref(py).into_py(py));
                    }
                } else if piece.__len__() == 4 {
                    if (!bit(0) && !(bit(1) && bit(2))) || (!bit(4) && !(bit(1) && bit(6))) || (!bit(5) && !(bit(2) && bit(6))) {
                        result.push(p.clone_ref(py).into_py(py));
                    }
                } else {
                    if (bit(0) && bit(1) && (bit(2) || bit(4))) || (bit(2) && bit(5) && (bit(0) || bit(6))) || (bit(4) && bit(6) && (bit(1) || bit(5))) {
                        result.push(p.clone_ref(py).into_py(py));
                    }
                }
            }
            let pylist = pyo3::types::PyList::new_bound(py, result);
            Ok(pylist.into_py(py))
        })
    }
    
}

#[pyclass]
#[derive(Eq, Clone)]
/// The HexEngine class provides a complete engine for managing a two-dimensional hexagonal
/// block grid used for constructing and interacting with hex-based shapes in the game.
///
/// The engine does not actually contain any blocks, but instead contains a list of booleans
/// representing the occupancy state of each block in the hexagonal grid. The correspondence is achieved
/// through optimized indexing and coordinate transformations.
///
/// Grid Structure:
/// - Uses an axial coordinate system (i, k), where i - j + k = 0, and j is derived as j = i + k.
/// - Three axes: I, J, K. I+ is 60° from J+, J+ is 60° from K+, K+ is 60° from I-.
/// - Raw coordinates: distance along an axis multiplied by 2.
/// - Line-coordinates (I, K) are perpendicular distances to axes, calculated from raw coordinates.
/// - Blocks are stored in a sorted array by increasing raw coordinate i, then k.
///
/// Grid Size:
/// - Total blocks for radius r: Aₖ = 1 + 3*r*(r-1)
/// - Derived from: Aₖ = Aₖ₋₁ + 6*(k-1); A₁ = 1
///
/// Machine Learning:
/// - Supports reward functions for evaluating action quality.
/// - check_add discourages invalid moves (e.g., overlaps).
/// - compute_dense_index evaluates placement density for rewarding efficient gap-filling.
///
/// Attributes:
/// - radius (int): The radius of the hexagonal grid, defining the size of the grid.
/// - states (list[bool]): A list of booleans representing the occupancy state of each block in the grid.
pub struct HexEngine {
    radius: usize,
    states: Vec<bool>,
}

impl PartialEq for HexEngine {
    fn eq(&self, other: &Self) -> bool {
        self.radius == other.radius && self.states == other.states
    }
}

// This is the backend scope, nothing is exposed to Python here
impl HexEngine {
    /// Converts linear index to coordinate
    /// 
    /// This method provides efficient conversion from a linear index in the internal state vector to a `Hex` coordinate.
    /// 
    /// Arguments:
    /// - `index`: The linear index to convert
    /// Returns:
    /// - A result containing the corresponding `Hex` coordinate, or an IndexError if the index is out of bounds.
    #[deprecated = "frontend only function that invokes Python GIL in backend scope"]
    fn coordinate_of(&self, mut index: usize) -> PyResult<Py<Hex>> {
        if index >= self.states.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err("Index out of bounds"));
        }

        let r = self.radius as i32;
        
        // First half
        for i in 0..r {
            let len = (i + r) as usize;
            if index < len {
                return Ok(get_hex(i, index as i32));
            }
            index -= len;
        }
        
        // Second half
        for i in 0..(r - 1) {
            let len = (2 * r - 2 - i) as usize;
            if index < len {
                return Ok(get_hex(i + r, index as i32 + i + 1));
            }
            index -= len;
        }
        
        Err(pyo3::exceptions::PyIndexError::new_err("Index out of bounds"))
    }
    /// Converts linear index to coordinate
    /// 
    /// This method provides efficient conversion from a linear index in the internal state vector to a `Hex` coordinate.
    /// 
    /// Arguments:
    /// - `index`: The linear index to convert
    /// Returns:
    /// - A result containing the corresponding `Hex` coordinate, or an IndexError if the index is out of bounds.
    fn hex_coordinate_of(&self, mut index: usize) -> PyResult<Hex> {
        if index >= self.states.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err("Index out of bounds"));
        }

        let r = self.radius as i32;
        
        // First half
        for i in 0..r {
            let len = (i + r) as usize;
            if index < len {
                return Ok(Hex { i, k: index as i32 });
            }
            index -= len;
        }
        
        // Second half
        for i in 0..(r - 1) {
            let len = (2 * r - 2 - i) as usize;
            if index < len {
                return Ok(Hex { i: i + r, k: index as i32 + i + 1 });
            }
            index -= len;
        }
        
        Err(pyo3::exceptions::PyIndexError::new_err("Index out of bounds"))
    }
    /// Converts coordinate to linear index
    /// 
    /// This method provides efficient conversion from a `Hex` coordinate to a linear index in the internal state vector.
    /// 
    /// Arguments:
    /// - `i`: The I-line coordinate
    /// - `k`: The K-line coordinate
    /// Returns:
    /// - A result containing the corresponding linear index, or -1 if the coordinate is out of range.
    fn linear_index_of(&self, i: i32, k: i32) -> PyResult<isize> {
        let r = self.radius as i32;
        if Self::check_range_coords(i, k, self.radius)? {
            if i < r {
                Ok((k + i * r + i * (i - 1) / 2) as isize)
            } else {
                Ok((k - (r - 1).pow(2) + i * r * 3 - i * (i + 5) / 2) as isize)
            }
        } else {
            Ok(-1)
        }
    }
    /// Check if a Hex coordinate is within the specified radius of the hexagonal grid.
    ///
    /// Arguments:
    /// - i: I-line coordinate.
    /// - k: K-line coordinate.
    /// - radius: Radius of the hexagonal grid.
    /// Returns:
    /// - bool: True if the coordinate is within range, False otherwise.
    #[inline]
    fn check_range_coords(i: i32, k: i32, radius: usize) -> PyResult<bool> {
        let j = k - i;
        Ok(0 <= i && i < (radius as i32) * 2 - 1 &&
           -((radius as i32)) < j && j < (radius as i32) &&
           0 <= k && k < (radius as i32) * 2 - 1)
    }

    /// Get the occupancy state of a block at the given linear index.
    ///
    /// Arguments:
    /// - index: Linear index of the block to get.
    /// Returns:
    /// - bool: The occupancy state of the block (True for occupied, False for unoccupied).
    fn get_state_from_index(&self, index: usize) -> PyResult<bool> {
        if index >= self.states.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err("Index out of bounds"));
        }
        Ok(self.states[index])
    }

    /// Set the occupancy state of a block at the given linear index.
    ///
    /// Arguments:
    /// - index: Linear index of the block to set.
    /// - state: The occupancy state to set (True for occupied, False for unoccupied
    fn set_state_from_index(&mut self, index: usize, state: bool) -> PyResult<()> {
        if index >= self.states.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err("Index out of bounds"));
        }
        self.states[index] = state;
        Ok(())
    }

    /// Get the occupancy state of a block at the given Hex coordinate.
    ////
    /// Arguments:
    /// - i: I-line coordinate of the block to get.
    /// - k: K-line coordinate of the block to get.
    /// Returns:
    /// - bool: The occupancy state of the block (True for occupied, False for unoccupied).
    fn state_of(&self, i: i32, k: i32) -> PyResult<bool> {
        let index = self.linear_index_of(i, k)?;
        if index == -1 {
            return Err(pyo3::exceptions::PyIndexError::new_err("Coordinate out of bounds"));
        }
        Ok(self.states[index as usize])
    }

    /// Count occupied neighboring Blocks around the given Hex position.
    ///
    /// Checks up to six adjacent positions to the block at Hex coordinate.
    /// A neighbor is occupied if the block is non-null and its state is True.
    ///
    /// Parameters:
    /// - i: I-line coordinate of the block to check for neighbors.
    /// - k: K-line coordinate of the block to check for neighbors.
    /// Returns:
    /// - int: The count of occupied neighboring Blocks.
    /// Raises:
    /// - TypeError: If coo is not a Hex.
    fn count_neighbors_coordinate(&self, i: i32, k: i32) -> PyResult<usize> {
        let mut count = 0;
        for pos in &Piece::positions {
            let target_i = pos.i + i;
            let target_k = pos.k + k;
            if Self::check_range_coords(target_i, target_k, self.radius).unwrap_or(false) {
                if self.state_of(target_i, target_k).unwrap_or(false) {
                    count += 1;
                }
            } else {
                count += 1;
            }
        }
        Ok(count)
    }
    /// Get a byte representation of the blocks around the given Hex position.
    /// 
    /// Each bit in the byte represents the occupancy state of a block in the Piece,
    /// with the most significant bit corresponding to the first block in the Piece's positions.
    /// 
    /// Parameters:
    /// - i: I-line coordinate of the block to check.
    /// - k: K-line coordinate of the block to check.
    /// Returns:
    /// - int: A byte representation of the blocks around the given Hex position.
    fn pattern_of(&self, i: i32, k: i32) -> PyResult<u8> {
        let mut pattern: u8 = 0;
        for (idx, pos) in Piece::positions.iter().enumerate() {
            let target_i = pos.i + i;
            let target_k = pos.k + k;
            if Self::check_range_coords(target_i, target_k, self.radius).unwrap_or(false) {
                if self.state_of(target_i, target_k).unwrap_or(false) {
                    pattern |= 1 << (6 - idx);
                }
            }
        }
        Ok(pattern)
    }

    /// Identify coordinates along I axis that can be eliminated and return them as Vec<Hex>
    ///
    /// Arguments:
    /// - eliminate (list[Hex]): Mutable list to append eliminated coordinates
    fn eliminate_i(&self) -> Vec<Hex> {
        let r = self.radius as i32;
        let mut eliminated = Vec::new();
        // First half
        for i in 0..r {
            let start_idx = (i * (r * 2 + i - 1) / 2) as usize;
            let len = (r + i) as usize;
            if (0..len).all(|b| self.states.get(start_idx + b) == Some(&true)) {
                for b in 0..len {
                    if let Ok(coo) = self.hex_coordinate_of(start_idx + b) {
                        eliminated.push(coo);
                    }
                }
            }
        }
        // Second half
        let const_term = (r * (r * 3 - 1) / 2) as usize;
        for i in (0..=(r - 2)).rev() {
            let start_idx = const_term + ((r - i - 2) * (r * 3 - 1 + i) / 2) as usize;
            let len = (r + i) as usize;
            if (0..len).all(|b| self.states.get(start_idx + b) == Some(&true)) {
                for b in 0..len {
                    if let Ok(coo) = self.hex_coordinate_of(start_idx + b) {
                        eliminated.push(coo);
                    }
                }
            }
        }
        eliminated
    }

    /// Identify coordinates along J axis that can be eliminated and insert them into the input list
    ///
    /// Arguments:
    /// - eliminate (list[Hex]): Mutable list to append eliminated coordinates
    fn eliminate_j(&self) -> Vec<Hex> {
        let radius = self.radius as i32;
        let mut eliminated = Vec::new();
        for r in 0..radius {
            let mut idx = r as usize;
            let mut all_valid = true;
            // Check first part (1 to radius-1)
            for c in 1..radius {
                if idx >= self.states.len() || !self.states[idx] {
                    all_valid = false;
                    break;
                }
                idx += (radius + c) as usize;
            }
            // Check second part (radius - r blocks)
            if all_valid {
                for c in 0..(radius - r) {
                    if idx >= self.states.len() || !self.states[idx] {
                        all_valid = false;
                        break;
                    }
                    idx += (2 * radius - c - 1) as usize;
                }
            }
            // If all blocks are occupied, add them to eliminated list
            if all_valid {
                let mut idx = r as usize;
                for c in 1..radius {
                    if let Ok(coo) = self.hex_coordinate_of(idx) {
                        eliminated.push(coo);
                    }
                    idx += (radius + c) as usize;
                }
                for c in 0..(radius - r) {
                    if let Ok(coo) = self.hex_coordinate_of(idx) {
                        eliminated.push(coo);
                    }
                    idx += (2 * radius - c - 1) as usize;
                }
            }
        }
        for r in 1..radius {
            let start_idx = (radius * r + r * (r - 1) / 2) as usize;
            let mut idx = start_idx;
            let mut all_valid = true;
            // Check first part (1 to radius-r-1)
            for c in 1..(radius - r) {
                if idx >= self.states.len() || !self.states[idx] {
                    all_valid = false;
                    break;
                }
                idx += (radius + c + r) as usize;
            }
            // Check second part (radius blocks)
            if all_valid {
                for c in 0..radius {
                    if idx >= self.states.len() || !self.states[idx] {
                        all_valid = false;
                        break;
                    }
                    idx += (2 * radius - c - 1) as usize;
                }
            }
            // If all blocks are occupied, add them to eliminated list
            if all_valid {
                let mut idx = start_idx;
                for c in 1..(radius - r) {
                    if let Ok(coo) = self.hex_coordinate_of(idx) {
                        eliminated.push(coo);
                    }
                    idx += (radius + c + r) as usize;
                }
                for c in 0..radius {
                    if let Ok(coo) = self.hex_coordinate_of(idx) {
                        eliminated.push(coo);
                    }
                    idx += (2 * radius - c - 1) as usize;
                }
            }
        }
        eliminated
    }

    /// Identify coordinates along K axis that can be eliminated and return them as Vec<Hex>
    /// 
    /// Arguments:
    /// - eliminate (list[Hex]): Mutable list to append eliminated coordinates
    fn eliminate_k(&self) -> Vec<Hex> {
        let radius = self.radius as i32;
        let mut eliminated = Vec::new();
        for r in 0..radius {
            let mut idx = r as usize;
            let mut all_valid = true;
            // Check first part (radius-1 blocks)
            for c in 0..(radius - 1) {
                if idx >= self.states.len() || !self.states[idx] {
                    all_valid = false;
                    break;
                }
                idx += (radius + c) as usize;
            }
            // Check second part (r+1 blocks)
            if all_valid {
                for c in 0..(r + 1) {
                    if idx >= self.states.len() || !self.states[idx] {
                        all_valid = false;
                        break;
                    }
                    idx += (2 * radius - c - 2) as usize;
                }
            }
            // If all blocks are occupied, add them to eliminated list
            if all_valid {
                let mut idx = r as usize;
                for c in 0..(radius - 1) {
                    if let Ok(coo) = self.hex_coordinate_of(idx) {
                        eliminated.push(coo);
                    }
                    idx += (radius + c) as usize;
                }
                for c in 0..(r + 1) {
                    if let Ok(coo) = self.hex_coordinate_of(idx) {
                        eliminated.push(coo);
                    }
                    idx += (2 * radius - c - 2) as usize;
                }
            }
        }
        for r in 1..radius {
            let start_idx = (radius * (r + 1) + r * (r + 1) / 2 - 1) as usize;
            let mut idx = start_idx;
            let mut all_valid = true;
            // Check first part (r to radius-2)
            for c in r..(radius - 1) {
                if idx >= self.states.len() || !self.states[idx] {
                    all_valid = false;
                    break;
                }
                idx += (radius + c) as usize;
            }
            // Check second part (radius-1 down to 0)
            if all_valid {
                for c in (0..radius).rev() {
                    if idx >= self.states.len() || !self.states[idx] {
                        all_valid = false;
                        break;
                    }
                    idx += (radius + c - 1) as usize;
                }
            }
            // If all blocks are occupied, add them to eliminated list
            if all_valid {
                let mut idx = start_idx;
                for c in r..(radius - 1) {
                    if let Ok(coo) = self.hex_coordinate_of(idx) {
                        eliminated.push(coo);
                    }
                    idx += (radius + c) as usize;
                }
                for c in (0..radius).rev() {
                    if let Ok(coo) = self.hex_coordinate_of(idx) {
                        eliminated.push(coo);
                    }
                    idx += (radius + c - 1) as usize;
                }
            }
        }
        eliminated
    }
}

#[pymethods]
impl HexEngine {
    /// Check if a Hex coordinate is within the specified radius of the hexagonal grid.
    ///
    /// Arguments:
    /// - coo: Hex coordinate to check.
    /// - radius: Radius of the hexagonal grid.
    /// Returns:
    /// - bool: True if the coordinate is within range, False otherwise.
    #[staticmethod]
    pub fn __in_range(coo: &pyo3::Bound<'_, PyAny>, radius: usize) -> PyResult<bool> {
        let (i, k) = if let Ok(hex) = coo.extract::<PyRef<Hex>>() {
            (hex.i, hex.k)
        } else if let Ok(tuple) = coo.extract::<(i32, i32)>() {
            let (i, k) = tuple;
            (i, k)
        } else if let Ok(tuple3) = coo.extract::<(i32, i32, i32)>() {
            if tuple3.0 + tuple3.1 + tuple3.2 != 0 {
                return Ok(false);
            }
            (tuple3.0, tuple3.2)
        } else {
            return Ok(false);
        };
        HexEngine::check_range_coords(i, k, radius)
    }

    /// Solves for the length of a HexEngine based on its radius.
    /// 
    /// Arguments:
    /// - radius (int): The radius of the hexagonal grid.
    /// Returns:
    /// - int: The length of the hexagonal grid, or -1 if the radius is invalid.
    #[staticmethod]
    pub fn solve_length(radius: usize) -> isize {
        if radius < 1 {
            -1
        } else {
            1 + 3 * (radius as isize) * ((radius as isize) - 1)
        }
    }

    /// Solves for the radius of a HexEngine based on its length.
    /// 
    /// Arguments:
    /// - radius (int): The radius of the hexagonal grid.
    /// Returns:
    /// - int: The radius of the hexagonal grid, or -1 if the length is invalid.
    #[staticmethod]
    pub fn solve_radius(length: usize) -> isize {
        if length == 0 {
            return 0;
        }
        if length % 3 != 1 {
            return -1;
        }
        let target = (length - 1) / 3;
        let u = target * 4 + 1;
        let r = ((u as f64).sqrt() as usize + 1) / 2;
        if r > 0 && r * (r - 1) == target {
            r as isize
        } else {
            -1
        }
    }

    /// Construct a HexEngine with the specified radius, states, or string.
    /// 
    /// This method initializes the hexagonal grid with a given radius,
    /// creating an array of booleans to represent the grid.
    /// 
    /// Arguments:
    /// - arg (int | str | list[bool]):
    ///     - An integer representing the radius of the hexagonal grid.
    ///     - A list of booleans representing the occupancy state of each block.
    ///     - A string representation of the occupancy state, either as 'X'/'O' or '1'/'0'.
    /// Raises:
    /// - TypeError: If radius is not an integer greater than 0, or if the list contains non-boolean values.
    /// - ValueError: If radius is less than 1, or if the length of the list/string does not match a valid hexagonal grid size.
    #[new]
    pub fn new(arg: &pyo3::Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(radius) = arg.extract::<usize>() {
            if radius < 1 {
                return Err(pyo3::exceptions::PyValueError::new_err("Radius must be greater than 0"));
            }
            Ok(HexEngine {
                radius,
                states: vec![false; 1 + 3 * radius * (radius - 1)],
            })
        } else if let Ok(s) = arg.extract::<String>() {
            let s = s.trim();
            if !s.chars().all(|c| c == '0' || c == '1' || c == 'X' || c == 'O') {
                return Err(pyo3::exceptions::PyValueError::new_err("String must contain only '0' or '1', or 'X' or 'O'"));
            }
            let radius = HexEngine::solve_radius(s.len()) as usize;
            if radius < 1 {
                return Err(pyo3::exceptions::PyValueError::new_err("Invalid length for hexagonal grid"));
            }
            let states = s.chars().map(|c| c == '1' || c == 'X').collect();
            Ok(HexEngine { radius, states })
        } else if let Ok(list) = arg.extract::<Vec<bool>>() {
            let radius = HexEngine::solve_radius(list.len()) as usize;
            if radius < 1 {
                return Err(pyo3::exceptions::PyValueError::new_err("Invalid length for hexagonal grid"));
            }
            Ok(HexEngine { radius, states: list })
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err("Invalid type for HexEngine initialization"))
        }
    }

    /// Get the radius of the hexagonal grid.
    /// 
    /// Returns:
    /// - int: The radius of the hexagonal grid.
    #[getter]
    pub fn radius(&self) -> usize {
        self.radius
    }

    /// Get the occupancy states of the hexagonal grid blocks.
    /// 
    /// Returns:
    /// - list[bool]: The occupancy states of the hexagonal grid blocks.
    #[getter]
    pub fn states(&self) -> Vec<bool> {
        self.states.clone()
    }

    /// Check equality with another HexEngine or a list of booleans.
    /// Returns True if the states match, False otherwise.
    /// 
    /// Arguments:
    /// - value (HexEngine | list[bool]): The HexEngine or list of booleans to compare with.
    /// Returns:
    /// - bool: True if the states match, False otherwise.
    pub fn __eq__(&self, value: &pyo3::Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(other) = value.extract::<PyRef<HexEngine>>() {
            Ok(self.states == other.states)
        } else if let Ok(list) = value.extract::<Vec<bool>>() {
            Ok(self.states == list)
        } else {
            Ok(false)
        }
    }

    /// Return a hash of the HexEngine's occupancy states.
    /// This method uses the tuple representation of the states for hashing.
    /// 
    /// Returns:
    /// - int: The hash value of the HexEngine.
    pub fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        self.states.hash(&mut hasher);
        hasher.finish()
    }

    /// Get the number of blocks in the hexagonal grid.
    /// 
    /// Returns:
    /// - int: The number of blocks in the grid.
    pub fn __len__(&self) -> usize {
        self.states.len()
    }

    /// Return an iterator over the occupancy states of the hexagonal grid blocks.
    /// 
    /// Yields:
    /// - bool: The occupancy state of each block in the grid.
    pub fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let tuple = pyo3::types::PyTuple::new_bound(py, &slf.states);
            Ok(tuple.into_py(py))
        })
    }

    /// Return a string representation of the grid block states.
    /// This representation is useful for debugging and serialization.
    /// Format: "1" for occupied blocks, "0" for unoccupied blocks.
    /// 
    /// Returns:
    /// - str: A string representation of the grid block states.
    pub fn __repr__(&self) -> String {
        self.states.iter().map(|&b| if b { '1' } else { '0' }).collect()
    }

    /// Return a string representation of the grid block states.
    /// Format: "HexEngine[blocks = {block1, block2, ...}]",
    /// where each block is represented by its string representation.
    /// 
    /// Returns:
    /// - str: The string representation of the HexEngine.
    pub fn __str__(&self) -> String {
        let mut s = String::from("HexEngine[blocks = {");
        for (i, &state) in self.states.iter().enumerate() {
            let hex = Python::with_gil(|py| {
                let hex = self.coordinate_of(i).unwrap();
                let h = hex.borrow(py);
                (h.i, h.k)
            });
            if i > 0 {
                s.push_str(", ");
            }
            s.push_str(&format!("({}, {}, {})", hex.0, hex.1, state));
        }
        s.push_str("}]");
        s
    }

    /// Create a deep copy of the HexEngine.
    /// 
    /// Returns:
    /// HexEngine: A new HexEngine with the same radius and states.
    pub fn __copy__(&self) -> Self {
        HexEngine { radius: self.radius, states: self.states.clone() }
    }

    /// Create a deep copy of the HexEngine.
    /// Arguments:
    /// - memo (dict): A dictionary to keep track of copied objects.
    /// Returns:
    /// - HexEngine: A new HexEngine instance with the same radius and blocks.
    pub fn __deepcopy__(&self, memo: Option<&pyo3::Bound<'_, PyAny>>) -> Self {
        // Check if already copied
        if let Some(m) = memo {
            if let Ok(existing) = m.get_item(self as *const _ as usize) {
                if let Ok(hex_engine) = existing.extract::<PyRef<HexEngine>>() {
                    return hex_engine.clone();
                }
            }
        }
        // Create a new instance
        let new_instance = HexEngine {
            radius: self.radius,
            states: self.states.clone(),
        };
        // Store in memo
        if let Some(m) = memo {
            let _ = m.set_item(self as *const _ as usize, Python::with_gil(|py| Py::new(py, new_instance.clone()).unwrap()));
        }
        new_instance
    }

    /// Reset the HexEngine grid to its initial state, clearing all blocks.
    /// This method reinitializes the grid, setting all blocks to unoccupied.
    /// 
    /// Returns:
    /// - None
    pub fn reset(&mut self) {
        self.states = vec![false; 1 + 3 * self.radius * (self.radius - 1)];
    }

    /// Check if a Hex coordinate is within the radius of the hexagonal grid.
    /// 
    /// Arguments:
    /// - coo: Hex coordinate to check.
    /// Returns:
    /// - bool: True if the coordinate is within range, False otherwise.
    pub fn in_range(&self, coo: &pyo3::Bound<'_, PyAny>) -> PyResult<bool> {
        Self::__in_range(coo, self.radius)
    }

    /// Get the index of the Block at the specified Hex coordinate.
    /// 
    /// This method is heavily optimized for performance and achieves O(1) complexity by using direct index formulas
    /// based on the hexagonal grid's structure. It calculates the index based on the I and K coordinates of the Hex.
    /// 
    /// Arguments:
    /// - coo: The Hex coordinate.
    /// Returns:
    /// - int: The index of the Block, or -1 if out of range.
    pub fn index_block(&self, coo: &pyo3::Bound<'_, PyAny>) -> PyResult<isize> {
        let r = self.radius as i32;
        let (i, k) = if let Ok(hex) = coo.extract::<PyRef<Hex>>() {
            (hex.i, hex.k)
        } else if let Ok(tuple) = coo.extract::<(i32, i32)>() {
            (tuple.0, tuple.1)
        } else if let Ok(tuple3) = coo.extract::<(i32, i32, i32)>() {
            (tuple3.0, tuple3.2)
        } else {
            return Ok(-1);
        };
        let py_hex: Option<Py<Hex>> = if let Ok(hex) = coo.extract::<PyRef<Hex>>() {
            Some(hex.into())
        } else if let Ok(tuple) = coo.extract::<(i32, i32)>() {
            Some(get_hex(tuple.0, tuple.1)) // TODO: unnecessary use of get_hex
        } else if let Ok(tuple3) = coo.extract::<(i32, i32, i32)>() {
            Some(get_hex(tuple3.0, tuple3.2))  // TODO: unnecessary use of get_hex
        } else {
            None
        };
        // To explain why the former are unnecessary use of get_hex:
        // We only need the i and k values, which we already have. Frontend functions should
        // Only be used at the end of the process to reduce overhead.
        match py_hex {
            Some(_) => {
                if Self::check_range_coords(i, k, self.radius)? {
                    if i < r {
                        Ok((k + i * r + i * (i - 1) / 2) as isize)
                    } else {
                        Ok((k - (r - 1).pow(2) + i * r * 3 - i * (i + 5) / 2) as isize)
                    }
                } else {
                    Ok(-1)
                }
            },
            None => Ok(-1),
        }
    }

    /// Get the Hex coordinate of the Block at the specified index.
    /// 
    /// This method retrieves the Hex coordinate based on the index in the hexagonal grid.
    /// If the index is out of range, raise ValueError.
    /// 
    /// Arguments:
    /// - index (int): The index of the Block.
    /// Returns:
    /// - Hex: The Hex coordinate of the Block.
    /// Raises:
    /// - TypeError: If index is not an integer.
    /// - ValueError: If the index is out of range.
    pub fn coordinate_block(&self, index: usize) -> PyResult<Py<Hex>> {
        if index < self.states.len() {
            let hex = self.coordinate_of(index).unwrap();
            Python::with_gil(|py| Ok(Py::new(py, hex).unwrap()))
        } else {
            Err(pyo3::exceptions::PyValueError::new_err("Index out of range"))
        }
    }

    /// Get the Block occupancy state at the specified Hex coordinate or index.
    /// 
    /// This method retrieves the Block state based on either a Hex coordinate or an index.
    /// If the coordinate or index is out of range, raise ValueError.
    /// 
    /// Arguments:
    /// - coo (Hex | tuple | int): The Hex coordinate or index of the Block.
    /// Returns:
    /// - bool: The occupancy state of the Block.
    /// Raises:
    /// - TypeError: If coo is not a Hex, tuple, or integer.
    /// - ValueError: If the coordinate or index is out of range.
    pub fn get_state(&self, coo: &pyo3::Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(idx) = coo.extract::<usize>() {
            if idx < self.states.len() {
                Ok(self.states[idx])
            } else {
                Err(pyo3::exceptions::PyValueError::new_err("Coordinate out of range"))
            }
        } else {
            let idx = self.index_block(coo)?;
            if idx == -1 {
                Err(pyo3::exceptions::PyValueError::new_err("Coordinate out of range"))
            } else {
                Ok(self.states[idx as usize])
            }
        }
    }

    /// Set the occupancy state of the Block at the specified Hex coordinate.
    /// 
    /// This method updates the state of a Block at the given coordinate.
    /// If the coordinate is out of range, raise ValueError.
    /// 
    /// Arguments:
    /// - coo (Hex | tuple | int): The Hex coordinate or index of the block to set.
    /// - state (bool): The new occupancy state to set for the Block.
    /// Raises:
    /// - ValueError: If the coordinate is out of range.
    /// - TypeError: If the coordinate type is unsupported, or state is not a boolean.
    pub fn set_state(&mut self, coo: &pyo3::Bound<'_, PyAny>, state: bool) -> PyResult<()> {
        if let Ok(idx) = coo.extract::<usize>() {
            if idx < self.states.len() {
                self.states[idx] = state;
                Ok(())
            } else {
                Err(pyo3::exceptions::PyValueError::new_err("Coordinate out of range"))
            }
        } else {
            let idx = self.index_block(coo)?;
            if idx == -1 {
                Err(pyo3::exceptions::PyValueError::new_err("Coordinate out of range"))
            } else {
                self.states[idx as usize] = state;
                Ok(())
            }
        }
    }

    /// Check if a Piece can be added to the hexagonal grid without overlaps.
    /// 
    /// This method checks if the Piece can be placed on the grid without overlapping
    /// any existing occupied blocks. It returns True if the Piece can be added,
    /// otherwise returns False.
    /// 
    /// Arguments:
    /// - coo (Hex | tuple): The Hex coordinate to check for addition.
    /// - piece (Piece | int): The Piece to check for addition.
    /// Returns:
    /// - bool: True if the Piece can be added, False otherwise.
    /// Raises:
    /// - TypeError: If piece is not a Piece instance.
    pub fn check_add(&self, coo: &pyo3::Bound<'_, PyAny>, piece: &pyo3::Bound<'_, PyAny>) -> PyResult<bool> {
        let piece = if let Ok(p) = piece.extract::<PyRef<Piece>>() {
            p
        } else if let Ok(state) = piece.extract::<u8>() {
            Python::with_gil(|py| {
                PIECE_CACHE.get_or_init(|| initialize_piece_cache())[state as usize].borrow(py)
            })
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err("Piece must be an instance of Piece or an integer representing a Piece state"));
        };
        for i in 0..7 {
            if piece.states()[i] {
                let (hex_i, hex_k) = Python::with_gil(|_py| {
                    let pos = &Piece::positions[i];
                    let coo_val = coo.extract::<PyRef<Hex>>().ok();
                    let base = if let Some(c) = coo_val { (c.i, c.k) } else if let Ok(tuple) = coo.extract::<(i32, i32)>() { (tuple.0, tuple.1) } else { (0, 0) };
                    (pos.i + base.0, pos.k + base.1)
                });
                if let Ok(idx) = self.linear_index_of(hex_i, hex_k) {
                    if idx == -1 { return Ok(false); }
                    if self.states[idx as usize] { return Ok(false); }
                } else {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Add a Piece to the hexagonal grid at the specified Hex coordinate.
    /// 
    /// This method places the Piece on the grid, updating the occupancy state of
    /// the blocks based on the Piece's states. If the Piece cannot be added due to
    /// overlaps or out-of-range coordinates, it raises a ValueError.
    /// 
    /// Arguments:
    /// - coo (Hex | tuple): The Hex coordinate to add the Piece.
    /// - piece (Piece | int): The Piece to add to the grid.
    /// Raises:
    /// - ValueError: If the Piece cannot be added due to overlaps or out-of-range coordinates.
    /// - TypeError: If piece is not a valid Piece instance.
    pub fn add_piece(&mut self, coo: &pyo3::Bound<'_, PyAny>, piece: &pyo3::Bound<'_, PyAny>) -> PyResult<()> {
        if !self.check_add(coo, piece)? {
            return Err(pyo3::exceptions::PyValueError::new_err("Cannot add piece due to overlaps or out-of-range coordinates"));
        }
        let piece = if let Ok(p) = piece.extract::<PyRef<Piece>>() {
            p
        } else if let Ok(state) = piece.extract::<u8>() {
            Python::with_gil(|py| PIECE_CACHE.get_or_init(|| initialize_piece_cache())[state as usize].borrow(py))
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err("Piece must be an instance of Piece or an integer representing a Piece state"));
        };
        for i in 0..7 {
            if piece.states()[i] {
                let (placed_i, placed_k) = Python::with_gil(|_py| {
                    let pos = &Piece::positions[i];
                    let base = if let Ok(c) = coo.extract::<PyRef<Hex>>() {
                        (c.i, c.k)
                    } else if let Ok(tuple) = coo.extract::<(i32, i32)>() {
                        (tuple.0, tuple.1)
                    } else {
                        (0, 0)
                    };
                    (pos.i + base.0, pos.k + base.1)
                });
                let idx = self.linear_index_of(placed_i, placed_k)?;
                if idx == -1 {
                    return Err(pyo3::exceptions::PyValueError::new_err("Coordinate out of range"));
                }
                self.states[idx as usize] = true;
            }
        }
        Ok(())
    }

    /// Return all valid positions where another grid can be added.
    /// 
    /// This method returns a list of Hex coordinate candidates where the Piece can be added
    /// without overlaps. It checks each position in the Piece and returns the Hex coordinates
    /// of the occupied blocks.
    /// If the Piece is not valid, it raises a ValueError.
    ///
    /// Arguments:
    /// - piece (Piece): The Piece to check for occupied positions.
    /// Returns:
    /// - list[Hex]: A list of Hex coordinates for the occupied blocks in the Piece.
    /// Raises:
    /// - TypeError: If the piece is not a valid Piece instance.
    pub fn check_positions(&self, piece: &pyo3::Bound<'_, PyAny>) -> PyResult<Vec<Py<Hex>>> {
        let piece = if let Ok(p) = piece.extract::<PyRef<Piece>>() {
            p
        } else if let Ok(state) = piece.extract::<u8>() {
            Python::with_gil(|py| PIECE_CACHE.get_or_init(|| initialize_piece_cache())[state as usize].borrow(py))
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err("Piece must be an instance of Piece or an integer representing a Piece state"));
        };
        let mut positions = Vec::new();
        for a in 0..(self.radius * 2) as i32 {
            for b in 0..(self.radius * 2) as i32 {
                let hex = get_hex(a, b); // TODO: could be unnecessary use of get_hex
                Python::with_gil(|py| { // TODO: could be unnecessary use of GIL
                    // TO optimize this, only create hex_any once check succeeds
                    let hex_any = hex.clone_ref(py).into_bound(py);
                    if self.check_add(&hex_any, &piece.into_py(py).into_bound(py)).unwrap_or(false) {
                        positions.push(hex);
                    }
                });
            }
        }
        Ok(positions)
    }

    /// Eliminate fully occupied lines along I, J, or K axes and return eliminated coordinates.
    /// 
    /// Modifies the grid permanently.
    /// 
    /// Returns:
    /// - list[Hex]: A list of Hex coordinates that were eliminated.
    // REDESIGN: see note on redesign elimination in eliminate_i (line 1025)
    pub fn eliminate(&mut self) -> PyResult<Vec<Py<Hex>>> {
        let i_coords = self.eliminate_i();
        let j_coords = self.eliminate_j();
        let k_coords = self.eliminate_k();
        let joint = [i_coords, j_coords, k_coords].concat();
        // Set to false and gather
        let mut res: Vec<Py<Hex>> = Vec::new();
        for coo in &joint {
            let index = self.linear_index_of(coo.i, coo.k)?;
            if index == -1 {
                continue;
            }
            self.set_state_from_index(index as usize, false)?;
            // Convert to Py<Hex>
            let py_hex = get_hex(coo.i, coo.k);
            res.push(py_hex);
        }
        Ok(res)
    }

    /// Identify coordinates along I axis that can be eliminated and insert them into the input list
    ///
    /// Arguments:
    /// - eliminate (list[Hex]): Mutable list to append eliminated coordinates
    // REDESIGN: see note on redesign elimination in eliminate_i (line 1025)
    pub fn __eliminate_i(&mut self, eliminated: &pyo3::Bound<'_, PyList>) -> PyResult<()> {
        let i_coords = self.eliminate_i();
        // Convert and insert
        for coo in i_coords {
            eliminated.append(get_hex(coo.i, coo.k))?;
        }
        Ok(())
    }

    /// Identify coordinates along J axis that can be eliminated and insert them into the input list
    ///
    /// Arguments:
    /// - eliminate (list[Hex]): Mutable list to append eliminated coordinates
    // REDESIGN: see note on redesign elimination in eliminate_i (line 1025)
    pub fn __eliminate_j(&mut self, eliminated: &pyo3::Bound<'_, PyList>) -> PyResult<()> {
        let j_coords = self.eliminate_j();
        // Convert and insert
        for coo in j_coords {
            eliminated.append(get_hex(coo.i, coo.k))?;
        }
        Ok(())
    }

    /// Identify coordinates along K axis that can be eliminated and insert them into the input list
    ///
    /// Arguments:
    /// - eliminate (list[Hex]): Mutable list to append eliminated coordinates
    // REDESIGN: see note on redesign elimination in eliminate_i (line 1025)
    pub fn __eliminate_k(&mut self, eliminated: &pyo3::Bound<'_, PyList>) -> PyResult<()> {
        let k_coords = self.eliminate_k();
        // Convert and insert
        for coo in k_coords {
            eliminated.append(get_hex(coo.i, coo.k))?;
        }
        Ok(())
    }

    /// Count occupied neighboring Blocks around the given Hex position.
    /// 
    /// Checks up to six adjacent positions to the block at Hex coordinate.
    /// A neighbor is occupied if the block is null or its state is True 
    ///
    /// Arguments:
    /// - coo (Hex | tuple): The Hex coordinate to check for neighbors.
    /// Returns:
    /// - int: The count of occupied neighboring Blocks.
    /// Raises:
    /// - TypeError: If coo is not a Hex or a tuple of coordinates.
    pub fn count_neighbors(&self, coo: &pyo3::Bound<'_, PyAny>) -> PyResult<usize> {
        let (hex_i, hex_k) = if let Ok(h) = coo.extract::<PyRef<Hex>>() {
            (h.i, h.k)
        } else if let Ok(tuple) = coo.extract::<(i32, i32)>() {
            (tuple.0, tuple.1)
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err("Invalid type for Hex coordinates"));
        };
        let mut count = 0;
        for pos in &Piece::positions {
            let target_i = pos.i + hex_i;
            let target_k = pos.k + hex_k;
            if Self::check_range_coords(target_i, target_k, self.radius)? {
                if self.state_of(target_i, target_k).unwrap_or(false) {
                    count += 1;
                }
            } else {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Determine the pattern of blocks around the given position in the hexagonal grid, including the block itself.
    /// 
    /// This method checks up to seven positions in a hexagonal box centered at coordinates (i, k).
    /// It returns a value representing the pattern of occupied/unoccupied blocks, ignoring block colors.
    /// The pattern is encoded as a 7-bit integer (0 to 127) based on the state of the central block
    /// and its six neighbors. If a neighboring position is out of range or contains a None block,
    /// it is treated as occupied or unoccupied based on the include_null flag.
    /// 
    /// Arguments:
    /// - coo (Hex | tuple): The hex coordinate of the block at the center of the box.
    /// Returns:
    /// - pattern (int): A number in the range [0, 127] representing the pattern of blocks in the hexagonal box.
    pub fn get_pattern(&self, coo: &pyo3::Bound<'_, PyAny>) -> PyResult<u8> {
        let mut pattern = 0u8;
        for i in 0..7 {
            pattern <<= 1;
            let (coo_i, coo_k) = Python::with_gil(|_py| {
                let pos = &Piece::positions[i];
                let base = if let Ok(h) = coo.extract::<PyRef<Hex>>() {
                    (h.i, h.k)
                } else if let Ok(tuple) = coo.extract::<(i32, i32)>() {
                    (tuple.0, tuple.1)
                } else {
                    (0, 0)
                };
                (pos.i + base.0, pos.k + base.1)
            });
            if let Ok(state) = self.state_of(coo_i, coo_k) {
                if state {
                    pattern |= 1;
                }
            }
        }
        Ok(pattern)
    }

    /// Compute a density index score for hypothetically placing another piece.
    /// 
    /// Returns a value between 0 and 1 representing surrounding density.
    /// A score of 1 means all surrounding blocks would be filled, 0 means the grid would be alone.
    /// 
    /// Arguments:
    /// - coo (Hex): Position for hypothetical placement.
    /// - piece (Piece): The Piece to evaluate for placement.
    /// Returns:
    /// - float: Density index (0 to 1), or 0 if placement is invalid or no neighbors exist.
    pub fn compute_dense_index(&self, coo: &pyo3::Bound<'_, PyAny>, piece: &pyo3::Bound<'_, PyAny>) -> PyResult<f64> {
        let piece = if let Ok(p) = piece.extract::<PyRef<Piece>>() {
            p
        } else if let Ok(state) = piece.extract::<u8>() {
            Python::with_gil(|py| PIECE_CACHE.get_or_init(|| initialize_piece_cache())[state as usize].borrow(py))
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err("Piece must be an instance of Piece or an integer representing a Piece state"));
        };
        let mut total_possible = 0;
        let mut total_populated = 0;
        for i in 0..7 {
            if piece.states()[i] {
                let (placed_i, placed_k) = Python::with_gil(|_py| {
                    let pos = &Piece::positions[i];
                    let base = if let Ok(h) = coo.extract::<PyRef<Hex>>() {
                        (h.i, h.k)
                    } else if let Ok(tuple) = coo.extract::<(i32, i32)>() {
                        (tuple.0, tuple.1)
                    } else {
                        (0, 0)
                    };
                    (pos.i + base.0, pos.k + base.1)
                });
                let placed_block = get_hex(placed_i, placed_k); // TODO: unnecessary use of get_hex involking Python GIL
                if !HexEngine::check_range_coords(placed_i, placed_k, self.radius).unwrap_or(false) || self.state_of(placed_i, placed_k).unwrap_or(false) {
                    return Ok(0.0);
                }
                total_possible += 6 - piece.count_neighbors(&placed_block.borrow(Python::with_gil(|py| py)));
                total_populated += self.count_neighbors_coordinate(placed_i, placed_k).unwrap_or(0);
            }
        }
        Ok(if total_possible > 0 { total_populated as f64 / total_possible as f64 } else { 0.0 })
    }

    /// Compute the entropy of the hexagonal grid based on the distribution of 7-block patterns.
    /// 
    /// Entropy is calculated using the Shannon entropy formula, measuring the randomness of block
    /// arrangements in the grid. Each pattern consists of a central block and its six neighbors,
    /// forming a 7-block hexagonal box, as defined by the _get_pattern method. The entropy reflects
    /// the diversity of these patterns: a grid with randomly distributed filled and empty blocks
    /// has higher entropy than one with structured patterns (e.g., all blocks in a line or cluster).
    /// A grid with all blocks filled or all empty has zero entropy. Inverting the grid (swapping
    /// filled and empty states) results in the same entropy, as the pattern distribution is unchanged.
    ///
    /// The method iterates over all blocks within the grid's radius (excluding the outermost layer
    /// to ensure all neighbors are in range), counts the frequency of each possible 7-block pattern
    /// (2^7 = 128 patterns), and computes the entropy using the Shannon entropy formula:
    ///     H = -Σ (p * log₂(p))
    /// where p is the probability of each pattern (frequency divided by total patterns counted).
    /// Blocks on the grid's boundary (beyond radius - 1) are excluded to avoid incomplete patterns.
    ///
    /// Returns:
    /// - entropy (float): The entropy of the grid in bits, a non-negative value representing the randomness
    ///     - of block arrangements. Returns 0.0 for a uniform grid (all filled or all empty) or if no valid patterns are counted.
    pub fn compute_entropy(&self) -> PyResult<f64> {
        let mut pattern_counts = [0usize; 128];
        let mut pattern_total = 0usize;
        let radius = self.radius as i32 - 1;
        for i in 0..self.states.len() {
            let center = match self.hex_coordinate_of(i) {
                Ok(h) => h,
                Err(_) => continue,
            };
            let (center_i, center_k) = center.into();
            let shifted_i = center_i - 1;
            let shifted_k = center_k + 1;
            if HexEngine::check_range_coords(shifted_i, shifted_k, radius as usize)? {
                let pattern = self.pattern_of(center_i, center_k)?;
                pattern_counts[pattern as usize] += 1;
                pattern_total += 1;
            }
        }
        let mut entropy = 0.0;
        for &count in &pattern_counts {
            if count > 0 {
                let p = count as f64 / pattern_total as f64;
                entropy -= p * p.log2();
            }
        }
        Ok(entropy)
    }

    /// Generate all possible HexEngine instances representing valid occupancy states for a given radius.
    /// All generated HexEngines will have eliminations already applied, meaning they will not contain any fully occupied lines.
    /// 
    /// For large radius values, this method may take a long time and significant resource to compute due to the exponential growth of possible states.
    /// It is recommended to cache the results for specific radius values to avoid recomputation. HexEngine does not provide a dictionary for caching such data.
    /// 
    /// Arguments:
    /// - radius (int): The radius of the hexagonal grid for which to generate all possible HexEngines.
    /// Returns:
    /// - list[HexEngine]: A list of HexEngine instances representing all valid occupancy states for the specified radius.
    /// Raises:
    /// - TypeError: If radius is not an integer greater than 1. Only empty engine is valid for radius 1.
    #[classmethod]
    pub fn all_engines(_cls: &pyo3::Bound<'_, PyType>, radius: usize) -> PyResult<Vec<HexEngine>> {
        if radius < 2 {
            return Err(pyo3::exceptions::PyTypeError::new_err("Radius must be an integer greater than 1"));
        }
        let length = 1 + 3 * radius * (radius - 1);
        let mut result = Vec::new();
        for i in 0..(1 << length) {
            let mut states = Vec::with_capacity(length);
            for j in 0..length {
                states.push(((i >> j) & 1) == 1);
            }
            let mut engine = HexEngine { radius, states };
            let eliminated = engine.eliminate().unwrap_or_default();
            if !eliminated.is_empty() {
                continue;
            }
            result.push(engine);
        }
        Ok(result)
    }
}
