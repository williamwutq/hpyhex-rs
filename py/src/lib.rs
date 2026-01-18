#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;

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
