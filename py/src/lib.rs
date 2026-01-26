#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "numpy")]
use numpy::{PyArray1, PyArray2, PyArray, PyArrayMethods};
#[cfg(feature = "numpy")]
use numpy::ndarray::array;
#[cfg(feature = "numpy")]
use numpy::ndarray;
#[cfg(all(feature = "numpy", feature = "half"))]
use numpy::{PyArrayDescr, dtype_bound};

use pyo3::prelude::*;
use pyo3::types::{PyList, PyAny, PyType};

#[pymodule]
fn hpyhex(_py: Python, m: &pyo3::Bound<'_, PyModule>) -> PyResult<()> {
    m.add("version", "hpyhex-rs-0.2.1")?;
    m.add_class::<Hex>()?;
    m.add_class::<Piece>()?;
    m.add_class::<HexEngine>()?;
    m.add_function(wrap_pyfunction!(random_engine, m)?)?;
    m.add_class::<PieceFactory>()?;
    m.add_class::<Game>()?;
    Ok(())
}

#[cfg(all(feature = "numpy", feature = "half"))]
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct F16(pub half::f16);

#[cfg(all(feature = "numpy", feature = "half"))]
impl F16 {
    #[inline]
    pub fn from_f32(v: f32) -> Self {
        F16(half::f16::from_f32(v))
    }
}

#[cfg(all(feature = "numpy", feature = "half"))]
impl PartialOrd for F16 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.to_f32().partial_cmp(&other.0.to_f32())
    }
}

#[cfg(all(feature = "numpy", feature = "half"))]
unsafe impl numpy::Element for F16 {
    const IS_COPY: bool = true;
    fn get_dtype_bound(py: pyo3::Python<'_>) -> pyo3::Bound<'_, PyArrayDescr> {
        dtype_bound::<F16>(py)
    }
}

#[cfg(all(feature = "numpy", feature = "half"))]
impl From<half::f16> for F16 {
    #[inline]
    fn from(v: half::f16) -> Self { F16(v) }
}

#[cfg(all(feature = "numpy", feature = "half"))]
impl From<F16> for half::f16 {
    #[inline]
    fn from(v: F16) -> Self { v.0 }
}

#[cfg(all(feature = "numpy", feature = "half"))]
impl std::ops::Deref for F16 {
    type Target = half::f16;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub trait BitScalar: Copy {
    /// Returns the zero value for the scalar type.
    fn zero() -> Self;
    /// Returns the one value for the scalar type.
    fn one() -> Self;
    /// Evaluates the scalar as a boolean predicate.
    /// 
    /// Returns true if the scalar represents a truthy value, false if falsy.
    fn predicate(self) -> bool;
}

pub trait SizeScalar: Copy {
    /// Converts the scalar to a usize.
    fn to_usize(self) -> Option<usize>;
    /// Creates the scalar from a usize.
    fn from_usize(v: usize) -> Self;
    /// Returns a sentinel value for the scalar type.
    fn sentinel() -> Self;
}

macro_rules! int_bitscalar {
    ($($t:ty),*) => {
        $(impl BitScalar for $t {
            #[inline] fn zero() -> Self { 0 }
            #[inline] fn one() -> Self { 1 }
            #[inline] fn predicate(self) -> bool { self > 0 }
            // For unsigned integers, non-zero means true, only zero means false
            // For signed integers, positive means true, zero and negative means false
        })*
    };
}

macro_rules! float_bitscalar {
    ($($t:ty),*) => {
        $(impl BitScalar for $t {
            #[inline] fn zero() -> Self { 0.0 }
            #[inline] fn one() -> Self { 1.0 }
            #[inline] fn predicate(self) -> bool { self > 0.0 }
        })*
    };
}

macro_rules! iint_sizescalar {
    ($($t:ty),*) => {
        $(impl SizeScalar for $t {
            #[inline] fn to_usize(self) -> Option<usize> {
                if self >= 0 {
                    Some(self as usize)
                } else {
                    None
                }
            }
            #[inline] fn from_usize(v: usize) -> Self { v as Self }
            #[inline] fn sentinel() -> Self { -1 as Self }
        })*
    };
}

int_bitscalar!(i8, u8, i16, u16, i32, u32, i64, u64);
float_bitscalar!(f32, f64);
// u8 and i8 are not used since they are usually too small for sizes
iint_sizescalar!(i16, i32, i64);
impl SizeScalar for u16 {
    #[inline] fn to_usize(self) -> Option<usize> {
        match self {
            u16::MAX => None,
            v => Some(v as usize),
        }
    }
    #[inline] fn from_usize(v: usize) -> Self { v as Self }
    #[inline] fn sentinel() -> Self { u16::MAX }
}

impl SizeScalar for u32 {
    #[inline] fn to_usize(self) -> Option<usize> {
        match self {
            u32::MAX => None,
            v => Some(v as usize),
        }
    }
    #[inline] fn from_usize(v: usize) -> Self { v as Self }
    #[inline] fn sentinel() -> Self { u32::MAX }
}

impl SizeScalar for u64 {
    #[inline] fn to_usize(self) -> Option<usize> {
        if self == u64::MAX {
            None
        } else {
            usize::try_from(self).ok()
        }
    }
    #[inline] fn from_usize(v: usize) -> Self { v as Self }
    #[inline] fn sentinel() -> Self { u64::MAX }
}

impl BitScalar for bool {
    #[inline] fn zero() -> Self { false }
    #[inline] fn one() -> Self { true }
    #[inline] fn predicate(self) -> bool { self }
}

#[cfg(feature = "half")]
impl BitScalar for F16 {
    #[inline] fn zero() -> Self { F16(half::f16::from_f32(0.0)) }
    #[inline] fn one() -> Self { F16(half::f16::from_f32(1.0)) }
    #[inline] fn predicate(self) -> bool { self.0.to_f32() > 0.0 }
}

use std::sync::OnceLock;
use std::hash::{Hash, Hasher};

// Note: Hex has no need for serialization to numpy arrays, as it is just a coordinate container.

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
/// 
/// Hpyhex-rs Serialization:
/// The class is inter-operable with the Rust hpyhex-rs crate through the following methods:
/// - hpyhex_rs_serialize(): Serialize the Hex to a byte vector in the format used by the Rust hpyhex-rs crate.
/// - hpyhex_rs_deserialize(data): Deserialize a Hex from a byte vector in the format used by the Rust hpyhex-rs crate.
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
    /* ------------------------------------- HPYHEX-RS ------------------------------------- */

    /// Serialize the Piece to a byte vector according the format used by the Rust hpyhex-rs crate.
    /// 
    /// The serialization format is 4 bytes for i (i32) followed by 4 bytes for k (i32), both in little-endian order.
    /// 
    /// Returns:
    /// - bytes: A byte vector containing the serialized Piece data.
    pub fn hpyhex_rs_serialize<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
        let mut bytes = Vec::with_capacity(8);
        bytes.extend_from_slice(&self.i.to_le_bytes());
        bytes.extend_from_slice(&self.k.to_le_bytes());
        pyo3::types::PyBytes::new_bound(py, &bytes)
    }

    /// Deserialize a Hex from a byte vector according the format used by the Rust hpyhex-rs crate.
    /// 
    /// The deserialization format expects 8 bytes: 4 bytes for i (i32) followed by 4 bytes for k (i32), both in little-endian order.
    /// 
    /// Arguments:
    /// - data (bytes, bytearray, or list[int]): The byte vector containing the serialized Hex data.
    /// Returns:
    /// - Hex: The deserialized Hex instance.
    #[staticmethod]
    pub fn hpyhex_rs_deserialize(data: &Bound<'_, PyAny>) -> PyResult<Py<Hex>> {
        use pyo3::types::{PyBytes, PyByteArray};
        // Extract PyBytes, PyByteArray, or Vec<u8>
        let value: Vec<u8> = if let Ok(py_bytes) = data.extract::<&PyBytes>() {
            py_bytes.as_bytes().to_vec()
        } else if let Ok(py_bytearray) = data.extract::<&PyByteArray>() {
            unsafe { py_bytearray.as_bytes() }.to_vec()
        } else if let Ok(vec_u8) = data.extract::<Vec<u8>>() {
            vec_u8
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err("Input must be bytes, bytearray, or list of integers"));
        };

        if value.len() != 8 {
            return Err(pyo3::exceptions::PyValueError::new_err("Invalid data length for Hex deserialization"));
        }

        let i = i32::from_le_bytes([value[0], value[1], value[2], value[3]]);
        let k = i32::from_le_bytes([value[4], value[5], value[6], value[7]]);
        Ok(get_hex(i, k))
    }

    /* ---------------------------------------- HPYHEX PYTHON API ---------------------------------------- */

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
    pub fn __iter__(slf: PyRef<'_, Self>) -> PyResult<HexIterator> {
        Ok(HexIterator {
            i: slf.i,
            k: slf.k,
            index: 0,
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

#[pyclass]
/// An iterator over Hex
pub struct HexIterator {
    i: i32,
    k: i32,
    index: usize,
}

#[pymethods]
impl HexIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<i32> {
        if slf.index == 0 {
            slf.index += 1;
            Some(slf.i)
        } else if slf.index == 1 {
            slf.index += 1;
            Some(slf.k)
        } else {
            None
        }
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
/// 
/// Hpyhex-rs Serialization:
/// The class is inter-operable with the Rust hpyhex-rs crate through the following methods:
/// - hpyhex_rs_serialize(): Serialize the Piece to a byte vector in the format used by the Rust hpyhex-rs crate.
/// - hpyhex_rs_deserialize(data): Deserialize a Piece from a byte vector in the format used by the Rust hpyhex-rs crate.
/// 
/// Numpy Support (Requires "numpy" feature):
/// 
/// Offer methods to convert the Piece's block states to NumPy ndarray representations and vice versa.
/// The conversion will always be to an ndarray of shape (7,).
/// 
/// In addition, provide methods to convert a list of Piece instances to stacked or flat NumPy ndarray representations,
/// and vice versa. The stacked representation will have shape (N, 7) with stride (8, 1), while the flat
/// representation will have shape (N*7,).
/// 
/// Support the following NumPy array types:
/// - bool
/// - int8
/// - uint8
/// - int16
/// - uint16
/// - int32
/// - uint32
/// - int64
/// - uint64
/// - half (f16) [Requires "half" feature, experimental]
/// - float32
/// - float64
/// 
/// The from_numpy_* methods will validate the input array shape and types, and raise a ValueError if the
/// input is invalid. The to_numpy_* methods will return a new NumPy ndarray representing the block states.
/// 
/// to_numpy() defaults to bool representation, but there are no from_numpy that can take in different types,
/// because numpy ndarrays cannot be easily casted into each other.
/// 
/// For a list of Piece instances, the vec_from_numpy_*_stacked and vec_from_numpy_*_flat functions
/// can be used to convert from NumPy ndarray representations to a list of Piece instances, and they will
/// raise a ValueError if the input is invalid. For converting a list of Piece instances to NumPy ndarray,
/// the vec_to_numpy_*_stacked and vec_to_numpy_*_flat functions can be used.
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

impl std::fmt::Display for Piece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..7 {
            let b = (self.state & (1 << (6 - i))) != 0;
            write!(f, "{}", if b { '1' } else { '0' })?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for Piece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::from("Piece{");
        for i in 0..7 {
            if i > 0 {
                s.push_str(", ");
            }
            let hex = &Piece::positions[i];
            let occupied = (self.state & (1 << (6 - i))) != 0;
            s.push_str(&format!(
                "({}, {}, {})",
                hex.i(),
                hex.k(),
                occupied
            ));
        }
        s.push('}');
        write!(f, "{s}")
    }
}

impl Piece {
    /// Returns the number of occupied blocks
    /// 
    /// # Returns
    /// The count of occupied blocks as a u32.
    #[inline]
    pub const fn count(&self) -> u32 {
        self.state.count_ones()
    }

    /// Get the cached Piece instance for the given state.
    /// 
    /// # Arguments
    /// - state (u8): The state byte representing the occupancy of the blocks.
    /// 
    /// # Returns
    /// - Py<Piece>: The cached Piece instance corresponding to the given state.
    #[inline]
    pub fn get_cached(state: u8) -> Py<Piece> {
        let truncated = state & 0b0111_1111; // Only 7 bits used
        let cache = PIECE_CACHE.get_or_init(|| initialize_piece_cache());
        cache[truncated as usize].clone()
    }
}

#[cfg(feature = "numpy")]
fn vec_to_numpy_flat_impl<'py, T>(
    py: Python<'py>,
    pieces: Vec<Py<Piece>>,
) -> Py<PyArray1<T>>
where
    T: BitScalar + Copy + numpy::Element,
{
    let mut arr = Vec::with_capacity(pieces.len() * 7);
    for piece in pieces.iter() {
        let piece_ref = piece.bind(py).extract::<PyRef<Piece>>().unwrap();
        for i in 0..7 {
            let b = if (piece_ref.state & (1 << (6 - i))) != 0 { T::one() } else { T::zero() };
            arr.push(b);
        }
    }
    PyArray1::from_vec_bound(py, arr).unbind()
}

#[cfg(feature = "numpy")]
fn vec_to_numpy_stacked_impl<'py, T>(
    py: Python<'py>,
    pieces: Vec<Py<Piece>>,
) -> Py<PyArray2<T>>
where
    T: BitScalar + Copy + numpy::Element,
{
    use ndarray::{Array2, ShapeBuilder};

    let n = pieces.len();
    let shape = (n, 7).strides((8, 1));

    let mut vec = Vec::with_capacity(n * 8);

    for piece in &pieces {
        let piece_ref = piece.bind(py).extract::<PyRef<Piece>>().unwrap();
        let s = piece_ref.state;

        let v = [
            if s & 0b1000000 != 0 { T::one() } else { T::zero() },
            if s & 0b0100000 != 0 { T::one() } else { T::zero() },
            if s & 0b0010000 != 0 { T::one() } else { T::zero() },
            if s & 0b0001000 != 0 { T::one() } else { T::zero() },
            if s & 0b0000100 != 0 { T::one() } else { T::zero() },
            if s & 0b0000010 != 0 { T::one() } else { T::zero() },
            if s & 0b0000001 != 0 { T::one() } else { T::zero() },
        ];

        vec.extend_from_slice(&v);
        vec.push(T::zero()); // padding for stride
    }

    let array = Array2::from_shape_vec(shape, vec).unwrap();
    PyArray2::from_owned_array_bound(py, array).unbind()
}

#[cfg(feature = "numpy")]
fn vec_from_numpy_flat_impl<T>(
    array: &Bound<'_, PyArray1<T>>,
) -> PyResult<Vec<Py<Piece>>>
where
    T: BitScalar + Copy + numpy::Element,
{
    let slice = unsafe { array.as_slice()? };
    if slice.len() % 7 != 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input array length must be a multiple of 7",
        ));
    }
    let mut pieces = Vec::with_capacity(slice.len() / 7);
    for chunk in slice.chunks(7) {
        let mut state: u8 = 0;
        for (i, &value) in chunk.iter().enumerate() {
            if T::predicate(value) {
                state |= 1 << (6 - i);
            }
        }
        pieces.push(Piece::get_cached(state));
    }
    Ok(pieces)
}

#[cfg(feature = "numpy")]
fn vec_from_numpy_stacked_impl<T>(
    array: &Bound<'_, PyArray2<T>>,
) -> PyResult<Vec<Py<Piece>>>
where
    T: BitScalar + Copy + numpy::Element,
{
    use numpy::PyUntypedArrayMethods;
    let shape = array.shape();
    if shape.len() != 2 || shape[1] != 7 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input array must have shape (n, 7)",
        ));
    }
    let n = shape[0];
    let c = array.strides()[0] as usize;
    let mut pieces = Vec::with_capacity(n);
    for i in 0..n {
        let row = unsafe { array.as_slice()?.get(
            i * c..(i * c + 7)
        ).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to access row in input array")
        })? };
        let mut state: u8 = 0;
        for (j, &value) in row.iter().enumerate() {
            if T::predicate(value) {
                state |= 1 << (6 - j);
            }
        }
        pieces.push(Piece::get_cached(state));
    }
    Ok(pieces)
}

#[cfg(feature = "numpy")]
fn to_numpy_piece_impl<'py, T>(
    py: Python<'py>,
    piece: &Piece,
) -> Py<PyArray1<T>>
where
    T: BitScalar + Copy + numpy::Element,
{
    let arr = array![
        if (piece.state & 0b1000000) != 0 { T::one() } else { T::zero() },
        if (piece.state & 0b0100000) != 0 { T::one() } else { T::zero() },
        if (piece.state & 0b0010000) != 0 { T::one() } else { T::zero() },
        if (piece.state & 0b0001000) != 0 { T::one() } else { T::zero() },
        if (piece.state & 0b0000100) != 0 { T::one() } else { T::zero() },
        if (piece.state & 0b0000010) != 0 { T::one() } else { T::zero() },
        if (piece.state & 0b0000001) != 0 { T::one() } else { T::zero() },
    ];
    PyArray1::from_array_bound(py, &arr).unbind()
}

#[cfg(feature = "numpy")]
fn from_numpy_piece_impl<T>(
    array: &Bound<'_, PyArray<T, ndarray::Dim<[usize; 1]>>>,
) -> PyResult<Py<Piece>>
where
    T: BitScalar + Copy + numpy::Element,
{
    let slice = unsafe { array.as_slice()? };
    if slice.len() != 7 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input array must have exactly 7 elements",
        ));
    }
    let mut state: u8 = 0;
    for (i, &value) in slice.iter().enumerate() {
        if T::predicate(value) {
            state |= 1 << (6 - i);
        }
    }
    Ok(Piece::get_cached(state))
}

#[pymethods]
impl Piece {
    /* ------------------------------------- HPYHEX-RS ------------------------------------- */

    /// Serialize the Piece to a byte vector according the format used by the Rust hpyhex-rs crate.
    /// 
    /// The serialization format is a single byte representing the occupancy state of the blocks.
    /// 
    /// Returns:
    /// - bytes: A byte vector containing the serialized Piece data.
    pub fn hpyhex_rs_serialize<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
        pyo3::types::PyBytes::new_bound(py, &[self.state])
    }

    /// Deserialize a Piece from a byte vector according the format used by the Rust hpyhex-rs crate.
    /// 
    /// The deserialization format expects a single byte representing the occupancy state of the blocks.
    /// 
    /// Arguments:
    /// - data (bytes): A byte vector containing the serialized Piece data.
    /// 
    /// Returns:
    /// - Piece: The deserialized Piece instance.
    /// Raises:
    /// - ValueError: If the input data length is invalid.
    #[staticmethod]
    pub fn hpyhex_rs_deserialize(data: &Bound<'_, PyAny>) -> PyResult<Py<Piece>> {
        use pyo3::types::{PyBytes, PyByteArray};
        // Extract PyBytes, PyByteArray, or Vec<u8>
        let value: Vec<u8> = if let Ok(py_bytes) = data.downcast::<PyBytes>() {
            py_bytes.as_bytes().to_vec()
        } else if let Ok(py_bytearray) = data.downcast::<PyByteArray>() {
            unsafe { py_bytearray.as_bytes() }.to_vec()
        } else if let Ok(vec_u8) = data.extract::<Vec<u8>>() {
            vec_u8
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err("Input must be bytes, bytearray, or list of integers"));
        };
        if value.len() != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err("Invalid data length for Piece deserialization"));
        }
        let state = value[0] & 0b0111_1111; // Only 7 bits used
        Ok(Piece::get_cached(state))
    }

    /* ---------------------------------------- NUMPY EXPORTS ---------------------------------------- */

    /// Get the default NumPy ndarray representation of the Piece's block states.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of boolean values representing the block states.
    #[cfg(feature = "numpy")]
    pub fn to_numpy(&self, py: Python) -> Py<PyArray1<bool>> {
        self.to_numpy_bool(py)
    }

    /// Get the NumPy ndarray boolean representation of the Piece's block states.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of boolean values representing the block states.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_bool(&self, py: Python) -> Py<PyArray1<bool>> {
        to_numpy_piece_impl::<bool>(py, self)
    }

    /// Get the NumPy ndarray int8 representation of the Piece's block states.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of int8 values (1 for occupied, 0 for unoccupied) representing the block states.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_int8(&self, py: Python) -> Py<PyArray1<i8>> {
        to_numpy_piece_impl::<i8>(py, self)
    }

    /// Get the NumPy ndarray uint8 representation of the Piece's block states.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of uint8 values (1 for occupied, 0 for unoccupied) representing the block states.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_uint8(&self, py: Python) -> Py<PyArray1<u8>> {
        to_numpy_piece_impl::<u8>(py, self)
    }

    /// Get the NumPy ndarray int16 representation of the Piece's block states.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of int16 values (1 for occupied, 0 for unoccupied) representing the block states.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_int16(&self, py: Python) -> Py<PyArray1<i16>> {
        to_numpy_piece_impl::<i16>(py, self)
    }

    /// Get the NumPy ndarray uint16 representation of the Piece's block states.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of uint16 values (1 for occupied, 0 for unoccupied) representing the block states.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_uint16(&self, py: Python) -> Py<PyArray1<u16>> {
        to_numpy_piece_impl::<u16>(py, self)
    }

    /// Get the NumPy ndarray int32 representation of the Piece's block states.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of int32 values (1 for occupied, 0 for unoccupied) representing the block states.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_int32(&self, py: Python) -> Py<PyArray1<i32>> {
        to_numpy_piece_impl::<i32>(py, self)
    }

    /// Get the NumPy ndarray uint32 representation of the Piece's block states.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of uint32 values (1 for occupied, 0 for unoccupied) representing the block states.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_uint32(&self, py: Python) -> Py<PyArray1<u32>> {
        to_numpy_piece_impl::<u32>(py, self)
    }

    /// Get the NumPy ndarray int64 representation of the Piece's block states.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of int64 values (1 for occupied, 0 for unoccupied) representing the block states.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_int64(&self, py: Python) -> Py<PyArray1<i64>> {
        to_numpy_piece_impl::<i64>(py, self)
    }

    /// Get the NumPy ndarray uint64 representation of the Piece's block states.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of uint64 values (1 for occupied, 0 for unoccupied) representing the block states.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_uint64(&self, py: Python) -> Py<PyArray1<u64>> {
        to_numpy_piece_impl::<u64>(py, self)
    }

    /// Get the NumPy ndarray float16 representation of the Piece's block states.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of float16 values (1.0 for occupied, 0.0 for unoccupied) representing the block states.
    /// Warning:
    /// - The 'half' feature, which add support for float16, is still experimental and may not be stable. On machines that does
    /// not support float16 or installed with a version of numpy that does not support float16, this function may lead to
    /// undefined behavior or crashes. Testing show that on some systems, this can result in memory misinterpretation issues
    /// causing incorrect values to be read, and on other systems, it cause the entire program to halt but not crash.
    /// Use with caution.
    #[cfg(all(feature = "numpy", feature = "half"))]
    pub fn to_numpy_float16(&self, py: Python) -> Py<PyArray1<F16>> {
        to_numpy_piece_impl::<F16>(py, self)
    }

    /// Get the NumPy ndarray float32 representation of the Piece's block states.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of float32 values (1.0 for occupied, 0.0 for unoccupied) representing the block states.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_float32(&self, py: Python) -> Py<PyArray1<f32>> {
        to_numpy_piece_impl::<f32>(py, self)
    }

    /// Get the NumPy ndarray float64 representation of the Piece's block states.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of float64 values (1.0 for occupied, 0.0 for unoccupied) representing the block states.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_float64(&self, py: Python) -> Py<PyArray1<f64>> {
        to_numpy_piece_impl::<f64>(py, self)
    }

    /// Convert a vector of Piece instances to a flat NumPy ndarray of boolean values.
    /// 
    /// Arguments:
    /// - pieces (list[Piece]): A list of Piece instances to convert.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of boolean values representing the block states of all pieces.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_to_numpy_flat<'py>(py: Python<'py>, pieces: Vec<Py<Piece>>) -> Py<PyArray1<bool>> {
        Self::vec_to_numpy_bool_flat(py, pieces)
    }

    /// Convert a vector of Piece instances to a stacked NumPy ndarray of boolean values.
    /// 
    /// Arguments:
    /// - pieces (list[Piece]): A list of Piece instances to convert.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array of boolean values with shape (num_pieces, 7)
    ///   representing the block states of all pieces. The array uses a stride of 8 for memory alignment.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_to_numpy_stacked<'py>(py: Python<'py>, pieces: Vec<Py<Piece>>) -> Py<PyArray2<bool>> {
        Self::vec_to_numpy_bool_stacked(py, pieces)
    }

    /// Convert a vector of Piece instances to a flat NumPy ndarray of boolean values.
    /// 
    /// Arguments:
    /// - pieces (list[Piece]): A list of Piece instances to convert.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of boolean values representing the block states of all pieces.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_to_numpy_bool_flat<'py>(py: Python<'py>, pieces: Vec<Py<Piece>>) -> Py<PyArray1<bool>> {
        vec_to_numpy_flat_impl::<bool>(py, pieces)
    }

    /// Convert a vector of Piece instances to a stacked NumPy ndarray of boolean values.
    /// 
    /// Arguments:
    /// - pieces (list[Piece]): A list of Piece instances to convert.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array of boolean values with shape (num_pieces, 7)
    ///   representing the block states of all pieces. The array uses a stride of 8 for memory alignment.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_to_numpy_bool_stacked<'py>(py: Python<'py>, pieces: Vec<Py<Piece>>) -> Py<PyArray2<bool>> {
        vec_to_numpy_stacked_impl::<bool>(py, pieces)
    }

    /// Convert a vector of Piece instances to a flat NumPy ndarray of int8 values.
    /// 
    /// Arguments:
    /// - pieces (list[Piece]): A list of Piece instances to convert.
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of int8 values representing the block states of all pieces.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_to_numpy_int8_flat<'py>(py: Python<'py>, pieces: Vec<Py<Piece>>) -> Py<PyArray1<i8>> {
        vec_to_numpy_flat_impl::<i8>(py, pieces)
    }

    /// Convert a vector of Piece instances to a stacked NumPy ndarray of int8 values.
    /// 
    /// Arguments:
    /// - pieces (list[Piece]): A list of Piece instances to convert.
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array of int8 values with shape (num_pieces, 7)
    ///   representing the block states of all pieces. The array uses a stride of 8 for memory alignment.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_to_numpy_int8_stacked<'py>(py: Python<'py>, pieces: Vec<Py<Piece>>) -> Py<PyArray2<i8>> {
        vec_to_numpy_stacked_impl::<i8>(py, pieces)
    }

    /// Convert a vector of Piece instances to a flat NumPy ndarray of uint8 values.
    /// 
    /// Arguments:
    /// - pieces (list[Piece]): A list of Piece instances to convert.
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of uint8 values representing the block states of all pieces.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_to_numpy_uint8_flat<'py>(py: Python<'py>, pieces: Vec<Py<Piece>>) -> Py<PyArray1<u8>> {
        vec_to_numpy_flat_impl::<u8>(py, pieces)
    }

    /// Convert a vector of Piece instances to a stacked NumPy ndarray of uint8 values.
    /// 
    /// Arguments:
    /// - pieces (list[Piece]): A list of Piece instances to convert.
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array of uint8 values with shape (num_pieces, 7)
    ///   representing the block states of all pieces. The array uses a stride of 8 for memory alignment.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_to_numpy_uint8_stacked<'py>(py: Python<'py>, pieces: Vec<Py<Piece>>) -> Py<PyArray2<u8>> {
        vec_to_numpy_stacked_impl::<u8>(py, pieces)
    }

    /// Convert a vector of Piece instances to a flat NumPy ndarray of int16 values.
    /// 
    /// Arguments:
    /// - pieces (list[Piece]): A list of Piece instances to convert.
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of int16 values representing the block states of all pieces.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_to_numpy_int16_flat<'py>(py: Python<'py>, pieces: Vec<Py<Piece>>) -> Py<PyArray1<i16>> {
        vec_to_numpy_flat_impl::<i16>(py, pieces)
    }

    /// Convert a vector of Piece instances to a stacked NumPy ndarray of int16 values.
    /// 
    /// Arguments:
    /// - pieces (list[Piece]): A list of Piece instances to convert.
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array of int16 values with shape (num_pieces, 7)
    ///   representing the block states of all pieces. The array uses a stride of 8 for memory alignment.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_to_numpy_int16_stacked<'py>(py: Python<'py>, pieces: Vec<Py<Piece>>) -> Py<PyArray2<i16>> {
        vec_to_numpy_stacked_impl::<i16>(py, pieces)
    }

    /// Convert a vector of Piece instances to a flat NumPy ndarray of uint16 values.
    /// 
    /// Arguments:
    /// - pieces (list[Piece]): A list of Piece instances to convert.
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of uint16 values representing the block states of all pieces.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_to_numpy_uint16_flat<'py>(py: Python<'py>, pieces: Vec<Py<Piece>>) -> Py<PyArray1<u16>> {
        vec_to_numpy_flat_impl::<u16>(py, pieces)
    }

    /// Convert a vector of Piece instances to a stacked NumPy ndarray of uint16 values.
    /// 
    /// Arguments:
    /// - pieces (list[Piece]): A list of Piece instances to convert.
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array of uint16 values with shape (num_pieces, 7)
    ///   representing the block states of all pieces. The array uses a stride of 8 for memory alignment.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_to_numpy_uint16_stacked<'py>(py: Python<'py>, pieces: Vec<Py<Piece>>) -> Py<PyArray2<u16>> {
        vec_to_numpy_stacked_impl::<u16>(py, pieces)
    }

    /// Convert a vector of Piece instances to a flat NumPy ndarray of int32 values.
    /// 
    /// Arguments:
    /// - pieces (list[Piece]): A list of Piece instances to convert.
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of int32 values representing the block states of all pieces.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_to_numpy_int32_flat<'py>(py: Python<'py>, pieces: Vec<Py<Piece>>) -> Py<PyArray1<i32>> {
        vec_to_numpy_flat_impl::<i32>(py, pieces)
    }

    /// Convert a vector of Piece instances to a stacked NumPy ndarray of int32 values.
    /// 
    /// Arguments:
    /// - pieces (list[Piece]): A list of Piece instances to convert.
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array of int32 values with shape (num_pieces, 7)
    ///   representing the block states of all pieces. The array uses a stride of 8 for memory alignment.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_to_numpy_int32_stacked<'py>(py: Python<'py>, pieces: Vec<Py<Piece>>) -> Py<PyArray2<i32>> {
        vec_to_numpy_stacked_impl::<i32>(py, pieces)
    }

    /// Convert a vector of Piece instances to a flat NumPy ndarray of uint32 values.
    /// 
    /// Arguments:
    /// - pieces (list[Piece]): A list of Piece instances to convert.
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of uint32 values representing the block states of all pieces.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_to_numpy_uint32_flat<'py>(py: Python<'py>, pieces: Vec<Py<Piece>>) -> Py<PyArray1<u32>> {
        vec_to_numpy_flat_impl::<u32>(py, pieces)
    }

    /// Convert a vector of Piece instances to a stacked NumPy ndarray of uint32 values.
    /// 
    /// Arguments:
    /// - pieces (list[Piece]): A list of Piece instances to convert.
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array of uint32 values with shape (num_pieces, 7)
    ///   representing the block states of all pieces. The array uses a stride of 8 for memory alignment.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_to_numpy_uint32_stacked<'py>(py: Python<'py>, pieces: Vec<Py<Piece>>) -> Py<PyArray2<u32>> {
        vec_to_numpy_stacked_impl::<u32>(py, pieces)
    }

    /// Create a vector of Piece instances to a flat NumPy ndarray of int64 values.
    /// 
    /// Arguments:
    /// - pieces (list[Piece]): A list of Piece instances to convert.
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of int64 values representing the block states of all pieces.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_to_numpy_int64_flat<'py>(py: Python<'py>, pieces: Vec<Py<Piece>>) -> Py<PyArray1<i64>> {
        vec_to_numpy_flat_impl::<i64>(py, pieces)
    }

    /// Create a vector of Piece instances to a stacked NumPy ndarray of int64 values.
    /// 
    /// Arguments:
    /// - pieces (list[Piece]): A list of Piece instances to convert.
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array of int64 values with shape (num_pieces, 7)
    ///   representing the block states of all pieces. The array uses a stride of 8 for memory alignment.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_to_numpy_int64_stacked<'py>(py: Python<'py>, pieces: Vec<Py<Piece>>) -> Py<PyArray2<i64>> {
        vec_to_numpy_stacked_impl::<i64>(py, pieces)
    }

    /// Create a vector of Piece instances to a flat NumPy ndarray of uint64 values.
    /// 
    /// Arguments:
    /// - pieces (list[Piece]): A list of Piece instances to convert.
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of uint64 values representing the block states of all pieces.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_to_numpy_uint64_flat<'py>(py: Python<'py>, pieces: Vec<Py<Piece>>) -> Py<PyArray1<u64>> {
        vec_to_numpy_flat_impl::<u64>(py, pieces)
    }

    /// Create a vector of Piece instances to a stacked NumPy ndarray of uint64 values.
    /// 
    /// Arguments:
    /// - pieces (list[Piece]): A list of Piece instances to convert.
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array of uint64 values with shape (num_pieces, 7)
    ///   representing the block states of all pieces. The array uses a stride of 8 for memory alignment.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_to_numpy_uint64_stacked<'py>(py: Python<'py>, pieces: Vec<Py<Piece>>) -> Py<PyArray2<u64>> {
        vec_to_numpy_stacked_impl::<u64>(py, pieces)
    }

    /// Create a vector of Piece instances to a flat NumPy ndarray of float32 values.
    /// 
    /// Arguments:
    /// - pieces (list[Piece]): A list of Piece instances to convert.
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of float32 values representing the block states of all pieces.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_to_numpy_float32_flat<'py>(py: Python<'py>, pieces: Vec<Py<Piece>>) -> Py<PyArray1<f32>> {
        vec_to_numpy_flat_impl::<f32>(py, pieces)
    }

    /// Create a vector of Piece instances to a stacked NumPy ndarray of float32 values.
    /// 
    /// Arguments:
    /// - pieces (list[Piece]): A list of Piece instances to convert.
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array of float32 values with shape (num_pieces, 7)
    ///   representing the block states of all pieces. The array uses a stride of 8 for memory alignment.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_to_numpy_float32_stacked<'py>(py: Python<'py>, pieces: Vec<Py<Piece>>) -> Py<PyArray2<f32>> {
        vec_to_numpy_stacked_impl::<f32>(py, pieces)
    }

    /// Create a vector of Piece instances to a flat NumPy ndarray of float64 values.
    /// 
    /// Arguments:
    /// - pieces (list[Piece]): A list of Piece instances to convert.
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of float64 values representing the block states of all pieces.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_to_numpy_float64_flat<'py>(py: Python<'py>, pieces: Vec<Py<Piece>>) -> Py<PyArray1<f64>> {
        vec_to_numpy_flat_impl::<f64>(py, pieces)
    }

    /// Create a vector of Piece instances to a stacked NumPy ndarray of float64 values.
    /// 
    /// Arguments:
    /// - pieces (list[Piece]): A list of Piece instances to convert.
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array of float64 values with shape (num_pieces, 7)
    ///   representing the block states of all pieces. The array uses a stride of 8 for memory alignment.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_to_numpy_float64_stacked<'py>(py: Python<'py>, pieces: Vec<Py<Piece>>) -> Py<PyArray2<f64>> {
        vec_to_numpy_stacked_impl::<f64>(py, pieces)
    }

    /// Create a vector of Piece instances to a flat NumPy ndarray of float16 values.
    /// 
    /// Arguments:
    /// - pieces (list[Piece]): A list of Piece instances to convert.
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of float16 values representing the block states of all pieces.
    /// Warning:
    /// - The 'half' feature, which add support for float16, is still experimental and may not be stable. On machines that does
    /// not support float16 or installed with a version of numpy that does not support float16, this function may lead to
    /// undefined behavior or crashes. Testing show that on some systems, this can result in memory misinterpretation issues
    /// causing incorrect values to be read, and on other systems, it cause the entire program to halt but not crash.
    /// Use with caution.
    #[cfg(all(feature = "numpy", feature = "half"))]
    #[staticmethod]
    pub fn vec_to_numpy_float16_flat<'py>(py: Python<'py>, pieces: Vec<Py<Piece>>) -> Py<PyArray1<F16>> {
        vec_to_numpy_flat_impl::<F16>(py, pieces)
    }

    /// Create a vector of Piece instances to a stacked NumPy ndarray of float16 values.
    /// 
    /// Arguments:
    /// - pieces (list[Piece]): A list of Piece instances to convert.
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array of float16 values with shape (num_pieces, 7)
    ///   representing the block states of all pieces. The array uses a stride of 8 for memory alignment.
    /// Warning:
    /// - The 'half' feature, which add support for float16, is still experimental and may not be stable. On machines that does
    /// not support float16 or installed with a version of numpy that does not support float16, this function may lead to
    /// undefined behavior or crashes. Testing show that on some systems, this can result in memory misinterpretation issues
    /// causing incorrect values to be read, and on other systems, it cause the entire program to halt but not crash.
    /// Use with caution.
    #[cfg(all(feature = "numpy", feature = "half"))]
    #[staticmethod]
    pub fn vec_to_numpy_float16_stacked<'py>(py: Python<'py>, pieces: Vec<Py<Piece>>) -> Py<PyArray2<F16>> {
        vec_to_numpy_stacked_impl::<F16>(py, pieces)
    }

    /// Create a vector of Piece instances from a flat NumPy ndarray of boolean values.
    /// 
    /// Arguments:
    /// - arr (numpy.ndarray): A 1D NumPy array of boolean values representing the block states.
    /// Returns:
    /// - list[Piece]: A list of Piece instances corresponding to the given block states.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_from_numpy_bool_flat<'py>(
        arr: Bound<'_, PyArray1<bool>>,
    ) -> PyResult<Vec<Py<Piece>>> {
        vec_from_numpy_flat_impl::<bool>(&arr)
    }

    /// Create a vector of Piece instances from a stacked NumPy ndarray of boolean values.
    /// 
    /// Arguments:
    /// - arr (numpy.ndarray): A 2D NumPy array of boolean values with shape (num_pieces, 7)
    ///   representing the block states.
    /// Returns:
    /// - list[Piece]: A list of Piece instances corresponding to the given block states.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_from_numpy_bool_stacked<'py>(
        arr: Bound<'_, PyArray2<bool>>,
    ) -> PyResult<Vec<Py<Piece>>> {
        vec_from_numpy_stacked_impl::<bool>(&arr)
    }

    /// Create a vector of Piece instances from a flat NumPy ndarray of int8 values.
    /// 
    /// Arguments:
    /// - arr (numpy.ndarray): A 1D NumPy array of int8 values representing the block states.
    /// Returns:
    /// - list[Piece]: A list of Piece instances corresponding to the given block states.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_from_numpy_int8_flat<'py>(
        arr: Bound<'_, PyArray1<i8>>,
    ) -> PyResult<Vec<Py<Piece>>> {
        vec_from_numpy_flat_impl::<i8>(&arr)
    }

    /// Create a vector of Piece instances from a stacked NumPy ndarray of int8 values.
    /// 
    /// Arguments:
    /// - arr (numpy.ndarray): A 2D NumPy array of int8 values with shape (num_pieces, 7)
    ///   representing the block states.
    /// Returns:
    /// - list[Piece]: A list of Piece instances corresponding to the given block states.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_from_numpy_int8_stacked<'py>(
        arr: Bound<'_, PyArray2<i8>>,
    ) -> PyResult<Vec<Py<Piece>>> {
        vec_from_numpy_stacked_impl::<i8>(&arr)
    }

    /// Create a vector of Piece instances from a flat NumPy ndarray of uint8 values.
    /// 
    /// Arguments:
    /// - arr (numpy.ndarray): A 1D NumPy array of uint8 values representing the block states.
    /// Returns:
    /// - list[Piece]: A list of Piece instances corresponding to the given block states.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_from_numpy_uint8_flat<'py>(
        arr: Bound<'_, PyArray1<u8>>,
    ) -> PyResult<Vec<Py<Piece>>> {
        vec_from_numpy_flat_impl::<u8>(&arr)
    }

    /// Create a vector of Piece instances from a stacked NumPy ndarray of uint8 values.
    /// 
    /// Arguments:
    /// - arr (numpy.ndarray): A 2D NumPy array of uint8 values with shape (num_pieces, 7)
    ///   representing the block states.
    /// Returns:
    /// - list[Piece]: A list of Piece instances corresponding to the given block states.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_from_numpy_uint8_stacked<'py>(
        arr: Bound<'_, PyArray2<u8>>,
    ) -> PyResult<Vec<Py<Piece>>> {
        vec_from_numpy_stacked_impl::<u8>(&arr)
    }

    /// Create a vector of Piece instances from a flat NumPy ndarray of int16 values.
    /// 
    /// Arguments:
    /// - arr (numpy.ndarray): A 1D NumPy array of int16 values representing the block states.
    /// Returns:
    /// - list[Piece]: A list of Piece instances corresponding to the given block states.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_from_numpy_int16_flat<'py>(
        arr: Bound<'_, PyArray1<i16>>,
    ) -> PyResult<Vec<Py<Piece>>> {
        vec_from_numpy_flat_impl::<i16>(&arr)
    }

    /// Create a vector of Piece instances from a stacked NumPy ndarray of int16 values.
    /// 
    /// Arguments:
    /// - arr (numpy.ndarray): A 2D NumPy array of int16 values with shape (num_pieces, 7)
    ///   representing the block states.
    /// Returns:
    /// - list[Piece]: A list of Piece instances corresponding to the given block states.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_from_numpy_int16_stacked<'py>(
        arr: Bound<'_, PyArray2<i16>>,
    ) -> PyResult<Vec<Py<Piece>>> {
        vec_from_numpy_stacked_impl::<i16>(&arr)
    }

    /// Create a vector of Piece instances from a flat NumPy ndarray of uint16 values.
    /// 
    /// Arguments:
    /// - arr (numpy.ndarray): A 1D NumPy array of uint16 values representing the block states.
    /// Returns:
    /// - list[Piece]: A list of Piece instances corresponding to the given block states.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_from_numpy_uint16_flat<'py>(
        arr: Bound<'_, PyArray1<u16>>,
    ) -> PyResult<Vec<Py<Piece>>> {
        vec_from_numpy_flat_impl::<u16>(&arr)
    }

    /// Create a vector of Piece instances from a stacked NumPy ndarray of uint16 values.
    /// 
    /// Arguments:
    /// - arr (numpy.ndarray): A 2D NumPy array of uint16 values with shape (num_pieces, 7)
    ///   representing the block states.
    /// Returns:
    /// - list[Piece]: A list of Piece instances corresponding to the given block states.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_from_numpy_uint16_stacked<'py>(
        arr: Bound<'_, PyArray2<u16>>,
    ) -> PyResult<Vec<Py<Piece>>> {
        vec_from_numpy_stacked_impl::<u16>(&arr)
    }

    /// Create a vector of Piece instances from a flat NumPy ndarray of int32 values.
    /// 
    /// Arguments:
    /// - arr (numpy.ndarray): A 1D NumPy array of int32 values representing the block states.
    /// Returns:
    /// - list[Piece]: A list of Piece instances corresponding to the given block states.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_from_numpy_int32_flat<'py>(
        arr: Bound<'_, PyArray1<i32>>,
    ) -> PyResult<Vec<Py<Piece>>> {
        vec_from_numpy_flat_impl::<i32>(&arr)
    }

    /// Create a vector of Piece instances from a stacked NumPy ndarray of int32 values.
    /// 
    /// Arguments:
    /// - arr (numpy.ndarray): A 2D NumPy array of int32 values with shape (num_pieces, 7)
    ///   representing the block states.
    /// Returns:
    /// - list[Piece]: A list of Piece instances corresponding to the given block states.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_from_numpy_int32_stacked<'py>(
        arr: Bound<'_, PyArray2<i32>>,
    ) -> PyResult<Vec<Py<Piece>>> {
        vec_from_numpy_stacked_impl::<i32>(&arr)
    }

    /// Create a vector of Piece instances from a flat NumPy ndarray of uint32 values.
    /// 
    /// Arguments:
    /// - arr (numpy.ndarray): A 1D NumPy array of uint32 values representing the block states.
    /// Returns:
    /// - list[Piece]: A list of Piece instances corresponding to the given block states.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_from_numpy_uint32_flat<'py>(
        arr: Bound<'_, PyArray1<u32>>,
    ) -> PyResult<Vec<Py<Piece>>> {
        vec_from_numpy_flat_impl::<u32>(&arr)
    }

    /// Create a vector of Piece instances from a stacked NumPy ndarray of uint32 values.
    /// 
    /// Arguments:
    /// - arr (numpy.ndarray): A 2D NumPy array of uint32 values with shape (num_pieces, 7)
    ///   representing the block states.
    /// Returns:
    /// - list[Piece]: A list of Piece instances corresponding to the given block states.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_from_numpy_uint32_stacked<'py>(
        arr: Bound<'_, PyArray2<u32>>,
    ) -> PyResult<Vec<Py<Piece>>> {
        vec_from_numpy_stacked_impl::<u32>(&arr)
    }

    /// Create a vector of Piece instances from a flat NumPy ndarray of int64 values.
    /// 
    /// Arguments:
    /// - arr (numpy.ndarray): A 1D NumPy array of int64 values representing the block states.
    /// Returns:
    /// - list[Piece]: A list of Piece instances corresponding to the given block states.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_from_numpy_int64_flat<'py>(
        arr: Bound<'_, PyArray1<i64>>,
    ) -> PyResult<Vec<Py<Piece>>> {
        vec_from_numpy_flat_impl::<i64>(&arr)
    }

    /// Create a vector of Piece instances from a stacked NumPy ndarray of int64 values.
    /// 
    /// Arguments:
    /// - arr (numpy.ndarray): A 2D NumPy array of int64 values with shape (num_pieces, 7)
    ///   representing the block states.
    /// Returns:
    /// - list[Piece]: A list of Piece instances corresponding to the given block states.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_from_numpy_int64_stacked<'py>(
        arr: Bound<'_, PyArray2<i64>>,
    ) -> PyResult<Vec<Py<Piece>>> {
        vec_from_numpy_stacked_impl::<i64>(&arr)
    }

    /// Create a vector of Piece instances from a flat NumPy ndarray of uint64 values.
    /// 
    /// Arguments:
    /// - arr (numpy.ndarray): A 1D NumPy array of uint64 values representing the block states.
    /// Returns:
    /// - list[Piece]: A list of Piece instances corresponding to the given block states.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_from_numpy_uint64_flat<'py>(
        arr: Bound<'_, PyArray1<u64>>,
    ) -> PyResult<Vec<Py<Piece>>> {
        vec_from_numpy_flat_impl::<u64>(&arr)
    }

    /// Create a vector of Piece instances from a stacked NumPy ndarray of uint64 values.
    /// 
    /// Arguments:
    /// - arr (numpy.ndarray): A 2D NumPy array of uint64 values with shape (num_pieces, 7)
    ///   representing the block states.
    /// Returns:
    /// - list[Piece]: A list of Piece instances corresponding to the given block states.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_from_numpy_uint64_stacked<'py>(
        arr: Bound<'_, PyArray2<u64>>,
    ) -> PyResult<Vec<Py<Piece>>> {
        vec_from_numpy_stacked_impl::<u64>(&arr)
    }

    /// Create a vector of Piece instances from a flat NumPy ndarray of float32 values.
    /// 
    /// Arguments:
    /// - arr (numpy.ndarray): A 1D NumPy array of float32 values representing the block states.
    /// Returns:
    /// - list[Piece]: A list of Piece instances corresponding to the given block states.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_from_numpy_float32_flat<'py>(
        arr: Bound<'_, PyArray1<f32>>,
    ) -> PyResult<Vec<Py<Piece>>> {
        vec_from_numpy_flat_impl::<f32>(&arr)
    }

    /// Create a vector of Piece instances from a stacked NumPy ndarray of float32 values.
    /// 
    /// Arguments:
    /// - arr (numpy.ndarray): A 2D NumPy array of float32 values with shape (num_pieces, 7)
    ///   representing the block states.
    /// Returns:
    /// - list[Piece]: A list of Piece instances corresponding to the given block states.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_from_numpy_float32_stacked<'py>(
        arr: Bound<'_, PyArray2<f32>>,
    ) -> PyResult<Vec<Py<Piece>>> {
        vec_from_numpy_stacked_impl::<f32>(&arr)
    }

    /// Create a vector of Piece instances from a flat NumPy ndarray of float64 values.
    /// 
    /// Arguments:
    /// - arr (numpy.ndarray): A 1D NumPy array of float64 values representing the block states.
    /// Returns:
    /// - list[Piece]: A list of Piece instances corresponding to the given block states.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_from_numpy_float64_flat<'py>(
        arr: Bound<'_, PyArray1<f64>>,
    ) -> PyResult<Vec<Py<Piece>>> {
        vec_from_numpy_flat_impl::<f64>(&arr)
    }

    /// Create a vector of Piece instances from a stacked NumPy ndarray of float64 values.
    /// 
    /// Arguments:
    /// - arr (numpy.ndarray): A 2D NumPy array of float64 values with shape (num_pieces, 7)
    ///   representing the block states.
    /// Returns:
    /// - list[Piece]: A list of Piece instances corresponding to the given block states.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn vec_from_numpy_float64_stacked<'py>(
        arr: Bound<'_, PyArray2<f64>>,
    ) -> PyResult<Vec<Py<Piece>>> {
        vec_from_numpy_stacked_impl::<f64>(&arr)
    }

    /// Create a vector of Piece instances from a flat NumPy ndarray of float16 values.
    /// 
    /// Arguments:
    /// - arr (numpy.ndarray): A 1D NumPy array of float16 values representing the block states.
    /// Returns:
    /// - list[Piece]: A list of Piece instances corresponding to the given block states.
    /// Warning:
    /// - The 'half' feature, which add support for float16, is still experimental and may not be stable. On machines that does
    /// not support float16 or installed with a version of numpy that does not support float16, this function may lead to
    /// undefined behavior or crashes. Testing show that on some systems, this can result in memory misinterpretation issues
    /// causing incorrect values to be read, and on other systems, it cause the entire program to halt but not crash.
    /// Use with caution.
    #[cfg(all(feature = "numpy", feature = "half"))]
    #[staticmethod]
    pub fn vec_from_numpy_float16_flat<'py>(
        arr: Bound<'_, PyArray1<F16>>,
    ) -> PyResult<Vec<Py<Piece>>> {
        vec_from_numpy_flat_impl::<F16>(&arr)
    }

    /// Create a vector of Piece instances from a stacked NumPy ndarray of float16 values.
    /// 
    /// Arguments:
    /// - arr (numpy.ndarray): A 2D NumPy array of float16 values with shape (num_pieces, 7)
    ///   representing the block states.
    /// Returns:
    /// - list[Piece]: A list of Piece instances corresponding to the given block states.
    /// Warning:
    /// - The 'half' feature, which add support for float16, is still experimental and may not be stable. On machines that does
    /// not support float16 or installed with a version of numpy that does not support float16, this function may lead to
    /// undefined behavior or crashes. Testing show that on some systems, this can result in memory misinterpretation issues
    /// causing incorrect values to be read, and on other systems, it cause the entire program to halt but not crash.
    /// Use with caution.
    #[cfg(all(feature = "numpy", feature = "half"))]
    #[staticmethod]
    pub fn vec_from_numpy_float16_stacked<'py>(
        arr: Bound<'_, PyArray2<F16>>,
    ) -> PyResult<Vec<Py<Piece>>> {
        vec_from_numpy_stacked_impl::<F16>(&arr)
    }

    /// Create a Piece instance from a NumPy ndarray of boolean values.
    /// 
    /// Arguments:
    /// - arr (numpy.ndarray): A 1D NumPy array of boolean values representing the block states.
    /// Returns:
    /// - Piece: A Piece instance corresponding to the given block states.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_bool(arr: Bound<'_, PyArray1<bool>>) -> PyResult<Py<Piece>> {
        from_numpy_piece_impl::<bool>(&arr)
    }

    /// Create a Piece instance from a NumPy ndarray of int8 values.
    ///
    /// Arguments:
    /// - arr (numpy.ndarray): A 1D NumPy array of int8 values representing the block states.
    /// Returns:
    /// - Piece: A Piece instance corresponding to the given block states (0 and negative are false, all others true).
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_int8(arr: Bound<'_, PyArray1<i8>>) -> PyResult<Py<Piece>> {
        from_numpy_piece_impl::<i8>(&arr)
    }

    /// Create a Piece instance from a NumPy ndarray of uint8 values.
    ///
    /// Arguments:
    /// - arr (numpy.ndarray): A 1D NumPy array of uint8 values representing the block states.
    /// Returns:
    /// - Piece: A Piece instance corresponding to the given block states (0 is false, all others true).
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_uint8(arr: Bound<'_, PyArray1<u8>>) -> PyResult<Py<Piece>> {
        from_numpy_piece_impl::<u8>(&arr)
    }

    /// Create a Piece instance from a NumPy ndarray of int16 values.
    ///
    /// Arguments:
    /// - arr (numpy.ndarray): A 1D NumPy array of int16 values representing the block states.
    /// Returns:
    /// - Piece: A Piece instance corresponding to the given block states (0 and negative are false, all others true).
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_int16(arr: Bound<'_, PyArray1<i16>>) -> PyResult<Py<Piece>> {
        from_numpy_piece_impl::<i16>(&arr)
    }

    /// Create a Piece instance from a NumPy ndarray of uint16 values.
    ///
    /// Arguments:
    /// - arr (numpy.ndarray): A 1D NumPy array of uint16 values representing the block states.
    /// Returns:
    /// - Piece: A Piece instance corresponding to the given block states (0 is false, all others true).
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_uint16(arr: Bound<'_, PyArray1<u16>>) -> PyResult<Py<Piece>> {
        from_numpy_piece_impl::<u16>(&arr)
    }

    /// Create a Piece instance from a NumPy ndarray of int32 values.
    ///
    /// Arguments:
    /// - arr (numpy.ndarray): A 1D NumPy array of int32 values representing the block states.
    /// Returns:
    /// - Piece: A Piece instance corresponding to the given block states (0 and negative are false, all others true).
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_int32(arr: Bound<'_, PyArray1<i32>>) -> PyResult<Py<Piece>> {
        from_numpy_piece_impl::<i32>(&arr)
    }

    /// Create a Piece instance from a NumPy ndarray of uint32 values.
    ///
    /// Arguments:
    /// - arr (numpy.ndarray): A 1D NumPy array of uint32 values representing the block states.
    /// Returns:
    /// - Piece: A Piece instance corresponding to the given block states (0 is false, all others true).
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_uint32(arr: Bound<'_, PyArray1<u32>>) -> PyResult<Py<Piece>> {
        from_numpy_piece_impl::<u32>(&arr)
    }

    /// Create a Piece instance from a NumPy ndarray of int64 values.
    ///
    /// Arguments:
    /// - arr (numpy.ndarray): A 1D NumPy array of int64 values representing the block states.
    ///
    /// Returns:
    /// - Piece: A Piece instance corresponding to the given block states (0 and negative are false, all others true).
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_int64(arr: Bound<'_, PyArray1<i64>>) -> PyResult<Py<Piece>> {
        from_numpy_piece_impl::<i64>(&arr)
    }

    /// Create a Piece instance from a NumPy ndarray of uint64 values.
    ///
    /// Arguments:
    /// - arr (numpy.ndarray): A 1D NumPy array of uint64 values representing the block states.
    /// Returns:
    /// - Piece: A Piece instance corresponding to the given block states (0 is false, all others true).
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_uint64(arr: Bound<'_, PyArray1<u64>>) -> PyResult<Py<Piece>> {
        from_numpy_piece_impl::<u64>(&arr)
    }

    /// Create a Piece instance from a NumPy ndarray of float16 values.
    ///
    /// Arguments:
    /// - arr (numpy.ndarray): A 1D NumPy array of float16 values representing the block states.
    /// Returns:
    /// - Piece: A Piece instance corresponding to the given block states (values > 0.0 are true, else false).
    /// Warning:
    /// - The 'half' feature, which add support for float16, is still experimental and may not be stable. On machines that does
    /// not support float16 or installed with a version of numpy that does not support float16, this function may lead to
    /// undefined behavior or crashes. Testing show that on some systems, this can result in memory misinterpretation issues
    /// causing incorrect values to be read, and on other systems, it cause a segmentation fault.
    #[cfg(all(feature = "numpy", feature = "half"))]
    #[staticmethod]
    pub fn from_numpy_float16(arr: Bound<'_, PyArray1<F16>>) -> PyResult<Py<Piece>> {
        from_numpy_piece_impl::<F16>(&arr)
    }

    /// Create a Piece instance from a NumPy ndarray of float32 values.
    ///
    /// Arguments:
    /// - arr (numpy.ndarray): A 1D NumPy array of float32 values representing the block states.
    /// Returns:
    /// - Piece: A Piece instance corresponding to the given block states (values > 0.0 are true, else false).
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_float32(arr: Bound<'_, PyArray1<f32>>) -> PyResult<Py<Piece>> {
        from_numpy_piece_impl::<f32>(&arr)
    }

    /// Create a Piece instance from a NumPy ndarray of float64 values.
    ///
    /// Arguments:
    /// - arr (numpy.ndarray): A 1D NumPy array of float64 values representing the block states.
    /// Returns:
    /// - Piece: A Piece instance corresponding to the given block states (values > 0.0 are true, else false).
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_float64(arr: Bound<'_, PyArray1<f64>>) -> PyResult<Py<Piece>> {
        from_numpy_piece_impl::<f64>(&arr)
    }

    /* ---------------------------------------- HPYHEX PYTHON API ---------------------------------------- */

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
    pub fn __iter__(&self) -> PyResult<PieceIterator> {
        Ok(PieceIterator {
            states: self.states(),
            index: 0,
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

fn piece_neighbors_of(p: Piece, target_i: i32, target_k: i32) -> usize {
    let mut count = 0;
    let idx = Piece::positions.iter().position(|h| h.i == target_i && h.k == target_k);
    if let Some(idx) = idx {
        if (p.state & (1 << (6 - idx))) != 0 {
            for (__i, pos) in Piece::positions.iter().enumerate() {
                let neighbor = Hex { i: pos.i + target_i, k: pos.k + target_k };
                if let Some(nidx) = Piece::positions.iter().position(|h| h == &neighbor) {
                    if (p.state & (1 << (6 - nidx))) != 0 {
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

#[pyclass]
/// An iterator over Piece, returning states as booleans.
pub struct PieceIterator {
    states: [bool; 7],
    index: usize,
}

#[pymethods]
impl PieceIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<bool> {
        if slf.index < 7 {
            let result = slf.states[slf.index];
            slf.index += 1;
            Some(result)
        } else {
            None
        }
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
/// 
/// Hpyhex-rs Serialization:
/// The class is inter-operable with the Rust hpyhex-rs crate through the following methods:
/// - hpyhex_rs_serialize(): Serialize the Piece to a byte vector in the format used by the Rust hpyhex-rs crate.
/// - hpyhex_rs_deserialize(data): Deserialize a Piece from a byte vector in the format used by the Rust hpyhex-rs crate.
/// 
/// Numpy Support (Requires "numpy" feature):
/// 
/// Offer methods to convert the Piece's block states to NumPy ndarray representations and vice versa.
/// These include safe conversion to and from, unchecked conversion, and even zero-copy views where applicable.
///
/// Support the following NumPy array types:
/// - bool
/// - int8
/// - uint8
/// - int16
/// - uint16
/// - int32
/// - uint32
/// - int64
/// - uint64
/// - half (f16) [Requires "half" feature, experimental]
/// - float32
/// - float64
/// 
/// The from_numpy_* methods will validate the input array shape and types, and raise a ValueError if the
/// input is invalid. The from_numpy_*_unchecked methods will skip validation for performance, but may lead to
/// undefined behavior if the input length does not correspond to a valid hexagonal grid size. The to_numpy_*
/// methods will return a new NumPy ndarray representing the block states of the HexEngine.
/// 
/// to_numpy() defaults to bool representation, but there are no from_numpy that can take in different types,
/// because numpy ndarrays cannot be easily casted into each other.
/// 
/// The from_numpy_raw_view and to_numpy_raw_view methods, which are *extremely unsafe*, provide zero-copy views
/// into the internal state vector as a NumPy ndarray of the specified type. These methods should
/// be used with caution, as they can lead to undefined behavior if the internal state vector is modified
/// while the view is still in use, or if the type does not match the expected representation. They may not work
/// correctly on all platforms due to differences in memory alignment and representation, but should work correctly
/// on most common platforms with correct installation of Python and NumPy and carefully managed memory usage.
/// 
/// In addition to states, placement positions can also be converted to NumPy ndarrays
/// using the to_numpy_positions_mask methods, which support all the same types as the state conversions.
/// These methods return a NumPy ndarray where each element corresponds to a block in the hexagonal grid,
/// with True (or one) indicating an occupied block and False (or zero) indicating an empty block.
pub struct HexEngine {
    radius: usize,
    states: Vec<bool>,
}

impl PartialEq for HexEngine {
    fn eq(&self, other: &Self) -> bool {
        self.radius == other.radius && self.states == other.states
    }
}

impl std::fmt::Display for HexEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HexEngine[blocks = {{")?;
        for (i, &state) in self.states.iter().enumerate() {
            let hex = self.hex_coordinate_of(i).unwrap_or(Hex { i: -1, k: -1 });
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "({}, {}, {})", hex.i, hex.k, state)?;
        }
        write!(f, "}}]")
    }
}

impl std::fmt::Debug for HexEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HexEngine[blocks = {{")?;
        for (i, &state) in self.states.iter().enumerate() {
            let hex = self.hex_coordinate_of(i).unwrap_or(Hex { i: -1, k: -1 });
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "({}, {}, {})", hex.i, hex.k, state)?;
        }
        write!(f, "}}]")
    }
}

impl TryFrom<Vec<bool>> for HexEngine {
    type Error = PyErr;

    fn try_from(value: Vec<bool>) -> Result<Self, Self::Error> {
        let radius = HexEngine::calc_radius(value.len()).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Invalid state length: {}", value.len())
            )
        })?;
        Ok(HexEngine {
            radius,
            states: value
        })
    }
}

impl TryFrom<usize> for HexEngine {
    type Error = PyErr;

    fn try_from(radius: usize) -> Result<Self, Self::Error> {
        let value = if radius == 0 {
            0
        } else {
            1 + 3 * radius * (radius - 1)
        };
        Ok(HexEngine {
            radius,
            states: vec![false; value]
        })
    }
}

// This is the backend scope, nothing is exposed to Python here
impl HexEngine {
    /// Create a HexEngine instance from a raw state vector without validation.
    /// 
    /// This unsafe method assumes that the provided state vector length corresponds to a valid hexagonal grid
    /// size and does not perform any checks. It calculates the radius based on the length of the state vector.
    /// 
    /// If the length of the state vector does not correspond to a valid hexagonal grid size or is zero,
    /// the behavior is undefined and may cause runtime errors or panics down the line.
    /// 
    /// Arguments:
    /// - value: A vector of booleans representing the occupancy state of each block in the hexagonal grid.
    /// Returns:
    /// - HexEngine: A HexEngine instance initialized with the given state vector.
    pub unsafe fn from_raw_state(value: Vec<bool>) -> Self {
        let t = (value.len() - 1) / 3;
        let radius = (((t * 4 + 1) as f64).sqrt() as usize + 1) >> 1;
        HexEngine {
            radius,
            states: value
        }
    }
    /// Calculate the radius of the hexagonal grid from the length of the state vector.
    /// 
    /// This method derives the radius based on the formula for the total number of blocks
    /// in a hexagonal grid of radius r: A_r = 1 + 3*r*(r-1).
    /// 
    /// Arguments:
    /// - length: The length of the state vector
    /// Returns:
    /// - An Option containing the radius if valid, or None if the length does not correspond
    ///   to a valid hexagonal grid size.
    fn calc_radius(length: usize) -> Option<usize> {
        if length == 0 {
            return Some(0);
        }
        if length % 3 != 1 {
            return None;
        }
        
        let target = (length - 1) / 3;
        let u = target * 4 + 1;
        let r = ((u as f64).sqrt() as usize + 1) / 2;
        
        if r > 0 && r * (r - 1) == target {
            Some(r)
        } else {
            None
        }
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
    /// Converts linear index to coordinate for a static radius.
    /// 
    /// This method provides efficient conversion from a linear index in the internal state vector to a `Hex` coordinate,
    /// assuming a static radius is provided.
    /// 
    /// Arguments:
    /// - `radius`: The radius of the hexagonal grid.
    /// - `index`: The linear index to convert
    /// Returns:
    /// - A result containing the corresponding `Hex` coordinate, or an IndexError if the index is out of bounds.
    fn static_hex_coordinate_of(radius: usize, mut index: usize) -> PyResult<Hex> {
        let r = radius as i32;
        let total_blocks = if radius == 0 {
            0
        } else {
            1 + 3 * radius * (radius - 1)
        };
        if index >= total_blocks {
            return Err(pyo3::exceptions::PyIndexError::new_err("Index out of bounds"));
        }

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
    /// Converts coordinate to linear index for a static radius.
    /// 
    /// This method provides efficient conversion from a `Hex` coordinate to a linear index in the internal state vector,
    /// assuming a static radius is provided.
    /// 
    /// Arguments:
    /// - `radius`: The radius of the hexagonal grid.
    /// - `i`: The I-line coordinate
    /// - `k`: The K-line coordinate
    /// Returns:
    /// - A result containing the corresponding linear index, or -1 if the coordinate is out of range.
    fn linear_index_of_static(radius: usize, i: i32, k: i32) -> PyResult<isize> {
        let r = radius as i32;
        if Self::check_range_coords(i, k, radius)? {
            if i < r {
                Ok((k + i * r + i * (i - 1) / 2) as isize)
            } else {
                Ok((k - (r - 1).pow(2) + i * r * 3 - i * (i + 5) / 2) as isize)
            }
        } else {
            Ok(-1)
        }
    }
    /// Generate the adjacency list for blocks in a hexagonal grid of the specified radius.
    /// 
    /// Each block is connected to its six neighboring blocks, if they exist within the grid.
    /// 
    /// Arguments:
    /// - radius: The radius of the hexagonal grid.
    /// Returns:
    /// - A vector of vectors, where each inner vector contains the indices of neighboring blocks for
    ///   the corresponding block.
    fn adjacency_list_static(radius: usize) -> Vec<Vec<usize>> {
        let total_blocks = if radius == 0 {
            return Vec::new();
        } else {
            1 + 3 * radius * (radius - 1)
        };
        let mut adjacency_list: Vec<Vec<usize>> = vec![Vec::new(); total_blocks];
        for index in 0..total_blocks {
            if let Ok(hex) = Self::static_hex_coordinate_of(radius, index) {
                for pos in &Piece::positions {
                    let neighbor_i = hex.i + pos.i;
                    let neighbor_k = hex.k + pos.k;
                    if let Ok(neighbor_index) = Self::linear_index_of_static(radius, neighbor_i, neighbor_k) {
                        if neighbor_index != -1 && neighbor_index as usize != index {
                            adjacency_list[index].push(neighbor_index as usize);
                        }
                    }
                }
            }
        }
        adjacency_list
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
    #[allow(unused)]
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
    /// Check if a Piece can be added at the given Hex coordinate without overlapping existing occupied blocks.
    /// 
    /// Arguments:
    /// - coo: Hex coordinate where the Piece is to be added.
    /// - piece: The Piece to be added.
    /// Returns:
    /// - bool: True if the Piece can be added without overlap, False otherwise.
    fn check_add_of(&self, coo: &Hex, piece: &Piece) -> PyResult<bool> {
        for (i, state) in piece.states().iter().enumerate() {
            let pos = &Piece::positions[i];
            let target_i = pos.i + coo.i;
            let target_k = pos.k + coo.k;
            if *state {
                if !Self::check_range_coords(target_i, target_k, self.radius)? {
                    return Ok(false);
                }
                let index = self.linear_index_of(target_i, target_k)?;
                if index == -1 || self.states[index as usize] {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Check if there exists at least one position in the HexEngine where the given Piece can be added without overlap
    /// or going out of bounds.
    /// 
    /// Arguments:
    /// - piece: The Piece to be checked for possible addition.
    /// Returns:
    /// - bool: True if there is at least one valid position for the Piece, False otherwise.
    fn check_has_positions(&self, piece: &Piece) -> bool {
        for a in 0..(self.radius * 2) as i32 {
            for b in 0..(self.radius * 2) as i32 {
                let hex = Hex{i: a, k: b};
                if self.check_add_of(&hex, &piece).unwrap_or(false) {
                    return true;
                }
            }
        }
        false
    }

    #[cfg(feature = "numpy")]
    fn from_numpy_engine_impl<T>(array: Bound<'_, PyArray1<T>>)
        -> PyResult<HexEngine>
    where
        T: BitScalar + Copy + numpy::Element,
    {
        let slice = unsafe { array.as_slice()? };
        let vec: Vec<bool> = slice.iter().map(|&b| T::predicate(b)).collect::<Vec<bool>>();
        Self::try_from(vec)
    }

    #[cfg(feature = "numpy")]
    unsafe fn from_numpy_engine_unchecked_impl<T>(array: Bound<'_, PyArray1<T>>) -> Self
    where
        T: BitScalar + Copy + numpy::Element,
    {
        let slice = unsafe { array.as_slice().unwrap() };
        let vec: Vec<bool> = slice.iter().map(|&b| T::predicate(b)).collect::<Vec<bool>>();
        unsafe { HexEngine::from_raw_state(vec) }
    }

    #[cfg(feature = "numpy")]
    fn to_numpy_engine_impl<T>(&self, py: Python) -> Py<PyArray1<T>>
    where
        T: BitScalar + Copy + numpy::Element,
    {
        let int_states: Vec<T> = self.states.iter().map(|&b| if b { T::one() } else { T::zero() }).collect();
        PyArray1::from_vec_bound(py, int_states).into()
    }

    #[allow(deprecated)]
    #[cfg(feature = "numpy")]
    fn to_numpy_engine_unboxed_impl<'py, T>(&self, py: Python<'py>) -> &'py PyArray1<T>
    where
        T: BitScalar + Copy + numpy::Element,
    {
        let int_states: Vec<T> = self.states.iter().map(|&b| if b { T::one() } else { T::zero() }).collect();
        &PyArray1::from_vec(py, int_states)
    }

    #[cfg(feature = "numpy")]
    fn from_numpy_engine_unboxed_explicit_radius_impl<T>(
        slice: &[T],
        radius: usize,
    ) -> PyResult<HexEngine>
    where
        T: BitScalar + Copy + numpy::Element,
    {
        let vec: Vec<bool> = slice.iter().map(|&b| T::predicate(b)).collect::<Vec<bool>>();
        let expected_len = if radius == 0 {
            0
        } else {
            1 + 3 * radius * (radius - 1)
        };
        if vec.len() != expected_len {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Input array length {} does not match expected length {} for radius {}",
                vec.len(),
                expected_len,
                radius
            )));
        }
        Ok(HexEngine {
            radius,
            states: vec,
        })
    }

    #[cfg(feature = "numpy")]
    fn to_numpy_positions_mask_impl<T>(&self, py: Python, piece: Piece) -> Py<PyArray1<T>>
    where
        T: BitScalar + Copy + numpy::Element,
    {
        // run check_add_of(hex_coordinate_of(i)) for all i in self.states
        let mut mask: Vec<T> = Vec::with_capacity(self.states.len());
        for i in 0..self.states.len() {
            let hex = self.hex_coordinate_of(i).unwrap(); // safe because i is in range
            let can_add = self.check_add_of(&hex, &piece).unwrap_or(false);
            if can_add {
                mask.push(T::one());
            } else {
                mask.push(T::zero());
            }
        }
        PyArray1::from_vec_bound(py, mask).into()
    }

    #[cfg(feature = "numpy")]
    fn to_numpy_adjacency_matrix_impl<T>(py: Python, radius: usize) -> Py<PyArray2<T>>
    where
        T: BitScalar + Copy + numpy::Element,
    {
        use ndarray::Array2;
        let n = if radius == 0 {
            0
        } else {
            1 + 3 * radius * (radius - 1)
        };
        let shape = (n, n);
        let mut vec: Vec<T> = vec![T::zero(); n * n];
        let adjacency_list = Self::adjacency_list_static(radius);
        for (i, neighbors) in adjacency_list.iter().enumerate() {
            for &j in neighbors {
                vec[i * n + j] = T::one();
            }
        }
        let array = Array2::from_shape_vec(shape, vec).unwrap();
        PyArray2::from_owned_array_bound(py, array).unbind()
    }

    #[cfg(feature = "numpy")]
    fn to_numpy_adjacency_list_impl<T>(py: Python, radius: usize) -> Py<PyArray2<T>>
    where
        T: SizeScalar + Copy + numpy::Element,
    {
        use ndarray::{Array2};
        let n = if radius == 0 {
            0
        } else {
            1 + 3 * radius * (radius - 1)
        };
        let positions_exclude_self = Piece::positions
            .iter()
            .filter(|&pos| !(pos.i == 0 && pos.k == 0))
            .collect::<Vec<&Hex>>();
        let shape = (n, 6);
        let mut vec: Vec<T> = vec![T::sentinel(); n * 6];
        for index in 0..n {
            if let Ok(hex) = Self::static_hex_coordinate_of(radius, index) {
                for (i, pos) in positions_exclude_self.iter().enumerate() {
                    let neighbor_i = hex.i + pos.i;
                    let neighbor_k = hex.k + pos.k;
                    if let Ok(neighbor_index) = Self::linear_index_of_static(radius, neighbor_i, neighbor_k) {
                        if neighbor_index != -1 && neighbor_index as usize != index {
                            vec[index * 6 + i] = T::from_usize(neighbor_index as usize);
                        }
                    }
                }
            }
        }
        let array = Array2::from_shape_vec(shape, vec).unwrap();
        PyArray2::from_owned_array_bound(py, array).unbind()
    }
}

impl Into<Vec<u8>> for &HexEngine {
    fn into(self) -> Vec<u8> {
        let mut bytes = Vec::new();
        let radius_u16 = self.radius as u32;
        bytes.extend_from_slice(&radius_u16.to_le_bytes());
        
        let mut byte: u8 = 0;
        for (i, &state) in self.states.iter().enumerate() {
            if state {
                byte |= 1 << (i % 8);
            }
            if i % 8 == 7 {
                bytes.push(byte);
                byte = 0;
            }
        }
        if self.states.len() % 8 != 0 {
            bytes.push(byte);
        }
        
        bytes
    }
}

impl TryFrom<&[u8]> for HexEngine {
    type Error = PyErr;
    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        if bytes.len() < 4 {
            return Err(pyo3::exceptions::PyValueError::new_err("Byte vector too short to contain radius"));
        }
        let radius = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let total_blocks = if radius == 0 {
            0
        } else {
            1 + 3 * radius * (radius - 1)
        };
        
        let mut states = Vec::with_capacity(total_blocks);
        for i in 0..total_blocks {
            let byte_index = 4 + (i / 8);
            let bit_index = i % 8;
            if byte_index < bytes.len() {
                let state = (bytes[byte_index] & (1 << bit_index)) != 0;
                states.push(state);
            } else {
                states.push(false);
            }
        }
        
        Ok(HexEngine {
            radius,
            states,
        })
    }
}

#[pymethods]
impl HexEngine {
    /* ------------------------------------- HPYHEX-RS ------------------------------------- */

    /// Serialize the HexEngine state into a byte vector according the format used by the Rust hpyhex-rs crate.
    /// 
    /// The serialization format is as follows:
    /// - The first four bytes represent the radius of the hexagonal grid as a little-endian u16.
    /// - The subsequent bytes represent the occupancy states of the blocks in the grid, packed into bits.
    ///   Each byte contains the states of up to 8 blocks, with the least significant bit corresponding to the first block.
    /// 
    /// Returns:
    /// - bytes: A byte vector containing the serialized state of the HexEngine.
    pub fn hpyhex_rs_serialize<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
        let bytes: Vec<u8> = self.into();
        
        pyo3::types::PyBytes::new_bound(py, &bytes)
    }

    /// Deserialize a byte vector into a HexEngine instance according to the format used by the Rust hpyhex-rs crate.
    /// 
    /// The deserialization format is as follows:
    /// - The first four bytes represent the radius of the hexagonal grid as a little-endian u16.
    /// - The subsequent bytes represent the occupancy states of the blocks in the grid, packed into bits.
    ///   Each byte contains the states of up to 8 blocks, with the least significant bit corresponding to the first block.
    /// 
    /// Arguments:
    /// - data: A byte vector containing the serialized state of the HexEngine.
    /// Returns:
    /// - HexEngine: A HexEngine instance reconstructed from the byte vector.
    #[staticmethod]
    pub fn hpyhex_rs_deserialize(data: Bound<'_, PyAny>) -> PyResult<Self> {
        use pyo3::types::{PyBytes, PyByteArray};
        // Extract PyBytes, PyByteArray, or Vec<u8>
        let bytes: Vec<u8> = if let Ok(py_bytes) = data.downcast::<PyBytes>() {
            py_bytes.as_bytes().to_vec()
        } else if let Ok(py_bytearray) = data.downcast::<PyByteArray>() {
            unsafe { py_bytearray.as_bytes() }.to_vec()
        } else if let Ok(vec_u8) = data.extract::<Vec<u8>>() {
            vec_u8
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err("Input must be bytes, bytearray, or list of integers"));
        };
        if bytes.len() < 4 {
            return Err(pyo3::exceptions::PyValueError::new_err("Byte vector too short to contain radius"));
        }
        
        HexEngine::try_from(bytes.as_slice())
    }

    /// Get the index of the Block at the specified Hex coordinate for a static radius.
    /// 
    /// This static method allows for index retrieval without needing an instance of HexEngine.
    /// It calculates the index based on the provided radius and Hex coordinates.
    /// 
    /// This method exists because the original API does not expose the index calculation from radius directly,
    /// requiring an instance of HexEngine to be created solely to retrieve it. This static method simplifies
    /// that process and may improve performance by avoiding unnecessary instance creation.
    /// 
    /// Arguments:
    /// - radius (int): The radius of the hexagonal grid.
    /// - coo: The Hex coordinate.
    /// Returns:
    /// - int: The index of the Block, or -1 if out of range.
    #[staticmethod]
    pub fn hpyhex_rs_index_block(radius: usize, coo: &pyo3::Bound<'_, PyAny>) -> isize {
        let (i, k) = if let Ok(hex) = coo.extract::<PyRef<Hex>>() {
            (hex.i, hex.k)
        } else if let Ok(tuple) = coo.extract::<(i32, i32)>() {
            (tuple.0, tuple.1)
        } else if let Ok(tuple3) = coo.extract::<(i32, i32, i32)>() {
            (tuple3.0, tuple3.2)
        } else {
            return -1;
        };
        HexEngine::linear_index_of_static(radius, i, k).unwrap_or(-1)
    }

    /// Get the Hex coordinate of the Block at the specified index for a static radius.
    /// 
    /// This static method allows for coordinate retrieval without needing an instance of HexEngine.
    /// It calculates the Hex coordinate based on the provided radius and index.
    /// 
    /// This method exists because the original API does not expose the coordinate calculation from radius directly,
    /// requiring an instance of HexEngine to be created solely to retrieve it. This static method simplifies
    /// that process and may improve performance by avoiding unnecessary instance creation.
    /// 
    /// Arguments:
    /// - radius (int): The radius of the hexagonal grid.
    /// - index (int): The linear index of the Block.
    /// Returns:
    /// - Hex: The Hex coordinate of the Block.
    #[staticmethod]
    pub fn hpyhex_rs_coordinate_block(radius: usize, index: usize) -> PyResult<Hex> {
        HexEngine::static_hex_coordinate_of(radius, index)
    }

    /// Generate the adjacency list for blocks in a hexagonal grid of the specified radius.
    /// 
    /// Each block is connected to its six neighboring blocks, if they exist within the grid.
    /// 
    /// This static method allows for adjacency list generation without needing an instance of HexEngine.
    /// It calculates the adjacency list based on the provided radius.
    /// 
    /// This method exists because the original API does not expose the adjacency list directly.
    /// As a result, batch operations require creating a `HexEngine` instance and repeatedly
    /// querying neighbors for each block, which is both cumbersome and inefficient.
    ///
    /// This static method provides direct access to the adjacency list, simplifying batch
    /// workflows and potentially improving performance by avoiding unnecessary engine
    /// instantiation and repeated neighbor lookups.
    ///
    /// Because the method is static, the resulting adjacency list can be reused across
    /// multiple `HexEngine` instances with the same radius, eliminating redundant
    /// calculations. In real applications, this can save thousands of hex-coordinate
    /// computations and index lookups.
    /// 
    /// Arguments:
    /// - radius (int): The radius of the hexagonal grid.
    /// Returns:
    /// - List[List[int]]: A list of lists, where each inner list contains the indices of neighboring blocks for
    ///   the corresponding block.
    #[staticmethod]
    pub fn hpyhex_rs_adjacency_list(radius: usize) -> Vec<Vec<usize>> {
        HexEngine::adjacency_list_static(radius)
    }

    /* ---------------------------------------- NUMPY ---------------------------------------- */

    /// Get a NumPy adjacency list as a 2D ndarray.
    /// 
    /// The adjacency list represents the connections between blocks in a hexagonal grid of the specified radius.
    /// Each row corresponds to a block, and contains up to six indices of adjacent blocks.
    /// If a block has fewer than six neighbors (e.g., edge blocks), the remaining entries are filled with a sentinel value.
    /// 
    /// The sentinel value used here is -1 for int64 representation.
    /// 
    /// Arguments:
    /// - radius: The radius of the hexagonal grid.
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array representing the adjacency list of the hexagonal grid of the given radius.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn to_numpy_adjacency_list(py: Python, radius: usize) -> Py<PyArray2<i64>> {
        Self::to_numpy_adjacency_list_int64(py, radius)
    }

    /// Get a NumPy adjacency list as a 2D ndarray of uint16.
    /// 
    /// The adjacency list represents the connections between blocks in a hexagonal grid of the specified radius.
    /// Each row corresponds to a block, and contains up to six indices of adjacent blocks.
    /// If a block has fewer than six neighbors (e.g., edge blocks), the remaining entries are filled with a sentinel value.
    /// 
    /// The sentinel value used here is u16::MAX for uint16 representation.
    /// - Python: numpy.iinfo(numpy.uint16).max.
    /// - C: UINT16_MAX of <stdint.h>.
    /// - C++: std::numeric_limits<uint16_t>::max().
    /// - Rust: std::u16::MAX.
    /// 
    /// Arguments:
    /// - radius: The radius of the hexagonal grid.
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array representing the adjacency list of the hexagonal grid of the given radius.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn to_numpy_adjacency_list_uint16(py: Python, radius: usize) -> Py<PyArray2<u16>> {
        Self::to_numpy_adjacency_list_impl::<u16>(py, radius)
    }

    /// Get a NumPy adjacency list as a 2D ndarray of uint32.
    /// 
    /// The adjacency list represents the connections between blocks in a hexagonal grid of the specified radius.
    /// Each row corresponds to a block, and contains up to six indices of adjacent blocks.
    /// If a block has fewer than six neighbors (e.g., edge blocks), the remaining entries are filled with a sentinel value.
    /// 
    /// The sentinel value used here is u32::MAX for uint32 representation.
    /// - Python: numpy.iinfo(numpy.uint32).max.
    /// - C: UINT32_MAX of <stdint.h>.
    /// - C++: std::numeric_limits<uint32_t>::max().
    /// - Rust: std::u32::MAX.
    /// 
    /// Arguments:
    /// - radius: The radius of the hexagonal grid.
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array representing the adjacency list of the hexagonal grid of the given radius.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn to_numpy_adjacency_list_uint32(py: Python, radius: usize) -> Py<PyArray2<u32>> {
        Self::to_numpy_adjacency_list_impl::<u32>(py, radius)
    }

    /// Get a NumPy adjacency list as a 2D ndarray of uint64.
    /// 
    /// The adjacency list represents the connections between blocks in a hexagonal grid of the specified radius.
    /// Each row corresponds to a block, and contains up to six indices of adjacent blocks.
    /// If a block has fewer than six neighbors (e.g., edge blocks), the remaining entries are filled with a sentinel value.
    /// 
    /// The sentinel value used here is u64::MAX for uint64 representation.
    /// - Python: numpy.iinfo(numpy.uint64).max.
    /// - C: UINT64_MAX of <stdint.h>.
    /// - C++: std::numeric_limits<uint64_t>::max().
    /// - Rust: std::u64::MAX.
    /// 
    /// Arguments:
    /// - radius: The radius of the hexagonal grid.
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array representing the adjacency list of the hexagonal grid of the given radius.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn to_numpy_adjacency_list_uint64(py: Python, radius: usize) -> Py<PyArray2<u64>> {
        Self::to_numpy_adjacency_list_impl::<u64>(py, radius)
    }

    /// Get a NumPy adjacency list as a 2D ndarray of int16.
    /// 
    /// The adjacency list represents the connections between blocks in a hexagonal grid of the specified radius.
    /// Each row corresponds to a block, and contains up to six indices of adjacent blocks.
    /// If a block has fewer than six neighbors (e.g., edge blocks), the remaining entries are filled with a sentinel value.
    /// 
    /// The sentinel value used here is -1 for int16 representation.
    /// 
    /// Arguments:
    /// - radius: The radius of the hexagonal grid.
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array representing the adjacency list of the hexagonal grid of the given radius.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn to_numpy_adjacency_list_int16(py: Python, radius: usize) -> Py<PyArray2<i16>> {
        Self::to_numpy_adjacency_list_impl::<i16>(py, radius)
    }

    /// Get a NumPy adjacency list as a 2D ndarray of int32.
    /// 
    /// The adjacency list represents the connections between blocks in a hexagonal grid of the specified radius.
    /// Each row corresponds to a block, and contains up to six indices of adjacent blocks.
    /// If a block has fewer than six neighbors (e.g., edge blocks), the remaining entries are filled with a sentinel value.
    /// 
    /// The sentinel value used here is -1 for int32 representation.
    /// 
    /// Arguments:
    /// - radius: The radius of the hexagonal grid.
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array representing the adjacency list of the hexagonal grid of the given radius.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn to_numpy_adjacency_list_int32(py: Python, radius: usize) -> Py<PyArray2<i32>> {
        Self::to_numpy_adjacency_list_impl::<i32>(py, radius)
    }

    /// Get a NumPy adjacency list as a 2D ndarray of int64.
    /// 
    /// The adjacency list represents the connections between blocks in a hexagonal grid of the specified radius.
    /// Each row corresponds to a block, and contains up to six indices of adjacent blocks.
    /// If a block has fewer than six neighbors (e.g., edge blocks), the remaining entries are filled with a sentinel value.
    /// 
    /// The sentinel value used here is -1 for int64 representation.
    /// 
    /// Arguments:
    /// - radius: The radius of the hexagonal grid.
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array representing the adjacency list of the hexagonal grid of the given radius.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn to_numpy_adjacency_list_int64(py: Python, radius: usize) -> Py<PyArray2<i64>> {
        Self::to_numpy_adjacency_list_impl::<i64>(py, radius)
    }

    /// Get a NumPy adjacency matrix as a 2D ndarray.
    /// 
    /// The adjacency matrix represents the connections between blocks in a hexagonal grid of the specified radius.
    /// Each entry (i, j) in the matrix is True if block i is adjacent to block j, and False otherwise.
    /// 
    /// The adjacency matrix is a 2D array representation of the connections (edges) between nodes (blocks) in a graph.
    /// In the context of a hexagonal grid, each block is a node, and an entry at (i, j) is true if block i is adjacent to block j.
    /// This representation is useful for applying graph algorithms for postprocessing, or applying convolution-like operations
    /// over the grid structure.
    /// 
    /// Arguments:
    /// - radius: The radius of the hexagonal grid.
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array representing the adjacency matrix of the hexagonal grid of the given radius.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn to_numpy_adjacency_matrix(py: Python, radius: usize) -> Py<PyArray2<bool>> {
        Self::to_numpy_adjacency_matrix_impl::<bool>(py, radius)
    }

    /// Get a NumPy adjacency matrix as a 2D ndarray of bools.
    /// 
    /// The adjacency matrix represents the connections between blocks in a hexagonal grid of the specified radius.
    /// Each entry (i, j) in the matrix is True if block i is adjacent to block j, and False otherwise.
    /// 
    /// The adjacency matrix is a 2D array representation of the connections (edges) between nodes (blocks) in a graph.
    /// In the context of a hexagonal grid, each block is a node, and an entry at (i, j) is true if block i is adjacent to block j.
    /// This representation is useful for applying graph algorithms for postprocessing, or applying convolution-like operations
    /// over the grid structure.
    /// 
    /// Arguments:
    /// - radius: The radius of the hexagonal grid.
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array representing the adjacency matrix of the hexagonal grid of the given radius.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn to_numpy_adjacency_matrix_bool(py: Python, radius: usize) -> Py<PyArray2<bool>> {
        Self::to_numpy_adjacency_matrix_impl::<bool>(py, radius)
    }

    /// Get a NumPy adjacency matrix as a 2D ndarray of int8.
    /// 
    /// The adjacency matrix represents the connections between blocks in a hexagonal grid of the specified radius.
    /// Each entry (i, j) in the matrix is 1 if block i is adjacent to block j, and 0 otherwise.
    /// 
    /// The adjacency matrix is a 2D array representation of the connections (edges) between nodes (blocks) in a graph.
    /// In the context of a hexagonal grid, each block is a node, and an entry at (i, j) is 1 if block i is adjacent to block j.
    /// This representation is useful for applying graph algorithms for postprocessing, or applying convolution-like operations
    /// over the grid structure.
    /// 
    /// Arguments:
    /// - radius: The radius of the hexagonal grid.
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array representing the adjacency matrix of the hexagonal grid of the given radius.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn to_numpy_adjacency_matrix_int8(py: Python, radius: usize) -> Py<PyArray2<i8>> {
        Self::to_numpy_adjacency_matrix_impl::<i8>(py, radius)
    }

    /// Get a NumPy adjacency matrix as a 2D ndarray of uint8.
    /// 
    /// The adjacency matrix represents the connections between blocks in a hexagonal grid of the specified radius.
    /// Each entry (i, j) in the matrix is 1 if block i is adjacent to block j, and 0 otherwise.
    /// 
    /// The adjacency matrix is a 2D array representation of the connections (edges) between nodes (blocks) in a graph.
    /// In the context of a hexagonal grid, each block is a node, and an entry at (i, j) is 1 if block i is adjacent to block j.
    /// This representation is useful for applying graph algorithms for postprocessing, or applying convolution-like operations
    /// over the grid structure.
    /// 
    /// Arguments:
    /// - radius: The radius of the hexagonal grid.
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array representing the adjacency matrix of the hexagonal grid of the given radius.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn to_numpy_adjacency_matrix_uint8(py: Python, radius: usize) -> Py<PyArray2<u8>> {
        Self::to_numpy_adjacency_matrix_impl::<u8>(py, radius)
    }

    /// Get a NumPy adjacency matrix as a 2D ndarray of int16.
    /// 
    /// The adjacency matrix represents the connections between blocks in a hexagonal grid of the specified radius.
    /// Each entry (i, j) in the matrix is 1 if block i is adjacent to block j, and 0 otherwise.
    /// 
    /// The adjacency matrix is a 2D array representation of the connections (edges) between nodes (blocks) in a graph.
    /// In the context of a hexagonal grid, each block is a node, and an entry at (i, j) is 1 if block i is adjacent to block j.
    /// This representation is useful for applying graph algorithms for postprocessing, or applying convolution-like operations
    /// over the grid structure.
    /// 
    /// Arguments:
    /// - radius: The radius of the hexagonal grid.
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array representing the adjacency matrix of the hexagonal grid of the given radius.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn to_numpy_adjacency_matrix_int16(py: Python, radius: usize) -> Py<PyArray2<i16>> {
        Self::to_numpy_adjacency_matrix_impl::<i16>(py, radius)
    }

    /// Get a NumPy adjacency matrix as a 2D ndarray of uint16.
    /// 
    /// The adjacency matrix represents the connections between blocks in a hexagonal grid of the specified radius.
    /// Each entry (i, j) in the matrix is 1 if block i is adjacent to block j, and 0 otherwise.
    /// 
    /// The adjacency matrix is a 2D array representation of the connections (edges) between nodes (blocks) in a graph.
    /// In the context of a hexagonal grid, each block is a node, and an entry at (i, j) is 1 if block i is adjacent to block j.
    /// This representation is useful for applying graph algorithms for postprocessing, or applying convolution-like operations
    /// over the grid structure.
    /// 
    /// Arguments:
    /// - radius: The radius of the hexagonal grid.
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array representing the adjacency matrix of the hexagonal grid of the given radius.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn to_numpy_adjacency_matrix_uint16(py: Python, radius: usize) -> Py<PyArray2<u16>> {
        Self::to_numpy_adjacency_matrix_impl::<u16>(py, radius)
    }

    /// Get a NumPy adjacency matrix as a 2D ndarray of int32.
    /// 
    /// The adjacency matrix represents the connections between blocks in a hexagonal grid of the specified radius.
    /// Each entry (i, j) in the matrix is 1 if block i is adjacent to block j, and 0 otherwise.
    /// 
    /// The adjacency matrix is a 2D array representation of the connections (edges) between nodes (blocks) in a graph.
    /// In the context of a hexagonal grid, each block is a node, and an entry at (i, j) is 1 if block i is adjacent to block j.
    /// This representation is useful for applying graph algorithms for postprocessing, or applying convolution-like operations
    /// over the grid structure.
    /// 
    /// Arguments:
    /// - radius: The radius of the hexagonal grid.
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array representing the adjacency matrix of the hexagonal grid of the given radius.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn to_numpy_adjacency_matrix_int32(py: Python, radius: usize) -> Py<PyArray2<i32>> {
        Self::to_numpy_adjacency_matrix_impl::<i32>(py, radius)
    }

    /// Get a NumPy adjacency matrix as a 2D ndarray of uint32.
    /// 
    /// The adjacency matrix represents the connections between blocks in a hexagonal grid of the specified radius.
    /// Each entry (i, j) in the matrix is 1 if block i is adjacent to block j, and 0 otherwise.
    /// 
    /// The adjacency matrix is a 2D array representation of the connections (edges) between nodes (blocks) in a graph.
    /// In the context of a hexagonal grid, each block is a node, and an entry at (i, j) is 1 if block i is adjacent to block j.
    /// This representation is useful for applying graph algorithms for postprocessing, or applying convolution-like operations
    /// over the grid structure.
    /// 
    /// Arguments:
    /// - radius: The radius of the hexagonal grid.
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array representing the adjacency matrix of the hexagonal grid of the given radius.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn to_numpy_adjacency_matrix_uint32(py: Python, radius: usize) -> Py<PyArray2<u32>> {
        Self::to_numpy_adjacency_matrix_impl::<u32>(py, radius)
    }

    /// Get a NumPy adjacency matrix as a 2D ndarray of int64.
    /// 
    /// The adjacency matrix represents the connections between blocks in a hexagonal grid of the specified radius.
    /// Each entry (i, j) in the matrix is 1 if block i is adjacent to block j, and 0 otherwise.
    /// 
    /// The adjacency matrix is a 2D array representation of the connections (edges) between nodes (blocks) in a graph.
    /// In the context of a hexagonal grid, each block is a node, and an entry at (i, j) is 1 if block i is adjacent to block j.
    /// This representation is useful for applying graph algorithms for postprocessing, or applying convolution-like operations
    /// over the grid structure.
    /// 
    /// Arguments:
    /// - radius: The radius of the hexagonal grid.
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array representing the adjacency matrix of the hexagonal grid of the given radius.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn to_numpy_adjacency_matrix_int64(py: Python, radius: usize) -> Py<PyArray2<i64>> {
        Self::to_numpy_adjacency_matrix_impl::<i64>(py, radius)
    }

    /// Get a NumPy adjacency matrix as a 2D ndarray of uint64.
    /// 
    /// The adjacency matrix represents the connections between blocks in a hexagonal grid of the specified radius.
    /// Each entry (i, j) in the matrix is 1 if block i is adjacent to block j, and 0 otherwise.
    /// 
    /// The adjacency matrix is a 2D array representation of the connections (edges) between nodes (blocks) in a graph.
    /// In the context of a hexagonal grid, each block is a node, and an entry at (i, j) is 1 if block i is adjacent to block j.
    /// This representation is useful for applying graph algorithms for postprocessing, or applying convolution-like operations
    /// over the grid structure.
    /// 
    /// Arguments:
    /// - radius: The radius of the hexagonal grid.
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array representing the adjacency matrix of the hexagonal grid of the given radius.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn to_numpy_adjacency_matrix_uint64(py: Python, radius: usize) -> Py<PyArray2<u64>> {
        Self::to_numpy_adjacency_matrix_impl::<u64>(py, radius)
    }

    /// Get a NumPy adjacency matrix as a 2D ndarray of float32.
    /// 
    /// The adjacency matrix represents the connections between blocks in a hexagonal grid of the specified radius.
    /// Each entry (i, j) in the matrix is 1.0 if block i is adjacent to block j, and 0.0 otherwise.
    /// 
    /// The adjacency matrix is a 2D array representation of the connections (edges) between nodes (blocks) in a graph.
    /// In the context of a hexagonal grid, each block is a node, and an entry at (i, j) is 1.0 if block i is adjacent to block j.
    /// This representation is useful for applying graph algorithms for postprocessing, or applying convolution-like operations
    /// over the grid structure.
    /// 
    /// Arguments:
    /// - radius: The radius of the hexagonal grid.
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array representing the adjacency matrix of the hexagonal grid of the given radius.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn to_numpy_adjacency_matrix_float32(py: Python, radius: usize) -> Py<PyArray2<f32>> {
        Self::to_numpy_adjacency_matrix_impl::<f32>(py, radius)
    }

    /// Get a NumPy adjacency matrix as a 2D ndarray of float64.
    /// 
    /// The adjacency matrix represents the connections between blocks in a hexagonal grid of the specified radius.
    /// Each entry (i, j) in the matrix is 1.0 if block i is adjacent to block j, and 0.0 otherwise.
    /// 
    /// The adjacency matrix is a 2D array representation of the connections (edges) between nodes (blocks) in a graph.
    /// In the context of a hexagonal grid, each block is a node, and an entry at (i, j) is 1.0 if block i is adjacent to block j.
    /// This representation is useful for applying graph algorithms for postprocessing, or applying convolution-like operations
    /// over the grid structure.
    /// 
    /// Arguments:
    /// - radius: The radius of the hexagonal grid.
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array representing the adjacency matrix of the hexagonal grid of the given radius.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn to_numpy_adjacency_matrix_float64(py: Python, radius: usize) -> Py<PyArray2<f64>> {
        Self::to_numpy_adjacency_matrix_impl::<f64>(py, radius)
    }

    /// Get a NumPy adjacency matrix as a 2D ndarray of float16.
    /// 
    /// The adjacency matrix represents the connections between blocks in a hexagonal grid of the specified radius.
    /// Each entry (i, j) in the matrix is 1.0 if block i is adjacent to block j, and 0.0 otherwise.
    /// 
    /// The adjacency matrix is a 2D array representation of the connections (edges) between nodes (blocks) in a graph.
    /// In the context of a hexagonal grid, each block is a node, and an entry at (i, j) is 1.0 if block i is adjacent to block j.
    /// This representation is useful for applying graph algorithms for postprocessing, or applying convolution-like operations
    /// over the grid structure.
    /// 
    /// Arguments:
    /// - radius: The radius of the hexagonal grid.
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array representing the adjacency matrix of the hexagonal grid of the given radius.
    /// Warning:
    /// - The 'half' feature, which add support for float16, is still experimental and may not be stable. On machines that does
    /// not support float16 or installed with a version of numpy that does not support float16, this function may lead to
    /// undefined behavior or crashes. Testing show that on some systems, this can result in memory misinterpretation issues
    /// causing incorrect values to be read, and on other systems, it cause the entire program to halt but not crash.
    /// Use with caution.
    #[cfg(all(feature = "numpy", feature = "half"))]
    #[staticmethod]
    pub fn to_numpy_adjacency_matrix_float16(py: Python, radius: usize) -> Py<PyArray2<F16>> {
        Self::to_numpy_adjacency_matrix_impl::<F16>(py, radius)
    }

    /// Get a NumPy bool ndarray mask indicating valid positions for adding the given Piece.
    /// 
    /// Arguments:
    /// - piece: The Piece to check for valid positions.
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of boolean values where True indicates a valid position for adding the Piece.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_positions_mask(&self, py: Python, piece: Piece) -> Py<PyArray1<bool>> {
        self.to_numpy_positions_mask_impl::<bool>(py, piece)
    }

    /// Get a NumPy int8 ndarray mask indicating valid positions for adding the given Piece.
    /// 
    /// Arguments:
    /// - piece: The Piece to check for valid positions.
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of int8 values (0 or 1) where 1 indicates a valid position for adding the Piece.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_positions_mask_int8(&self, py: Python, piece: Piece) -> Py<PyArray1<i8>> {
        self.to_numpy_positions_mask_impl::<i8>(py, piece)
    }

    /// Get a NumPy uint8 ndarray mask indicating valid positions for adding the given Piece.
    /// 
    /// Arguments:
    /// - piece: The Piece to check for valid positions.
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of uint8 values (0 or 1) where 1 indicates a valid position for adding the Piece.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_positions_mask_uint8(&self, py: Python, piece: Piece) -> Py<PyArray1<u8>> {
        self.to_numpy_positions_mask_impl::<u8>(py, piece)
    }

    /// Get a NumPy int16 ndarray mask indicating valid positions for adding the given Piece.
    /// 
    /// Arguments:
    /// - piece: The Piece to check for valid positions.
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of int16 values (0 or 1) where 1 indicates a valid position for adding the Piece.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_positions_mask_int16(&self, py: Python, piece: Piece) -> Py<PyArray1<i16>> {
        self.to_numpy_positions_mask_impl::<i16>(py, piece)
    }

    /// Get a NumPy uint16 ndarray mask indicating valid positions for adding the given Piece.
    /// 
    /// Arguments:
    /// - piece: The Piece to check for valid positions.
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of uint16 values (0 or 1) where 1 indicates a valid position for adding the Piece.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_positions_mask_uint16(&self, py: Python, piece: Piece) -> Py<PyArray1<u16>> {
        self.to_numpy_positions_mask_impl::<u16>(py, piece)
    }

    /// Get a NumPy int32 ndarray mask indicating valid positions for adding the given Piece.
    /// 
    /// Arguments:
    /// - piece: The Piece to check for valid positions.
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of int32 values (0 or 1) where 1 indicates a valid position for adding the Piece.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_positions_mask_int32(&self, py: Python, piece: Piece) -> Py<PyArray1<i32>> {
        self.to_numpy_positions_mask_impl::<i32>(py, piece)
    }

    /// Get a NumPy uint32 ndarray mask indicating valid positions for adding the given Piece.
    /// 
    /// Arguments:
    /// - piece: The Piece to check for valid positions.
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of uint32 values (0 or 1) where 1 indicates a valid position for adding the Piece.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_positions_mask_uint32(&self, py: Python, piece: Piece) -> Py<PyArray1<u32>> {
        self.to_numpy_positions_mask_impl::<u32>(py, piece)
    }

    /// Get a NumPy int64 ndarray mask indicating valid positions for adding the given Piece.
    /// 
    /// Arguments:
    /// - piece: The Piece to check for valid positions.
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of int64 values (0 or 1) where 1 indicates a valid position for adding the Piece.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_positions_mask_int64(&self, py: Python, piece: Piece) -> Py<PyArray1<i64>> {
        self.to_numpy_positions_mask_impl::<i64>(py, piece)
    }

    /// Get a NumPy uint64 ndarray mask indicating valid positions for adding the given Piece.
    /// 
    /// Arguments:
    /// - piece: The Piece to check for valid positions.
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of uint64 values (0 or 1) where 1 indicates a valid position for adding the Piece.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_positions_mask_uint64(&self, py: Python, piece: Piece) -> Py<PyArray1<u64>> {
        self.to_numpy_positions_mask_impl::<u64>(py, piece)
    }

    /// Get a NumPy float32 ndarray mask indicating valid positions for adding the given Piece.
    /// 
    /// Arguments:
    /// - piece: The Piece to check for valid positions.
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of float32 values (0.0 or 1.0) where 1.0 indicates a valid position for adding the Piece.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_positions_mask_float32(&self, py: Python, piece: Piece) -> Py<PyArray1<f32>> {
        self.to_numpy_positions_mask_impl::<f32>(py, piece)
    }

    /// Get a NumPy float64 ndarray mask indicating valid positions for adding the given Piece.
    /// 
    /// Arguments:
    /// - piece: The Piece to check for valid positions.
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of float64 values (0.0 or 1.0) where 1.0 indicates a valid position for adding the Piece.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_positions_mask_float64(&self, py: Python, piece: Piece) -> Py<PyArray1<f64>> {
        self.to_numpy_positions_mask_impl::<f64>(py, piece)
    }

    /// Get a NumPy float16 ndarray mask indicating valid positions for adding the given Piece.
    /// 
    /// Arguments:
    /// - piece: The Piece to check for valid positions.
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of float16 values (0.0 or 1.0) where 1.0 indicates a valid position for adding the Piece.
    /// Warning:
    /// - The 'half' feature, which add support for float16, is still experimental and may not be stable. On machines that does
    /// not support float16 or installed with a version of numpy that does not support float16, this function may lead to
    /// undefined behavior or crashes. Testing show that on some systems, this can result in memory misinterpretation issues
    /// causing incorrect values to be read, and on other systems, it cause the entire program to halt but not crash.
    /// Use with caution.
    #[cfg(all(feature = "numpy", feature = "half"))]
    pub fn to_numpy_positions_mask_float16(&self, py: Python, piece: Piece) -> Py<PyArray1<F16>> {
        self.to_numpy_positions_mask_impl::<F16>(py, piece)
    }

    /// Get the default NumPy ndarray representation of the HexEngine's block states.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of boolean values representing the block states.
    #[cfg(feature = "numpy")]
    pub fn to_numpy(&self, py: Python) -> Py<PyArray1<bool>> {
        self.to_numpy_bool(py)
    }

    /// Get the NumPy ndarray boolean representation of the HexEngine's block states.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of boolean values representing the block states.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_bool(&self, py: Python) -> Py<PyArray1<bool>> {
        PyArray1::from_vec_bound(py, self.states.clone()).into()
    }

    /// Get the NumPy ndarray int8 representation of the HexEngine's block states.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of int8 values (0 or 1) representing the block states.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_int8(&self, py: Python) -> Py<PyArray1<i8>> {
        Self::to_numpy_engine_impl::<i8>(self, py)
    }

    /// Get the NumPy ndarray uint8 representation of the HexEngine's block states.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of uint8 values (0 or 1) representing the block states.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_uint8(&self, py: Python) -> Py<PyArray1<u8>> {
        Self::to_numpy_engine_impl::<u8>(self, py)
    }

    /// Get the NumPy ndarray int16 representation of the HexEngine's block states.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of int16 values (0 or 1) representing the block states.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_int16(&self, py: Python) -> Py<PyArray1<i16>> {
        Self::to_numpy_engine_impl::<i16>(self, py)
    }

    /// Get the NumPy ndarray uint16 representation of the HexEngine's block states.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of uint16 values (0 or 1) representing the block states.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_uint16(&self, py: Python) -> Py<PyArray1<u16>> {
        Self::to_numpy_engine_impl::<u16>(self, py)
    }

    /// Get the NumPy ndarray int32 representation of the HexEngine's block states.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of int32 values (0 or 1) representing the block states.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_int32(&self, py: Python) -> Py<PyArray1<i32>> {
        Self::to_numpy_engine_impl::<i32>(self, py)
    }

    /// Get the NumPy ndarray uint32 representation of the HexEngine's block states.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of uint32 values (0 or 1) representing the block states.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_uint32(&self, py: Python) -> Py<PyArray1<u32>> {
        Self::to_numpy_engine_impl::<u32>(self, py)
    }

    /// Get the NumPy ndarray int64 representation of the HexEngine's block states.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of int64 values (0 or 1) representing the block states.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_int64(&self, py: Python) -> Py<PyArray1<i64>> {
        Self::to_numpy_engine_impl::<i64>(self, py)
    }

    /// Get the NumPy ndarray uint64 representation of the HexEngine's block states.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of uint64 values (0 or 1) representing the block states.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_uint64(&self, py: Python) -> Py<PyArray1<u64>> {
        Self::to_numpy_engine_impl::<u64>(self, py)
    }

    /// Get the NumPy ndarray float32 representation of the HexEngine's block states.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of float32 values (0.0 or 1.0) representing the block states.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_float32(&self, py: Python) -> Py<PyArray1<f32>> {
        Self::to_numpy_engine_impl::<f32>(self, py)
    }

    /// Get the NumPy ndarray float64 representation of the HexEngine's block states.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of float64 values (0.0 or 1.0) representing the block states.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_float64(&self, py: Python) -> Py<PyArray1<f64>> {
        Self::to_numpy_engine_impl::<f64>(self, py)
    }

    /// Get the NumPy ndarray float16 representation of the HexEngine's block states.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of float16 values (0.0 or 1.0) representing the block states.
    /// Warning:
    /// - The 'half' feature, which add support for float16, is still experimental and may not be stable. On machines that does
    /// not support float16 or installed with a version of numpy that does not support float16, this function may lead to
    /// undefined behavior or crashes. Testing show that on some systems, this can result in memory misinterpretation issues
    /// causing incorrect values to be read, and on other systems, it cause the entire program to halt but not crash.
    /// Use with caution.
    #[cfg(all(feature = "numpy", feature = "half"))]
    pub fn to_numpy_float16(&self, py: Python) -> Py<PyArray1<F16>> {
        Self::to_numpy_engine_impl::<F16>(self, py)
    }

    /// Create a NumPy ndarray boolean representation of the HexEngine's block states
    /// without copying the underlying data. The method is extremely unsafe
    /// and should be used with caution.
    /// 
    /// The following conditions must be met for safe usage:
    /// 
    /// It is assumed that the HexEngine contains a valid hexagonal grid state and does not perform any checks.
    /// 
    /// The method also assumes that the memory of the HexEngine's states:
    /// - Is compatible with NumPy's memory layout. This means that NumPy must be able to interpret the HexEngine's internal
    ///   memory representation correctly as a NumPy array of the expected dtype and shape, and must not expect special padding
    ///   or alignment that is not present.
    /// - Is not used elsewhere after this function is called. Since the function takes a view of the data,
    ///   any further use of the original HexEngine will lead to undefined behavior, including potential crashes
    ///   or data corruption.
    /// - Is mutable and not shared. If the HexEngine's states are shared across multiple references or threads,
    ///   modifying it in NumPy could lead to data corruption or race conditions.
    /// 
    /// After conversion, the data is technically still held by Rust. So it is necessary to ensure that
    /// the lifetime of the NumPy array does not exceed that of the HexEngine in both Python and Rust memory management.
    /// If this is violated, it is highly likely that garbage data or segmentation faults will occur when accessing
    /// the NumPy array's data.
    /// 
    /// IMPORTANT: Under normal conditions, even if all the above conditions are met, this method will eventually
    /// lead to a double-free error when both Rust and Python attempt to free the same memory during their respective
    /// deallocation processes. To prevent this, manually increment the reference count of either the NumPy array or the
    /// HexEngine instance in Python using methods like `ctypes.pythonapi.Py_IncRef` to ensure that only one of them is
    /// responsible for freeing the memory. If this is undesirable, consider holding references to both objects until the end
    /// of the program execution so that all double free errors occur only at program termination.
    /// 
    /// For these reasons, unless performance is absolutely critical and you are certain that all the above
    /// conditions are met, it is strongly recommended to use the safe alternative `to_numpy_bool` method instead,
    /// which copies the data and performs necessary operations safely.
    ///
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of boolean values representing the block states.
    /// Warning:
    /// - This function is highly unsafe and can lead to undefined behavior if the above conditions are not met.
    #[cfg(feature = "numpy")]
    pub unsafe fn to_numpy_raw_view<'t>(&self, py: Python<'t>) -> Bound<'t, PyArray1<bool>> {
        // First, we violate safety by create two Vec<bool> pointers to the same data
        let ptr = self.states.as_ptr();
        let len = self.states.len();
        let vec = unsafe { Vec::from_raw_parts(ptr as *mut bool, len, len) };
        PyArray1::from_vec_bound(py, vec).into()
    }

    /// Construct a HexEngine from a NumPy ndarray boolean representation of the block states
    /// without validation and taking a view of the underlying data. The method is extremely unsafe
    /// and should be used with caution.
    /// 
    /// The following conditions must be met for safe usage:
    /// 
    /// It is assumed that when this method is used, the provided NumPy array length corresponds to a
    /// valid hexagonal grid size and does not perform any checks. It calculates the radius based on the
    /// length of the array.
    /// 
    /// If the length of the array does not correspond to a valid hexagonal grid size or is zero,
    /// the behavior is undefined and may cause runtime errors or panics down the line.
    /// 
    /// The method also assumes that the memory of the NumPy array:
    /// - Is allocated on the host (CPU) memory. If the array is allocated on a different device (e.g., GPU),
    ///   accessing its memory directly from Rust will lead to undefined behavior or mysterious crashes.
    /// - Is allocated in a way that is compatible with Rust's Vec<bool> memory layout. This means that it 
    ///   it not padded or aligned in a way that would be incompatible with Rust's expectations for Vec<bool>.
    /// - Is contiguous. If it is not contiguous, the function will panic.
    /// - Is not used elsewhere after this function is called. Since the function takes a view of the data,
    ///   any further use of the original NumPy array will lead to undefined behavior, including potential crashes
    ///   or data corruption.
    /// - Is mutable and not shared. If the NumPy array is shared across multiple references or threads,
    ///   modifying it in Rust could lead to data corruption or race conditions.
    /// 
    /// After conversion, the data is technically still held by NumPy. So it is necessary to ensure that
    /// the lifetime of the HexEngine does not exceed that of the original NumPy array in both Python and NumPy memory management.
    /// If this is violated, it is highly likely that garbage data or segmentation faults will occur when accessing
    /// the HexEngine's states.
    /// 
    /// IMPORTANT: Under normaly conditions, even if all the above conditions are met, this method will eventually
    /// lead to a double-free error when both Rust and Python attempt to free the same memory during their respective
    /// deallocation processes. To prevent this, manually increment the reference count of either the NumPy array or the
    /// HexEngine instance in Python using methods like `ctypes.pythonapi.Py_IncRef` to ensure that only one of them is
    /// responsible for freeing the memory. If this is undesireable, consider hold reference to both objects until the end
    /// of the program execution so that all double free errors occur only at program termination.
    /// 
    /// For these reasons, unless performance is absolutely critical and you are certain that all the above
    /// conditions are met, it is strongly recommended to use the safe alternative `from_numpy_bool` method instead,
    /// which copies the data and performs necessary validations. Or if you are sure about the data validity,
    /// the no validation (but still copying) version `from_numpy_bool_unchecked`.
    ///
    /// Arguments:
    /// - array: A 1D NumPy array of boolean values representing the block states.
    /// Returns:
    /// - HexEngine: A HexEngine instance initialized with the given state vector.
    /// Warning:
    /// - This function is highly unsafe and can lead to undefined behavior if the above conditions are not met.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub unsafe fn from_numpy_raw_view<'t>(array: Bound<'t, PyArray1<bool>>) -> Self {
        let slice = unsafe { array.as_slice_mut().unwrap() };
        let ptr = slice.as_mut_ptr();
        let len = slice.len();
        let vec = unsafe { Vec::from_raw_parts(ptr, len, len) };
        unsafe { HexEngine::from_raw_state(vec) }
    }

    /// Construct a HexEngine from a NumPy ndarray boolean representation of the block states.
    /// 
    /// Arguments:
    /// - array: A 1D NumPy array of boolean values representing the block states.
    /// Returns:
    /// - HexEngine: A new HexEngine instance initialized with the provided block states.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_bool(array: Bound<'_, PyArray1<bool>>) -> PyResult<Self> {
        let vec = array.to_vec()?;
        HexEngine::try_from(vec)
    }

    /// Construct a HexEngine from a NumPy ndarray boolean representation of the block states without validation.
    /// 
    /// This unsafe method assumes that the provided NumPy array length corresponds to a valid hexagonal grid
    /// size and does not perform any checks. It calculates the radius based on the length of the array.
    /// 
    /// If the length of the array does not correspond to a valid hexagonal grid size or is zero,
    /// the behavior is undefined and may cause runtime errors or panics down the line.
    /// 
    /// Arguments:
    /// - array: A 1D NumPy array of boolean values representing the block states.
    /// Returns:
    /// - HexEngine: A HexEngine instance initialized with the given state vector.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub unsafe fn from_numpy_bool_unchecked(array: Bound<'_, PyArray1<bool>>) -> Self {
        let slice = unsafe { array.as_slice().unwrap() };
        let vec: Vec<bool> = slice.to_vec();
        unsafe { HexEngine::from_raw_state(vec) }
    }

    /// Construct a HexEngine from a NumPy ndarray int8 representation of the block states.
    /// 
    /// Arguments:
    /// - array: A 1D NumPy array of int8 values (0 or 1) representing the block states.
    /// Returns:
    /// - HexEngine: A new HexEngine instance initialized with the provided block states (values > 0 are true, else false).
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_int8(array: Bound<'_, PyArray1<i8>>) -> PyResult<Self> {
        Self::from_numpy_engine_impl::<i8>(array)
    }

    /// Construct a HexEngine from a NumPy ndarray int8 representation of the block states without validation.
    /// 
    /// This unsafe method assumes that the provided NumPy array length corresponds to a valid hexagonal grid
    /// size and does not perform any checks. It calculates the radius based on the length of the array.
    /// 
    /// If the length of the array does not correspond to a valid hexagonal grid size or is zero,
    /// the behavior is undefined and may cause runtime errors or panics down the line.
    /// 
    /// Arguments:
    /// - array: A 1D NumPy array of int8 values (0 or 1) representing the block states.
    /// Returns:
    /// - HexEngine: A HexEngine instance initialized with the given state vector.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub unsafe fn from_numpy_int8_unchecked(array: Bound<'_, PyArray1<i8>>) -> Self {
        Self::from_numpy_engine_unchecked_impl::<i8>(array)
    }

    /// Construct a HexEngine from a NumPy ndarray uint8 representation of the block states.
    /// 
    /// Arguments:
    /// - array: A 1D NumPy array of uint8 values (0 or 1) representing the block states.
    /// Returns:
    /// - HexEngine: A new HexEngine instance initialized with the provided block states (0 is false, all others true).
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_uint8(array: Bound<'_, PyArray1<u8>>) -> PyResult<Self> {
        Self::from_numpy_engine_impl::<u8>(array)
    }

    /// Construct a HexEngine from a NumPy ndarray uint8 representation of the block states without validation.
    /// 
    /// This unsafe method assumes that the provided NumPy array length corresponds to a valid hexagonal grid
    /// size and does not perform any checks. It calculates the radius based on the length of the array.
    /// 
    /// If the length of the array does not correspond to a valid hexagonal grid size or is zero,
    /// the behavior is undefined and may cause runtime errors or panics down the line.
    /// 
    /// Arguments:
    /// - array: A 1D NumPy array of uint8 values (0 or 1) representing the block states.
    /// Returns:
    /// - HexEngine: A HexEngine instance initialized with the given state vector.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub unsafe fn from_numpy_uint8_unchecked(array: Bound<'_, PyArray1<u8>>) -> Self {
        Self::from_numpy_engine_unchecked_impl::<u8>(array)
    }

    /// Construct a HexEngine from a NumPy ndarray int16 representation of the block states.
    /// 
    /// Arguments:
    /// - array: A 1D NumPy array of int16 values (0 or 1) representing the block states.
    /// Returns:
    /// - HexEngine: A new HexEngine instance initialized with the provided block states (values > 0 are true, else false).
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_int16(array: Bound<'_, PyArray1<i16>>) -> PyResult<Self> {
        Self::from_numpy_engine_impl::<i16>(array)
    }

    /// Construct a HexEngine from a NumPy ndarray int16 representation of the block states without validation.
    /// 
    /// This unsafe method assumes that the provided NumPy array length corresponds to a valid hexagonal grid
    /// size and does not perform any checks. It calculates the radius based on the length of the array.
    /// 
    /// If the length of the array does not correspond to a valid hexagonal grid size or is zero,
    /// the behavior is undefined and may cause runtime errors or panics down the line.
    /// 
    /// Arguments:
    /// - array: A 1D NumPy array of int16 values (0 or 1) representing the block states.
    /// Returns:
    /// - HexEngine: A HexEngine instance initialized with the given state vector.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub unsafe fn from_numpy_int16_unchecked(array: Bound<'_, PyArray1<i16>>) -> Self {
        Self::from_numpy_engine_unchecked_impl::<i16>(array)
    }

    /// Construct a HexEngine from a NumPy ndarray uint16 representation of the block states.
    /// 
    /// Arguments:
    /// - array: A 1D NumPy array of uint16 values (0 or 1) representing the block states.
    /// Returns:
    /// - HexEngine: A new HexEngine instance initialized with the provided block states (0 is false, all others true).
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_uint16(array: Bound<'_, PyArray1<u16>>) -> PyResult<Self> {
        Self::from_numpy_engine_impl::<u16>(array)
    }

    /// Construct a HexEngine from a NumPy ndarray uint16 representation of the block states without validation.
    /// 
    /// This unsafe method assumes that the provided NumPy array length corresponds to a valid hexagonal grid
    /// size and does not perform any checks. It calculates the radius based on the length of the array.
    /// 
    /// If the length of the array does not correspond to a valid hexagonal grid size or is zero,
    /// the behavior is undefined and may cause runtime errors or panics down the line.
    /// 
    /// Arguments:
    /// - array: A 1D NumPy array of uint16 values (0 or 1) representing the block states.
    /// Returns:
    /// - HexEngine: A HexEngine instance initialized with the given state vector.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub unsafe fn from_numpy_uint16_unchecked(array: Bound<'_, PyArray1<u16>>) -> Self {
        Self::from_numpy_engine_unchecked_impl::<u16>(array)
    }

    /// Construct a HexEngine from a NumPy ndarray int32 representation of the block states.
    ///
    /// Arguments:
    /// - array: A 1D NumPy array of int32 values (0 or 1) representing the block states.
    /// Returns:
    /// - HexEngine: A new HexEngine instance initialized with the provided block states (values > 0 are true, else false).
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_int32(array: Bound<'_, PyArray1<i32>>) -> PyResult<Self> {
        Self::from_numpy_engine_impl::<i32>(array)
    }

    /// Construct a HexEngine from a NumPy ndarray int32 representation of the block states without validation.
    /// 
    /// This unsafe method assumes that the provided NumPy array length corresponds to a valid hexagonal grid
    /// size and does not perform any checks. It calculates the radius based on the length of the array.
    /// 
    /// If the length of the array does not correspond to a valid hexagonal grid size or is zero,
    /// the behavior is undefined and may cause runtime errors or panics down the line.
    /// 
    /// Arguments:
    /// - array: A 1D NumPy array of int32 values (0 or 1) representing the block states.
    /// Returns:
    /// - HexEngine: A HexEngine instance initialized with the given state vector.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub unsafe fn from_numpy_int32_unchecked(array: Bound<'_, PyArray1<i32>>) -> Self {
        Self::from_numpy_engine_unchecked_impl::<i32>(array)
    }

    /// Construct a HexEngine from a NumPy ndarray uint32 representation of the block states.
    /// 
    /// Arguments:
    /// - array: A 1D NumPy array of uint32 values (0 or 1) representing the block states.
    /// Returns:
    /// - HexEngine: A new HexEngine instance initialized with the provided block states (0 is false, all others true).
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_uint32(array: Bound<'_, PyArray1<u32>>) -> PyResult<Self> {
        Self::from_numpy_engine_impl::<u32>(array)
    }

    /// Construct a HexEngine from a NumPy ndarray uint32 representation of the block states without validation.
    /// 
    /// This unsafe method assumes that the provided NumPy array length corresponds to a valid hexagonal grid
    /// size and does not perform any checks. It calculates the radius based on the length of the array.
    /// 
    /// If the length of the array does not correspond to a valid hexagonal grid size or is zero,
    /// the behavior is undefined and may cause runtime errors or panics down the line.
    /// 
    /// Arguments:
    /// - array: A 1D NumPy array of uint32 values (0 or 1) representing the block states.
    /// Returns:
    /// - HexEngine: A HexEngine instance initialized with the given state vector.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub unsafe fn from_numpy_uint32_unchecked(array: Bound<'_, PyArray1<u32>>) -> Self {
        Self::from_numpy_engine_unchecked_impl::<u32>(array)
    }

    /// Construct a HexEngine from a NumPy ndarray int64 representation of the block states.
    /// 
    /// Arguments:
    /// - array: A 1D NumPy array of int64 values (0 or 1) representing the block states.
    /// Returns:
    /// - HexEngine: A new HexEngine instance initialized with the provided block states (values > 0 are true, else false).
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_int64(array: Bound<'_, PyArray1<i64>>) -> PyResult<Self> {
        Self::from_numpy_engine_impl::<i64>(array)
    }

    /// Construct a HexEngine from a NumPy ndarray int64 representation of the block states without validation.
    /// 
    /// This unsafe method assumes that the provided NumPy array length corresponds to a valid hexagonal grid
    /// size and does not perform any checks. It calculates the radius based on the length of the array.
    /// 
    /// If the length of the array does not correspond to a valid hexagonal grid size or is zero,
    /// the behavior is undefined and may cause runtime errors or panics down the line.
    /// 
    /// Arguments:
    /// - array: A 1D NumPy array of int64 values (0 or 1) representing the block states.
    /// Returns:
    /// - HexEngine: A HexEngine instance initialized with the given state vector.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub unsafe fn from_numpy_int64_unchecked(array: Bound<'_, PyArray1<i64>>) -> Self {
        Self::from_numpy_engine_unchecked_impl::<i64>(array)
    }

    /// Construct a HexEngine from a NumPy ndarray uint64 representation of the block states.
    /// 
    /// Arguments:
    /// - array: A 1D NumPy array of uint64 values (0 or 1) representing the block states.
    /// Returns:
    /// - HexEngine: A new HexEngine instance initialized with the provided block states (0 is false, all others true).
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_uint64(array: Bound<'_, PyArray1<u64>>) -> PyResult<Self> {
        Self::from_numpy_engine_impl::<u64>(array)
    }

    /// Construct a HexEngine from a NumPy ndarray uint64 representation of the block states without validation.
    /// 
    /// This unsafe method assumes that the provided NumPy array length corresponds to a valid hexagonal grid
    /// size and does not perform any checks. It calculates the radius based on the length of the array.
    /// 
    /// If the length of the array does not correspond to a valid hexagonal grid size or is zero,
    /// the behavior is undefined and may cause runtime errors or panics down the line.
    /// 
    /// Arguments:
    /// - array: A 1D NumPy array of uint64 values (0 or 1) representing the block states.
    /// Returns:
    /// - HexEngine: A HexEngine instance initialized with the given state vector.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub unsafe fn from_numpy_uint64_unchecked(array: Bound<'_, PyArray1<u64>>) -> Self {
        Self::from_numpy_engine_unchecked_impl::<u64>(array)
    }

    /// Construct a HexEngine from a NumPy ndarray float32 representation of the block states.
    /// 
    /// Arguments:
    /// - array: A 1D NumPy array of float32 values (0.0 or 1.0) representing the block states.
    /// Returns:
    /// - HexEngine: A new HexEngine instance initialized with the provided block states (values > 0.0 are true, else false).
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_float32(array: Bound<'_, PyArray1<f32>>) -> PyResult<Self> {
        Self::from_numpy_engine_impl::<f32>(array)
    }

    /// Construct a HexEngine from a NumPy ndarray float32 representation of the block states without validation.
    /// 
    /// This unsafe method assumes that the provided NumPy array length corresponds to a valid hexagonal grid
    /// size and does not perform any checks. It calculates the radius based on the length of the array.
    /// 
    /// If the length of the array does not correspond to a valid hexagonal grid size or is zero,
    /// the behavior is undefined and may cause runtime errors or panics down the line.
    /// 
    /// Arguments:
    /// - array: A 1D NumPy array of float32 values (0.0 or 1.0) representing the block states.
    /// Returns:
    /// - HexEngine: A HexEngine instance initialized with the given state vector.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub unsafe fn from_numpy_float32_unchecked(array: Bound<'_, PyArray1<f32>>) -> Self {
        Self::from_numpy_engine_unchecked_impl::<f32>(array)
    }

    /// Construct a HexEngine from a NumPy ndarray float64 representation of the block states.
    /// 
    /// Arguments:
    /// - array: A 1D NumPy array of float64 values (0.0 or 1.0) representing the block states.
    /// Returns:
    /// - HexEngine: A new HexEngine instance initialized with the provided block states (values > 0.0 are true, else false).
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_float64(array: Bound<'_, PyArray1<f64>>) -> PyResult<Self> {
        Self::from_numpy_engine_impl::<f64>(array)
    }

    /// Construct a HexEngine from a NumPy ndarray float64 representation of the block states without validation.
    /// 
    /// This unsafe method assumes that the provided NumPy array length corresponds to a valid hexagonal grid
    /// size and does not perform any checks. It calculates the radius based on the length of the array.
    /// 
    /// If the length of the array does not correspond to a valid hexagonal grid size or is zero,
    /// the behavior is undefined and may cause runtime errors or panics down the line.
    /// 
    /// Arguments:
    /// - array: A 1D NumPy array of float64 values (0.0 or 1.0) representing the block states.
    /// Returns:
    /// - HexEngine: A HexEngine instance initialized with the given state vector.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub unsafe fn from_numpy_float64_unchecked(array: Bound<'_, PyArray1<f64>>) -> Self {
        Self::from_numpy_engine_unchecked_impl::<f64>(array)
    }

    /// Construct a HexEngine from a NumPy ndarray float16 representation of the block states.
    /// 
    /// Arguments:
    /// - array: A 1D NumPy array of float16 values (0.0 or 1.0) representing the block states.
    /// Returns:
    /// - HexEngine: A new HexEngine instance initialized with the provided block states (values > 0.0 are true, else false).
    /// Warning:
    /// - The 'half' feature, which add support for float16, is still experimental and may not be stable. On machines that does
    /// not support float16 or installed with a version of numpy that does not support float16, this function may lead to
    /// undefined behavior or crashes. Testing show that on some systems, this can result in memory misinterpretation issues
    /// causing incorrect values to be read, and on other systems, it may cause a segmentation fault.
    /// Use with caution.
    #[cfg(all(feature = "numpy", feature = "half"))]
    #[staticmethod]
    pub fn from_numpy_float16(array: Bound<'_, PyArray1<F16>>) -> PyResult<Self> {
        Self::from_numpy_engine_impl::<F16>(array)
    }

    /// Construct a HexEngine from a NumPy ndarray float16 representation of the block states without validation.
    /// 
    /// This unsafe method assumes that the provided NumPy array length corresponds to a valid hexagonal grid
    /// size and does not perform any checks. It calculates the radius based on the length of the array.
    /// 
    /// If the length of the array does not correspond to a valid hexagonal grid size or is zero,
    /// the behavior is undefined and may cause runtime errors or panics down the line.
    /// 
    /// Arguments:
    /// - array: A 1D NumPy array of float16 values (0.0 or 1.0) representing the block states.
    /// Returns:
    /// - HexEngine: A HexEngine instance initialized with the given state vector.
    /// Warning:
    /// - The 'half' feature, which add support for float16, is still experimental and may not be stable. On machines that does
    /// not support float16 or installed with a version of numpy that does not support float16, this function may lead to
    /// undefined behavior or crashes. Testing show that on some systems, this can result in memory misinterpretation issues
    /// causing incorrect values to be read, and on other systems, it may cause a segmentation fault. Those unintended behaviors
    /// may still occur even when the arguments to the function are valid.
    /// Use with caution.
    #[cfg(all(feature = "numpy", feature = "half"))]
    #[staticmethod]
    pub unsafe fn from_numpy_float16_unchecked(array: Bound<'_, PyArray1<F16>>) -> Self {
        Self::from_numpy_engine_unchecked_impl::<F16>(array)
    }

    /* ---------------------------------------- HPYHEX PYTHON API ---------------------------------------- */

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
    pub fn __iter__(slf: PyRef<'_, Self>) -> PyResult<HexEngineIterator> {
        Ok(HexEngineIterator {
            states: slf.states.clone(),
            index: 0,
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
            let hex = self.hex_coordinate_of(i).unwrap_or(Hex { i: -1, k: -1 });
            // It is certain that the later will not happen but just in case
            if i > 0 {
                s.push_str(", ");
            }
            s.push_str(&format!("({}, {}, {})", hex.i, hex.k, state));
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
    pub fn index_block(&self, coo: &pyo3::Bound<'_, PyAny>) -> isize {
        let (i, k) = if let Ok(hex) = coo.extract::<PyRef<Hex>>() {
            (hex.i, hex.k)
        } else if let Ok(tuple) = coo.extract::<(i32, i32)>() {
            (tuple.0, tuple.1)
        } else if let Ok(tuple3) = coo.extract::<(i32, i32, i32)>() {
            (tuple3.0, tuple3.2)
        } else {
            return -1;
        };
        self.linear_index_of(i, k).unwrap_or(-1)
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
            let hex = self.hex_coordinate_of(index)?;
            let hex = get_hex(hex.i, hex.k);
            Ok(hex)
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
            let idx = self.index_block(coo);
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
            let idx = self.index_block(coo);
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
        if let Ok(piece_ref) = piece.extract::<PyRef<Piece>>() {
            for i in 0..7 {
                if piece_ref.states()[i] {
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
        } else if let Ok(state) = piece.extract::<u8>() {
            Python::with_gil(|py| {
                let piece_obj = PIECE_CACHE.get_or_init(|| initialize_piece_cache())[state as usize].clone();
                let piece_ref = piece_obj.borrow(py);
                for i in 0..7 {
                    if piece_ref.states()[i] {
                        let (hex_i, hex_k) = {
                            let pos = &Piece::positions[i];
                            let coo_val = coo.extract::<PyRef<Hex>>().ok();
                            let base = if let Some(c) = coo_val { (c.i, c.k) } else if let Ok(tuple) = coo.extract::<(i32, i32)>() { (tuple.0, tuple.1) } else { (0, 0) };
                            (pos.i + base.0, pos.k + base.1)
                        };
                        if let Ok(idx) = self.linear_index_of(hex_i, hex_k) {
                            if idx == -1 { return Ok(false); }
                            if self.states[idx as usize] { return Ok(false); }
                        } else {
                            return Ok(false);
                        }
                    }
                }
                Ok(true)
            })
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err("Piece must be an instance of Piece or an integer representing a Piece state"));
        }
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
        if let Ok(piece_ref) = piece.extract::<PyRef<Piece>>() {
            for i in 0..7 {
                if piece_ref.states()[i] {
                    let (placed_i, placed_k) = Python::with_gil(|_py| {
                        let pos = &Piece::positions[i];
                        let coo_val = coo.extract::<PyRef<Hex>>().ok();
                        let base = if let Some(c) = coo_val { (c.i, c.k) } else if let Ok(tuple) = coo.extract::<(i32, i32)>() { (tuple.0, tuple.1) } else { (0, 0) };
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
        } else if let Ok(state) = piece.extract::<u8>() {
            Python::with_gil(|py| {
                let piece_obj = PIECE_CACHE.get_or_init(|| initialize_piece_cache())[state as usize].clone();
                let piece_ref = piece_obj.borrow(py);
                for i in 0..7 {
                    if piece_ref.states()[i] {
                        let (placed_i, placed_k) = {
                            let pos = &Piece::positions[i];
                            let coo_val = coo.extract::<PyRef<Hex>>().ok();
                            let base = if let Some(c) = coo_val { (c.i, c.k) } else if let Ok(tuple) = coo.extract::<(i32, i32)>() { (tuple.0, tuple.1) } else { (0, 0) };
                            (pos.i + base.0, pos.k + base.1)
                        };
                        let idx = self.linear_index_of(placed_i, placed_k)?;
                        if idx == -1 {
                            return Err(pyo3::exceptions::PyValueError::new_err("Coordinate out of range"));
                        }
                        self.states[idx as usize] = true;
                    }
                }
                Ok(())
            })
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err("Piece must be an instance of Piece or an integer representing a Piece state"))
        }
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
        if let Ok(piece) = piece.extract::<Piece>() {
            let mut positions = Vec::new();
            for a in 0..(self.radius * 2) as i32 {
                for b in 0..(self.radius * 2) as i32 {
                    let hex = Hex{i: a, k: b};
                    if self.check_add_of(&hex, &piece)? {
                        positions.push(get_hex(hex.i, hex.k));
                    }
                }
            }
            Ok(positions)
        } else if let Ok(state) = piece.extract::<u8>() {
            Python::with_gil(|py|
                {
                    let piece = PIECE_CACHE.get_or_init(|| initialize_piece_cache())[state as usize].borrow(py);
                    let mut positions = Vec::new();
                    for a in 0..(self.radius * 2) as i32 {
                        for b in 0..(self.radius * 2) as i32 {
                            let hex = Hex{i: a, k: b};
                            if self.check_add_of(&hex, &piece)? {
                                positions.push(get_hex(hex.i, hex.k));
                            }
                        }
                    }
                    Ok(positions)
                }
            )
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err("Piece must be an instance of Piece or an integer representing a Piece state"))
        }
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
        if let Ok(piece_ref) = piece.extract::<PyRef<Piece>>() {
            let mut total_possible = 0;
            let mut total_populated = 0;
            let piece_piece = piece.extract::<Piece>()?;
            for i in 0..7 {
                if piece_ref.states()[i] {
                    let (placed_i, placed_k) = Python::with_gil(|_py| {
                        let pos = &Piece::positions[i];
                        let coo_val = coo.extract::<PyRef<Hex>>().ok();
                        let base = if let Some(c) = coo_val { (c.i, c.k) } else if let Ok(tuple) = coo.extract::<(i32, i32)>() { (tuple.0, tuple.1) } else { (0, 0) };
                        (pos.i + base.0, pos.k + base.1)
                    });
                    if !HexEngine::check_range_coords(placed_i, placed_k, self.radius).unwrap_or(false) || self.state_of(placed_i, placed_k).unwrap_or(false) {
                        return Ok(0.0);
                    }
                    total_possible += 6 - piece_neighbors_of(piece_piece.clone(), placed_i, placed_k);
                    total_populated += self.count_neighbors_coordinate(placed_i, placed_k).unwrap_or(0);
                }
            }
            Ok(if total_possible > 0 { total_populated as f64 / total_possible as f64 } else { 0.0 })
        } else if let Ok(state) = piece.extract::<u8>() {
            Python::with_gil(|py| {
                let piece_obj = PIECE_CACHE.get_or_init(|| initialize_piece_cache())[state as usize].clone();
                let piece_ref = piece_obj.borrow(py);
                // let piece = Piece {state: state};
                let mut total_possible = 0;
                let mut total_populated = 0;
                for i in 0..7 {
                    if piece_ref.states()[i] {
                        let (placed_i, placed_k) = {
                            let pos = &Piece::positions[i];
                            let coo_val = coo.extract::<PyRef<Hex>>().ok();
                            let base = if let Some(c) = coo_val { (c.i, c.k) } else if let Ok(tuple) = coo.extract::<(i32, i32)>() { (tuple.0, tuple.1) } else { (0, 0) };
                            (pos.i + base.0, pos.k + base.1)
                        };
                        if !HexEngine::check_range_coords(placed_i, placed_k, self.radius).unwrap_or(false) || self.state_of(placed_i, placed_k).unwrap_or(false) {
                            return Ok(0.0);
                        }
                        total_possible += 6 - piece_neighbors_of(Piece {state: state}, placed_i, placed_k);
                        total_populated += self.count_neighbors_coordinate(placed_i, placed_k).unwrap_or(0);
                    }
                }
                Ok(if total_possible > 0 { total_populated as f64 / total_possible as f64 } else { 0.0 })
            })
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err("Piece must be an instance of Piece or an integer representing a Piece state"))
        }
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

#[pyclass]
/// An iterator over HexEngine, returning states as booleans.
pub struct HexEngineIterator {
    states: Vec<bool>,
    index: usize,
}

#[pymethods]
impl HexEngineIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<bool> {
        if slf.index < slf.states.len() {
            let result = slf.states[slf.index];
            slf.index += 1;
            Some(result)
        } else {
            None
        }
    }
}

use rand::Rng;

#[pyfunction]
/// Generate a random HexEngine with a given radius. True randomness or random distribution is not guaranteed
/// as elimination is applied to the engine, reducing some instances to other instances.
/// 
/// This is superior than HexEngine.all_engines(radius) because it does not consume significant memory and time.
/// 
/// Arguments:
/// - radius (int): The radius of the hexagonal game board.
/// Returns:
/// - HexEngine: A new randomized HexEngine instance with the specified radius.
/// Raises:
/// - TypeError: If radius is not an integer or is less than 2.
pub fn random_engine(radius: usize) -> PyResult<HexEngine> {
    if radius < 2 {
        return Err(pyo3::exceptions::PyTypeError::new_err("Radius must be an integer greater than 1"));
    }
    let length = 1 + 3 * radius * (radius - 1);
    let mut rng = rand::rng();
    let mut states = Vec::with_capacity(length);
    for _ in 0..length {
        states.push(rng.random_bool(0.5));
    }
    let mut engine = HexEngine { radius, states };
    let _ = engine.eliminate();
    Ok(engine)
}

use std::collections::HashMap;
use once_cell::sync::Lazy;

/// PieceFactory is a utility class that provides methods for creating and managing game pieces.
/// It includes a predefined set of pieces, their corresponding byte values, and reverse mappings
/// to retrieve piece names from byte values. The class also supports generating random pieces
/// based on predefined probabilities.
///
/// Attributes:
/// - pieces (dict): A dictionary mapping piece names (str) to their corresponding byte values (int).
/// - reverse_pieces (dict): A reverse mapping of `pieces`, mapping byte values (int) to piece names (str).
#[pyclass]
pub struct PieceFactory {}

static PIECE_MAP: Lazy<HashMap<&'static str, u8>> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert("uno", 8);
    m.insert("full", 127);
    m.insert("hallow", 119);
    m.insert("triangle_3_a", 13);
    m.insert("triangle_3_b", 88);
    m.insert("line_3_i", 28);
    m.insert("line_3_j", 73);
    m.insert("line_3_k", 42);
    m.insert("corner_3_i_l", 74);
    m.insert("corner_3_i_r", 41);
    m.insert("corner_3_j_l", 56);
    m.insert("corner_3_j_r", 14);
    m.insert("corner_3_k_l", 76);
    m.insert("corner_3_k_r", 25);
    m.insert("fan_4_a", 78);
    m.insert("fan_4_b", 57);
    m.insert("rhombus_4_i", 27);
    m.insert("rhombus_4_j", 120);
    m.insert("rhombus_4_k", 90);
    m.insert("corner_4_i_l", 39);
    m.insert("corner_4_i_r", 114);
    m.insert("corner_4_j_l", 101);
    m.insert("corner_4_j_r", 83);
    m.insert("corner_4_k_l", 23);
    m.insert("corner_4_k_r", 116);
    m.insert("asym_4_i_a", 92);
    m.insert("asym_4_i_b", 30);
    m.insert("asym_4_i_c", 60);
    m.insert("asym_4_i_d", 29);
    m.insert("asym_4_j_a", 75);
    m.insert("asym_4_j_b", 77);
    m.insert("asym_4_j_c", 89);
    m.insert("asym_4_j_d", 105);
    m.insert("asym_4_k_a", 46);
    m.insert("asym_4_k_b", 106);
    m.insert("asym_4_k_c", 43);
    m.insert("asym_4_k_d", 58);
    m
});

static REVERSE_PIECE_MAP: Lazy<HashMap<u8, &'static str>> = Lazy::new(|| {
    let mut rev = HashMap::new();
    for (&k, &v) in PIECE_MAP.iter() {
        rev.insert(v, k);
    }
    rev
});

#[pymethods]
impl PieceFactory {
    #[classattr]
    /// Mapping from piece name to byte value
    fn pieces(py: pyo3::Python<'_>) -> pyo3::PyObject {
        let dict = pyo3::types::PyDict::new_bound(py);
        for (&k, &v) in PIECE_MAP.iter() {
            dict.set_item(k, v).unwrap();
        }
        dict.into()
    }

    #[classattr]
    /// Reverse mapping from byte value to piece name
    fn reverse_pieces(py: pyo3::Python<'_>) -> pyo3::PyObject {
        let dict = pyo3::types::PyDict::new_bound(py);
        for (&k, &v) in REVERSE_PIECE_MAP.iter() {
            dict.set_item(k, v).unwrap();
        }
        dict.into()
    }

    /// Get a piece by its name.
    ///
    /// # Arguments
    /// - name (str): The name of the piece to retrieve.
    /// # Returns
    /// - Piece: The piece object corresponding to the given name.
    /// # Raises
    /// - ValueError: If the piece name is not found in the factory.
    #[staticmethod]
    pub fn get_piece(name: &str) -> PyResult<Piece> {
        if let Some(&byte) = PIECE_MAP.get(name) {
            Ok(Piece { state: byte })
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(format!("Piece '{}' not found in factory.", name)))
        }
    }

    /// Get the name of a piece based on its byte value.
    ///
    /// # Arguments
    /// - p (Piece): The piece object whose name is to be retrieved.
    /// # Returns
    /// - String: The name of the piece corresponding to the given byte value.
    /// # Raises
    /// - ValueError: If the piece byte value is not found in the factory.
    #[staticmethod]
    pub fn get_piece_name(p: &Piece) -> PyResult<String> {
        let byte = p.state as u8;
        if let Some(name) = REVERSE_PIECE_MAP.get(&byte) {
            Ok(name.to_string())
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(format!("Piece with byte value {} not found in factory.", byte)))
        }
    }

    // Rust note: can also be used as a backend function.

    /// Generate a random piece based on frequency distribution.
    ///
    /// # Returns
    /// - Piece: A randomly selected piece object.
    #[staticmethod]
    pub fn generate_piece() -> PyResult<Piece> {
        use rand::Rng;
        let mut rng = rand::rng();
        if rng.random_bool(0.5) {
            let i = rng.random_range(0..74);
            if i < 8 {
                return Self::get_piece("triangle_3_a");
            } else if i < 16 {
                return Self::get_piece("triangle_3_b");
            } else if i < 22 {
                return Self::get_piece("line_3_i");
            } else if i < 28 {
                return Self::get_piece("line_3_j");
            } else if i < 34 {
                return Self::get_piece("line_3_k");
            } else if i < 37 {
                return Self::get_piece("corner_3_i_r");
            } else if i < 40 {
                return Self::get_piece("corner_3_j_r");
            } else if i < 43 {
                return Self::get_piece("corner_3_k_r");
            } else if i < 46 {
                return Self::get_piece("corner_3_i_l");
            } else if i < 49 {
                return Self::get_piece("corner_3_j_l");
            } else if i < 52 {
                return Self::get_piece("corner_3_k_l");
            } else if i < 56 {
                return Self::get_piece("rhombus_4_i");
            } else if i < 60 {
                return Self::get_piece("rhombus_4_j");
            } else if i < 64 {
                return Self::get_piece("rhombus_4_k");
            }
            let j = rng.random_range(0..25);
            if j == 0 || j == 1 {
                return Self::get_piece("fan_4_a");
            } else if j == 2 || j == 3 {
                return Self::get_piece("fan_4_b");
            } else if j == 4 {
                return Self::get_piece("corner_4_i_l");
            } else if j == 5 {
                return Self::get_piece("corner_4_i_r");
            } else if j == 6 {
                return Self::get_piece("corner_4_j_l");
            } else if j == 7 {
                return Self::get_piece("corner_4_j_r");
            } else if j == 8 {
                return Self::get_piece("corner_4_k_l");
            } else if j == 9 {
                return Self::get_piece("corner_4_k_r");
            } else if j < 14 {
                let c = (b'a' + (j - 10) as u8) as char;
                return Self::get_piece(&format!("asym_4_i_{}", c));
            } else if j < 18 {
                let c = (b'a' + (j - 14) as u8) as char;
                return Self::get_piece(&format!("asym_4_j_{}", c));
            } else if j < 22 {
                let c = (b'a' + (j - 18) as u8) as char;
                return Self::get_piece(&format!("asym_4_k_{}", c));
            } else {
                return Self::get_piece("uno");
            }
        } else {
            let i = rng.random_range(0..86);
            if i < 6 {
                return Self::get_piece("triangle_3_a");
            } else if i < 12 {
                return Self::get_piece("triangle_3_b");
            } else if i < 16 {
                return Self::get_piece("line_3_i");
            } else if i < 20 {
                return Self::get_piece("line_3_j");
            } else if i < 24 {
                return Self::get_piece("line_3_k");
            } else if i < 26 {
                return Self::get_piece("corner_3_i_r");
            } else if i < 28 {
                return Self::get_piece("corner_3_j_r");
            } else if i < 30 {
                return Self::get_piece("corner_3_k_r");
            } else if i < 32 {
                return Self::get_piece("corner_3_i_l");
            } else if i < 34 {
                return Self::get_piece("corner_3_j_l");
            } else if i < 36 {
                return Self::get_piece("corner_3_k_l");
            } else if i < 40 {
                return Self::get_piece("rhombus_4_i");
            } else if i < 44 {
                return Self::get_piece("rhombus_4_j");
            } else if i < 48 {
                return Self::get_piece("rhombus_4_k");
            } else if i < 54 {
                return Self::get_piece("fan_4_a");
            } else if i < 60 {
                return Self::get_piece("fan_4_b");
            } else if i < 62 {
                return Self::get_piece("corner_4_i_l");
            } else if i < 64 {
                return Self::get_piece("corner_4_i_r");
            } else if i < 66 {
                return Self::get_piece("corner_4_j_l");
            } else if i < 68 {
                return Self::get_piece("corner_4_j_r");
            } else if i < 70 {
                return Self::get_piece("corner_4_k_l");
            } else if i < 72 {
                return Self::get_piece("corner_4_k_r");
            } else if i < 76 {
                let c = (b'a' + (i - 72) as u8) as char;
                return Self::get_piece(&format!("asym_4_i_{}", c));
            } else if i < 80 {
                let c = (b'a' + (i - 76) as u8) as char;
                return Self::get_piece(&format!("asym_4_j_{}", c));
            } else if i < 84 {
                let c = (b'a' + (i - 80) as u8) as char;
                return Self::get_piece(&format!("asym_4_k_{}", c));
            } else {
                return Self::get_piece("full");
            }
        }
    }

    // Rust note: can also be used as a backend function.

    /// Return all pieces that are defined in this factory.
    ///
    /// # Returns
    /// - list[Piece]: A list of all Piece instances defined in the factory.
    #[staticmethod]
    pub fn all_pieces() -> Vec<Piece> {
        PIECE_MAP.values().map(|&byte| Piece { state: byte }).collect()
    }
}

#[allow(non_snake_case)]
// Note for underscoring and getter, setters in Game class:
// The python implementation has those private attributes with double underscores
// Although attributes in Game are never intended to be modified directly from outside,
// Some code with bad practice may depend on them. To not break compatibility with the
// original hpyhex, we provide getters and setters with the same names.

/// Game is a class that represents the game environment for Hex.
/// It manages the game engine, the queue of pieces, and the game state.
/// It provides methods to add pieces, make moves, and check the game status.
/// Its methods are intended to catch exceptions and handle errors gracefully.
///
/// Attributes:
/// - engine (HexEngine): The game engine that manages the game state.
/// - queue (list[Piece]): The queue of pieces available for placement.
/// - result (tuple[int, int]): The current result of the game, including the score and turn number.
/// - score (int): The current score of the game.
/// - turn (int): The current turn number in the game.
/// - end (bool): Whether the game has ended.
/// 
/// Special Methods:
/// - hpyhex_rs_add_piece_with_index: Adds a piece to the queue at a specified index.
/// 
/// NumPy Integration:
/// If the 'numpy' feature is enabled, Game provides methods to convert the piece queue, the engine,
/// and the whole game state into NumPy ndarray representations for efficient numerical processing.
/// 
/// Queue Conversion Methods:
/// - queue_to_numpy_flat: Converts the piece queue into a flat NumPy ndarray.
/// - queue_to_numpy_stacked: Converts the piece queue into a stacked NumPy ndarray.
/// 
/// For specific dtype, the methods are named queue_to_numpy_{dtype}_flat and queue_to_numpy_{dtype}_stacked,
/// where {dtype} can be 'bool', 'int8', etc.
/// 
/// Engine Conversion Methods:
/// 
/// There are no special engine conversion methods in Game, but the engine attribute can be accessed
/// directly to perform conversions using HexEngine's methods via game.engine.to_numpy. The converted
/// ndarray will be an one-dimensional array representing the engine's state with the order of blocks according
/// to the linear indexing of HexEngine, which can be found in HexEngine documentation.
/// 
/// For specific dtype, the methods are named engine_to_numpy_{dtype}, where {dtype} can be 'bool', 'int8', etc.
/// 
/// See the "NumPy Integration" of HexEngine for more details on engine conversion methods.
/// 
/// Game Conversion Methods:
/// - game_to_numpy: Converts the entire game state (engine and queue) into a NumPy ndarray.
/// - game_from_numpy_with_radius: Creates a Game instance from a NumPy ndarray representation with explicit radius.
/// - game_from_numpy_with_queue_length: Creates a Game instance from a NumPy ndarray representation with explicit queue length.
/// 
/// There is no generic game_from_numpy method because the radius or queue length must be specified to
/// correctly interpret the ndarray representation.
/// 
/// For specific dtype, the methods are named game_to_numpy_{dtype}, game_from_numpy_with_radius_{dtype},
/// and game_from_numpy_with_queue_length_{dtype}, where {dtype} can be 'bool', 'int8', etc.
/// 
/// Accept a move as a NumPy ndarray mask or max tensor:
/// - move_with_numpy_mask_*: Accepts a move represented as a NumPy ndarray mask.
/// - move_with_numpy_max_*: Accepts a move represented as a NumPy ndarray max tensor.
/// 
/// For specific dtype, the methods are named move_with_numpy_mask_{dtype} and move_with_numpy_max_{dtype},
/// where {dtype} can be 'bool', 'int8', etc.
#[pyclass]
#[derive(Clone)]
pub struct Game {
    #[pyo3(get, set)]
    _Game__engine: Py<HexEngine>,
    #[pyo3(get, set)]
    _Game__queue: Vec<Piece>,
    #[pyo3(get, set)]
    _Game__score: u64,
    #[pyo3(get, set)]
    _Game__turn: u64,
    #[pyo3(get, set)]
    _Game__end: bool,
}

#[cfg(feature = "numpy")]
fn queue_to_numpy_flat_impl<'py, T>(
    py: Python<'py>,
    pieces: &Vec<Piece>,
) -> Py<PyArray1<T>>
where
    T: BitScalar + Copy + numpy::Element,
{
    let mut arr = Vec::with_capacity(pieces.len() * 7);
    for piece in pieces.iter() {
        for i in 0..7 {
            let b = if (piece.state & (1 << (6 - i))) != 0 { T::one() } else { T::zero() };
            arr.push(b);
        }
    }
    PyArray1::from_vec_bound(py, arr).unbind()
}

#[cfg(feature = "numpy")]
fn queue_to_numpy_stacked_impl<'py, T>(
    py: Python<'py>,
    pieces: &Vec<Piece>,
) -> Py<PyArray2<T>>
where
    T: BitScalar + Copy + numpy::Element,
{
    use ndarray::{Array2, ShapeBuilder};

    let n = pieces.len();
    let shape = (n, 7).strides((8, 1));

    let mut vec = Vec::with_capacity(n * 8);

    for piece in pieces {
        let s = piece.state;

        let v = [
            if s & 0b1000000 != 0 { T::one() } else { T::zero() },
            if s & 0b0100000 != 0 { T::one() } else { T::zero() },
            if s & 0b0010000 != 0 { T::one() } else { T::zero() },
            if s & 0b0001000 != 0 { T::one() } else { T::zero() },
            if s & 0b0000100 != 0 { T::one() } else { T::zero() },
            if s & 0b0000010 != 0 { T::one() } else { T::zero() },
            if s & 0b0000001 != 0 { T::one() } else { T::zero() },
        ];

        vec.extend_from_slice(&v);
        vec.push(T::zero()); // padding for stride
    }

    let array = Array2::from_shape_vec(shape, vec).unwrap();
    PyArray2::from_owned_array_bound(py, array).unbind()
}

#[allow(non_snake_case)]
impl Game {
    /* ---------------------------------------- NUMPY ---------------------------------------- */
    /// Get the NumPy ndarray boolean representation of the entire Game state.
    /// Because there is no way for engine and queue to have the same shape,
    /// the returned array is a 1D array representing the engine followed by the queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array representing the game engine followed by the queue.
    #[cfg(feature = "numpy")]
    #[allow(deprecated)]
    fn game_to_numpy_impl<'py, T>(
        &self,
        py: Python<'py>,
    ) -> Py<PyArray1<T>>
    where
        T: BitScalar + Copy + numpy::Element,
    {
        let engine = self._Game__engine.borrow(py);
        let engine_array: &PyArray1<T> = engine.to_numpy_engine_unboxed_impl::<T>(py);
        let engine_slice = unsafe { engine_array.as_slice().unwrap() };

        let mut arr = Vec::with_capacity(engine_slice.len() + self._Game__queue.len() * 7);
        arr.extend_from_slice(engine_slice);
        for piece in self._Game__queue.iter() {
            for i in 0..7 {
                let b = if (piece.state & (1 << (6 - i))) != 0 { T::one() } else { T::zero() };
                arr.push(b);
            }
        }
        PyArray1::from_vec_bound(py, arr).unbind()
    }

    /// Create a Game instance from a NumPy ndarray representation with explicit radius.
    /// The input array should represent the engine followed by the queue.
    /// 
    /// Arguments:
    /// - radius (usize): The radius of the hexagonal grid.
    /// - arr (numpy.ndarray): A 1D NumPy array representing the game engine followed by the queue.
    /// Returns:
    /// - Game: A new Game instance initialized from the provided NumPy array.
    #[cfg(feature = "numpy")]
    fn game_from_numpy_with_radius_impl<'py, T>(
        py: Python<'py>,
        radius: usize,
        arr: Bound<'_, PyArray1<T>>,
    ) -> PyResult<Py<Game>>
    where
        T: BitScalar + Copy + numpy::Element,
    {
        // Gather slice of length
        let slice = unsafe { arr.as_slice().unwrap() };
        // Check length
        let engine_length = 1 + 3 * radius * (radius - 1);
        let expected_length = engine_length + (slice.len() - engine_length) / 7 * 7;
        if slice.len() != expected_length {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Input array length {} does not match expected length {} for radius {}", slice.len(), expected_length, radius)));
        }
        let engine_slice = &slice[0..engine_length];
        let queue_length = (slice.len() - engine_length) / 7;
        let mut queue = Vec::with_capacity(queue_length);
        for i in 0..queue_length {
            let mut state: u8 = 0;
            for j in 0..7 {
                let val = slice[engine_length + i * 7 + j];
                if T::predicate(val) {
                    state |= 1 << (6 - j);
                }
            }
            queue.push(Piece { state });
        }
        let engine = HexEngine::from_numpy_engine_unboxed_explicit_radius_impl::<T>(engine_slice, radius)?;
        let game = Game {
            _Game__engine: Py::new(py, engine)?,
            _Game__queue: queue,
            _Game__score: 0,
            _Game__turn: 0,
            _Game__end: false,
        };
        Ok(Py::new(py, game)?.to_owned())
    }

    /// Create a Game instance from a NumPy ndarray representation with explicit queue length.
    /// The input array should represent the engine followed by the queue.
    /// 
    /// Arguments:
    /// - length (usize): The length of the piece queue.
    /// - arr (numpy.ndarray): A 1D NumPy array representing the game engine followed by the queue.
    /// Returns:
    /// - Game: A new Game instance initialized from the provided NumPy array.
    #[cfg(feature = "numpy")]
    fn game_from_numpy_with_queue_length_impl<'py, T>(
        py: Python<'py>,
        length: usize,
        arr: Bound<'_, PyArray1<T>>,
    ) -> PyResult<Py<Game>>
    where
        T: BitScalar + Copy + numpy::Element,
    {
        // Gather slice of length
        let slice = unsafe { arr.as_slice().unwrap() };
        // Check length
        let total_queue_length = length * 7;
        let engine_length = slice.len() - total_queue_length;
        let radius = match HexEngine::calc_radius(engine_length) {
            Some(r) => r,
            None => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!("Input array length {} does not correspond to a valid engine length for any radius", slice.len())));
            }
        };
        let engine_slice = &slice[0..engine_length];
        let mut queue = Vec::with_capacity(length);
        for i in 0..length {
            let mut state: u8 = 0;
            for j in 0..7 {
                let val = slice[engine_length + i * 7 + j];
                if T::predicate(val) {
                    state |= 1 << (6 - j);
                }
            }
            queue.push(Piece { state });
        }
        let engine = HexEngine::from_numpy_engine_unboxed_explicit_radius_impl::<T>(engine_slice, radius)?;
        let game = Game {
            _Game__engine: Py::new(py, engine)?,
            _Game__queue: queue,
            _Game__score: 0,
            _Game__turn: 0,
            _Game__end: false,
        };
        Ok(Py::new(py, game)?.to_owned())
    }

    /// Make a move in the game using a NumPy ndarray mask.
    /// The mask should be a 2D array where the first dimension corresponds to the piece index
    /// in the queue and the second dimension corresponds to the coordinate index on the engine.
    /// A value of true in the mask indicates the position where the piece should be placed.
    /// 
    /// The mask MUST be of shape (M, N), where M is the length of the queue and N is the length of the engine,
    /// and there MUST be exactly one true value in the entire mask. If there are multiple or no true values,
    /// an error will be raised.
    /// 
    /// Arguments:
    /// - array (numpy.ndarray): A 2D NumPy array representing the move mask.
    /// Returns:
    /// - bool: True if the move was successfully made, False otherwise.
    /// Raises:
    /// - ValueError: If the input array shape does not match the expected shape or if the mask does not contain exactly one true value,
    ///   or if the move is invalid due to collisions or out-of-bounds placement.
    #[cfg(feature = "numpy")]
    pub fn move_with_numpy_mask<T>(&mut self, py: Python, array: Bound<'_, PyArray2<T>>) -> PyResult<bool>
    where
        T: BitScalar + Copy + numpy::Element,
    {
        // Let's say the mask is of shape (M, N), we find the position (m, n) where the value is true
        // Then call hpyhex_rs_add_piece_with_index with m as the piece index and n as the coordinate index
        use numpy::PyUntypedArrayMethods;
        let slice = unsafe { array.as_slice().unwrap() };
        let shape = array.shape();
        // Validate shape
        let expected_queue_length = self._Game__queue.len();
        let expected_engine_length = {
            let engine = self._Game__engine.borrow(py);
            engine.__len__()
        };
        if shape[0] != expected_queue_length || shape[1] != expected_engine_length {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Input array shape {:?} does not match expected shape ({}, {})", shape, expected_queue_length, expected_engine_length)));
        }
        let mut piece_index: Option<usize> = None;
        let mut coord_index: Option<usize> = None;
        let mut found = false;
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                let val = slice[i * shape[1] + j];
                if T::predicate(val) {
                    if found {
                        return Err(pyo3::exceptions::PyValueError::new_err("Mask contains multiple true values; exactly one is required"));
                    }
                    found = true;
                    piece_index = Some(i);
                    coord_index = Some(j);
                    break;
                }
            }
            if piece_index.is_some() {
                break;
            }
        }
        if piece_index.is_none() || coord_index.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err("No valid move found in the mask"));
        }
        let piece_index = piece_index.unwrap();
        let coord_index = coord_index.unwrap();

        Self::hpyhex_rs_add_piece_with_index(self, py, piece_index, coord_index)
    }

    /// Make a move in the game by selecting the maximum value in a NumPy ndarray.
    /// The array should be a 2D array where the first dimension corresponds to the piece index
    /// in the queue and the second dimension corresponds to the coordinate index on the engine.
    /// The position of the maximum value in the array indicates where the piece should be placed.
    /// 
    /// The array MUST be of shape (M, N), where M is the length of the queue and N is the length of the engine.
    /// If there are multiple maximum values, the first occurrence will be used.
    /// 
    /// Arguments:
    /// - array (numpy.ndarray): A 2D NumPy array representing the move values.
    /// Returns:
    /// - bool: True if the move was successfully made, False otherwise.
    /// Raises:
    /// - ValueError: If the input array shape does not match the expected shape or if the move is invalid due to collisions or out-of-bounds placement.
    #[cfg(feature = "numpy")]
    pub fn move_with_numpy_max<T>(&mut self, py: Python, array: Bound<'_, PyArray2<T>>) -> PyResult<bool>
    where
        T: BitScalar + Copy + numpy::Element + PartialOrd,
    {
        // Find the maximum value in the array and its position
        use numpy::PyUntypedArrayMethods;
        let slice = unsafe { array.as_slice().unwrap() };
        let shape = array.shape();
        // Validate shape
        let expected_queue_length = self._Game__queue.len();
        let expected_engine_length = {
            let engine = self._Game__engine.borrow(py);
            engine.__len__()
        };
        if shape[0] != expected_queue_length || shape[1] != expected_engine_length {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Input array shape {:?} does not match expected shape ({}, {})", shape, expected_queue_length, expected_engine_length)));
        }
        let mut max_value: Option<T> = None;
        let mut piece_index: Option<usize> = None;
        let mut coord_index: Option<usize> = None;
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                let val = slice[i * shape[1] + j];
                if max_value.is_none() || val > max_value.unwrap() {
                    max_value = Some(val);
                    piece_index = Some(i);
                    coord_index = Some(j);
                }
            }
        }
        if piece_index.is_none() || coord_index.is_none() {
            return Err(pyo3::exceptions::PyValueError::new_err("Input array is empty"));
        }
        let piece_index = piece_index.unwrap();
        let coord_index = coord_index.unwrap();

        Self::hpyhex_rs_add_piece_with_index(self, py, piece_index, coord_index)
    }

    /* ---------------------------------------- HPYHEX PYTHON API ---------------------------------------- */

    /// Add a piece from the queue to the game engine at the specified coordinates.
    /// 
    /// This method updates the game state, including the score and turn number,
    /// and checks for eliminations and game end conditions.
    /// 
    /// Arguments:
    /// - python (Python): The Python interpreter context.
    /// - piece_index (usize): The index of the piece in the queue to add.
    /// - coord (Bound[Hex]): The coordinates where the piece should be added.
    /// Returns:
    /// - bool: True if the piece was successfully added, False otherwise.
    pub fn add_piece_checked(&mut self, python: Python, piece_index: usize, coord: &Bound<'_,Hex>) -> PyResult<bool> {
        // Check piece exists
        if piece_index >= self._Game__queue.len() {
            return Ok(false);
        }
        let piece = self._Game__queue[piece_index].clone();
        // Add piece to engine and increment score and turn
        let add_result = { self._Game__engine.bind(python) }.call_method1("add_piece", (coord, piece.clone()));
        if let Err(ref e) = add_result {
            if e.is_instance_of::<pyo3::exceptions::PyValueError>(python) {
                return Ok(false);
            } else {
                return Err(e.clone_ref(python));
            }
        }
        self._Game__score += piece.count() as u64;
        // Replace used piece
        let new_piece = PieceFactory::generate_piece()?;
        self._Game__queue[piece_index] = new_piece;
        let mut engine = self._Game__engine.borrow_mut(python);
        let eliminated = engine.eliminate()?;
        let eliminated_len = eliminated.len();
        self._Game__score += (eliminated_len as u64) * 5;
        self._Game__turn += 1;
        // Check whether the game has ended
        let mut has_move = false;
        for p in &self._Game__queue {
            if HexEngine::check_has_positions(&engine, p) {
                has_move = true;
                break;
            }
        }
        if !has_move {
            self._Game__end = true;
        }
        Ok(true)
    }
}

#[allow(non_snake_case)]
#[pymethods]
impl Game {
    /* ------------------------------------- HPYHEX-RS ------------------------------------- */

    /// Serialize the Game instance into a byte vector according to the format used by the Rust hpyhex-rs crate.
    /// 
    /// The serialization format is as follows:
    /// - First 4 bytes: Little-endian u32 representing the score.
    /// - Next 4 bytes: Little-endian u32 representing the turn number.
    /// - Next 4 bytes: Little-endian u32 representing the length of the piece queue.
    /// - Next 'length' bytes: Each byte represents a Piece in the queue. See `hpyhex_rs_serialize` method of Piece for details.
    /// - Remaining bytes: Binary representation of the HexEngine. See `hpyhex_rs_serialize` method of HexEngine for details.
    /// 
    /// Returns:
    /// - bytes: A byte vector containing the serialized state of the Game.
    pub fn hpyhex_rs_serialize<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
        let mut vec = Vec::<u8>::new();
        vec.extend_from_slice(&(self._Game__score as u32).to_le_bytes());
        vec.extend_from_slice(&(self._Game__turn as u32).to_le_bytes());
        vec.extend_from_slice(&(self._Game__queue.len() as u32).to_le_bytes());
        for piece in &self._Game__queue {
            vec.push(piece.state);
        }
        // Add engine binary representation
        let engine_vec: Vec<u8> = {(&*self._Game__engine.borrow(py)).into()};
        vec.extend_from_slice(&engine_vec);
        pyo3::types::PyBytes::new_bound(py, &vec)
    }

    /// Deserialize a byte vector into a Game instance according to the format used by the Rust hpyhex-rs crate.
    /// 
    /// The deserialization format is as follows:
    /// - First 4 bytes: Little-endian u32 representing the score.
    /// - Next 4 bytes: Little-endian u32 representing the turn number.
    /// - Next 4 bytes: Little-endian u32 representing the length of the piece queue.
    /// - Next 'length' bytes: Each byte represents a Piece in the queue. See `hpyhex_rs_deserialize` method of Piece for details.
    /// - Remaining bytes: Binary representation of the HexEngine. See `hpyhex_rs_deserialize` method of HexEngine for details.
    /// 
    /// Arguments:
    /// - data: A byte vector containing the serialized state of the Game.
    /// Returns:
    /// - Game: A Game instance reconstructed from the byte vector.
    #[staticmethod]
    pub fn hpyhex_rs_deserialize<'py>(py: Python<'py>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let bytes: &pyo3::types::PyBytes = data.extract()?;
        let slice = bytes.as_bytes();
        if slice.len() < 12 {
            return Err(pyo3::exceptions::PyValueError::new_err("Data too short to deserialize Game"));
        }
        let score = u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]) as u64;
        let turn = u32::from_le_bytes([slice[4], slice[5], slice[6], slice[7]]) as u64;
        let queue_length = u32::from_le_bytes([slice[8], slice[9], slice[10], slice[11]]) as usize;
        let expected_length = 12 + queue_length + slice.len() - 12 - queue_length;
        if slice.len() < expected_length {
            return Err(pyo3::exceptions::PyValueError::new_err("Data too short to deserialize Game with given queue length"));
        }
        let mut queue = Vec::with_capacity(queue_length);
        for i in 0..queue_length {
            let state = slice[12 + i];
            queue.push(Piece { state });
        }
        let engine_slice = &slice[12 + queue_length..];
        let engine = HexEngine::try_from(engine_slice)?;
        let game = Game {
            _Game__engine: Py::new(py, engine)?,
            _Game__queue: queue,
            _Game__score: score,
            _Game__turn: turn,
            _Game__end: false,
        };
        Ok(game)
    }

    /* ---------------------------------------- NUMPY ---------------------------------------- */
    /// Make a move in the game using a NumPy ndarray boolean mask.
    /// The mask should be a 2D array where the first dimension corresponds to the piece index
    /// in the queue and the second dimension corresponds to the coordinate index on the engine.
    /// A value of true in the mask indicates the position where the piece should be placed.
    /// 
    /// The mask MUST be of shape (M, N), where M is the length of the queue and N is the length of the engine,
    /// and there MUST be exactly one true value in the entire mask. If there are multiple or no true values,
    /// an error will be raised.
    /// 
    /// Arguments:
    /// - array (numpy.ndarray): A 2D NumPy array representing the move mask.
    /// Returns:
    /// - bool: True if the move was successfully made, False otherwise.
    /// Raises:
    /// - ValueError: If the input array shape does not match the expected shape or if the mask does not contain exactly one true value,
    ///   or if the move is invalid due to collisions or out-of-bounds placement.
    #[cfg(feature = "numpy")]
    pub fn move_with_numpy_mask_bool(&mut self, py: Python, array: Bound<'_, PyArray2<bool>>) -> PyResult<bool> {
        self.move_with_numpy_mask::<bool>(py, array)
    }

    /// Make a move in the game by selecting the maximum value in a NumPy bool ndarray.
    /// The array should be a 2D array where the first dimension corresponds to the piece index
    /// in the queue and the second dimension corresponds to the coordinate index on the engine.
    /// The position of the maximum value in the array indicates where the piece should be placed.
    /// 
    /// The array MUST be of shape (M, N), where M is the length of the queue and N is the length of the engine.
    /// If there are multiple maximum values, the first occurrence will be used.
    /// 
    /// Arguments:
    /// - array (numpy.ndarray): A 2D NumPy array representing the move values.
    /// Returns:
    /// - bool: True if the move was successfully made, False otherwise.
    /// Raises:
    /// - ValueError: If the input array shape does not match the expected shape or if the move is invalid due to collisions or out-of-bounds placement.
    #[cfg(feature = "numpy")]
    pub fn move_with_numpy_max_bool(&mut self, py: Python, array: Bound<'_, PyArray2<bool>>) -> PyResult<bool> {
        self.move_with_numpy_max::<bool>(py, array)
    }

    /// Make a move in the game using a NumPy ndarray int8 mask.
    /// The mask should be a 2D array where the first dimension corresponds to the piece index
    /// in the queue and the second dimension corresponds to the coordinate index on the engine.
    /// A non-zero value in the mask indicates the position where the piece should be placed.
    /// 
    /// The mask MUST be of shape (M, N), where M is the length of the queue and N is the length of the engine,
    /// and there MUST be exactly one non-zero value in the entire mask. If there are multiple or no non-zero values,
    /// an error will be raised.
    /// 
    /// Arguments:
    /// - array (numpy.ndarray): A 2D NumPy array representing the move mask.
    /// Returns:
    /// - bool: True if the move was successfully made, False otherwise.
    /// Raises:
    /// - ValueError: If the input array shape does not match the expected shape or if the mask does not contain exactly one non-zero value,
    ///   or if the move is invalid due to collisions or out-of-bounds placement.
    #[cfg(feature = "numpy")]
    pub fn move_with_numpy_mask_int8(&mut self, py: Python, array: Bound<'_, PyArray2<i8>>) -> PyResult<bool> {
        self.move_with_numpy_mask::<i8>(py, array)
    }

    /// Make a move in the game by selecting the maximum value in a NumPy int8 ndarray.
    /// The array should be a 2D array where the first dimension corresponds to the piece index
    /// in the queue and the second dimension corresponds to the coordinate index on the engine.
    /// The position of the maximum value in the array indicates where the piece should be placed.
    /// 
    /// The array MUST be of shape (M, N), where M is the length of the queue and N is the length of the engine.
    /// If there are multiple maximum values, the first occurrence will be used.
    /// 
    /// Arguments:
    /// - array (numpy.ndarray): A 2D NumPy array representing the move values.
    /// Returns:
    /// - bool: True if the move was successfully made, False otherwise.
    /// Raises:
    /// - ValueError: If the input array shape does not match the expected shape or if the move is invalid due to collisions or out-of-bounds placement.
    #[cfg(feature = "numpy")]
    pub fn move_with_numpy_max_int8(&mut self, py: Python, array: Bound<'_, PyArray2<i8>>) -> PyResult<bool> {
        self.move_with_numpy_max::<i8>(py, array)
    }

    /// Make a move in the game using a NumPy ndarray uint8 mask.
    /// The mask should be a 2D array where the first dimension corresponds to the piece index
    /// in the queue and the second dimension corresponds to the coordinate index on the engine.
    /// A non-zero value in the mask indicates the position where the piece should be placed.
    /// 
    /// The mask MUST be of shape (M, N), where M is the length of the queue and N is the length of the engine,
    /// and there MUST be exactly one non-zero value in the entire mask. If there are multiple or no non-zero values,
    /// an error will be raised.
    /// 
    /// Arguments:
    /// - array (numpy.ndarray): A 2D NumPy array representing the move mask.
    /// Returns:
    /// - bool: True if the move was successfully made, False otherwise.
    /// Raises:
    /// - ValueError: If the input array shape does not match the expected shape or if the mask does not contain exactly one non-zero value,
    ///   or if the move is invalid due to collisions or out-of-bounds placement.
    #[cfg(feature = "numpy")]
    pub fn move_with_numpy_mask_uint8(&mut self, py: Python, array: Bound<'_, PyArray2<u8>>) -> PyResult<bool> {
        self.move_with_numpy_mask::<u8>(py, array)
    }

    /// Make a move in the game by selecting the maximum value in a NumPy uint8 ndarray.
    /// The array should be a 2D array where the first dimension corresponds to the piece index
    /// in the queue and the second dimension corresponds to the coordinate index on the engine.
    /// The position of the maximum value in the array indicates where the piece should be placed.
    /// 
    /// The array MUST be of shape (M, N), where M is the length of the queue and N is the length of the engine.
    /// If there are multiple maximum values, the first occurrence will be used.
    /// 
    /// Arguments:
    /// - array (numpy.ndarray): A 2D NumPy array representing the move values.
    /// Returns:
    /// - bool: True if the move was successfully made, False otherwise.
    /// Raises:
    /// - ValueError: If the input array shape does not match the expected shape or if the move is invalid due to collisions or out-of-bounds placement.
    #[cfg(feature = "numpy")]
    pub fn move_with_numpy_max_uint8(&mut self, py: Python, array: Bound<'_, PyArray2<u8>>) -> PyResult<bool> {
        self.move_with_numpy_max::<u8>(py, array)
    }

    /// Make a move in the game using a NumPy ndarray int16 mask.
    /// The mask should be a 2D array where the first dimension corresponds to the piece index
    /// in the queue and the second dimension corresponds to the coordinate index on the engine.
    /// A non-zero value in the mask indicates the position where the piece should be placed.
    /// 
    /// The mask MUST be of shape (M, N), where M is the length of the queue and N is the length of the engine,
    /// and there MUST be exactly one non-zero value in the entire mask. If there are multiple or no non-zero values,
    /// an error will be raised.
    /// 
    /// Arguments:
    /// - array (numpy.ndarray): A 2D NumPy array representing the move mask.
    /// Returns:
    /// - bool: True if the move was successfully made, False otherwise.
    /// Raises:
    /// - ValueError: If the input array shape does not match the expected shape or if the mask does not contain exactly one non-zero value,
    ///   or if the move is invalid due to collisions or out-of-bounds placement.
    #[cfg(feature = "numpy")]
    pub fn move_with_numpy_mask_int16(&mut self, py: Python, array: Bound<'_, PyArray2<i16>>) -> PyResult<bool> {
        self.move_with_numpy_mask::<i16>(py, array)
    }

    /// Make a move in the game by selecting the maximum value in a NumPy int16 ndarray.
    /// The array should be a 2D array where the first dimension corresponds to the piece index
    /// in the queue and the second dimension corresponds to the coordinate index on the engine.
    /// The position of the maximum value in the array indicates where the piece should be placed.
    /// 
    /// The array MUST be of shape (M, N), where M is the length of the queue and N is the length of the engine.
    /// If there are multiple maximum values, the first occurrence will be used.
    /// 
    /// Arguments:
    /// - array (numpy.ndarray): A 2D NumPy array representing the move values.
    /// Returns:
    /// - bool: True if the move was successfully made, False otherwise.
    /// Raises:
    /// - ValueError: If the input array shape does not match the expected shape or if the move is invalid due to collisions or out-of-bounds placement.
    #[cfg(feature = "numpy")]
    pub fn move_with_numpy_max_int16(&mut self, py: Python, array: Bound<'_, PyArray2<i16>>) -> PyResult<bool> {
        self.move_with_numpy_max::<i16>(py, array)
    }

    /// Make a move in the game using a NumPy ndarray uint16 mask.
    /// The mask should be a 2D array where the first dimension corresponds to the piece index
    /// in the queue and the second dimension corresponds to the coordinate index on the engine.
    /// A non-zero value in the mask indicates the position where the piece should be placed.
    /// 
    /// The mask MUST be of shape (M, N), where M is the length of the queue and N is the length of the engine,
    /// and there MUST be exactly one non-zero value in the entire mask. If there are multiple or no non-zero values,
    /// an error will be raised.
    /// 
    /// Arguments:
    /// - array (numpy.ndarray): A 2D NumPy array representing the move mask.
    /// Returns:
    /// - bool: True if the move was successfully made, False otherwise.
    /// Raises:
    /// - ValueError: If the input array shape does not match the expected shape or if the mask does not contain exactly one non-zero value,
    ///   or if the move is invalid due to collisions or out-of-bounds placement.
    #[cfg(feature = "numpy")]
    pub fn move_with_numpy_mask_uint16(&mut self, py: Python, array: Bound<'_, PyArray2<u16>>) -> PyResult<bool> {
        self.move_with_numpy_mask::<u16>(py, array)
    }

    /// Make a move in the game by selecting the maximum value in a NumPy uint16 ndarray.
    /// The array should be a 2D array where the first dimension corresponds to the piece index
    /// in the queue and the second dimension corresponds to the coordinate index on the engine.
    /// The position of the maximum value in the array indicates where the piece should be placed.
    /// 
    /// The array MUST be of shape (M, N), where M is the length of the queue and N is the length of the engine.
    /// If there are multiple maximum values, the first occurrence will be used.
    /// 
    /// Arguments:
    /// - array (numpy.ndarray): A 2D NumPy array representing the move values.
    /// Returns:
    /// - bool: True if the move was successfully made, False otherwise.
    /// Raises:
    /// - ValueError: If the input array shape does not match the expected shape or if the move is invalid due to collisions or out-of-bounds placement.
    #[cfg(feature = "numpy")]
    pub fn move_with_numpy_max_uint16(&mut self, py: Python, array: Bound<'_, PyArray2<u16>>) -> PyResult<bool> {
        self.move_with_numpy_max::<u16>(py, array)
    }

    /// Make a move in the game using a NumPy ndarray int32 mask.
    /// The mask should be a 2D array where the first dimension corresponds to the piece index
    /// in the queue and the second dimension corresponds to the coordinate index on the engine.
    /// A non-zero value in the mask indicates the position where the piece should be placed.
    /// 
    /// The mask MUST be of shape (M, N), where M is the length of the queue and N is the length of the engine,
    /// and there MUST be exactly one non-zero value in the entire mask. If there are multiple or no non-zero values,
    /// an error will be raised.
    /// 
    /// Arguments:
    /// - array (numpy.ndarray): A 2D NumPy array representing the move mask.
    /// Returns:
    /// - bool: True if the move was successfully made, False otherwise.
    /// Raises:
    /// - ValueError: If the input array shape does not match the expected shape or if the mask does not contain exactly one non-zero value,
    ///   or if the move is invalid due to collisions or out-of-bounds placement.
    #[cfg(feature = "numpy")]
    pub fn move_with_numpy_mask_int32(&mut self, py: Python, array: Bound<'_, PyArray2<i32>>) -> PyResult<bool> {
        self.move_with_numpy_mask::<i32>(py, array)
    }

    /// Make a move in the game by selecting the maximum value in a NumPy int32 ndarray.
    /// The array should be a 2D array where the first dimension corresponds to the piece index
    /// in the queue and the second dimension corresponds to the coordinate index on the engine.
    /// The position of the maximum value in the array indicates where the piece should be placed.
    /// 
    /// The array MUST be of shape (M, N), where M is the length of the queue and N is the length of the engine.
    /// If there are multiple maximum values, the first occurrence will be used.
    /// 
    /// Arguments:
    /// - array (numpy.ndarray): A 2D NumPy array representing the move values.
    /// Returns:
    /// - bool: True if the move was successfully made, False otherwise.
    /// Raises:
    /// - ValueError: If the input array shape does not match the expected shape or if the move is invalid due to collisions or out-of-bounds placement.
    #[cfg(feature = "numpy")]
    pub fn move_with_numpy_max_int32(&mut self, py: Python, array: Bound<'_, PyArray2<i32>>) -> PyResult<bool> {
        self.move_with_numpy_max::<i32>(py, array)
    }

    /// Make a move in the game using a NumPy ndarray uint32 mask.
    /// The mask should be a 2D array where the first dimension corresponds to the piece index
    /// in the queue and the second dimension corresponds to the coordinate index on the engine.
    /// A non-zero value in the mask indicates the position where the piece should be placed.
    /// 
    /// The mask MUST be of shape (M, N), where M is the length of the queue and N is the length of the engine,
    /// and there MUST be exactly one non-zero value in the entire mask. If there are multiple or no non-zero values,
    /// an error will be raised.
    /// 
    /// Arguments:
    /// - array (numpy.ndarray): A 2D NumPy array representing the move mask.
    /// Returns:
    /// - bool: True if the move was successfully made, False otherwise.
    /// Raises:
    /// - ValueError: If the input array shape does not match the expected shape or if the mask does not contain exactly one non-zero value,
    ///   or if the move is invalid due to collisions or out-of-bounds placement.
    #[cfg(feature = "numpy")]
    pub fn move_with_numpy_mask_uint32(&mut self, py: Python, array: Bound<'_, PyArray2<u32>>) -> PyResult<bool> {
        self.move_with_numpy_mask::<u32>(py, array)
    }

    /// Make a move in the game by selecting the maximum value in a NumPy uint32 ndarray.
    /// The array should be a 2D array where the first dimension corresponds to the piece index
    /// in the queue and the second dimension corresponds to the coordinate index on the engine.
    /// The position of the maximum value in the array indicates where the piece should be placed.
    /// 
    /// The array MUST be of shape (M, N), where M is the length of the queue and N is the length of the engine.
    /// If there are multiple maximum values, the first occurrence will be used.
    /// 
    /// Arguments:
    /// - array (numpy.ndarray): A 2D NumPy array representing the move values.
    /// Returns:
    /// - bool: True if the move was successfully made, False otherwise.
    /// Raises:
    /// - ValueError: If the input array shape does not match the expected shape or if the move is invalid due to collisions or out-of-bounds placement.
    #[cfg(feature = "numpy")]
    pub fn move_with_numpy_max_uint32(&mut self, py: Python, array: Bound<'_, PyArray2<u32>>) -> PyResult<bool> {
        self.move_with_numpy_max::<u32>(py, array)
    }

    /// Make a move in the game using a NumPy ndarray float16 mask.
    /// The mask should be a 2D array where the first dimension corresponds to the piece index
    /// in the queue and the second dimension corresponds to the coordinate index on the engine.
    /// A non-zero value in the mask indicates the position where the piece should be placed.
    /// 
    /// The mask MUST be of shape (M, N), where M is the length of the queue and N is the length of the engine,
    /// and there MUST be exactly one non-zero value in the entire mask. If there are multiple or no non-zero values,
    /// an error will be raised.
    /// 
    /// Arguments:
    /// - array (numpy.ndarray): A 2D NumPy array representing the move mask.
    /// Returns:
    /// - bool: True if the move was successfully made, False otherwise.
    /// Raises:
    /// - ValueError: If the input array shape does not match the expected shape or if the mask does not contain exactly one non-zero value,
    ///   or if the move is invalid due to collisions or out-of-bounds placement.
    /// Warning:
    /// - The 'half' feature, which add support for float16, is still experimental and may not be stable. On machines that does
    /// not support float16 or installed with a version of numpy that does not support float16, this function may lead to
    /// undefined behavior or crashes. Testing show that on some systems, this can result in memory misinterpretation issues
    /// causing incorrect values to be read, and on other systems, it cause the entire program to halt but not crash.
    /// Use with caution.
    #[cfg(all(feature = "numpy", feature = "half"))]
    pub fn move_with_numpy_mask_float16(&mut self, py: Python, array: Bound<'_, PyArray2<F16>>) -> PyResult<bool> {
        self.move_with_numpy_mask::<F16>(py, array)
    }

    /// Make a move in the game by selecting the maximum value in a NumPy float16 ndarray.
    /// The array should be a 2D array where the first dimension corresponds to the piece index
    /// in the queue and the second dimension corresponds to the coordinate index on the engine.
    /// The position of the maximum value in the array indicates where the piece should be placed.
    /// 
    /// The array MUST be of shape (M, N), where M is the length of the queue and N is the length of the engine.
    /// If there are multiple maximum values, the first occurrence will be used.
    /// 
    /// Arguments:
    /// - array (numpy.ndarray): A 2D NumPy array representing the move values.
    /// Returns:
    /// - bool: True if the move was successfully made, False otherwise.
    /// Raises:
    /// - ValueError: If the input array shape does not match the expected shape or if the move is invalid due to collisions or out-of-bounds placement.
    /// Warning:
    /// - The 'half' feature, which add support for float16, is still experimental and may not be stable. On machines that does
    /// not support float16 or installed with a version of numpy that does not support float16, this function may lead to
    /// undefined behavior or crashes. Testing show that on some systems, this can result in memory misinterpretation issues
    /// causing incorrect values to be read, and on other systems, it cause the entire program to halt but not crash.
    /// Use with caution.
    #[cfg(all(feature = "numpy", feature = "half"))]
    pub fn move_with_numpy_max_float16(&mut self, py: Python, array: Bound<'_, PyArray2<F16>>) -> PyResult<bool> {
        self.move_with_numpy_max::<F16>(py, array)
    }

    /// Make a move in the game using a NumPy ndarray float32 mask.
    /// The mask should be a 2D array where the first dimension corresponds to the piece index
    /// in the queue and the second dimension corresponds to the coordinate index on the engine.
    /// A non-zero value in the mask indicates the position where the piece should be placed.
    /// 
    /// The mask MUST be of shape (M, N), where M is the length of the queue and N is the length of the engine,
    /// and there MUST be exactly one non-zero value in the entire mask. If there are multiple or no non-zero values,
    /// an error will be raised.
    /// 
    /// Arguments:
    /// - array (numpy.ndarray): A 2D NumPy array representing the move mask.
    /// Returns:
    /// - bool: True if the move was successfully made, False otherwise.
    /// Raises:
    /// - ValueError: If the input array shape does not match the expected shape or if the mask does not contain exactly one non-zero value,
    ///   or if the move is invalid due to collisions or out-of-bounds placement.
    #[cfg(feature = "numpy")]
    pub fn move_with_numpy_mask_float32(&mut self, py: Python, array: Bound<'_, PyArray2<f32>>) -> PyResult<bool> {
        self.move_with_numpy_mask::<f32>(py, array)
    }

    /// Make a move in the game by selecting the maximum value in a NumPy float32 ndarray.
    /// The array should be a 2D array where the first dimension corresponds to the piece index
    /// in the queue and the second dimension corresponds to the coordinate index on the engine.
    /// The position of the maximum value in the array indicates where the piece should be placed.
    /// 
    /// The array MUST be of shape (M, N), where M is the length of the queue and N is the length of the engine.
    /// If there are multiple maximum values, the first occurrence will be used.
    /// 
    /// Arguments:
    /// - array (numpy.ndarray): A 2D NumPy array representing the move values.
    /// Returns:
    /// - bool: True if the move was successfully made, False otherwise.
    /// Raises:
    /// - ValueError: If the input array shape does not match the expected shape or if the move is invalid due to collisions or out-of-bounds placement.
    #[cfg(feature = "numpy")]
    pub fn move_with_numpy_max_float32(&mut self, py: Python, array: Bound<'_, PyArray2<f32>>) -> PyResult<bool> {
        self.move_with_numpy_max::<f32>(py, array)
    }

    /// Make a move in the game using a NumPy ndarray float64 mask.
    /// The mask should be a 2D array where the first dimension corresponds to the piece index
    /// in the queue and the second dimension corresponds to the coordinate index on the engine.
    /// A non-zero value in the mask indicates the position where the piece should be placed.
    /// 
    /// The mask MUST be of shape (M, N), where M is the length of the queue and N is the length of the engine,
    /// and there MUST be exactly one non-zero value in the entire mask. If there are multiple or no non-zero values,
    /// an error will be raised.
    /// 
    /// Arguments:
    /// - array (numpy.ndarray): A 2D NumPy array representing the move mask.
    /// Returns:
    /// - bool: True if the move was successfully made, False otherwise.
    /// Raises:
    /// - ValueError: If the input array shape does not match the expected shape or if the mask does not contain exactly one non-zero value,
    ///   or if the move is invalid due to collisions or out-of-bounds placement.
    #[cfg(feature = "numpy")]
    pub fn move_with_numpy_mask_float64(&mut self, py: Python, array: Bound<'_, PyArray2<f64>>) -> PyResult<bool> {
        self.move_with_numpy_mask::<f64>(py, array)
    }

    /// Make a move in the game by selecting the maximum value in a NumPy float64 ndarray.
    /// The array should be a 2D array where the first dimension corresponds to the piece index
    /// in the queue and the second dimension corresponds to the coordinate index on the engine.
    /// The position of the maximum value in the array indicates where the piece should be placed.
    /// 
    /// The array MUST be of shape (M, N), where M is the length of the queue and N is the length of the engine.
    /// If there are multiple maximum values, the first occurrence will be used.
    /// 
    /// Arguments:
    /// - array (numpy.ndarray): A 2D NumPy array representing the move values.
    /// Returns:
    /// - bool: True if the move was successfully made, False otherwise.
    /// Raises:
    /// - ValueError: If the input array shape does not match the expected shape or if the move is invalid due to collisions or out-of-bounds placement.
    #[cfg(feature = "numpy")]
    pub fn move_with_numpy_max_float64(&mut self, py: Python, array: Bound<'_, PyArray2<f64>>) -> PyResult<bool> {
        self.move_with_numpy_max::<f64>(py, array)
    }

    
    /// Get the NumPy ndarray boolean representation of the entire Game state.
    /// Because there is no way for engine and queue to have the same shape,
    /// the returned array is a 1D array representing the engine followed by the queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array representing the game engine followed by the queue.
    #[cfg(feature = "numpy")]
    pub fn to_numpy<'py>(&self, py: Python<'py>) -> Py<PyArray1<bool>> {
        self.game_to_numpy_impl::<bool>(py)
    }

    /// Get the NumPy ndarray boolean representation of the entire Game state.
    /// Because there is no way for engine and queue to have the same shape,
    /// the returned array is a 1D array representing the engine followed by the queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array representing the game engine followed by the queue.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_bool<'py>(&self, py: Python<'py>) -> Py<PyArray1<bool>> {
        self.game_to_numpy_impl::<bool>(py)
    }

    /// Get the NumPy ndarray int8 representation of the entire Game state.
    /// Because there is no way for engine and queue to have the same shape,
    /// the returned array is a 1D array representing the engine followed by the queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array representing the game engine followed by the queue.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_int8<'py>(&self, py: Python<'py>) -> Py<PyArray1<i8>> {
        self.game_to_numpy_impl::<i8>(py)
    }

    /// Get the NumPy ndarray uint8 representation of the entire Game state.
    /// Because there is no way for engine and queue to have the same shape,
    /// the returned array is a 1D array representing the engine followed by the queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array representing the game engine followed by the queue.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_uint8<'py>(&self, py: Python<'py>) -> Py<PyArray1<u8>> {
        self.game_to_numpy_impl::<u8>(py)
    }

    /// Get the NumPy ndarray int16 representation of the entire Game state.
    /// Because there is no way for engine and queue to have the same shape,
    /// the returned array is a 1D array representing the engine followed by the queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array representing the game engine followed by the queue.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_int16<'py>(&self, py: Python<'py>) -> Py<PyArray1<i16>> {
        self.game_to_numpy_impl::<i16>(py)
    }

    /// Get the NumPy ndarray uint16 representation of the entire Game state.
    /// Because there is no way for engine and queue to have the same shape,
    /// the returned array is a 1D array representing the engine followed by the queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array representing the game engine followed by the queue.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_uint16<'py>(&self, py: Python<'py>) -> Py<PyArray1<u16>> {
        self.game_to_numpy_impl::<u16>(py)
    }

    /// Get the NumPy ndarray int32 representation of the entire Game state.
    /// Because there is no way for engine and queue to have the same shape,
    /// the returned array is a 1D array representing the engine followed by the queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array representing the game engine followed by the queue.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_int32<'py>(&self, py: Python<'py>) -> Py<PyArray1<i32>> {
        self.game_to_numpy_impl::<i32>(py)
    }

    /// Get the NumPy ndarray uint32 representation of the entire Game state.
    /// Because there is no way for engine and queue to have the same shape,
    /// the returned array is a 1D array representing the engine followed by the queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array representing the game engine followed by the queue.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_uint32<'py>(&self, py: Python<'py>) -> Py<PyArray1<u32>> {
        self.game_to_numpy_impl::<u32>(py)
    }

    /// Get the NumPy ndarray int64 representation of the entire Game state.
    /// Because there is no way for engine and queue to have the same shape,
    /// the returned array is a 1D array representing the engine followed by the queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array representing the game engine followed by the queue.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_int64<'py>(&self, py: Python<'py>) -> Py<PyArray1<i64>> {
        self.game_to_numpy_impl::<i64>(py)
    }

    /// Get the NumPy ndarray uint64 representation of the entire Game state.
    /// Because there is no way for engine and queue to have the same shape,
    /// the returned array is a 1D array representing the engine followed by the queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array representing the game engine followed by the queue.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_uint64<'py>(&self, py: Python<'py>) -> Py<PyArray1<u64>> {
        self.game_to_numpy_impl::<u64>(py)
    }

    /// Get the NumPy ndarray float32 representation of the entire Game state.
    /// Because there is no way for engine and queue to have the same shape,
    /// the returned array is a 1D array representing the engine followed by the queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array representing the game engine followed by the queue.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_float32<'py>(&self, py: Python<'py>) -> Py<PyArray1<f32>> {
        self.game_to_numpy_impl::<f32>(py)
    }

    /// Get the NumPy ndarray float64 representation of the entire Game state.
    /// Because there is no way for engine and queue to have the same shape,
    /// the returned array is a 1D array representing the engine followed by the queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array representing the game engine followed by the queue.
    #[cfg(feature = "numpy")]
    pub fn to_numpy_float64<'py>(&self, py: Python<'py>) -> Py<PyArray1<f64>> {
        self.game_to_numpy_impl::<f64>(py)
    }

    /// Get the NumPy ndarray float16 representation of the entire Game state.
    /// Because there is no way for engine and queue to have the same shape,
    /// the returned array is a 1D NumPy array representing the game engine followed by
    /// the queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array representing the game engine followed by the queue.
    /// Warning:
    /// - The 'half' feature, which add support for float16, is still experimental and may not be stable. On machines that does
    /// not support float16 or installed with a version of numpy that does not support float16, this function may lead to
    /// undefined behavior or crashes. Testing show that on some systems, this can result in memory misinterpretation issues
    /// causing incorrect values to be read, and on other systems, it cause the entire program to halt but not crash.
    /// Use with caution.
    #[cfg(all(feature = "numpy", feature = "half"))]
    pub fn to_numpy_float16<'py>(&self, py: Python<'py>) -> Py<PyArray1<F16>> {
        self.game_to_numpy_impl::<F16>(py)
    }

    /// Create a Game instance from a NumPy ndarray bool representation with explicit queue length.
    /// The input array should represent the engine followed by the queue.
    /// 
    /// Arguments:
    /// - length (usize): The length of the piece queue.
    /// - arr (numpy.ndarray): A 1D NumPy array representing the game engine followed by the queue.
    /// Returns:
    /// - Game: A new Game instance initialized from the provided NumPy array.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_with_queue_length_bool<'py>(py: Python<'py>, length: usize, arr: Bound<'_, PyArray1<bool>>) -> PyResult<Py<Game>>{
        Self::game_from_numpy_with_queue_length_impl::<bool>(py, length, arr)
    }

    /// Create a Game instance from a NumPy ndarray bool representation with explicit radius.
    /// The input array should represent the engine followed by the queue.
    /// 
    /// Arguments:
    /// - radius (usize): The radius of the hexagonal grid.
    /// - arr (numpy.ndarray): A 1D NumPy array representing the game engine followed by the queue.
    /// Returns:
    /// - Game: A new Game instance initialized from the provided NumPy array.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_with_radius_bool<'py>(py: Python<'py>, radius: usize, arr: Bound<'_, PyArray1<bool>>) -> PyResult<Py<Game>>{
        Self::game_from_numpy_with_radius_impl::<bool>(py, radius, arr)
    }

    /// Create a Game instance from a NumPy ndarray int8 representation with explicit queue length.
    /// The input array should represent the engine followed by the queue.
    /// 
    /// Arguments:
    /// - length (usize): The length of the piece queue.
    /// - arr (numpy.ndarray): A 1D NumPy array representing the game engine followed by the queue.
    /// Returns:
    /// - Game: A new Game instance initialized from the provided NumPy array.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_with_queue_length_int8<'py>(py: Python<'py>, length: usize, arr: Bound<'_, PyArray1<i8>>) -> PyResult<Py<Game>>{
        Self::game_from_numpy_with_queue_length_impl::<i8>(py, length, arr)
    }

    /// Create a Game instance from a NumPy ndarray int8 representation with explicit radius.
    /// The input array should represent the engine followed by the queue.
    /// 
    /// Arguments:
    /// - radius (usize): The radius of the hexagonal grid.
    /// - arr (numpy.ndarray): A 1D NumPy array representing the game engine followed by the queue.
    /// Returns:
    /// - Game: A new Game instance initialized from the provided NumPy array.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_with_radius_int8<'py>(py: Python<'py>, radius: usize, arr: Bound<'_, PyArray1<i8>>) -> PyResult<Py<Game>>{
        Self::game_from_numpy_with_radius_impl::<i8>(py, radius, arr)
    }

    /// Create a Game instance from a NumPy ndarray uint8 representation with explicit queue length.
    /// The input array should represent the engine followed by the queue.
    /// 
    /// Arguments:
    /// - length (usize): The length of the piece queue.
    /// - arr (numpy.ndarray): A 1D NumPy array representing the game engine followed by the queue.
    /// Returns:
    /// - Game: A new Game instance initialized from the provided NumPy array.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_with_queue_length_uint8<'py>(py: Python<'py>, length: usize, arr: Bound<'_, PyArray1<u8>>) -> PyResult<Py<Game>>{
        Self::game_from_numpy_with_queue_length_impl::<u8>(py, length, arr)
    }

    /// Create a Game instance from a NumPy ndarray uint8 representation with explicit radius.
    /// The input array should represent the engine followed by the queue.
    /// 
    /// Arguments:
    /// - radius (usize): The radius of the hexagonal grid.
    /// - arr (numpy.ndarray): A 1D NumPy array representing the game engine followed by the queue.
    /// Returns:
    /// - Game: A new Game instance initialized from the provided NumPy array.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_with_radius_uint8<'py>(py: Python<'py>, radius: usize, arr: Bound<'_, PyArray1<u8>>) -> PyResult<Py<Game>>{
        Self::game_from_numpy_with_radius_impl::<u8>(py, radius, arr)
    }

    /// Create a Game instance from a NumPy ndarray int16 representation with explicit queue length.
    /// The input array should represent the engine followed by the queue.
    /// 
    /// Arguments:
    /// - length (usize): The length of the piece queue.
    /// - arr (numpy.ndarray): A 1D NumPy array representing the game engine followed by the queue.
    /// Returns:
    /// - Game: A new Game instance initialized from the provided NumPy array.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_with_queue_length_int16<'py>(py: Python<'py>, length: usize, arr: Bound<'_, PyArray1<i16>>) -> PyResult<Py<Game>>{
        Self::game_from_numpy_with_queue_length_impl::<i16>(py, length, arr)
    }

    /// Create a Game instance from a NumPy ndarray int16 representation with explicit radius.
    /// The input array should represent the engine followed by the queue.
    /// 
    /// Arguments:
    /// - radius (usize): The radius of the hexagonal grid.
    /// - arr (numpy.ndarray): A 1D NumPy array representing the game engine followed by the queue.
    /// Returns:
    /// - Game: A new Game instance initialized from the provided NumPy array.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_with_radius_int16<'py>(py: Python<'py>, radius: usize, arr: Bound<'_, PyArray1<i16>>) -> PyResult<Py<Game>>{
        Self::game_from_numpy_with_radius_impl::<i16>(py, radius, arr)
    }

    /// Create a Game instance from a NumPy ndarray uint16 representation with explicit queue length.
    /// The input array should represent the engine followed by the queue.
    /// 
    /// Arguments:
    /// - length (usize): The length of the piece queue.
    /// - arr (numpy.ndarray): A 1D NumPy array representing the game engine followed by the queue.
    /// Returns:
    /// - Game: A new Game instance initialized from the provided NumPy array.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_with_queue_length_uint16<'py>(py: Python<'py>, length: usize, arr: Bound<'_, PyArray1<u16>>) -> PyResult<Py<Game>>{
        Self::game_from_numpy_with_queue_length_impl::<u16>(py, length, arr)
    }

    /// Create a Game instance from a NumPy ndarray uint16 representation with explicit radius.
    /// The input array should represent the engine followed by the queue.
    /// 
    /// Arguments:
    /// - radius (usize): The radius of the hexagonal grid.
    /// - arr (numpy.ndarray): A 1D NumPy array representing the game engine followed by the queue.
    /// Returns:
    /// - Game: A new Game instance initialized from the provided NumPy array.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_with_radius_uint16<'py>(py: Python<'py>, radius: usize, arr: Bound<'_, PyArray1<u16>>) -> PyResult<Py<Game>>{
        Self::game_from_numpy_with_radius_impl::<u16>(py, radius, arr)
    }

    /// Create a Game instance from a NumPy ndarray int32 representation with explicit queue length.
    /// The input array should represent the engine followed by the queue.
    /// 
    /// Arguments:
    /// - length (usize): The length of the piece queue.
    /// - arr (numpy.ndarray): A 1D NumPy array representing the game engine followed by the queue.
    /// Returns:
    /// - Game: A new Game instance initialized from the provided NumPy array.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_with_queue_length_int32<'py>(py: Python<'py>, length: usize, arr: Bound<'_, PyArray1<i32>>) -> PyResult<Py<Game>>{
        Self::game_from_numpy_with_queue_length_impl::<i32>(py, length, arr)
    }

    /// Create a Game instance from a NumPy ndarray int32 representation with explicit radius.
    /// The input array should represent the engine followed by the queue.
    /// 
    /// Arguments:
    /// - radius (usize): The radius of the hexagonal grid.
    /// - arr (numpy.ndarray): A 1D NumPy array representing the game engine followed by the queue.
    /// Returns:
    /// - Game: A new Game instance initialized from the provided NumPy array.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_with_radius_int32<'py>(py: Python<'py>, radius: usize, arr: Bound<'_, PyArray1<i32>>) -> PyResult<Py<Game>>{
        Self::game_from_numpy_with_radius_impl::<i32>(py, radius, arr)
    }

    /// Create a Game instance from a NumPy ndarray uint32 representation with explicit queue length.
    /// The input array should represent the engine followed by the queue.
    /// 
    /// Arguments:
    /// - length (usize): The length of the piece queue.
    /// - arr (numpy.ndarray): A 1D NumPy array representing the game engine followed by the queue.
    /// Returns:
    /// - Game: A new Game instance initialized from the provided NumPy array.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_with_queue_length_uint32<'py>(py: Python<'py>, length: usize, arr: Bound<'_, PyArray1<u32>>) -> PyResult<Py<Game>>{
        Self::game_from_numpy_with_queue_length_impl::<u32>(py, length, arr)
    }

    /// Create a Game instance from a NumPy ndarray uint32 representation with explicit radius.
    /// The input array should represent the engine followed by the queue.
    /// 
    /// Arguments:
    /// - radius (usize): The radius of the hexagonal grid.
    /// - arr (numpy.ndarray): A 1D NumPy array representing the game engine followed by the queue.
    /// Returns:
    /// - Game: A new Game instance initialized from the provided NumPy array.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_with_radius_uint32<'py>(py: Python<'py>, radius: usize, arr: Bound<'_, PyArray1<u32>>) -> PyResult<Py<Game>>{
        Self::game_from_numpy_with_radius_impl::<u32>(py, radius, arr)
    }

    /// Create a Game instance from a NumPy ndarray int64 representation with explicit queue length.
    /// The input array should represent the engine followed by the queue.
    /// 
    /// Arguments:
    /// - length (usize): The length of the piece queue.
    /// - arr (numpy.ndarray): A 1D NumPy array representing the game engine followed by the queue.
    /// Returns:
    /// - Game: A new Game instance initialized from the provided NumPy array.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_with_queue_length_int64<'py>(py: Python<'py>, length: usize, arr: Bound<'_, PyArray1<i64>>) -> PyResult<Py<Game>>{
        Self::game_from_numpy_with_queue_length_impl::<i64>(py, length, arr)
    }

    /// Create a Game instance from a NumPy ndarray int64 representation with explicit radius.
    /// The input array should represent the engine followed by the queue.
    /// 
    /// Arguments:
    /// - radius (usize): The radius of the hexagonal grid.
    /// - arr (numpy.ndarray): A 1D NumPy array representing the game engine followed by the queue.
    /// Returns:
    /// - Game: A new Game instance initialized from the provided NumPy array.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_with_radius_int64<'py>(py: Python<'py>, radius: usize, arr: Bound<'_, PyArray1<i64>>) -> PyResult<Py<Game>>{
        Self::game_from_numpy_with_radius_impl::<i64>(py, radius, arr)
    }

    /// Create a Game instance from a NumPy ndarray uint64 representation with explicit queue length.
    /// The input array should represent the engine followed by the queue.
    /// 
    /// Arguments:
    /// - length (usize): The length of the piece queue.
    /// - arr (numpy.ndarray): A 1D NumPy array representing the game engine followed by the queue.
    /// Returns:
    /// - Game: A new Game instance initialized from the provided NumPy array.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_with_queue_length_uint64<'py>(py: Python<'py>, length: usize, arr: Bound<'_, PyArray1<u64>>) -> PyResult<Py<Game>>{
        Self::game_from_numpy_with_queue_length_impl::<u64>(py, length, arr)
    }

    /// Create a Game instance from a NumPy ndarray uint64 representation with explicit radius.
    /// The input array should represent the engine followed by the queue.
    /// 
    /// Arguments:
    /// - radius (usize): The radius of the hexagonal grid.
    /// - arr (numpy.ndarray): A 1D NumPy array representing the game engine followed by the queue.
    /// Returns:
    /// - Game: A new Game instance initialized from the provided NumPy array.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_with_radius_uint64<'py>(py: Python<'py>, radius: usize, arr: Bound<'_, PyArray1<u64>>) -> PyResult<Py<Game>>{
        Self::game_from_numpy_with_radius_impl::<u64>(py, radius, arr)
    }

    /// Create a Game instance from a NumPy ndarray float32 representation with explicit queue length.
    /// The input array should represent the engine followed by the queue.
    /// 
    /// Arguments:
    /// - length (usize): The length of the piece queue.
    /// - arr (numpy.ndarray): A 1D NumPy array representing the game engine followed by the queue.
    /// Returns:
    /// - Game: A new Game instance initialized from the provided NumPy array.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_with_queue_length_float32<'py>(py: Python<'py>, length: usize, arr: Bound<'_, PyArray1<f32>>) -> PyResult<Py<Game>>{
        Self::game_from_numpy_with_queue_length_impl::<f32>(py, length, arr)
    }

    /// Create a Game instance from a NumPy ndarray float32 representation with explicit radius.
    /// The input array should represent the engine followed by the queue.
    /// 
    /// Arguments:
    /// - radius (usize): The radius of the hexagonal grid.
    /// - arr (numpy.ndarray): A 1D NumPy array representing the game engine followed by the queue.
    /// Returns:
    /// - Game: A new Game instance initialized from the provided NumPy array.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_with_radius_float32<'py>(py: Python<'py>, radius: usize, arr: Bound<'_, PyArray1<f32>>) -> PyResult<Py<Game>>{
        Self::game_from_numpy_with_radius_impl::<f32>(py, radius, arr)
    }

    /// Create a Game instance from a NumPy ndarray float64 representation with explicit queue length.
    /// The input array should represent the engine followed by the queue.
    /// 
    /// Arguments:
    /// - length (usize): The length of the piece queue.
    /// - arr (numpy.ndarray): A 1D NumPy array representing the game engine followed by the queue.
    /// Returns:
    /// - Game: A new Game instance initialized from the provided NumPy array.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_with_queue_length_float64<'py>(py: Python<'py>, length: usize, arr: Bound<'_, PyArray1<f64>>) -> PyResult<Py<Game>>{
        Self::game_from_numpy_with_queue_length_impl::<f64>(py, length, arr)
    }

    /// Create a Game instance from a NumPy ndarray float64 representation with explicit radius.
    /// The input array should represent the engine followed by the queue.
    /// 
    /// Arguments:
    /// - radius (usize): The radius of the hexagonal grid.
    /// - arr (numpy.ndarray): A 1D NumPy array representing the game engine followed by the queue.
    /// Returns:
    /// - Game: A new Game instance initialized from the provided NumPy array.
    #[cfg(feature = "numpy")]
    #[staticmethod]
    pub fn from_numpy_with_radius_float64<'py>(py: Python<'py>, radius: usize, arr: Bound<'_, PyArray1<f64>>) -> PyResult<Py<Game>>{
        Self::game_from_numpy_with_radius_impl::<f64>(py, radius, arr)
    }

    /// Create a Game instance from a NumPy ndarray float16 representation with explicit queue length.
    /// The input array should represent the engine followed by the queue.
    /// 
    /// Arguments:
    /// - length (usize): The length of the piece queue.
    /// - arr (numpy.ndarray): A 1D NumPy array representing the game engine followed by the queue.
    /// Returns:
    /// - Game: A new Game instance initialized from the provided NumPy array.
    /// Warning:
    /// - The 'half' feature, which add support for float16, is still experimental and may not be stable. On machines that does
    /// not support float16 or installed with a version of numpy that does not support float16, this function may lead to
    /// undefined behavior or crashes. Testing show that on some systems, this can result in memory misinterpretation issues
    /// causing incorrect values to be read, and on other systems, it cause the entire program to halt but not crash.
    /// Use with caution.
    #[cfg(all(feature = "numpy", feature = "half"))]
    #[staticmethod]
    pub fn from_numpy_with_queue_length_float16<'py>(py: Python<'py>, length: usize, arr: Bound<'_, PyArray1<F16>>) -> PyResult<Py<Game>>{
        Self::game_from_numpy_with_queue_length_impl::<F16>(py, length, arr)
    }

    /// Create a Game instance from a NumPy ndarray float16 representation with explicit radius.
    /// The input array should represent the engine followed by the queue.
    /// 
    /// Arguments:
    /// - radius (usize): The radius of the hexagonal grid.
    /// - arr (numpy.ndarray): A 1D NumPy array representing the game engine followed by the queue.
    /// Returns:
    /// - Game: A new Game instance initialized from the provided NumPy array.
    /// Warning:
    /// - The 'half' feature, which add support for float16, is still experimental and may not be stable. On machines that does
    /// not support float16 or installed with a version of numpy that does not support float16, this function may lead to
    /// undefined behavior or crashes. Testing show that on some systems, this can result in memory misinterpretation issues
    /// causing incorrect values to be read, and on other systems, it cause the entire program to halt but not crash.
    /// Use with caution.
    #[cfg(all(feature = "numpy", feature = "half"))]
    #[staticmethod]
    pub fn from_numpy_with_radius_float16<'py>(py: Python<'py>, radius: usize, arr: Bound<'_, PyArray1<F16>>) -> PyResult<Py<Game>>{
        Self::game_from_numpy_with_radius_impl::<F16>(py, radius, arr)
    }

    /// Get the flat NumPy ndarray boolean representation of the Game queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of shape (n * 7,) representing the game queue.
    #[cfg(feature = "numpy")]
    pub fn queue_to_numpy_flat(&self, py: Python) -> Py<PyArray1<bool>> {
        queue_to_numpy_flat_impl::<bool>(py, &self._Game__queue)
    }

    /// Get the stacked NumPy ndarray boolean representation of the Game queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array of shape (n, 7) representing the game queue.
    #[cfg(feature = "numpy")]
    pub fn queue_to_numpy_stacked(&self, py: Python) -> Py<PyArray2<bool>> {
        queue_to_numpy_stacked_impl::<bool>(py, &self._Game__queue)
    }

    /// Get the flat NumPy ndarray boolean representation of the Game queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of shape (n * 7,) representing the game queue.
    #[cfg(feature = "numpy")]
    pub fn queue_to_numpy_bool_flat(&self, py: Python) -> Py<PyArray1<bool>> {
        queue_to_numpy_flat_impl::<bool>(py, &self._Game__queue)
    }

    /// Get the stacked NumPy ndarray boolean representation of the Game queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array of shape (n, 7) representing the game queue.
    #[cfg(feature = "numpy")]
    pub fn queue_to_numpy_bool_stacked(&self, py: Python) -> Py<PyArray2<bool>> {
        queue_to_numpy_stacked_impl::<bool>(py, &self._Game__queue)
    }

    /// Get the flat NumPy ndarray int8 representation of the Game queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of shape (n * 7,) representing the game queue.
    #[cfg(feature = "numpy")]
    pub fn queue_to_numpy_int8_flat(&self, py: Python) -> Py<PyArray1<i8>> {
        queue_to_numpy_flat_impl::<i8>(py, &self._Game__queue)
    }

    /// Get the stacked NumPy ndarray int8 representation of the Game queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array of shape (n, 7) representing the game queue.
    #[cfg(feature = "numpy")]
    pub fn queue_to_numpy_int8_stacked(&self, py: Python) -> Py<PyArray2<i8>> {
        queue_to_numpy_stacked_impl::<i8>(py, &self._Game__queue)
    }

    /// Get the flat NumPy ndarray uint8 representation of the Game queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of shape (n * 7,) representing the game queue.
    #[cfg(feature = "numpy")]
    pub fn queue_to_numpy_uint8_flat(&self, py: Python) -> Py<PyArray1<u8>> {
        queue_to_numpy_flat_impl::<u8>(py, &self._Game__queue)
    }

    /// Get the stacked NumPy ndarray uint8 representation of the Game queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array of shape (n, 7) representing the game queue.
    #[cfg(feature = "numpy")]
    pub fn queue_to_numpy_uint8_stacked(&self, py: Python) -> Py<PyArray2<u8>> {
        queue_to_numpy_stacked_impl::<u8>(py, &self._Game__queue)
    }

    /// Get the flat NumPy ndarray int16 representation of the Game queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of shape (n * 7,) representing the game queue.
    #[cfg(feature = "numpy")]
    pub fn queue_to_numpy_int16_flat(&self, py: Python) -> Py<PyArray1<i16>> {
        queue_to_numpy_flat_impl::<i16>(py, &self._Game__queue)
    }

    /// Get the stacked NumPy ndarray int16 representation of the Game queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array of shape (n, 7) representing the game queue.
    #[cfg(feature = "numpy")]
    pub fn queue_to_numpy_int16_stacked(&self, py: Python) -> Py<PyArray2<i16>> {
        queue_to_numpy_stacked_impl::<i16>(py, &self._Game__queue)
    }

    /// Get the flat NumPy ndarray uint16 representation of the Game queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of shape (n * 7,) representing the game queue.
    #[cfg(feature = "numpy")]
    pub fn queue_to_numpy_uint16_flat(&self, py: Python) -> Py<PyArray1<u16>> {
        queue_to_numpy_flat_impl::<u16>(py, &self._Game__queue)
    }

    /// Get the stacked NumPy ndarray uint16 representation of the Game queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array of shape (n, 7) representing the game queue.
    #[cfg(feature = "numpy")]
    pub fn queue_to_numpy_uint16_stacked(&self, py: Python) -> Py<PyArray2<u16>> {
        queue_to_numpy_stacked_impl::<u16>(py, &self._Game__queue)
    }

    /// Get the flat NumPy ndarray int32 representation of the Game queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of shape (n * 7,) representing the game queue.
    #[cfg(feature = "numpy")]
    pub fn queue_to_numpy_int32_flat(&self, py: Python) -> Py<PyArray1<i32>> {
        queue_to_numpy_flat_impl::<i32>(py, &self._Game__queue)
    }

    /// Get the stacked NumPy ndarray int32 representation of the Game queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array of shape (n, 7) representing the game queue.
    #[cfg(feature = "numpy")]
    pub fn queue_to_numpy_int32_stacked(&self, py: Python) -> Py<PyArray2<i32>> {
        queue_to_numpy_stacked_impl::<i32>(py, &self._Game__queue)
    }

    /// Get the flat NumPy ndarray uint32 representation of the Game queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of shape (n * 7,) representing the game queue.
    #[cfg(feature = "numpy")]
    pub fn queue_to_numpy_uint32_flat(&self, py: Python) -> Py<PyArray1<u32>> {
        queue_to_numpy_flat_impl::<u32>(py, &self._Game__queue)
    }

    /// Get the stacked NumPy ndarray uint32 representation of the Game queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array of shape (n, 7) representing the game queue.
    #[cfg(feature = "numpy")]
    pub fn queue_to_numpy_uint32_stacked(&self, py: Python) -> Py<PyArray2<u32>> {
        queue_to_numpy_stacked_impl::<u32>(py, &self._Game__queue)
    }

    /// Get the flat NumPy ndarray int64 representation of the Game queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of shape (n * 7,) representing the game queue.
    #[cfg(feature = "numpy")]
    pub fn queue_to_numpy_int64_flat(&self, py: Python) -> Py<PyArray1<i64>> {
        queue_to_numpy_flat_impl::<i64>(py, &self._Game__queue)
    }

    /// Get the stacked NumPy ndarray int64 representation of the Game queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array of shape (n, 7) representing the game queue.
    #[cfg(feature = "numpy")]
    pub fn queue_to_numpy_int64_stacked(&self, py: Python) -> Py<PyArray2<i64>> {
        queue_to_numpy_stacked_impl::<i64>(py, &self._Game__queue)
    }

    /// Get the flat NumPy ndarray uint64 representation of the Game queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of shape (n * 7,) representing the game queue.
    #[cfg(feature = "numpy")]
    pub fn queue_to_numpy_uint64_flat(&self, py: Python) -> Py<PyArray1<u64>> {
        queue_to_numpy_flat_impl::<u64>(py, &self._Game__queue)
    }

    /// Get the stacked NumPy ndarray uint64 representation of the Game queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array of shape (n, 7) representing the game queue.
    #[cfg(feature = "numpy")]
    pub fn queue_to_numpy_uint64_stacked(&self, py: Python) -> Py<PyArray2<u64>> {
        queue_to_numpy_stacked_impl::<u64>(py, &self._Game__queue)
    }

    /// Get the flat NumPy ndarray float32 representation of the Game queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of shape (n * 7,) representing the game queue.
    #[cfg(feature = "numpy")]
    pub fn queue_to_numpy_float32_flat(&self, py: Python) -> Py<PyArray1<f32>> {
        queue_to_numpy_flat_impl::<f32>(py, &self._Game__queue)
    }

    /// Get the stacked NumPy ndarray float32 representation of the Game queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array of shape (n, 7) representing the game queue.
    #[cfg(feature = "numpy")]
    pub fn queue_to_numpy_float32_stacked(&self, py: Python) -> Py<PyArray2<f32>> {
        queue_to_numpy_stacked_impl::<f32>(py, &self._Game__queue)
    }

    /// Get the flat NumPy ndarray float64 representation of the Game queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of shape (n * 7,) representing the game queue.
    #[cfg(feature = "numpy")]
    pub fn queue_to_numpy_float64_flat(&self, py: Python) -> Py<PyArray1<f64>> {
        queue_to_numpy_flat_impl::<f64>(py, &self._Game__queue)
    }

    /// Get the stacked NumPy ndarray float64 representation of the Game queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array of shape (n, 7) representing the game queue.
    #[cfg(feature = "numpy")]
    pub fn queue_to_numpy_float64_stacked(&self, py: Python) -> Py<PyArray2<f64>> {
        queue_to_numpy_stacked_impl::<f64>(py, &self._Game__queue)
    }

    /// Get the flat NumPy ndarray float16 representation of the Game queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 1D NumPy array of shape (n * 7,) representing the game queue.
    /// Warning:
    /// - The 'half' feature, which add support for float16, is still experimental and may not be stable. On machines that does
    /// not support float16 or installed with a version of numpy that does not support float16, this function may lead to
    /// undefined behavior or crashes. Testing show that on some systems, this can result in memory misinterpretation issues
    /// causing incorrect values to be read, and on other systems, it cause the entire program to halt but not crash.
    /// Use with caution.
    #[cfg(all(feature = "numpy", feature = "half"))]
    pub fn queue_to_numpy_float16_flat(&self, py: Python) -> Py<PyArray1<F16>> {
        queue_to_numpy_flat_impl::<F16>(py, &self._Game__queue)
    }

    /// Get the stacked NumPy ndarray float16 representation of the Game queue.
    /// 
    /// Returns:
    /// - numpy.ndarray: A 2D NumPy array of shape (n, 7) representing the game queue.
    /// Warning:
    /// - The 'half' feature, which add support for float16, is still experimental and may not be stable. On machines that does
    /// not support float16 or installed with a version of numpy that does not support float16, this function may lead to
    /// undefined behavior or crashes. Testing show that on some systems, this can result in memory misinterpretation issues
    /// causing incorrect values to be read, and on other systems, it cause the entire program to halt but not crash.
    /// Use with caution.
    #[cfg(all(feature = "numpy", feature = "half"))]
    pub fn queue_to_numpy_float16_stacked(&self, py: Python) -> Py<PyArray2<F16>> {
        queue_to_numpy_stacked_impl::<F16>(py, &self._Game__queue)
    }

    /* ------------------------------------- HPYHEX-RS ------------------------------------- */

    /// Special convenient method not provided in the standard hpyhex API. This method allows adding a piece using
    /// the piece index in the queue and the position index in the engine directly.
    ///
    /// Parameters:
    /// - piece_index (int): The index of the piece in the queue to be added.
    /// - position_index (int): The index of the position in the engine where the piece should be placed.
    /// Returns:
    /// - bool: True if the piece was successfully added, False otherwise.
    pub fn hpyhex_rs_add_piece_with_index(&mut self, python: Python, piece_index: usize, position_index: usize) -> PyResult<bool> {
        let engine_bound = self._Game__engine.bind(python);
        let mut engine = engine_bound.extract::<HexEngine>()?;
        // Check piece and position exists
        if piece_index >= self._Game__queue.len() {
            return Ok(false);
        }
        if position_index >= engine.__len__() {
            return Ok(false);
        }
        let piece = self._Game__queue[piece_index].clone();
        let coo = engine.hex_coordinate_of(position_index)?;
        // Add piece to engine and increment score and turn
        let add_result = engine_bound.call_method1("add_piece", (coo, piece.clone()));
        if let Err(ref e) = add_result {
            if e.is_instance_of::<pyo3::exceptions::PyValueError>(python) {
                return Ok(false);
            } else {
                return Err(e.clone_ref(python));
            }
        }
        self._Game__score += piece.count() as u64;
        // Replace used piece
        let new_piece = PieceFactory::generate_piece()?;
        self._Game__queue[piece_index] = new_piece;
        // Eliminate and add score
        let eliminated = engine.eliminate();
        let eliminated_len = eliminated?.len();
        self._Game__score += (eliminated_len as u64) * 5;
        self._Game__turn += 1;
        // Check whether the game has ended
        let mut has_move = false;
        for p in &self._Game__queue {
            if engine.check_has_positions(p) {
                has_move = true;
                break;
            }
        }
        if !has_move {
            self._Game__end = true;
        }
        Ok(true)
    }

    /* ---------------------------------------- HPYHEX PYTHON API ---------------------------------------- */
    
    /// Initialize the game with a game engine of radius r and game queue of length q.
    ///
    /// Parameters:
    /// - engine (HexEngine | int): The game engine to use, either as a HexEngine instance or an integer representing the radius.
    /// - queue (list[Piece] | int): The queue of pieces to use, either as a list of Piece instances or an integer representing the size of the queue.
    /// - initial_turn (int): The initial turn number of the game, default is 0.
    /// - initial_score (int): The initial score of the game, default is 0.
    /// Returns:
    /// - None
    /// Raises:
    /// - ValueError: If the engine radius is less than 2 or if the queue size is less than 1.
    /// - TypeError: If the engine is not a HexEngine instance or an integer, or if the queue is not a list of Piece instances or an integer, or if initial_turn or initial_score is not a non-negative integer.
    #[new]
    pub fn new(
        engine: &pyo3::Bound<'_, pyo3::PyAny>,
        queue: &pyo3::Bound<'_, pyo3::PyAny>,
        initial_turn: Option<i64>,
        initial_score: Option<i64>,
        // Those accept i64 to check for negative values
    ) -> pyo3::PyResult<Self> {
        // Engine: HexEngine or int (radius)
        let _Game__engine = if let Ok(engine_ref) = engine.extract::<Py<HexEngine>>() {
            engine_ref
        } else if let Ok(radius) = engine.extract::<usize>() {
            if radius < 2 {
                return Err(pyo3::exceptions::PyValueError::new_err("Radius must be greater than or equals two"));
            }
            let engine_raw = HexEngine::try_from(radius)?;
            Py::new(engine.py(), engine_raw)?
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err("Engine must be a HexEngine instance or an integer representing the radius"));
        };

        // Queue: list[Piece] or int (queue size)
        let _Game__queue: Vec<Piece> = if let Ok(list) = queue.extract::<Vec<Piece>>() {
            list
        } else if let Ok(qsize) = queue.extract::<usize>() {
            if qsize < 1 {
                return Err(pyo3::exceptions::PyValueError::new_err("Queue size must be greater than or equals one"));
            }
            let mut pieces = Vec::with_capacity(qsize);
            for _ in 0..qsize {
                pieces.push(PieceFactory::generate_piece()?);
            }
            pieces
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err("Queue must be a list of Piece instances or an integer representing the size of the queue"));
        };

        // initial_turn: Option<u32> (default 0)
        let _Game__turn = match initial_turn {
            Some(t) => t,
            None => 0,
        };

        // initial_score: Option<u64> (default 0)
        let _Game__score = match initial_score {
            Some(s) => s,
            None => 0,
        };

        // Validate initial_turn and initial_score
        let _Game__turn = if _Game__turn < 0 {
            return Err(pyo3::exceptions::PyTypeError::new_err("Initial turn must be a non-negative integer"));
        } else {
            _Game__turn as u64
        };
        let _Game__score = if _Game__score < 0 {
            return Err(pyo3::exceptions::PyTypeError::new_err("Initial score must be a non-negative integer"));
        } else {
            _Game__score as u64
        };

        Ok(Game {
            _Game__engine,
            _Game__queue,
            _Game__score,
            _Game__turn,
            _Game__end: false,
        })
    }

    /// Add a piece to the game engine at the specified coordinates.
    ///
    /// Parameters:
    /// - piece_index (int): The index of the piece in the queue to be added.
    /// - coord (Hex): The coordinates where the piece should be placed.
    /// Returns:
    /// - bool: True if the piece was successfully added, False otherwise.
    pub fn add_piece(&mut self, python: Python, piece_index: &Bound<'_, PyAny>, coord: &Bound<'_, PyAny>) -> PyResult<bool> {
        // Extract piece_index
        let piece_index: usize = match piece_index.extract() {
            Ok(idx) => idx,
            Err(_) => return Ok(false),
        };
        // Extract coord
        let coord: Bound<'_, Hex> = match coord.extract() {
            Ok(c) => c,
            Err(_) => return Ok(false),
        };
        self.add_piece_checked(python, piece_index, &coord)
    }

    /// Make a move using the specified algorithm.
    ///
    /// Parameters:
    /// - algorithm (callable): The algorithm to use for making the move.
    ///   The algorithm should follow the signature: `algorithm(engine: HexEngine, queue: list[Piece]) -> tuple[int, Hex]`.
    /// Returns:
    /// - bool: True if the move was successfully made, False otherwise.
    pub fn make_move(&mut self, python: Python, algorithm: &Bound<'_, PyAny>) -> PyResult<bool> {
        if self._Game__end { return Ok(false); }
        let engine_bound = self._Game__engine.bind(python);
        let queue_py = self.queue(python)?;
        let result = algorithm.call1((engine_bound, queue_py));
        let (index, coord) = match result {
            Ok(tuple) => {
                let item0 = tuple.get_item(0);
                let index = match &item0 {
                    Ok(val) => val.extract::<usize>().unwrap_or(usize::MAX),
                    Err(_) => usize::MAX,
                };
                let item1 = tuple.get_item(1);
                let coord: Bound<'_, Hex>;
                match &item1 {
                    Ok(val) => {
                        let py_hex = val.extract::<Py<Hex>>()?;
                        coord = py_hex.bind(python).clone();
                    },
                    Err(_) => return Ok(false),
                };
                (index, coord)
            }
            Err(_) => return Ok(false),
        };
        if index == usize::MAX {
            return Ok(false);
        }
        self.add_piece_checked(python, index, &coord)
    }

    /// Return a string representation of the game state.
    /// 
    /// Returns:
    /// - str: A string representation of the game state, including engine, queue, score, turn, and whether the game has ended.
    fn __str__(&self, py: Python) -> String {
        // Format queue with []
        let mut queue_str = String::from("[");
        for (i, piece) in self._Game__queue.iter().enumerate() {
            if i > 0 {
                queue_str.push_str(", ");
            }
            queue_str.push_str(&format!("{:?}", piece));
        }
        queue_str.push(']');
        format!(
            "Game(engine={:?}, queue={}, score={}, turn={}, end={})",
            (&*self._Game__engine.borrow(py)).__str__(), queue_str, self._Game__score, self._Game__turn, self._Game__end
        )
    }

    /// Return a string representation of the game state.
    /// 
    /// Returns:
    /// - str: A string representation of the game state.
    fn __repr__(&self, py: Python) -> String {
        format!("({}, {:?})", (&*self._Game__engine.borrow(py)).__repr__(), self._Game__queue)
    }

    /// Check equality between this game and another game.
    /// 
    /// Parameters:
    /// - other (Game): The other game to compare with.
    /// Returns:
    /// - bool: True if the games are equal, False otherwise.
    fn __eq__(&self, py: Python, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let other_game: Game = other.extract::<Game>()?;
        let self_engine = &*self._Game__engine.borrow(py);
        let other_engine = &*other_game._Game__engine.borrow(py);
        Ok(
            self_engine == other_engine &&
            self._Game__queue == other_game._Game__queue &&
            self._Game__score == other_game._Game__score &&
            self._Game__turn == other_game._Game__turn &&
            self._Game__end == other_game._Game__end
        )
    }

    /// Returns whether this game has ended.
    /// 
    /// Returns:
    /// - is_end (bool): True if the game has ended, False otherwise.
    #[getter]
    fn end(&self) -> bool {
        self._Game__end
    }

    /// Returns the current result of this game.
    /// 
    /// Returns:
    /// - result (tuple[int, int]): A tuple containing the current turn number and score, in the order (turn, score).
    #[getter]
    fn result(&self) -> (u64, u64) {
        (self._Game__turn, self._Game__score)
    }

    /// Returns the current turn number of this game.
    /// 
    /// Returns:
    /// - turn (int): The current turn number in the game.
    #[getter]
    fn turn(&self) -> u64 {
        self._Game__turn
    }

    /// Returns the current score of this game.
    /// 
    /// Returns:
    /// - score (int): The current score in the game.
    #[getter]
    fn score(&self) -> u64 {
        self._Game__score
    }

    /// Returns the reference to game engine of this game.
    ///
    /// Returns:
    /// - engine (HexEngine): The HexEngine instance used in this game.
    #[getter]
    fn engine<'py>(&self, py: Python<'py>) -> Py<HexEngine> {
        self._Game__engine.clone_ref(py)
    }

    /// Returns the reference to the queue of pieces available in this game.
    /// 
    /// Returns:
    /// - queue (list[Piece]): The list of pieces currently in the queue.
    #[getter]
    fn queue<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty_bound(py);
        for piece in &self._Game__queue {
            match Py::new(py, piece.clone()) {
                Ok(obj) => {
                    if let Err(e) = list.append(obj.bind(py)) {
                        return Err(e);
                    }
                }
                Err(e) => return Err(e),
            }
        }
        Ok(list)
    }
}
