#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;

#[pymodule]
fn hpyhex(_py: Python, m: &pyo3::Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Hex>()?;
    Ok(())
}


use std::collections::HashMap;
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
///     - Raw Coordinates (i, j, k): Three axes satisfying i + j + k = 0, where
///       each axis is diagonal to the others at 60° increments.
///     - Line Coordinates (i, k): Derived coordinates representing distances
///       perpendicular to axes, simplifying grid operations.
///
/// Note:
///     - This class is immutable and optimized with __slots__.
///     - Raw coordinate methods (__i__, __j__, __k__) are retained for backward compatibility.
///     - Only basic functionality is implemented; complex adjacency, iteration,
///       and mutability features are omitted for simplicity.
///
/// Attributes:
///     i (int): The line i coordinate.
///     j (int): The computed line j coordinate (k - i).
///     k (int): The line k coordinate.
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
static HEX_CACHE: OnceLock<HashMap<(i32, i32), Hex>> = OnceLock::new();

fn get_hex(i: i32, k: i32) -> Hex {
    if (CACHE_MIN..=CACHE_MAX).contains(&i) && (CACHE_MIN..=CACHE_MAX).contains(&k) {
        let cache = HEX_CACHE.get_or_init(|| HashMap::new());
        if let Some(hex) = cache.get(&(i, k)) {
            return hex.clone();
        }
        // Insert if not present
        let mut cache = HEX_CACHE.get_or_init(|| HashMap::new()).clone();
        let hex = Hex { i, k };
        cache.insert((i, k), hex.clone());
        HEX_CACHE.set(cache).ok();
        hex
    } else {
        Hex { i, k }
    }
}


#[pymethods]
impl Hex {
    /// Initialize a Hex coordinate at (i, k). Defaults to (0, 0).
    ///
    /// Arguments:
    ///     i (int): The I-line coordinate of the hex.
    ///     k (int): The K-line coordinate of the hex.
    /// Returns:
    ///     Hex
    /// Raises:
    ///     TypeError: If i or k is not an integer.
    #[new]
    #[pyo3(signature = (i = 0, k = 0))]
    pub fn new(i: i32, k: i32) -> Self {
        get_hex(i, k)
    }

    /// Get the I-line coordinate of the hex.
    ///
    /// Returns:
    ///     int: The I-line coordinate.
    #[inline]
    #[getter]
    pub fn i(&self) -> i32 {
        self.i
    }

    /// Get the J-line coordinate of the hex.
    ///
    /// Returns:
    ///     int: The J-line coordinate.
    #[inline]
    #[getter]
    pub fn j(&self) -> i32 {
        self.k - self.i
    }

    /// Get the K-line coordinate of the hex.
    ///
    /// Returns:
    ///     int: The K-line coordinate.
    #[inline]
    #[getter]
    pub fn k(&self) -> i32 {
        self.k
    }

    /// Return an iterator over the hex coordinates.
    ///
    /// Yields:
    ///     int: The I-line coordinate of the hex.
    ///     int: The K-line coordinate of the hex.
    pub fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let tuple = pyo3::types::PyTuple::new_bound(py, &[slf.i.into_py(py), slf.k.into_py(py)]);
            Ok(tuple.into_py(py))
        })
    }

    /// Return the raw i coordinate of the hex.
    ///
    /// Returns:
    ///     int: The raw i coordinate.
    #[inline]
    pub fn __i__(&self) -> i32 {
        self.k * 2 - self.i
    }

    /// Return the raw j coordinate of the hex.
    ///
    /// Returns:
    ///     int: The raw j coordinate.
    #[inline]
    pub fn __j__(&self) -> i32 {
        self.i + self.k
    }

    /// Return the raw k coordinate of the hex.
    ///
    /// Returns:
    ///     int: The raw k coordinate.
    #[inline]
    pub fn __k__(&self) -> i32 {
        self.i * 2 - self.k
    }

    /// Return a string representation of the hex coordinates.
    ///
    /// Format: Hex(i, j, k), where i, j, and k are the line coordinates.
    /// Returns:
    ///     str: The string representation of the hex.
    pub fn __str__(&self) -> String {
        format!("Hex({}, {}, {})", self.i, self.k - self.i, self.k)
    }

    /// Return a string representation of the hex coordinates for debugging.
    ///
    /// Format: Hex(i, j, k), where i, j, and k are the line coordinates.
    /// Returns:
    ///     str: The string representation of the hex.
    pub fn __repr__(&self) -> String {
        format!("({}, {})", self.i, self.k)
    }

    /// Check equality with another Hex or a tuple of coordinates.
    ///
    /// Arguments:
    ///     value (Hex or tuple): The value to compare with.
    /// Returns:
    ///     bool: True if the coordinates match, False otherwise.
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
    ///     int: The hash value of the hex coordinates.
    pub fn __hash__(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.i.hash(&mut hasher);
        self.k.hash(&mut hasher);
        hasher.finish()
    }

    /// Add another Hex or a tuple of coordinates to this hex.
    ///
    /// Arguments:
    ///     other (Hex or tuple): The value to add.
    /// Returns:
    ///     Hex: A new Hex with the added coordinates.
    /// Raises:
    ///     TypeError: If the other operand is not a Hex or a tuple of coordinates.
    pub fn __add__(&self, other: &pyo3::Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_hex) = other.extract::<PyRef<Hex>>() {
            Ok(get_hex(self.i + other_hex.i, self.k + other_hex.k))
        } else if let Ok(tuple) = other.extract::<(i32, i32)>() {
            Ok(get_hex(self.i + tuple.0, self.k + tuple.1))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err("Unsupported type for addition with Hex"))
        }
    }

    /// Reverse addition of this hex to another Hex or a tuple.
    ///
    /// Arguments:
    ///     other (Hex or tuple): The value to add this hex to.
    /// Returns:
    ///     Hex: A new Hex with the added coordinates.
    /// Raises:
    ///     TypeError: If the other operand is not a Hex or a tuple of coordinates.
    pub fn __radd__(&self, other: &pyo3::Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_hex) = other.extract::<PyRef<Hex>>() {
            Ok(get_hex(other_hex.i + self.i, other_hex.k + self.k))
        } else if let Ok(tuple) = other.extract::<(i32, i32)>() {
            Ok(get_hex(tuple.0 + self.i, tuple.1 + self.k))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err("Unsupported type for reverse addition with Hex"))
        }
    }

    /// Subtract another Hex or a tuple of coordinates from this hex.
    ///
    /// Arguments:
    ///     other (Hex or tuple): The value to subtract.
    /// Returns:
    ///     Hex: A new Hex with the subtracted coordinates.
    /// Raises:
    ///     TypeError: If the other operand is not a Hex or a tuple of coordinates.
    pub fn __sub__(&self, other: &pyo3::Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_hex) = other.extract::<PyRef<Hex>>() {
            Ok(get_hex(self.i - other_hex.i, self.k - other_hex.k))
        } else if let Ok(tuple) = other.extract::<(i32, i32)>() {
            Ok(get_hex(self.i - tuple.0, self.k - tuple.1))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err("Unsupported type for subtraction with Hex"))
        }
    }

    /// Reverse subtraction of this hex from another Hex or a tuple.
    ///
    /// Arguments:
    ///     other (Hex or tuple): The value to subtract this hex from.
    /// Returns:
    ///     Hex: A new Hex with the subtracted coordinates.
    /// Raises:
    ///     TypeError: If the other operand is not a Hex or a tuple of coordinates.
    pub fn __rsub__(&self, other: &pyo3::Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_hex) = other.extract::<PyRef<Hex>>() {
            Ok(get_hex(other_hex.i - self.i, other_hex.k - self.k))
        } else if let Ok(tuple) = other.extract::<(i32, i32)>() {
            Ok(get_hex(tuple.0 - self.i, tuple.1 - self.k))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err("Unsupported type for reverse subtraction with Hex"))
        }
    }

    /// Create a copy of this Hex.
    ///
    /// Returns:
    ///     Hex: A new Hex with the same coordinates.
    #[inline]
    pub fn __copy__(&self) -> Self {
        get_hex(self.i, self.k)
    }

    /// Create a deep copy of this Hex.
    ///
    /// Arguments:
    ///     memo (dict): A dictionary to keep track of copied objects.
    /// Returns:
    ///     Hex: A new Hex with the same coordinates.
    #[inline]
    pub fn __deepcopy__(&self, _memo: Option<&pyo3::Bound<'_, PyAny>>) -> Self {
        get_hex(self.i, self.k)
    }

    /// Check if the Hex is not at the origin (0, 0).
    ///
    /// Returns:
    ///     bool: True if the Hex is not at the origin, False otherwise.
    #[inline]
    pub fn __bool__(&self) -> bool {
        self.i != 0 || self.k != 0
    }

    /// Return a new Hex shifted along the i-axis by units.
    ///
    /// Arguments:
    ///     units (int): The number of units to shift along the i-axis.
    /// Returns:
    ///     Hex: A new Hex shifted by the specified units along the i-axis.
    /// Raises:
    ///     TypeError: If units is not an integer.
    #[inline]
    pub fn shift_i(&self, units: i32) -> Self {
        get_hex(self.i + units, self.k)
    }

    /// Return a new Hex shifted along the j-axis by units.
    ///
    /// Arguments:
    ///     units (int): The number of units to shift along the j-axis.
    /// Returns:
    ///     Hex: A new Hex shifted by the specified units along the j-axis.
    /// Raises:
    ///     TypeError: If units is not an integer.
    #[inline]
    pub fn shift_j(&self, units: i32) -> Self {
        get_hex(self.i - units, self.k + units)
    }

    /// Return a new Hex shifted along the k-axis by units.
    ///
    /// Arguments:
    ///     units (int): The number of units to shift along the k-axis.
    /// Returns:
    ///     Hex: A new Hex shifted by the specified units along the k-axis.
    /// Raises:
    ///     TypeError: If units is not an integer.
    #[inline]
    pub fn shift_k(&self, units: i32) -> Self {
        get_hex(self.i, self.k + units)
    }
}