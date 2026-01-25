//! # Hex Core for HappyHex
//! 
//! ## Feature Flag: core
//! 
//! ## Overview
//! 
//! This module implements a simplified and optimized hexagonal grid system designed for machine learning applications
//! and other performance-critical scenarios. It provides efficient representations and operations for hexagonal grids,
//! pieces, and coordinates, focusing on speed and low overhead.
//! 
//! This version is rewritten in Rust and optimized for performance, while maintaining the core functionality and
//! API design from the original Java implementation. The details of the implementation more closely resemble the Python version
//! of HappyHex ML, with adaptations for Rust's strengths.
//! 
//! # Key Components
//! 
//! - **Hex**: Represents a coordinate in the hexagonal grid using (i, k) coordinates.
//! - **Piece**: Represents a piece composed of 7 blocks in a hexagonal pattern using a bitfield representation.
//! - **HexEngine**: Manages the hexagonal grid, storing occupancy as a boolean vector with O(1) coordinate-to-index conversion.
//! 
//! # Performance Considerations
//! - Uses primitive types and bitwise operations for efficient storage and computation.
//! - Provides O(1) methods for coordinate-index conversions.
//! - Optimized algorithms for piece placement, elimination, and density calculations.

#![cfg(any(feature = "default", feature = "core"))]
use std::borrow::Borrow;
use std::fmt;
use std::ops::{Add, Sub};

/// Represents a 2D coordinate in a hexagonal grid system using a specialized integer coordinate model.
///
/// # Coordinate System
///
/// The `Hex` struct models a point in a hexagonal grid using (i, k) coordinates, supporting both raw coordinate access
/// and derived line-based computations across three axes: I, J, and K.
///
/// - The axes **I**, **J**, and **K** run diagonally through the hexagonal grid.
/// - I+ is 60° to J+, J+ is 60° to K+, and K+ is 60° to J-.
/// - Coordinates (i, k) form a basis for representing any hexagon.
/// - *Raw coordinates* (or hex coordinates) refer to the distance of a point along one of the axes, multiplied by 2.
///   The relationship is: `i - j + k = 0`.
/// - *Line coordinates* (or line-distance based coordinates) are based on the distance perpendicular to the axes,
///   with the relationship: `I + J - K = 0`.
/// - All line coordinates correspond to some raw coordinate, but not all raw coordinates correspond to valid line coordinates.
///   For most applications, line coordinates are preferred for their simplicity.
///
/// ## Coordinate System Visualization
///
/// Example points with raw coordinates (2i, 2j, 2k):
///
/// ```text
///    I
///   / * (5, 4, -1)
///  /     * (5, 7, 2)
/// o - - J
///  \ * (0, 3, 3)
///   \
///    K
/// ```
///
/// Example points with line coordinates (I, J, K):
///
/// ```text
///    I
///   / * (1, 2, 3)
///  /     * (3, 1, 4)
/// o - - J
///  \ * (2, -1, 1)
///   \
///    K
/// ```
///
/// ## Implementation Notes
///
/// - `i` and `k` are the base values stored in each `Hex` instance.
/// - `I = i`, `K = k`, and `J = k - i`.
/// - Raw coordinates can be accessed via [`raw_i`], [`raw_j`], and [`raw_k`].
/// - Line coordinates can be accessed via [`i`], [`j`], and [`k`].
///
/// ## Functionality
///
/// - Access and compute raw coordinates: [`raw_i`], [`raw_j`], [`raw_k`].
/// - Access and compute line-distance based coordinates: [`i`], [`j`], [`k`].
/// - Create hex objects via [`Hex::new`] or [`From<(i32, i32)>`].
/// - Move hex objects along I, J, or K axes: [`shift_i`], [`shift_j`], [`shift_k`].
/// - Addition and subtraction of coordinates: [`Add`], [`Sub`].
/// - Conversion to tuples: [`Into<(i32, i32)>`] and [`From<(i32, i32)>`].
/// 
/// ## Mutability
/// 
/// `Hex` instances are immutable. Operations that modify coordinates return new `Hex` instances.
/// Since the memory footprint is small (two `i32` values), copying is efficient.
///
/// # Usage Notes
///
/// Prefer using [`Hex::new`] or [`From<(i32, i32)>`] for construction, as these ensure correct coordinate logic.
///
/// Designed by William Wu. Adapted for Rust.
#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Hex {
    i: i32,
    k: i32,
}

impl Default for Hex {
    /// Creates a default Hex at (0, 0)
    #[inline]
    fn default() -> Self {
        Hex { i: 0, k: 0 }
    }
}

impl Hex {
    /// Creates a new Hex at position (i, k)
    /// 
    /// This represents a coordinate in a hexagonal grid using the (i, k) coordinate system.
    /// 
    /// ## Parameters
    /// - `i`: The i-line coordinate as an `i32`.
    /// - `k`: The k-line coordinate as an `i32`.
    /// 
    /// ## Returns
    /// A new `Hex` instance at the specified coordinates.
    /// 
    /// ## See Also
    /// - [`From<(i32, i32)>`] for an alternative construction method.
    #[inline]
    pub const fn new(i: i32, k: i32) -> Self {
        Hex { i, k }
    }

    /// Returns the i-line coordinate
    /// 
    /// ## Returns
    /// The i-line coordinate as an `i32`.
    #[inline]
    pub const fn i(&self) -> i32 {
        self.i
    }

    /// Returns the j-line coordinate
    /// 
    /// ## Returns
    /// The j-line coordinate as an `i32`.
    #[inline]
    pub const fn j(&self) -> i32 {
        self.k - self.i
    }

    /// Returns the k-line coordinate
    /// 
    /// ## Returns
    /// The k-line coordinate as an `i32`.
    #[inline]
    pub const fn k(&self) -> i32 {
        self.k
    }

    /// Returns raw i coordinate
    /// 
    /// ## Returns
    /// The raw i coordinate as an `i32`.
    #[inline]
    pub const fn raw_i(&self) -> i32 {
        self.k * 2 - self.i
    }

    /// Returns raw j coordinate
    /// 
    /// ## Returns
    /// The raw j coordinate as an `i32`.
    #[inline]
    pub const fn raw_j(&self) -> i32 {
        self.i + self.k
    }

    /// Returns raw k coordinate
    /// 
    /// ## Returns
    /// The raw k coordinate as an `i32`.
    #[inline]
    pub const fn raw_k(&self) -> i32 {
        self.i * 2 - self.k
    }

    /// Shifts along the i-axis
    /// 
    /// ## Parameters
    /// - `units`: Number of units to shift (positive or negative)
    /// 
    /// ## Returns
    /// A new `Hex` instance shifted along the i-axis.
    #[inline]
    pub const fn shift_i(self, units: i32) -> Self {
        Hex::new(self.i + units, self.k)
    }

    /// Shifts along the j-axis
    /// 
    /// ## Parameters
    /// - `units`: Number of units to shift (positive or negative)
    /// 
    /// ## Returns
    /// A new `Hex` instance shifted along the j-axis.
    #[inline]
    pub const fn shift_j(self, units: i32) -> Self {
        Hex::new(self.i - units, self.k + units)
    }

    /// Shifts along the k-axis
    /// 
    /// ## Parameters
    /// - `units`: Number of units to shift (positive or negative)
    /// 
    /// ## Returns
    /// A new `Hex` instance shifted along the k-axis.
    #[inline]
    pub const fn shift_k(self, units: i32) -> Self {
        Hex::new(self.i, self.k + units)
    }
}

impl Add for Hex {
    type Output = Self;
    
    /// Adds two Hex coordinates
    /// 
    /// ## Parameters
    /// - `other`: The other `Hex` to add
    /// 
    /// ## Returns
    /// A new `Hex` instance representing the sum of the two coordinates.
    #[inline]
    fn add(self, other: Self) -> Self {
        Hex::new(self.i + other.i, self.k + other.k)
    }
}

impl Sub for Hex {
    type Output = Self;
    
    /// Subtracts two Hex coordinates
    /// 
    /// ## Parameters
    /// - `other`: The other `Hex` to subtract
    /// 
    /// ## Returns
    /// A new `Hex` instance representing the difference of the two coordinates.
    #[inline]
    fn sub(self, other: Self) -> Self {
        Hex::new(self.i - other.i, self.k - other.k)
    }
}

impl From<(i32, i32)> for Hex {
    /// Creates a Hex from (i, k) tuple
    /// 
    /// ## Parameters
    /// - `(i, k)`: Tuple representing the (i, k) coordinates
    /// 
    /// ## Returns
    /// A new `Hex` instance at the specified coordinates.
    #[inline]
    fn from((i, k): (i32, i32)) -> Self {
        Hex::new(i, k)
    }
}

impl Into<(i32, i32)> for Hex {
    /// Converts Hex to (i, k) tuple
    /// 
    /// ## Returns
    /// A tuple representing the (i, k) coordinates of the `Hex`.
    #[inline]
    fn into(self) -> (i32, i32) {
        (self.i, self.k)
    }
}

impl TryFrom<(i32, i32, i32)> for Hex {
    type Error = String;

    /// Creates a Hex from (i, j, k) tuple
    /// 
    /// ## Parameters
    /// - `(i, j, k)`: Tuple representing the (i, j, k) coordinates
    /// 
    /// ## Returns
    /// A new `Hex` instance at the specified coordinates or an error if invalid.
    #[inline]
    fn try_from((i, j, k): (i32, i32, i32)) -> Result<Self, Self::Error> {
        if i - j + k != 0 {
            return Err("Invalid hex coordinates: i - j + k must equal 0".to_string());
        }
        Ok(Hex::new(i, k))
    }
}

impl Into<(i32, i32, i32)> for Hex {
    /// Converts Hex to (i, j, k) tuple
    /// 
    /// ## Returns
    /// A tuple representing the (i, j, k) coordinates of the `Hex`.
    #[inline]
    fn into(self) -> (i32, i32, i32) {
        (self.i, self.j(), self.k)
    }
}

impl fmt::Display for Hex {
    /// Formats Hex for display
    /// 
    /// ## Returns
    /// A string representation of the `Hex` in the format "Hex(i, j, k)".
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hex({}, {}, {})", self.i, self.j(), self.k)
    }
}

impl fmt::Debug for Hex {
    /// Formats Hex for debugging
    /// 
    /// ## Returns
    /// A string representation of the `Hex` in the format "{i, j, k}".
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{{}, {}, {}}}", self.i, self.j(), self.k)
    }
}

impl Into<Vec<u8>> for Hex {
    /// Converts Hex to a vector of bytes
    /// 
    /// The exact format is:
    /// - First 4 bytes: little-endian representation of `i`
    /// - Next 4 bytes: little-endian representation of `k`
    /// 
    /// ## Returns
    /// A vector containing the byte representation of the (i, k) coordinates.
    #[inline]
    fn into(self) -> Vec<u8> {
        let mut vec = Vec::with_capacity(8);
        vec.extend_from_slice(&self.i.to_le_bytes());
        vec.extend_from_slice(&self.k.to_le_bytes());
        vec
    }
}

impl TryFrom<Vec<u8>> for Hex {
    type Error = String;

    /// Creates a Hex from a vector of bytes
    /// 
    /// The expected format is:
    /// - First 4 bytes: little-endian representation of `i`
    /// - Next 4 bytes: little-endian representation of `k`
    /// 
    /// ## Parameters
    /// - `bytes`: Vector of bytes representing the (i, k) coordinates
    /// 
    /// ## Returns
    /// A new `Hex` instance or an error if the byte vector is invalid.
    fn try_from(bytes: Vec<u8>) -> Result<Self, Self::Error> {
        if bytes.len() != 8 {
            return Err("Byte vector must be exactly 8 bytes long".to_string());
        }
        let i = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let k = i32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        Ok(Hex::new(i, k))
    }
}

/// Represents a shape or unit made up of multiple blocks in a fixed hexagonal pattern.
///
/// A `Piece` is a compact, immutable structure representing a small, self-contained hexagonal grid of up to 7 blocks.
/// Each block is positioned at a standard location relative to the origin, and occupancy is encoded as a 7-bit bitfield.
///
/// # Structure
///
/// - The 7 possible block positions are defined by the `Piece::POSITIONS` array, using line coordinates (i, k) as in [`Hex`].
/// - The bitfield (`states`) encodes which of these positions are occupied (1 = occupied, 0 = empty).
/// - The piece is always centered at (0, 0) in line coordinates, and the positions form a radius-2 hex grid.
///
/// # Usage
///
/// - Pieces are typically created using [`Piece::new`] (from a bitfield) or [`Piece::from_bools`] (from a boolean array).
/// - Once created, a `Piece` is immutable. To modify, create a new piece or use the bitfield to construct a variant.
/// - Coordinate access and block queries use the line coordinate system (i, k), simplifying navigation and manipulation.
///
/// ## Example
///
/// ```rust
/// // Create a piece with three blocks at (0,0), (0,1), and (1,1):
/// let mut states = [false; 7];
/// states[3] = true; // (0,0)
/// states[4] = true; // (0,1)
/// states[6] = true; // (1,1)
/// let piece = Piece::from_bools(&states);
/// assert_eq!(piece.count(), 3);
/// ```
///
/// # Standard 7-block Piece
///
/// The standard piece covers the following positions (line coordinates):
///
/// - (-1, -1)
/// - (-1, 0)
/// - (0, -1)
/// - (0, 0) *(center)*
/// - (0, 1)
/// - (1, 0)
/// - (1, 1)
///
/// This forms a radius-2 hex grid centered at (0, 0).
///
/// # Notes
///
/// - Pieces can represent any subset of these 7 positions (including empty or single-block pieces).
/// - For efficiency, cloning is trivial and preferred over mutation; all methods return new values.
/// - The design is optimized for fast bitwise operations and compact storage.
///
/// Designed by William Wu. Adapted for Rust.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Piece {
    /// Bitfield: bits 0-6 represent occupancy of each position
    /// Bit 7 is unused (pieces only have 7 blocks)
    states: u8,
}

impl Default for Piece {
    /// Creates a default empty Piece
    #[inline]
    fn default() -> Self {
        Piece { states: 0 }
    }
}

impl Piece {
    /// Standard positions for the 7 blocks in a piece
    /// 
    /// The values are ordered to correspond to the bitfield representation.
    /// 
    /// ## Values
    /// - Index 0: (-1, -1)
    /// - Index 1: (-1, 0)
    /// - Index 2: (0, -1)
    /// - Index 3: (0, 0) (center)
    /// - Index 4: (0, 1)
    /// - Index 5: (1, 0)
    /// - Index 6: (1, 1)
    pub const POSITIONS: [Hex; 7] = [
        Hex { i: -1, k: -1 },
        Hex { i: -1, k: 0 },
        Hex { i: 0, k: -1 },
        Hex { i: 0, k: 0 },   // Center
        Hex { i: 0, k: 1 },
        Hex { i: 1, k: 0 },
        Hex { i: 1, k: 1 },
    ];

    /// Creates a new piece from a bitfield (0-127)
    /// 
    /// ## Parameters
    /// - `states`: Bitfield representing block occupancy (0-127)
    /// 
    /// ## Returns
    /// A new `Piece` instance with the specified occupancy.
    /// 
    /// ## See Also
    /// - [`Piece::from_bools`] for creating from a boolean array.
    #[inline]
    pub const fn new(states: u8) -> Self {
        debug_assert!(states < 128, "Piece state must be 0-127");
        Piece { states: states & 0x7F }
    }

    /// Creates a piece from a boolean array
    /// 
    /// ## Parameters
    /// - `states`: Array of 7 booleans representing block occupancy
    /// 
    /// ## Returns
    /// A new `Piece` instance with the specified occupancy.
    pub fn from_bools(states: &[bool; 7]) -> Self {
        let mut bits = 0u8;
        for (i, &state) in states.iter().enumerate() {
            if state {
                bits |= 1 << (6 - i);
            }
        }
        Piece { states: bits }
    }

    /// Returns the bitfield representation
    /// 
    /// ## Returns
    /// The bitfield as a u8 value.
    /// 
    /// ## Guarantees
    /// The returned value will always be in the range 0-127.
    #[inline]
    pub const fn as_u8(&self) -> u8 {
        self.states
    }

    /// Checks if a specific position is occupied
    /// 
    /// ## Parameters
    /// - `index`: Position index (0-6)
    /// 
    /// ## Returns
    /// - `Some(true)` if occupied
    /// - `Some(false)` if unoccupied
    /// - `None` if index is out of range
    #[inline]
    pub const fn is_occupied(&self, index: usize) -> Option<bool> {
        match index {
            0..=6 => Some((self.states & (1 << (6 - index))) != 0),
            _ => None,
        }
    }

    /// Checks if a specific position is occupied (unsafe)
    /// 
    /// ## Parameters
    /// - `index`: Position index (0-6)
    /// 
    /// ## Returns
    /// `true` if occupied, `false` if unoccupied.
    /// 
    /// ## Panics
    /// Panics if index is out of range.
    #[inline]
    pub const fn is_occupied_unsafe(&self, index: usize) -> bool {
        (self.states & (1 << (6 - index))) != 0
    }

    /// Returns the number of occupied blocks
    /// 
    /// ## Returns
    /// The count of occupied blocks as a u32.
    #[inline]
    pub const fn count(&self) -> u32 {
        self.states.count_ones()
    }

    /// Checks if piece has any occupied blocks
    /// 
    /// ## Returns
    /// `true` if no blocks are occupied, `false` otherwise.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.states == 0
    }

    /// Returns coordinates of all occupied blocks
    /// 
    /// ## Returns
    /// A vector of `Hex` coordinates for occupied blocks.
    pub fn coordinates(&self) -> Vec<Hex> {
        Self::POSITIONS
            .iter()
            .enumerate()
            .filter(|(i, _)| self.is_occupied_unsafe(*i))
            .map(|(_, &pos)| pos)
            .collect()
    }

    /// Counts occupied neighbors around a given coordinate within this piece
    /// 
    /// ## Parameters
    /// - `coo`: The center coordinate to check neighbors around
    /// 
    /// ## Returns
    /// The count of occupied neighboring blocks as a u32.
    pub fn count_neighbors(&self, coo: Hex) -> u32 {
        if let Some(center_idx) = Self::POSITIONS.iter().position(|&p| p == coo) {
            if !self.is_occupied_unsafe(center_idx) {
                return 0;
            }

            let mut count = 0;
            for (i, &pos) in Self::POSITIONS.iter().enumerate() {
                if i != center_idx && self.is_occupied_unsafe(i) {
                    // Check if this position is adjacent to coo
                    let diff = pos - coo;
                    let dist = diff.i.abs().max(diff.j().abs()).max(diff.k.abs());
                    if dist == 1 {
                        count += 1;
                    }
                }
            }
            count
        } else {
            0
        }
    }

    /// Returns an iterator over all non-empty pieces (1-127)
    /// 
    /// ## Returns
    /// An iterator yielding all possible non-empty `Piece` instances.
    pub fn all_pieces() -> impl Iterator<Item = Piece> {
        (1u8..128).map(Piece::new)
    }

    /// Checks if a piece is spatially contiguous
    /// 
    /// ## Returns
    /// `true` if the piece is contiguous, `false` otherwise.
    pub fn is_contiguous(&self) -> bool {
        if self.is_empty() {
            return false;
        }
        
        let count = self.count();
        
        // Center occupied = always contiguous
        if self.is_occupied_unsafe(3) {
            return true;
        }
        
        // 1, 5, or 6 blocks = always contiguous
        if count == 1 || count == 5 || count == 6 {
            return true;
        }
        
        // 2 blocks: check adjacency
        if count == 2 {
            let s = self.states;
            return (s & 0b1000000 != 0 && (s & 0b0100000 != 0 || s & 0b0010000 != 0))
                || (s & 0b0000100 != 0 && (s & 0b0100000 != 0 || s & 0b0000001 != 0))
                || (s & 0b0000010 != 0 && (s & 0b0010000 != 0 || s & 0b0000001 != 0));
        }
        
        // 4 blocks: check that missing blocks aren't adjacent
        if count == 4 {
            let s = !self.states & 0x7F;
            return (s & 0b1000000 == 0 || (s & 0b0100000 == 0 && s & 0b0010000 == 0))
                && (s & 0b0000100 == 0 || (s & 0b0100000 == 0 && s & 0b0000001 == 0))
                && (s & 0b0000010 == 0 || (s & 0b0010000 == 0 && s & 0b0000001 == 0));
        }
        
        // 3 blocks: check if all adjacent
        if count == 3 {
            let s = self.states;
            return (s & 0b1000000 != 0 && s & 0b0100000 != 0 && (s & 0b0010000 != 0 || s & 0b0000100 != 0))
                || (s & 0b0010000 != 0 && s & 0b0000010 != 0 && (s & 0b1000000 != 0 || s & 0b0000001 != 0))
                || (s & 0b0000100 != 0 && s & 0b0000001 != 0 && (s & 0b0100000 != 0 || s & 0b0000010 != 0));
        }
        
        false
    }

    /// Returns an iterator over all contiguous pieces
    /// 
    /// ## Returns
    /// An iterator yielding all contiguous `Piece` instances.
    pub fn contiguous_pieces() -> impl Iterator<Item = Piece> {
        Self::all_pieces().filter(|p| p.is_contiguous())
    }
}

impl fmt::Display for Piece {
    /// Formats Piece for display
    /// 
    /// The format is: `Piece{(i, k, occupied), ...}` for each position.
    /// 
    /// ## Returns
    /// A string representation of the `Piece` showing occupied positions.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Piece{{")?;
        for (i, &pos) in Self::POSITIONS.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "({}, {}, {})", pos.i, pos.k, self.is_occupied_unsafe(i))?;
        }
        write!(f, "}}")
    }
}

impl fmt::Debug for Piece {
    /// Formats Piece for debugging
    /// 
    /// The format is: `{ code: <bitfield>, count: <count>, blocks: {(i, k, occupied), ...} }`.
    ///
    /// ## Returns
    /// A string representation of the `Piece` with detailed information.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{ code: {}, count: {}, blocks: [", self.states, self.count())?;
        for (i, &pos) in Self::POSITIONS.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "({}, {}, {})", pos.i, pos.k, self.is_occupied_unsafe(i))?;
        }
        write!(f, "] }}")
    }
}

impl From<u8> for Piece {
    /// Creates a Piece from a bitfield
    /// 
    /// ## Parameters
    /// - `value`: Bitfield representing block occupancy (0-127)
    #[inline]
    fn from(value: u8) -> Self {
        Piece::new(value)
    }
}

impl Into<u8> for Piece {
    /// Converts Piece to a bitfield
    /// 
    /// ## Returns
    /// The bitfield as a u8 value.
    #[inline]
    fn into(self) -> u8 {
        self.as_u8()
    }
}

impl Into<[bool; 7]> for Piece {
    /// Converts Piece to a boolean array
    /// 
    /// ## Returns
    /// An array of 7 booleans representing block occupancy.
    #[inline]
    fn into(self) -> [bool; 7] {
        let mut states = [false; 7];
        for i in 0..7 {
            states[i] = self.is_occupied_unsafe(i);
        }
        states
    }
}

impl From <[bool; 7]> for Piece {
    /// Creates a Piece from a boolean array
    /// 
    /// ## Parameters
    /// - `states`: Array of 7 booleans representing block occupancy
    #[inline]
    fn from(states: [bool; 7]) -> Self {
        Piece::from_bools(&states)
    }
}

impl Into<Vec<Hex>> for Piece {
    /// Converts Piece to a vector of occupied coordinates
    /// 
    /// ## Returns
    /// A vector of `Hex` coordinates for occupied blocks.
    #[inline]
    fn into(self) -> Vec<Hex> {
        self.coordinates()
    }   
}

impl Into<Vec<u8>> for Piece {
    /// Converts Piece to a vector of bytes
    /// 
    /// The exact format is:
    /// - First byte: bitfield representing block occupancy (0-127)
    /// 
    /// ## Returns
    /// A vector containing the byte representation of the piece.
    #[inline]
    fn into(self) -> Vec<u8> {
        vec![self.states]
    }
}

impl TryFrom<Vec<u8>> for Piece {
    type Error = String;

    /// Creates a Piece from a vector of bytes
    /// 
    /// The expected format is:
    /// - First byte: bitfield representing block occupancy (0-127)
    /// 
    /// ## Parameters
    /// - `bytes`: Vector of bytes representing the piece
    /// 
    /// ## Returns
    /// A new `Piece` instance or an error if the byte vector is invalid.
    fn try_from(bytes: Vec<u8>) -> Result<Self, Self::Error> {
        if bytes.len() != 1 {
            return Err("Byte vector must be exactly 1 byte long".to_string());
        }
        let states = bytes[0];
        if states >= 128 {
            return Err("Piece state must be in range 0-127".to_string());
        }
        Ok(Piece::new(states))
    }
}

impl TryFrom<Vec<Hex>> for Piece {
    type Error = String;

    /// Creates a Piece from a vector of coordinates
    /// 
    /// ## Parameters
    /// - `coords`: Vector of `Hex` coordinates representing occupied blocks
    /// 
    /// ## Returns
    /// A new `Piece` instance or an error if any coordinate is invalid.
    fn try_from(coords: Vec<Hex>) -> Result<Self, Self::Error> {
        let mut bits = 0u8;
        let mut seen = [false; 7];
        for coo in coords {
            if let Some(idx) = Piece::POSITIONS.iter().position(|&p| p == coo) {
            if seen[idx] {
                return Err(format!("Duplicate coordinate for piece: {}", coo));
            }
            seen[idx] = true;
            bits |= 1 << (6 - idx);
            } else {
            return Err(format!("Invalid coordinate for piece: {}", coo));
            }
        }
        Ok(Piece { states: bits })
    }
}

/// The `HexEngine` struct provides a complete engine for managing a two-dimensional hexagonal block grid.
///
/// This engine is used for constructing and interacting with hex-based shapes, such as in games or ML environments.
/// It maintains a vector of block states (occupied/unoccupied) arranged in a hexagonal pattern, and provides efficient
/// operations for grid initialization, placement, validation, elimination, and analysis.
///
/// # Grid Structure
///
/// - The grid uses an axial coordinate system (i, k), where `i - j + k = 0` and `j = k - i` (see [`Hex`]).
/// - The three axes (I, J, K) are 60° apart, forming a true hexagonal lattice.
/// - Blocks are stored in a vector, sorted by increasing i then k (raw coordinates), allowing O(1) index-based access.
/// - The total number of blocks for radius `r` is: `1 + 3 * r * (r - 1)`.
///
/// # Key Operations
///
/// - Grid initialization and reset ([`HexEngine::new`], [`reset`])
/// - Efficient block lookup and coordinate-index conversion ([`get`], [`set`], [`index_of`], [`coordinate_of`])
/// - Placement validation and piece insertion ([`check_add`], [`add_piece`])
/// - Line detection and elimination across I/J/K axes ([`eliminate`])
/// - Deep copy via [`Clone`]
///
/// # Block Coloring
///
/// - By default, blocks are visually represented as boolean states (false = empty, true = filled).
/// - State changes (via [`set`] or [`reset`]) update the block's state; color management is left to the UI layer.
///
/// # Machine Learning Utility
///
/// The engine exposes utility methods for evaluating the quality and validity of in-game actions, supporting reinforcement learning:
///
/// - [`check_add`] returns false for invalid placements (e.g., overlap or out-of-bounds), useful for negative rewards.
/// - [`compute_density`] computes a normalized score (0 to 1) for how densely a piece would interact with neighbors, encouraging efficient placement.
/// - [`compute_entropy`] computes the Shannon entropy of the grid, rewarding moves that simplify the board.
///
/// # Example
///
/// ```rust
/// let mut engine = HexEngine::new(2);
/// let piece = Piece::new(0b1010101);
/// let pos = Hex::new(0, 0);
/// if engine.check_add(pos, piece) {
///     engine.add_piece(pos, piece).unwrap();
/// }
/// let eliminated = engine.eliminate();
/// let entropy = engine.compute_entropy();
/// ```
///
/// # Notes
///
/// - The coordinate system and grid structure are optimized for fast, index-based access and efficient algorithms.
/// - See [`Hex`] for more on the coordinate system.
///
/// Designed by William Wu. Adapted for Rust.
#[derive(PartialEq, Eq)]
pub struct HexEngine {
    radius: usize,
    states: Vec<bool>,
}

impl Default for HexEngine {
    /// Creates a default empty HexEngine with radius 0
    #[inline]
    fn default() -> Self {
        HexEngine::new(0)
    }
}

impl HexEngine {
    /// Calculates grid length from radius
    /// 
    /// ## Parameters
    /// - `radius`: The radius of the hexagonal grid
    #[inline]
    pub const fn calc_length(radius: usize) -> usize {
        if radius == 0 {
            0
        } else {
            1 + 3 * radius * (radius - 1)
        }
    }

    /// Calculates radius from grid length (returns None if invalid)
    /// 
    /// ## Parameters
    /// - `length`: The length of the grid
    /// 
    /// ## Returns
    /// An `Option<usize>` containing the radius if valid, or `None` if invalid.
    pub fn calc_radius(length: usize) -> Option<usize> {
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

    /// Creates a new empty grid with given radius
    /// 
    /// ## Parameters
    /// - `radius`: The radius of the hexagonal grid
    /// 
    /// ## Returns
    /// A new `HexEngine` instance with all blocks unoccupied.
    pub fn new(radius: usize) -> Self {
        let length = Self::calc_length(radius);
        HexEngine {
            radius,
            states: vec![false; length],
        }
    }

    /// Creates a grid from a boolean vector
    /// 
    /// ## Parameters
    /// - `states`: Vector of booleans representing block occupancy
    /// 
    /// ## Returns
    /// A `Result<HexEngine, String>` containing the new grid or an error if invalid.
    pub fn from_states(states: Vec<bool>) -> Result<Self, String> {
        let radius = Self::calc_radius(states.len())
            .ok_or_else(|| format!("Invalid state length: {}", states.len()))?;
        Ok(HexEngine { radius, states })
    }

    /// Creates a grid from a binary string ("0"/"1" or "X"/"O")
    /// 
    /// ## Parameters
    /// - `s`: String representing block occupancy
    /// 
    /// ## Returns
    /// A `Result<HexEngine, String>` containing the new grid or an error if invalid.
    pub fn from_string(s: &str) -> Result<Self, String> {
        let states: Vec<bool> = s
            .chars()
            .map(|c| match c {
                '1' | 'X' => Ok(true),
                '0' | 'O' => Ok(false),
                _ => Err(format!("Invalid character: {}", c)),
            })
            .collect::<Result<_, _>>()?;
        
        Self::from_states(states)
    }

    /// Returns the radius of the grid
    /// 
    /// ## Returns
    /// The radius as a usize.
    #[inline]
    pub fn radius(&self) -> usize {
        self.radius
    }

    /// Returns the length of the grid
    /// 
    /// This is the number of blocks in the hexagonal grid.
    /// 
    /// ## Returns
    /// The length as a usize.
    #[inline]
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Checks if the grid is empty
    ///
    /// ## Returns
    /// `true` if the grid has no blocks, `false` otherwise.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Checks if a coordinate is within grid bounds
    /// 
    /// ## Parameters
    /// - `coo`: The `Hex` coordinate to check
    /// 
    /// ## Returns
    /// `true` if the coordinate is within bounds, `false` otherwise.
    #[inline]
    pub fn in_range(&self, coo: Hex) -> bool {
        let r = self.radius as i32;
        coo.i >= 0
            && coo.i < 2 * r - 1
            && coo.j() > -r
            && coo.j() < r
            && coo.k >= 0
            && coo.k < 2 * r - 1
    }

    /// Converts coordinate to linear index
    /// 
    /// This method provides O(1) conversion from a `Hex` coordinate to a linear index in the internal state vector.
    /// 
    /// ## Parameters
    /// - `coo`: The `Hex` coordinate to convert
    /// 
    /// ## Returns
    /// An `Option<usize>` containing the index if in range, or `None` if out of bounds.
    #[inline]
    pub fn index_of(&self, coo: Hex) -> Option<usize> {
        if !self.in_range(coo) {
            return None;
        }

        let r = self.radius as i32;
        let i = coo.i;
        let k = coo.k;

        let idx = if i < r {
            k + i * r + i * (i - 1) / 2
        } else {
            k - (r - 1) * (r - 1) + i * r * 3 - i * (i + 5) / 2
        };

        Some(idx as usize)
    }

    /// Converts linear index to coordinate
    /// 
    /// This method provides efficient conversion from a linear index in the internal state vector to a `Hex` coordinate.
    /// 
    /// ## Parameters
    /// - `index`: The linear index to convert
    /// 
    /// ## Returns
    /// An `Option<Hex>` containing the coordinate if index is valid, or `None` if out of bounds.
    #[inline]
    pub fn coordinate_of(&self, mut index: usize) -> Option<Hex> {
        if index >= self.states.len() {
            return None;
        }

        let r = self.radius as i32;
        
        // First half
        for i in 0..r {
            let len = (i + r) as usize;
            if index < len {
                return Some(Hex::new(i, index as i32));
            }
            index -= len;
        }
        
        // Second half
        for i in 0..(r - 1) {
            let len = (2 * r - 2 - i) as usize;
            if index < len {
                return Some(Hex::new(i + r, index as i32 + i + 1));
            }
            index -= len;
        }
        
        None
    }

    /// Gets the state at a coordinate
    /// 
    /// ## Parameters
    /// - `coo`: The `Hex` coordinate to query
    /// 
    /// ## Returns
    /// An `Option<bool>` containing the state if in range, or `None` if out of bounds.
    #[inline]
    pub fn get(&self, coo: Hex) -> Option<bool> {
        self.index_of(coo).map(|idx| self.states[idx])
    }

    /// Sets the state at a coordinate
    /// 
    /// ## Parameters
    /// - `coo`: The `Hex` coordinate to set
    /// - `state`: The state to set (true = occupied, false = unoccupied)
    /// 
    /// ## Returns
    /// A `Result<(), String>` indicating success or an error if out of bounds.
    #[inline]
    pub fn set(&mut self, coo: Hex, state: bool) -> Result<(), String> {
        let idx = self.index_of(coo)
            .ok_or_else(|| format!("Coordinate out of range: {}", coo))?;
        self.states[idx] = state;
        Ok(())
    }

    /// Resets all blocks to unoccupied
    pub fn reset(&mut self) {
        self.states.fill(false);
    }

    /// Checks if a piece can be added at the given position
    /// 
    /// ## Parameters
    /// - `coo`: The `Hex` coordinate to add the piece at
    /// - `piece`: The `Piece` to add
    /// 
    /// ## Returns
    /// `true` if the piece can be added without overlap or going out of bounds, `false` otherwise.
    pub fn check_add(&self, coo: Hex, piece: Piece) -> bool {
        for (i, &pos) in Piece::POSITIONS.iter().enumerate() {
            if piece.is_occupied_unsafe(i) {
                let target = coo + pos;
                match self.get(target) {
                    Some(true) => return false,  // Overlap
                    None => return false,        // Out of bounds
                    Some(false) => {}            // OK
                }
            }
        }
        true
    }

    /// Adds a piece at the given position
    /// 
    /// ## Parameters
    /// - `coo`: The `Hex` coordinate to add the piece at
    /// - `piece`: The `Piece` to add
    /// 
    /// ## Returns
    /// A `Result<(), String>` indicating success or an error if the piece cannot be added.
    pub fn add_piece(&mut self, coo: Hex, piece: Piece) -> Result<(), String> {
        if !self.check_add(coo, piece) {
            return Err("Cannot add piece: overlap or out of bounds".to_string());
        }

        for (i, &pos) in Piece::POSITIONS.iter().enumerate() {
            if piece.is_occupied_unsafe(i) {
                self.set(coo + pos, true)?;
            }
        }
        Ok(())
    }

    /// Returns all valid positions where a piece can be added
    /// 
    /// ## Parameters
    /// - `piece`: The `Piece` to check for valid positions
    /// 
    /// ## Returns
    /// A vector of `Hex` coordinates where the piece can be added.
    pub fn valid_positions(&self, piece: Piece) -> Vec<Hex> {
        let r = self.radius as i32;
        let mut positions = Vec::new();
        
        for i in 0..(2 * r) {
            for k in 0..(2 * r) {
                let hex = Hex::new(i, k);
                if self.check_add(hex, piece) {
                    positions.push(hex);
                }
            }
        }
        positions
    }

    /// Counts occupied neighbors around a coordinate
    /// 
    /// ## Parameters
    /// - `coo`: The `Hex` coordinate to check neighbors around
    /// 
    /// ## Returns
    /// The count of occupied neighboring blocks as a u32.
    pub fn count_neighbors(&self, coo: Hex) -> u32 {
        let mut count = 0;
        for &offset in &Piece::POSITIONS {
            let target = coo + offset;
            match self.get(target) {
                Some(true) => count += 1,
                None => count += 1,  // Out of bounds counts as neighbor
                Some(false) => {}
            }
        }
        count
    }

    /// Computes density index for placing a piece at a position
    /// 
    /// ## Parameters
    /// - `coo`: The `Hex` coordinate to place the piece at
    /// - `piece`: The `Piece` to evaluate
    /// 
    /// ## Returns
    /// A density index as a `f32` between 0.0 and 1.0, or 0.0 if placement is invalid.
    pub fn compute_density(&self, coo: Hex, piece: Piece) -> f32 {
        let mut total_possible = 0;
        let mut total_populated = 0;

        for (i, &pos) in Piece::POSITIONS.iter().enumerate() {
            if piece.is_occupied_unsafe(i) {
                let placed = coo + pos;
                
                if self.get(placed) != Some(false) {
                    return 0.0;  // Invalid placement
                }

                total_possible += 6 - piece.count_neighbors(pos);
                total_populated += self.count_neighbors(placed);
            }
        }

        if total_possible > 0 {
            total_populated as f32 / total_possible as f32
        } else {
            0.0
        }
    }

    /// Eliminates fully occupied lines and returns eliminated coordinates
    /// 
    /// ## Returns
    /// A vector of `Hex` coordinates that were eliminated.
    pub fn eliminate(&mut self) -> Vec<Hex> {
        let mut eliminated = Vec::new();
        
        self.eliminate_i(&mut eliminated);
        self.eliminate_j(&mut eliminated);
        self.eliminate_k(&mut eliminated);
        
        for &coo in &eliminated {
            let _ = self.set(coo, false);
        }
        
        eliminated
    }

    /// Eliminates fully occupied i-lines and appends eliminated coordinates
    /// 
    /// ## Parameters
    /// - `eliminated`: Mutable vector to append eliminated coordinates
    fn eliminate_i(&self, eliminated: &mut Vec<Hex>) {
        let r = self.radius as i32;
        
        // First half
        for i in 0..r {
            let start_idx = (i * (r * 2 + i - 1) / 2) as usize;
            let len = (r + i) as usize;
            
            if (0..len).all(|b| self.states.get(start_idx + b) == Some(&true)) {
                for b in 0..len {
                    if let Some(coo) = self.coordinate_of(start_idx + b) {
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
                    if let Some(coo) = self.coordinate_of(start_idx + b) {
                        eliminated.push(coo);
                    }
                }
            }
        }
    }

    /// Eliminates fully occupied j-lines and appends eliminated coordinates
    /// 
    /// ## Parameters
    /// - `eliminated`: Mutable vector to append eliminated coordinates
    fn eliminate_j(&self, eliminated: &mut Vec<Hex>) {
        let radius = self.radius as i32;
        
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
                    if let Some(coo) = self.coordinate_of(idx) {
                        eliminated.push(coo);
                    }
                    idx += (radius + c) as usize;
                }
                for c in 0..(radius - r) {
                    if let Some(coo) = self.coordinate_of(idx) {
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
                    if let Some(coo) = self.coordinate_of(idx) {
                        eliminated.push(coo);
                    }
                    idx += (radius + c + r) as usize;
                }
                for c in 0..radius {
                    if let Some(coo) = self.coordinate_of(idx) {
                        eliminated.push(coo);
                    }
                    idx += (2 * radius - c - 1) as usize;
                }
            }
        }
    }

    /// Eliminates fully occupied lines in the k direction
    /// 
    /// ## Parameters
    /// - `eliminated`: Mutable reference to a vector to store eliminated coordinates
    fn eliminate_k(&self, eliminated: &mut Vec<Hex>) {
        let radius = self.radius as i32;
        
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
                    if let Some(coo) = self.coordinate_of(idx) {
                        eliminated.push(coo);
                    }
                    idx += (radius + c) as usize;
                }
                for c in 0..(r + 1) {
                    if let Some(coo) = self.coordinate_of(idx) {
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
                    if let Some(coo) = self.coordinate_of(idx) {
                        eliminated.push(coo);
                    }
                    idx += (radius + c) as usize;
                }
                for c in (0..radius).rev() {
                    if let Some(coo) = self.coordinate_of(idx) {
                        eliminated.push(coo);
                    }
                    idx += (radius + c - 1) as usize;
                }
            }
        }
    }

    /// Computes Shannon entropy of the grid
    /// 
    /// ## Returns
    /// The Shannon entropy as a `f32`.
    pub fn compute_entropy(&self) -> f32 {
        let mut pattern_counts = [0u32; 128];
        let mut total = 0u32;
        let radius = (self.radius - 1) as i32;

        for i in 0..self.states.len() {
            if let Some(center) = self.coordinate_of(i) {
                if self.in_range_with_radius(center.shift_j(1), radius) {
                    let pattern = self.get_pattern(center);
                    pattern_counts[pattern as usize] += 1;
                    total += 1;
                }
            }
        }

        if total == 0 {
            return 0.0;
        }

        pattern_counts
            .iter()
            .filter(|&&count| count > 0)
            .map(|&count| {
                let p = count as f32 / total as f32;
                -p * p.log2()
            })
            .sum()
    }

    /// Checks if a coordinate is within bounds given a radius
    fn in_range_with_radius(&self, coo: Hex, radius: i32) -> bool {
        coo.i >= 0
            && coo.i < 2 * radius - 1
            && coo.j() > -radius
            && coo.j() < radius
            && coo.k >= 0
            && coo.k < 2 * radius - 1
    }

    /// Gets the 7-bit pattern around a coordinate
    fn get_pattern(&self, coo: Hex) -> u8 {
        let mut pattern = 0u8;
        for &pos in &Piece::POSITIONS {
            pattern <<= 1;
            if self.get(coo + pos) == Some(true) {
                pattern |= 1;
            }
        }
        pattern
    }

    /// Returns the piece pattern around a coordinate
    /// 
    /// ## Parameters
    /// - `coo`: The `Hex` coordinate to get the pattern for
    /// 
    /// ## Returns
    /// An `Option<Piece>` containing the piece pattern if in range, or `None` if out of bounds.
    pub fn pattern_as_piece(&self, coo: Hex) -> Option<Piece> {
        if !self.in_range_with_radius(coo, (self.radius - 1) as i32) {
            return None;
        }
        let pattern = self.get_pattern(coo);
        Some(Piece::new(pattern))
    }

    /// Returns binary string representation
    /// 
    /// ## Returns
    /// A string of '0's and '1's representing block occupancy.
    pub fn to_binary_string(&self) -> String {
        self.states.iter().map(|&b| if b { '1' } else { '0' }).collect()
    }

    /// Returns an iterator over the grid states
    /// 
    /// ## Returns
    /// A `HexEngineIterator` for iterating over block states.
    pub fn iter(&self) -> HexEngineIterator<'_> {
        HexEngineIterator {
            engine: self,
            index: 0,
        }
    }
}

/// Iterator for HexEngine
/// 
/// Yields:
/// - `(Hex, bool)`: Tuple of coordinate and its state (occupied/unoccupied)
pub struct HexEngineIterator<'a> {
    engine: &'a HexEngine,
    index: usize,
}

impl<'a> HexEngineIterator<'a> {
    /// Creates a new HexEngineIterator
    /// 
    /// ## Parameters
    /// - `engine`: Reference to the HexEngine to iterate over
    pub fn new(engine: &'a HexEngine) -> Self {
        HexEngineIterator {
            engine,
            index: 0,
        }
    }
}

impl <'a> Iterator for HexEngineIterator<'a> {
    type Item = (Hex, bool);

    /// Returns the next coordinate and its state
    /// 
    /// ## Returns
    /// An `Option<(Hex, bool)>` containing the next coordinate and its state, or `None` if iteration is complete.
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.engine.states.len() {
            return None;
        }
        let coo = self.engine.coordinate_of(self.index)?;
        let state = self.engine.states[self.index];
        self.index += 1;
        Some((coo, state))
    }
}

impl Clone for HexEngine {
    /// Clones the HexEngine
    /// 
    /// ## Returns
    /// A new `HexEngine` instance with the same radius and states.
    fn clone(&self) -> Self {
        HexEngine {
            radius: self.radius,
            states: self.states.clone(),
        }
    }
}

impl TryFrom<&str> for HexEngine {
    type Error = String;

    /// Creates a HexEngine from a binary string
    /// 
    /// ## Parameters
    /// - `s`: String representing block occupancy
    /// 
    /// ## Returns
    /// A `Result<HexEngine, String>` containing the new grid or an error if invalid.
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        HexEngine::from_string(s)
    }
}

impl TryFrom<Vec<bool>> for HexEngine {
    type Error = String;

    /// Creates a HexEngine from a boolean vector
    /// 
    /// ## Parameters
    /// - `states`: Vector of booleans representing block occupancy
    /// 
    /// ## Returns
    /// A `Result<HexEngine, String>` containing the new grid or an error if invalid.
    fn try_from(states: Vec<bool>) -> Result<Self, Self::Error> {
        HexEngine::from_states(states)
    }
}

impl TryFrom<Vec<u8>> for HexEngine {
    type Error = String;

    /// Creates a HexEngine from a byte vector
    /// 
    /// The format is: 4 byte for radius, followed by packed bits for block states.
    /// 
    /// ## Parameters
    /// - `bytes`: Vector of bytes representing the grid
    /// 
    /// ## Returns
    /// A `Result<HexEngine, String>` containing the new grid or an error if invalid.
    fn try_from(bytes: Vec<u8>) -> Result<Self, Self::Error> {
        if bytes.len() < 4 {
            return Err("Byte vector too short".to_string());
        }
        let radius = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let expected_len = HexEngine::calc_length(radius);
        let mut states = Vec::with_capacity(expected_len);  
        let mut bit_index = 0;
        for byte in &bytes[4..] {
            for i in 0..8 {
                if bit_index >= expected_len {
                    break;
                }
                let bit = (byte >> (7 - i)) & 1;
                states.push(bit == 1);
                bit_index += 1;
            }
        }
        if states.len() != expected_len {
            return Err("Byte vector does not match expected length".to_string());
        }
        Ok(HexEngine { radius, states })
    }
}

impl TryFrom<Vec<Hex>> for HexEngine {
    type Error = String;

    /// Creates a HexEngine from a vector of occupied coordinates
    /// 
    /// ## Parameters
    /// - `coords`: Vector of `Hex` coordinates representing occupied blocks
    /// 
    /// ## Returns
    /// A `Result<HexEngine, String>` containing the new grid or an error if invalid.
    fn try_from(coords: Vec<Hex>) -> Result<Self, Self::Error> {
        let mut max_i = 0;
        let mut max_k = 0;

        for coo in &coords {
            if coo.i > max_i {
                max_i = coo.i;
            }
            if coo.k > max_k {
                max_k = coo.k;
            }
        }

        let radius = ((max_i.max(max_k) + 1) as f64 / 2.0).ceil() as usize;
        let mut engine = HexEngine::new(radius);

        for coo in coords {
            engine.set(coo, true)?;
        }

        Ok(engine)
    }
}

impl std::hash::Hash for HexEngine {
    /// Hashes the HexEngine
    /// 
    /// The hash is the same as hashing the internal state vector.
    /// 
    /// ## Parameters
    /// - `state`: The hasher to write to
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.states.hash(state);
    }
}

impl AsRef<Vec<bool>> for HexEngine {
    /// Returns a reference to the internal state vector
    /// 
    /// ## Returns
    /// A reference to the vector of booleans representing block occupancy.
    fn as_ref(&self) -> &Vec<bool> {
        &self.states
    }
}

impl AsMut<Vec<bool>> for HexEngine {
    /// Returns a mutable reference to the internal state vector
    /// 
    /// ## Returns
    /// A mutable reference to the vector of booleans representing block occupancy.
    fn as_mut(&mut self) -> &mut Vec<bool> {
        &mut self.states
    }
}

impl Borrow<Vec<bool>> for HexEngine {
    /// Borrows the internal state vector
    /// 
    /// The borrow can occur because HexEngine is just a wrapper around the vector,
    /// and they have the same hash and equality semantics.
    /// 
    /// However, any boolean vector cannot be borrowed as a HexEngine,
    /// since it may not satisfy the HexEngine invariants. Use `TryFrom` instead.
    /// 
    /// ## Returns
    /// A reference to the vector of booleans representing block occupancy.
    fn borrow(&self) -> &Vec<bool> {
        &self.states
    }
}

impl Into<String> for HexEngine {
    /// Converts HexEngine to binary string
    /// 
    /// ## Returns
    /// A string of '0's and '1's representing block occupancy.
    fn into(self) -> String {
        self.to_binary_string()
    }
}

impl Into<Vec<bool>> for HexEngine {
    /// Converts HexEngine to boolean vector
    /// 
    /// ## Returns
    /// A vector of booleans representing block occupancy.
    fn into(self) -> Vec<bool> {
        self.states
    }
}

impl Into<Vec<u8>> for HexEngine {
    /// Converts HexEngine to byte vector
    /// 
    /// The format is: 4 byte for radius, followed by packed bits for block states.
    /// 
    /// ## Returns
    /// A vector of bytes representing the grid.
    fn into(self) -> Vec<u8> {
        let mut bytes = Vec::new();
        let radius_bytes = (self.radius as u32).to_le_bytes();
        bytes.extend_from_slice(&radius_bytes);

        let mut current_byte = 0u8;
        let mut bit_count = 0;

        for &state in &self.states {
            current_byte <<= 1;
            if state {
                current_byte |= 1;
            }
            bit_count += 1;

            if bit_count == 8 {
                bytes.push(current_byte);
                current_byte = 0;
                bit_count = 0;
            }
        }

        if bit_count > 0 {
            current_byte <<= 8 - bit_count;
            bytes.push(current_byte);
        }

        bytes
    }
}

impl Into<Vec<Hex>> for HexEngine {
    /// Converts HexEngine to vector of occupied coordinates
    /// 
    /// ## Returns
    /// A vector of `Hex` coordinates representing occupied blocks.
    fn into(self) -> Vec<Hex> {
        let mut coords = Vec::new();
        for (i, &state) in self.states.iter().enumerate() {
            if state {
                if let Some(coo) = self.coordinate_of(i) {
                    coords.push(coo);
                }
            }
        }
        coords
    }
}

impl fmt::Display for HexEngine {
    /// Formats HexEngine for display
    /// 
    /// The format is: `HexEngine[blocks = {(i, k, state), ...}]` for each block.
    /// 
    /// ## Returns
    /// A string representation of the `HexEngine` showing block states.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HexEngine[blocks = {{")?;
        for (i, &state) in self.states.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            if let Some(coo) = self.coordinate_of(i) {
                write!(f, "({}, {}, {})", coo.i, coo.k, state)?;
            }
        }
        write!(f, "}}]")
    }
}

impl std::fmt::Debug for HexEngine {
    /// Formats HexEngine for debugging
    /// 
    /// The format is a visual representation of the hexagonal grid, with 'X' for occupied blocks and 'O' for unoccupied blocks.
    /// 
    /// ## Returns
    /// A string representation of the `HexEngine` in a hexagonal layout.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let r = self.radius;
        let mut idx = 0;
        // Top half
        for i in 0..r {
            // Leading spaces
            for _ in 0..(r - i - 1) {
                write!(f, " ")?;
            }
            // Hexagon blocks
            for _ in 0..(r + i) {
                let c = if self.states[idx] { 'X' } else { 'O' };
                write!(f, "{} ", c)?;
                idx += 1;
            }
            writeln!(f)?;
        }
        // Bottom half
        for i in 0..(r - 1) {
            // Leading spaces
            for _ in 0..(i + 1) {
                write!(f, " ")?;
            }
            // Hexagon blocks
            for _ in 0..(2 * r - 2 - i) {
                let c = if self.states[idx] { 'X' } else { 'O' };
                write!(f, "{} ", c)?;
                idx += 1;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_engine() -> HexEngine {
        HexEngine::new(2)
    }

    #[test]
    fn test_hex_basic() {
        let h = Hex::new(1, 2);
        assert_eq!(h.i(), 1);
        assert_eq!(h.k(), 2);
        assert_eq!(h.j(), 1);
    }

    #[test]
    fn print_engine_debug() {
        let mut engine = HexEngine::new(5);
        engine.set(Hex::new(1, 1), true).unwrap();
        engine.set(Hex::new(2, 2), true).unwrap();
        println!("{:?}", engine);
    }

    #[test]
    fn test_engine_solve_radius() {
        // 1: 1, 7: 2, 19: 3, 37: 4, 61: 5, 91: 6, 127: 7, 169: 8, 217: 9, 271: 10, 331: 11, 397: 12, 469: 13
        let test_cases = [
            (1, Some(1)),
            (7, Some(2)),
            (19, Some(3)),
            (37, Some(4)),
            (61, Some(5)),
            (91, Some(6)),
            (127, Some(7)),
            (169, Some(8)),
            (217, Some(9)),
            (271, Some(10)),
            (331, Some(11)),
            (397, Some(12)),
            (469, Some(13)),
            (10, None),
            (0, Some(0)),
        ];
        for (length, expected_radius) in test_cases.iter() {
            let radius = HexEngine::calc_radius(*length);
            assert_eq!(radius, *expected_radius, "Failed for length {}", length);
        }
    }

    #[test]
    fn test_piece_bitfield() {
        let p = Piece::new(0b1010101);
        assert_eq!(p.count(), 4);
        assert!(p.is_occupied(0).unwrap());
        assert!(!p.is_occupied(1).unwrap());
    }

    #[test]
    fn test_engine_basic() {
        let mut engine = HexEngine::new(2);
        assert_eq!(engine.len(), 7);
        
        let hex = Hex::new(0, 0);
        assert_eq!(engine.get(hex), Some(false));
        
        engine.set(hex, true).unwrap();
        assert_eq!(engine.get(hex), Some(true));
    }

    #[test]
    fn test_in_range() {
        let engine = create_test_engine();
        assert!(engine.in_range(Hex::new(0, 0)), "Coordinate (0,0) should be in range");
        assert!(engine.in_range(Hex::new(1, 1)), "Coordinate (1,1) should be in range");
        assert!(!engine.in_range(Hex::new(3, 3)), "Coordinate (3,3) should be out of range");
    }

    #[test]
    fn test_get_and_set_state() {
        let mut engine = create_test_engine();
        
        // Test initial state
        let state = engine.get(Hex::new(0, 0));
        assert_eq!(state, Some(false), "Block at (0,0) should initially be unoccupied");
        
        // Test coordinate retrieval
        let coo = engine.coordinate_of(0);
        assert!(coo.is_some(), "Should be able to get coordinate at index 0");
        assert_eq!(coo.unwrap().i(), 0, "Block I coordinate should be 0");
        assert_eq!(coo.unwrap().k(), 0, "Block K coordinate should be 0");
        
        // Test setting state
        engine.set(Hex::new(0, 0), true).unwrap();
        assert_eq!(engine.get(Hex::new(0, 0)), Some(true), "Block should be occupied after set");
    }

    #[test]
    fn test_set_state() {
        let mut engine = create_test_engine();
        
        engine.set(Hex::new(0, 0), true).unwrap();
        assert_eq!(engine.get(Hex::new(0, 0)), Some(true), "Block should be occupied");
        
        engine.set(Hex::new(0, 0), false).unwrap();
        assert_eq!(engine.get(Hex::new(0, 0)), Some(false), "Block should be unoccupied");
    }

    #[test]
    fn test_check_positions() {
        let mut engine = create_test_engine();
        
        // Create a piece with just the center block occupied
        let piece = Piece::new(0b0001000); // Only position 3 (center) occupied
        
        let positions = engine.valid_positions(piece);
        assert!(!positions.is_empty(), "There should be valid positions");
        assert!(positions.contains(&Hex::new(0, 0)), "Position (0,0) should be valid");
        
        // Occupy a block and check again
        engine.set(Hex::new(0, 0), true).unwrap();
        let positions = engine.valid_positions(piece);
        assert!(!positions.contains(&Hex::new(0, 0)), "Position (0,0) should be invalid after occupation");
    }

    #[test]
    fn test_eliminate() {
        let mut engine = create_test_engine();
        
        // Set up a full I-line (line 1 has 3 blocks at radius 2)
        engine.set(Hex::new(1, 0), true).unwrap();
        engine.set(Hex::new(1, 1), true).unwrap();
        engine.set(Hex::new(1, 2), true).unwrap();
        
        let eliminated = engine.eliminate();
        assert_eq!(eliminated.len(), 3, "Should eliminate 3 blocks");
        assert_eq!(engine.get(Hex::new(1, 0)), Some(false), "Block at (1,0) should be unoccupied");
        assert_eq!(engine.get(Hex::new(1, 1)), Some(false), "Block at (1,1) should be unoccupied");
        assert_eq!(engine.get(Hex::new(1, 2)), Some(false), "Block at (1,2) should be unoccupied");
        
        // Test with no full lines
        engine.set(Hex::new(0, 0), true).unwrap();
        let eliminated = engine.eliminate();
        assert_eq!(eliminated.len(), 0, "No blocks should be eliminated");
    }

    #[test]
    fn test_check_add() {
        let mut engine = create_test_engine();
        
        // Create a piece with two blocks: center and one to the right
        let piece = Piece::new(0b0001100); // Positions 3 and 4
        let origin = Hex::new(0, 0);
        
        assert!(engine.check_add(origin, piece), "Adding piece at (0,0) should be valid");
        
        engine.set(Hex::new(0, 0), true).unwrap();
        assert!(!engine.check_add(origin, piece), "Adding piece over occupied block should be invalid");
        
        let out_of_range = Hex::new(3, 3);
        assert!(!engine.check_add(out_of_range, piece), "Adding piece out of range should be invalid");
    }

    #[test]
    fn test_add_piece() {
        let mut engine = create_test_engine();
        
        // Create a piece with two blocks: center and one to the right
        let piece = Piece::new(0b0001100); // Positions 3 and 4
        let origin = Hex::new(0, 0);
        
        engine.add_piece(origin, piece).unwrap();
        assert_eq!(engine.get(Hex::new(0, 0)), Some(true), "Block at (0,0) should be occupied");
        assert_eq!(engine.get(Hex::new(0, 1)), Some(true), "Block at (0,1) should be occupied");
        
        // Try to add overlapping piece
        let overlapping_piece = Piece::new(0b0001000); // Just center
        let result = engine.add_piece(origin, overlapping_piece);
        assert!(result.is_err(), "Should fail to add overlapping piece");
        assert_eq!(
            result.unwrap_err(),
            "Cannot add piece: overlap or out of bounds",
            "Exception message should match"
        );
    }

    #[test]
    fn test_piece_contiguous() {
        // Single block - contiguous
        assert!(Piece::new(0b0001000).is_contiguous(), "Single center block should be contiguous");
        
        // Two adjacent blocks - contiguous
        assert!(Piece::new(0b0011000).is_contiguous(), "Two adjacent blocks should be contiguous");
        
        // Two non-adjacent blocks - not contiguous
        assert!(!Piece::new(0b1000001).is_contiguous(), "Two opposite corner blocks should not be contiguous");
        
        // All blocks - contiguous
        assert!(Piece::new(0b1111111).is_contiguous(), "All blocks should be contiguous");
        
        // Five blocks - always contiguous
        assert!(Piece::new(0b1111110).is_contiguous(), "Five blocks should be contiguous");
    }

    #[test]
    fn test_hex_operations() {
        let h1 = Hex::new(1, 2);
        let h2 = Hex::new(3, 4);
        
        // Addition
        let sum = h1 + h2;
        assert_eq!(sum.i(), 4);
        assert_eq!(sum.k(), 6);
        
        // Subtraction
        let diff = h2 - h1;
        assert_eq!(diff.i(), 2);
        assert_eq!(diff.k(), 2);
        
        // Shifts
        assert_eq!(h1.shift_i(5), Hex::new(6, 2));
        assert_eq!(h1.shift_k(3), Hex::new(1, 5));
        assert_eq!(h1.shift_j(1), Hex::new(0, 3));
    }

    #[test]
    fn test_into_bytes_and_back() {
        let mut engine = create_test_engine();
        engine.set(Hex::new(0, 0), true).unwrap();
        engine.set(Hex::new(1, 1), true).unwrap();
        engine.set(Hex::new(1, 0), true).unwrap();
        let bytes: Vec<u8> = engine.clone().into();
        let engine_from_bytes = HexEngine::try_from(bytes).unwrap();
        assert_eq!(engine.radius, engine_from_bytes.radius, "Radius should match after conversion");
        assert_eq!(engine.states, engine_from_bytes.states, "States should match after conversion");
    }

    #[test]
    fn test_engine_index_coordinate_conversion() {
        let engine = create_test_engine();
        
        // Test round-trip conversion
        for i in 0..engine.len() {
            let coo = engine.coordinate_of(i).unwrap();
            let idx = engine.index_of(coo).unwrap();
            assert_eq!(idx, i, "Round-trip conversion should preserve index");
        }
    }

    #[test]
    fn test_engine_from_string() {
        let engine = HexEngine::from_string("1010101").unwrap();
        assert_eq!(engine.len(), 7);
        assert_eq!(engine.get(Hex::new(0, 0)), Some(true));
        assert_eq!(engine.get(engine.coordinate_of(1).unwrap()), Some(false));
        
        // Test with X/O notation
        let engine2 = HexEngine::from_string("XOXOXOX").unwrap();
        assert_eq!(engine2.to_binary_string(), engine.to_binary_string());
    }

    #[test]
    fn test_compute_density() {
        let mut engine = create_test_engine();
        
        // Set some blocks to create a pattern
        engine.set(Hex::new(0, 1), true).unwrap();
        engine.set(Hex::new(1, 0), true).unwrap();
        
        let piece = Piece::new(0b0001000); // Just center block
        
        // Density at (0,0) should be higher because of neighbors
        let density = engine.compute_density(Hex::new(0, 0), piece);
        assert!(density > 0.0, "Density should be positive with neighbors");
        
        // Density at occupied position should be 0
        let density_occupied = engine.compute_density(Hex::new(0, 1), piece);
        assert_eq!(density_occupied, 0.0, "Density at occupied position should be 0");
    }
}