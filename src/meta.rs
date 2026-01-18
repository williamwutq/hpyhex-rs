
#![cfg(feature = "extended")]
use crate::hex::{Hex, HexEngine, Piece};

/// Extended HexEngine with per-block metadata
/// Generic over metadata type T
#[derive(Clone, Debug)]
pub struct ExtendedHexEngine<T> {
    base: HexEngine,
    metadata: Vec<T>,
}

impl <T>Default for ExtendedHexEngine<T>
where
    T: Default + Clone,
{
    fn default() -> Self {
        ExtendedHexEngine {
            base: HexEngine::default(),
            metadata: vec![],
        }
    }
}

/// # ExtendedHexEngine
///
/// `ExtendedHexEngine<T>` is a generic extension of the [`HexEngine`] grid engine, designed to associate
/// per-block metadata of type `T` with each cell in the hexagonal grid. This allows you to track additional
/// information (such as colors, scores, timestamps, or custom data) for each block, while maintaining the
/// core grid logic and API of [`HexEngine`].
///
/// ## API Compatibility
///
/// The API of `ExtendedHexEngine<T>` is intentionally designed to be mostly compatible with [`HexEngine`],
/// mirroring its methods for grid creation, coordinate conversion, state queries, and piece manipulation.
/// Most methods in `ExtendedHexEngine<T>` either directly delegate to the underlying [`HexEngine`] or
/// extend its functionality to handle metadata. This makes it easy to migrate code from [`HexEngine`] to
/// `ExtendedHexEngine<T>`, or to use both interchangeably where only grid state is needed.
///
/// - **Grid State:** All methods for querying and mutating the occupancy state of the grid (e.g., `get`,
/// `set`, `add_piece`, `eliminate`, etc.) are available and behave identically to [`HexEngine`], with the
/// addition that metadata is also managed alongside state changes.
/// - **Metadata Storage:** For every block in the grid, a value of type `T` is stored in a parallel vector.
///   Metadata is initialized to `T::default()` and can be set or queried independently of the occupancy
///   state using methods like `set_metadata` and `get`.
/// - **Piece Operations:** When adding or eliminating pieces, metadata is set or cleared in sync with the
///   block state, ensuring consistency between the grid and its associated metadata.
/// - **Pattern and Neighborhood Queries:** Methods such as `get_pattern_metadata` allow you to extract metadata
///   for all blocks in a piece pattern or neighborhood, enabling advanced analysis or visualization.
///
/// ## Example Use Cases
///
/// - **Game Development:** Track ownership, scoring, or effects for each cell in a puzzle or board game.
/// - **Simulation:** Store physical properties, timestamps, or other simulation data per block.
/// - **Visualization:** Attach color, label, or annotation data to each cell for rendering or analysis.
///
/// ## Type Parameter
///
/// - `T`: The metadata type to associate with each block. Must implement [`Default`] and [`Clone`].
///
/// ## Key Differences from HexEngine
///
/// - All state-changing operations (e.g., adding pieces, eliminating lines) also update metadata.
/// - Additional methods for setting and retrieving metadata per block or per pattern.
/// - The `get` method returns both occupancy state and metadata.
///
/// ## See Also
///
/// - [`HexEngine`]: The base grid engine for hexagonal block state.
/// - [`Piece`]: Piece patterns for placement and manipulation.
/// - [`Hex`]: Coordinate type for grid positions.
impl <T>ExtendedHexEngine<T> 
where
    T: Default + Clone,
{
    /// Calculates grid length from radius
    /// 
    /// ## Parameters
    /// - `radius`: The radius of the hexagonal grid
    #[inline]
    pub const fn calc_length(radius: usize) -> usize {
        return HexEngine::calc_length(radius);
    }

    /// Calculates radius from grid length (returns None if invalid)
    /// 
    /// ## Parameters
    /// - `length`: The length of the grid
    /// 
    /// ## Returns
    /// An `Option<usize>` containing the radius if valid, or `None` if invalid.
    pub fn calc_radius(length: usize) -> Option<usize> {
        HexEngine::calc_radius(length)
    }

    /// Creates a new empty grid with given radius
    /// 
    /// ## Parameters
    /// - `radius`: The radius of the hexagonal grid
    /// 
    /// ## Returns
    /// A new `ExtendedHexEngine<T>` instance with default metadata for each block.
    pub fn new(radius: usize) -> Self {
        let length = Self::calc_length(radius);
        ExtendedHexEngine {
            base: HexEngine::new(radius),
            metadata: vec![T::default(); length],
        }
    }

    /// Creates a new empty grid with given radius and default metadata
    /// 
    /// ## Parameters
    /// - `radius`: The radius of the hexagonal grid
    /// - `default_meta`: The default metadata value to initialize each block with
    /// 
    /// ## Returns
    /// A new `ExtendedHexEngine<T>` instance with specified default metadata for each block.
    pub fn with_metadata(radius: usize, default_meta: T) -> Self {
        let length = Self::calc_length(radius);
        ExtendedHexEngine {
            base: HexEngine::new(radius),
            metadata: vec![default_meta; length],
        }
    }

    /// Creates a grid from a boolean vector
    /// 
    /// ## Parameters
    /// - `states`: Vector of booleans representing block occupancy
    /// 
    /// ## Returns
    /// A `Result<ExtendedHexEngine, String>` containing the new grid or an error if invalid.
    /// Metadata is initialized to default values.
    pub fn from_states(states: Vec<bool>) -> Result<Self, String> {
        let engine = HexEngine::from_states(states.clone())?;
        Ok(ExtendedHexEngine {
            base: engine,
            metadata: vec![T::default(); states.len()],
        })
    }

    /// Creates a grid from a binary string ("0"/"1" or "X"/"O")
    /// 
    /// ## Parameters
    /// - `s`: String representing block occupancy
    /// 
    /// ## Returns
    /// A `Result<HexEngine, String>` containing the new grid or an error if invalid.
    /// Metadata is initialized to default values.
    pub fn from_binary_string(s: &str) -> Result<Self, String> {
        let engine = HexEngine::from_string(s)?;
        let len = engine.len();
        Ok(ExtendedHexEngine {
            base: engine,
            metadata: vec![T::default(); len],
        })
    }

    /// Returns the radius of the grid
    /// 
    /// ## Returns
    /// The radius as a usize.
    #[inline]
    pub fn radius(&self) -> usize {
        self.base.radius()
    }

    /// Returns the length of the grid
    /// 
    /// This is the number of blocks in the hexagonal grid.
    /// 
    /// ## Returns
    /// The length as a usize.
    #[inline]
    pub fn len(&self) -> usize {
        self.base.len()
    }

    /// Checks if the grid is empty
    ///
    /// ## Returns
    /// `true` if the grid has no blocks, `false` otherwise.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.base.is_empty()
    }

    /// Checks if the grid is in default state (all unoccupied and default metadata)
    /// 
    /// ## Returns
    /// `true` if all blocks are unoccupied and metadata is default, `false` otherwise.
    #[inline]
    pub fn is_default(&self) -> bool 
    where
        T: PartialEq,
    {
        self.base.is_empty() && self.metadata.iter().all(|m| *m == T::default())
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
        self.base.in_range(coo)
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
        self.base.index_of(coo)
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
    pub fn coordinate_of(&self, index: usize) -> Option<Hex> {
        self.base.coordinate_of(index)
    }

    /// Searches for all coordinates with matching metadata
    /// 
    /// ## Parameters
    /// - `target`: The metadata value to search for
    /// 
    /// ## Returns
    /// A vector of `Hex` coordinates where the metadata matches the target.
    /// 
    /// ## See Also
    /// - [`count_of`]: Counts the number of blocks with matching metadata.
    pub fn search_for(&self, target: T) -> Vec<Hex> 
    where
        T: PartialEq,
    {
        let mut results = Vec::new();
        for (i, meta) in self.metadata.iter().enumerate() {
            if *meta == target {
                if let Some(coo) = self.coordinate_of(i) {
                    results.push(coo);
                }
            }
        }
        results
    }

    /// Counts the number of blocks with matching metadata
    /// 
    /// ## Parameters
    /// - `target`: The metadata value to count
    /// 
    /// ## Returns
    /// The count of blocks where the metadata matches the target.
    /// 
    /// ## See Also
    /// - [`search_for`]: Searches for coordinates with matching metadata.
    pub fn count_of(&self, target: T) -> usize 
    where
        T: PartialEq,
    {
        let mut count = 0;
        for meta in self.metadata.iter() {
            if *meta == target {
                count += 1;
            }
        }
        count
    }

    /// Gets the state at a coordinate
    /// 
    /// ## Parameters
    /// - `coo`: The `Hex` coordinate to query
    /// 
    /// ## Returns
    /// An `Option<(bool, T)>` containing a tuple of the state (true = occupied, false = unoccupied) and associated metadata if in range, or `None` if out of bounds.
    #[inline]
    pub fn get(&self, coo: Hex) -> Option<(bool, T)> {
        let state = self.base.get(coo)?;
        let index = self.base.index_of(coo).unwrap(); // Safe unwrap due to previous check
        let meta = self.metadata[index].clone();
        Some((state, meta))
    }

    /// Gets the occupancy state at a coordinate
    /// 
    /// ## Parameters
    /// - `coo`: The `Hex` coordinate to query
    /// 
    /// ## Returns
    /// An `Option<bool>` containing the state if in range, or `None` if out of bounds.
    #[inline]
    pub fn get_state(&self, coo: Hex) -> Option<bool> {
        self.base.get(coo)
    }

    /// Gets the metadata at a coordinate
    /// 
    /// ## Parameters
    /// - `coo`: The `Hex` coordinate to query
    /// 
    /// ## Returns
    /// An `Option<T>` containing the metadata if in range, or `None` if out of bounds.
    #[inline]
    pub fn get_metadata(&self, coo: Hex) -> Option<T> {
        let index = self.base.index_of(coo)?;
        Some(self.metadata[index].clone())
    }

    /// Sets the state and metadata at a coordinate
    /// 
    /// ## Parameters
    /// - `coo`: The `Hex` coordinate to set
    /// - `value`: A tuple of the state (true = occupied, false = unoccupied) and metadata to set
    /// 
    /// ## Returns
    /// A `Result<(), String>` indicating success or an error if out of bounds.
    #[inline]
    pub fn set(&mut self, coo: Hex, value: (bool, T)) -> Result<(), String> {
        let (state, meta) = value;
        self.base.set(coo, state)?;
        let index = self.base.index_of(coo).ok_or("Coordinate out of bounds")?;
        self.metadata[index] = meta;
        Ok(())
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
    pub fn set_state(&mut self, coo: Hex, state: bool) -> Result<(), String> {
        self.base.set(coo, state)
    }

    /// Sets the metadata at a coordinate
    /// 
    /// ## Parameters
    /// - `coo`: The `Hex` coordinate to set metadata for
    /// - `meta`: The metadata to set
    /// 
    /// ## Returns
    /// A `Result<(), String>` indicating success or an error if out of bounds.
    #[inline]
    pub fn set_metadata(&mut self, coo: Hex, meta: T) -> Result<(), String> {
        let index = self.base.index_of(coo).ok_or("Coordinate out of bounds")?;
        self.metadata[index] = meta;
        Ok(())
    }

    /// Resets all blocks to unoccupied and default metadata
    pub fn reset(&mut self) {
        self.base.reset();
        for meta in &mut self.metadata {
            *meta = T::default();
        }
    }

    /// Fills all blocks to unoccupied and sets metadata to given value
    /// 
    /// ## Parameters
    /// - `meta`: The metadata value to set for all blocks
    pub fn fill(&mut self, meta: T) {
        self.base.reset();
        for m in &mut self.metadata {
            *m = meta.clone();
        }
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
        self.base.check_add(coo, piece)
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
        self.base.add_piece(coo, piece)?;
        for (i, &pos) in Piece::POSITIONS.iter().enumerate() {
            if piece.is_occupied_unsafe(i) {
                self.set_metadata(coo + pos, T::default())?;
            }
        }
        Ok(())
    }

    /// Adds a piece with associated metadata at the given position
    /// 
    /// ## Parameters
    /// - `coo`: The `Hex` coordinate to add the piece at
    /// - `piece`: The `Piece` to add
    /// - `meta`: The metadata to associate with each occupied block of the piece
    /// 
    /// ## Returns
    /// A `Result<(), String>` indicating success or an error if the piece cannot be added.
    pub fn add_piece_with_metadata(&mut self, coo: Hex, piece: Piece, meta: T) -> Result<(), String> {
        self.base.add_piece(coo, piece)?;
        for (i, &pos) in Piece::POSITIONS.iter().enumerate() {
            if piece.is_occupied_unsafe(i) {
                self.set_metadata(coo + pos, meta.clone())?;
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
        self.base.valid_positions(piece)
    }

    /// Counts occupied neighbors around a coordinate
    /// 
    /// ## Parameters
    /// - `coo`: The `Hex` coordinate to check neighbors around
    /// 
    /// ## Returns
    /// The count of occupied neighboring blocks as a u32.
    pub fn count_neighbors(&self, coo: Hex) -> u32 {
        self.base.count_neighbors(coo)
    }

    /// Counts occupied neighbors with specific metadata around a coordinate
    /// 
    /// ## Parameters
    /// - `coo`: The `Hex` coordinate to check neighbors around
    /// - `key`: The metadata key to match
    /// 
    /// ## Returns
    /// The count of occupied neighboring blocks with matching metadata as a u32.
    pub fn count_neighbors_with_metadata(&self, coo: Hex, key: T) -> u32
    where
        T: PartialEq,
    {
        let mut count = 0;
        for neighbor in Piece::POSITIONS.iter() {
            let neighbor_coo = coo + *neighbor;
            if let Some((state, meta)) = self.get(neighbor_coo) {
                if state && meta == key {
                    count += 1;
                }
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
        self.base.compute_density(coo, piece)
    }

    /// Eliminates fully occupied lines and returns eliminated coordinates
    /// Clears metadata for eliminated blocks.
    /// 
    /// ## Returns
    /// A vector of `Hex` coordinates that were eliminated.
    pub fn eliminate(&mut self) -> Vec<Hex> {
        let eliminated = self.base.eliminate();
        for coo in &eliminated {
            let index = self.base.index_of(*coo).unwrap();
            // Safe unwrap since eliminated coords are valid
            self.metadata[index] = T::default();
        }
        eliminated
    }

    /// Computes Shannon entropy of the grid
    /// 
    /// ## Returns
    /// The Shannon entropy as a `f32`.
    pub fn compute_entropy(&self) -> f32 {
        self.base.compute_entropy()
    }

    /// Returns the piece pattern around a coordinate
    /// 
    /// ## Parameters
    /// - `coo`: The `Hex` coordinate to get the pattern for
    /// 
    /// ## Returns
    /// An `Option<Piece>` containing the piece pattern if in range, or `None` if out of bounds.
    pub fn pattern_as_piece(&self, coo: Hex) -> Option<Piece> {
        self.base.pattern_as_piece(coo)
    }

    /// Returns metadata for occupied blocks in the piece pattern around a coordinate
    /// 
    /// ## Parameters
    /// - `coo`: The `Hex` coordinate to get metadata for
    /// 
    /// ## Returns
    /// An `Option<Vec<T>>` containing a vector of metadata for occupied blocks if in range, or `None` if out of bounds.
    pub fn get_pattern_metadata(&self, coo: Hex) -> Option<Vec<T>> {
        let piece = self.pattern_as_piece(coo)?;
        let mut metas = Vec::new();
        for (i, &pos) in Piece::POSITIONS.iter().enumerate() {
            if piece.is_occupied_unsafe(i) {
                let index = match self.base.index_of(coo + pos) {
                    Some(idx) => idx,
                    None => return None,
                };
                metas.push(self.metadata[index].clone());
            }
        }
        Some(metas)
    }
}

use crate::game::PieceFactory;

impl PieceFactory {

}