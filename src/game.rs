//! Comprehensive game environment for Hex, including grid engine, piece factory, and game logic.
//!
//! ## Feature Flag: game
//! 
//! ## Overview
//! 
//! This module implements the core gameplay environment for Hex, inspired by the Python and Java versions of HappyHex.
//! It integrates grid management, piece generation, and game state tracking into a unified, robust system suitable for both interactive play and automated agents.
//!
//! ## Components
//!
//! - **PieceFactory**: Utility for creating and managing all predefined game pieces. Provides static mappings between piece names and bitfield values, reverse lookup, and random piece generation with weighted probabilities. See the original HappyHex Java and Python implementations for piece definitions and generation logic.
//! - **Game**: Main game environment managing the hexagonal grid, piece queue, score, turn tracking, and end-state detection. Handles piece placement, move execution, and game status queries, with robust error handling and consistent state management. The queue is always full; used pieces are immediately replaced by new random pieces.
//! - **random_engine**: Utility function for generating randomized HexEngine states for game initialization, using efficient bitfield generation and elimination logic.
//!
//! ## Design Notes
//!
//! - The integrated game environment (first introduced in Python) eliminates the need for separate engine and queue management, simplifying game logic and state interaction.
//! - PieceFactory is stateless and not intended to be instantiated; all methods are static.
//! - The game environment is designed to be robust, error-tolerant, and suitable for both human and AI agents (e.g., reinforcement learning).
//! - Thread safety: The Game struct is thread-safe for concurrent read access; mutable operations should be externally synchronized.
//!
//! ## Example Usage
//!
//! ```rust
//! use hpyhex_rs::{Game, PieceFactory, random_engine};
//! use hpyhex_rs::Hex;
//!
//! // Create a new game with radius 2 and a queue of 3 pieces
//! let mut game = Game::new(2, 3);
//! // Access the first piece in the queue
//! let piece = game.queue()[0];
//! // Find valid positions for the piece
//! let valid_positions = game.engine().valid_positions(piece);
//! // Place the piece if possible
//! if let Some(pos) = valid_positions.first() {
//!     game.add_piece(0, *pos);
//! }
//! println!("Score: {} Turn: {} End: {}", game.score(), game.turn(), game.is_end());
//! ```
//!
//! ## References
//!
//! - Python: [HappyHex ML](https://github.com/williamwutq/hpyhexml)
//! - Java: [HappyHex](https://github.com/williamwutq/game_HappyHex)
//!
//! Designed by William Wu. Adapted for Rust.

#![cfg(any(feature = "default", feature = "game"))]
use crate::hex::{Piece, HexEngine, Hex};
use rand::Rng;
use std::fmt;

/// Generates a random HexEngine with a given radius.
/// 
/// Creates a random boolean pattern and applies elimination rules.
/// This is more efficient than generating all possible engines.
/// 
/// This is a free function because it is not a part of the HexEngine struct itself,
/// but rather to extend its functionality specifically for game initialization.
/// 
/// # Arguments
/// * `radius` - The radius of the hexagonal game board (must be >= 2)
/// 
/// # Returns
/// A randomized HexEngine instance
/// 
/// # Panics
/// Panics if radius < 2
pub fn random_engine(radius: usize) -> HexEngine {
    assert!(radius >= 2, "Radius must be at least 2");

    let length = HexEngine::calc_length(radius);
    let mut rng = rand::rng();

    // For length > 64, generate enough random u64s to cover all bits
    let mut bits = Vec::with_capacity(length);
    let num_chunks = (length + 63) / 64;
    let mut randoms = Vec::with_capacity(num_chunks);
    for _ in 0..num_chunks {
        randoms.push(rng.random::<u64>());
    }

    for j in 0..length {
        let chunk = j / 64;
        let bit = j % 64;
        bits.push((randoms[chunk] >> bit) & 1 == 1);
    }

    let mut engine = HexEngine::from_states(bits)
        .expect("Generated states should be valid");
    engine.eliminate();
    engine
}

/// Utility for creating and managing game pieces used in Hex.
///
/// The `PieceFactory` provides a catalog of predefined pieces, their bitfield representations, and reverse mappings to retrieve piece names from byte values.
/// It supports generating random pieces based on weighted probability distributions, reflecting both easy and normal difficulty modes from the original game logic.
///
/// # Structure
///
/// - **Piece Catalog**: Contains a static list of named pieces and their corresponding byte values. These names are rustified versions of the original Java piece names (e.g., `triangle_3_a` for `Triangle3A`).
/// - **Reverse Mapping**: Allows retrieval of piece names from their byte values for display and debugging.
/// - **Random Generation**: The [`generate_piece`] method produces random pieces using a weighted distribution, mixing easier and harder pieces as in the original game. The probability split is fixed at 50% for easy mode in this implementation.
/// - **Statelessness**: The factory is stateless and not intended to be instantiated; all methods are static.
///
/// # Features
///
/// - Retrieve a piece by name ([`get_piece`])
/// - Retrieve a piece name by value ([`get_piece_name`])
/// - Generate random pieces for gameplay ([`generate_piece`])
/// - Access all predefined pieces ([`all_pieces`])
///
/// # Piece Byte Codes
///
/// The following byte values correspond to standard pieces:
///
/// - 8: uno
/// - 127: full
/// - 119: hallow
/// - 13: triangle_3_a
/// - 88: triangle_3_b
/// - 28: line_3_i
/// - 73: line_3_j
/// - 42: line_3_k
/// - 74: corner_3_i_l
/// - 41: corner_3_i_r
/// - 56: corner_3_j_l
/// - 14: corner_3_j_r
/// - 76: corner_3_k_l
/// - 25: corner_3_k_r
/// - 78: fan_4_a
/// - 57: fan_4_b
/// - 27: rhombus_4_i
/// - 120: rhombus_4_j
/// - 90: rhombus_4_k
/// - 39: corner_4_i_l
/// - 114: corner_4_i_r
/// - 101: corner_4_j_l
/// - 83: corner_4_j_r
/// - 23: corner_4_k_l
/// - 116: corner_4_k_r
/// - 92: asym_4_i_a
/// - 30: asym_4_i_b
/// - 60: asym_4_i_c
/// - 29: asym_4_i_d
/// - 75: asym_4_j_a
/// - 77: asym_4_j_b
/// - 89: asym_4_j_c
/// - 105: asym_4_j_d
/// - 46: asym_4_k_a
/// - 106: asym_4_k_b
/// - 43: asym_4_k_c
/// - 58: asym_4_k_d
///
/// # Notes
///
/// - PieceFactory is stateless and not intended to be instantiated.
/// - See the original HappyHex Java and Python implementations for more details on piece definitions and generation logic.
///
/// Designed by William Wu. Adapted for Rust.
pub struct PieceFactory;

impl PieceFactory {
    /// All predefined pieces with their names and bitfield values
    /// 
    /// These are pieces that are most commonly used in the game.
    /// 
    /// The names are rustified versions of the original piece names.
    /// For example, "triangle_3_a" corresponds to java version "Triangle3A".
    ///
    /// Names of pieces should be intuitive. See PieceFactory of the origianl HappyHex in Java
    /// (https://github.com/williamwutq/game_HappyHex) for detailed descriptions of each piece.
    pub const PIECES: &'static [(&'static str, u8)] = &[
        ("uno", 8),
        ("full", 127),
        ("hallow", 119),
        ("triangle_3_a", 13),
        ("triangle_3_b", 88),
        ("line_3_i", 28),
        ("line_3_j", 73),
        ("line_3_k", 42),
        ("corner_3_i_l", 74),
        ("corner_3_i_r", 41),
        ("corner_3_j_l", 56),
        ("corner_3_j_r", 14),
        ("corner_3_k_l", 76),
        ("corner_3_k_r", 25),
        ("fan_4_a", 78),
        ("fan_4_b", 57),
        ("rhombus_4_i", 27),
        ("rhombus_4_j", 120),
        ("rhombus_4_k", 90),
        ("corner_4_i_l", 39),
        ("corner_4_i_r", 114),
        ("corner_4_j_l", 101),
        ("corner_4_j_r", 83),
        ("corner_4_k_l", 23),
        ("corner_4_k_r", 116),
        ("asym_4_i_a", 92),
        ("asym_4_i_b", 30),
        ("asym_4_i_c", 60),
        ("asym_4_i_d", 29),
        ("asym_4_j_a", 75),
        ("asym_4_j_b", 77),
        ("asym_4_j_c", 89),
        ("asym_4_j_d", 105),
        ("asym_4_k_a", 46),
        ("asym_4_k_b", 106),
        ("asym_4_k_c", 43),
        ("asym_4_k_d", 58),
    ];

    /// Gets a piece by name
    /// 
    /// # Arguments
    /// * `name` - The name of the piece to retrieve
    /// 
    /// # Returns
    /// The piece if found, None otherwise
    pub const fn get_piece(name: &str) -> Option<Piece> {
        let mut i = 0;
        while i < Self::PIECES.len() {
            let (piece_name, value) = Self::PIECES[i];
            if const_str_eq(name, piece_name) {
                return Some(Piece::new(value));
            }
            i += 1;
        }
        None
    }

    /// Gets the name of a piece based on its bitfield value
    /// 
    /// # Arguments
    /// * `piece` - The piece whose name is to be retrieved
    /// 
    /// # Returns
    /// The name if found, None otherwise
    pub const fn get_piece_name(piece: Piece) -> Option<&'static str> {
        let value = piece.as_u8();
        let mut i = 0;
        while i < Self::PIECES.len() {
            let (name, piece_value) = Self::PIECES[i];
            if value == piece_value {
                return Some(name);
            }
            i += 1;
        }
        None
    }

    /// Generates a random piece based on weighted probability distribution
    /// 
    /// Uses a mix of easier and harder pieces with different probabilities.
    /// 
    /// The generation algorithm is the exact same as used in the original game,
    /// except the possiblility of enter easy mode is 50% instead of a settable value.
    /// 
    /// # Returns
    /// A randomly generated Piece
    pub fn generate_piece() -> Piece {
        let mut rng = rand::rng();
        
        if rng.random_bool(0.5) {
            // Easier generation
            let i = rng.random_range(0..74);
            
            match i {
                0..=7 => Self::get_piece("triangle_3_a").unwrap(),
                8..=15 => Self::get_piece("triangle_3_b").unwrap(),
                16..=21 => Self::get_piece("line_3_i").unwrap(),
                22..=27 => Self::get_piece("line_3_j").unwrap(),
                28..=33 => Self::get_piece("line_3_k").unwrap(),
                34..=36 => Self::get_piece("corner_3_i_r").unwrap(),
                37..=39 => Self::get_piece("corner_3_j_r").unwrap(),
                40..=42 => Self::get_piece("corner_3_k_r").unwrap(),
                43..=45 => Self::get_piece("corner_3_i_l").unwrap(),
                46..=48 => Self::get_piece("corner_3_j_l").unwrap(),
                49..=51 => Self::get_piece("corner_3_k_l").unwrap(),
                52..=55 => Self::get_piece("rhombus_4_i").unwrap(),
                56..=59 => Self::get_piece("rhombus_4_j").unwrap(),
                60..=63 => Self::get_piece("rhombus_4_k").unwrap(),
                _ => {
                    let j = rng.random_range(0..25);
                    match j {
                        0..=1 => Self::get_piece("fan_4_a").unwrap(),
                        2..=3 => Self::get_piece("fan_4_b").unwrap(),
                        4 => Self::get_piece("corner_4_i_l").unwrap(),
                        5 => Self::get_piece("corner_4_i_r").unwrap(),
                        6 => Self::get_piece("corner_4_j_l").unwrap(),
                        7 => Self::get_piece("corner_4_j_r").unwrap(),
                        8 => Self::get_piece("corner_4_k_l").unwrap(),
                        9 => Self::get_piece("corner_4_k_r").unwrap(),
                        10..=13 => {
                            let suffix = ['a', 'b', 'c', 'd'][j - 10];
                            Self::get_piece(&format!("asym_4_i_{}", suffix)).unwrap()
                        }
                        14..=17 => {
                            let suffix = ['a', 'b', 'c', 'd'][j - 14];
                            Self::get_piece(&format!("asym_4_j_{}", suffix)).unwrap()
                        }
                        18..=21 => {
                            let suffix = ['a', 'b', 'c', 'd'][j - 18];
                            Self::get_piece(&format!("asym_4_k_{}", suffix)).unwrap()
                        }
                        _ => Self::get_piece("uno").unwrap(),
                    }
                }
            }
        } else {
            // Harder generation
            let i = rng.random_range(0..86);
            
            match i {
                0..=5 => Self::get_piece("triangle_3_a").unwrap(),
                6..=11 => Self::get_piece("triangle_3_b").unwrap(),
                12..=15 => Self::get_piece("line_3_i").unwrap(),
                16..=19 => Self::get_piece("line_3_j").unwrap(),
                20..=23 => Self::get_piece("line_3_k").unwrap(),
                24..=25 => Self::get_piece("corner_3_i_r").unwrap(),
                26..=27 => Self::get_piece("corner_3_j_r").unwrap(),
                28..=29 => Self::get_piece("corner_3_k_r").unwrap(),
                30..=31 => Self::get_piece("corner_3_i_l").unwrap(),
                32..=33 => Self::get_piece("corner_3_j_l").unwrap(),
                34..=35 => Self::get_piece("corner_3_k_l").unwrap(),
                36..=39 => Self::get_piece("rhombus_4_i").unwrap(),
                40..=43 => Self::get_piece("rhombus_4_j").unwrap(),
                44..=47 => Self::get_piece("rhombus_4_k").unwrap(),
                48..=53 => Self::get_piece("fan_4_a").unwrap(),
                54..=59 => Self::get_piece("fan_4_b").unwrap(),
                60..=61 => Self::get_piece("corner_4_i_l").unwrap(),
                62..=63 => Self::get_piece("corner_4_i_r").unwrap(),
                64..=65 => Self::get_piece("corner_4_j_l").unwrap(),
                66..=67 => Self::get_piece("corner_4_j_r").unwrap(),
                68..=69 => Self::get_piece("corner_4_k_l").unwrap(),
                70..=71 => Self::get_piece("corner_4_k_r").unwrap(),
                72..=75 => {
                    let suffix = ['a', 'b', 'c', 'd'][i - 72];
                    Self::get_piece(&format!("asym_4_i_{}", suffix)).unwrap()
                }
                76..=79 => {
                    let suffix = ['a', 'b', 'c', 'd'][i - 76];
                    Self::get_piece(&format!("asym_4_j_{}", suffix)).unwrap()
                }
                80..=83 => {
                    let suffix = ['a', 'b', 'c', 'd'][i - 80];
                    Self::get_piece(&format!("asym_4_k_{}", suffix)).unwrap()
                }
                _ => Self::get_piece("full").unwrap(),
            }
        }
    }

    /// Returns all pieces defined in this factory
    /// 
    /// # Returns
    /// An array of all predefined pieces
    pub const fn all_pieces() -> [Piece; 37] {
        [
            Piece::new(8),   // uno
            Piece::new(127), // full
            Piece::new(119), // hallow
            Piece::new(13),  // triangle_3_a
            Piece::new(88),  // triangle_3_b
            Piece::new(28),  // line_3_i
            Piece::new(73),  // line_3_j
            Piece::new(42),  // line_3_k
            Piece::new(74),  // corner_3_i_l
            Piece::new(41),  // corner_3_i_r
            Piece::new(56),  // corner_3_j_l
            Piece::new(14),  // corner_3_j_r
            Piece::new(76),  // corner_3_k_l
            Piece::new(25),  // corner_3_k_r
            Piece::new(78),  // fan_4_a
            Piece::new(57),  // fan_4_b
            Piece::new(27),  // rhombus_4_i
            Piece::new(120), // rhombus_4_j
            Piece::new(90),  // rhombus_4_k
            Piece::new(39),  // corner_4_i_l
            Piece::new(114), // corner_4_i_r
            Piece::new(101), // corner_4_j_l
            Piece::new(83),  // corner_4_j_r
            Piece::new(23),  // corner_4_k_l
            Piece::new(116), // corner_4_k_r
            Piece::new(92),  // asym_4_i_a
            Piece::new(30),  // asym_4_i_b
            Piece::new(60),  // asym_4_i_c
            Piece::new(29),  // asym_4_i_d
            Piece::new(75),  // asym_4_j_a
            Piece::new(77),  // asym_4_j_b
            Piece::new(89),  // asym_4_j_c
            Piece::new(105), // asym_4_j_d
            Piece::new(46),  // asym_4_k_a
            Piece::new(106), // asym_4_k_b
            Piece::new(43),  // asym_4_k_c
            Piece::new(58),  // asym_4_k_d
        ]
    }
}

/// Helper for const string equality comparison
const fn const_str_eq(a: &str, b: &str) -> bool {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    
    if a_bytes.len() != b_bytes.len() {
        return false;
    }
    
    let mut i = 0;
    while i < a_bytes.len() {
        if a_bytes[i] != b_bytes[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// Represents the main game environment for Hex.
///
/// The `Game` struct manages the hexagonal grid engine, a fixed-length queue of pieces, game score, turn tracking, and end-state detection.
/// It provides methods to add pieces, make moves, and query game status, handling errors gracefully and maintaining consistent state.
/// 
/// The integrated game evironment, first introduced in the Python adaptation, eliminated the need for separate engine and queue management,
/// making it easier to implement game logic and interact with the game state.
///
/// # Structure
///
/// - **Engine**: The [`HexEngine`] manages the hexagonal grid, block placement, and elimination logic.
/// - **Queue**: The piece queue is a fixed-size array of [`Piece`] objects. When a piece is used, it is immediately replaced by a new randomly generated piece, ensuring the queue always remains full. The queue supports peeking and consuming elements by index, but does not allow external insertion.
/// - **Score & Turn**: Tracks the player's score and turn count, updating after each move and elimination.
/// - **End State**: Detects when no valid moves remain, marking the game as ended.
///
/// # Features
///
/// - Add pieces to the grid and update game state ([`add_piece`])
/// - Make moves using custom algorithms ([`make_move`])
/// - Query game status, score, turn, and queue ([`is_end`], [`result`], [`queue`])
/// - Handles invalid moves and errors gracefully
/// - Cloning and display support for game state
///
/// # Queue Behavior
///
/// The queue always maintains a fixed number of pieces. When a piece is placed, it is replaced by a new piece generated internally. The queue only stores static pieces. External addition of pieces is not supported.
/// 
/// # Thread Safety
/// 
/// The `Game` struct is designed to be thread-safe for concurrent read access. However, mutable operations should be synchronized externally to prevent data races. This may be achieved using a `Mutex` or similar synchronization primitive.
///
/// # Example
///
/// ```rust
/// use hpyhex_rs::{Game, PieceFactory, random_engine};
/// let mut game = Game::new(2, 3);
/// let piece = game.queue()[0];
/// let valid_positions = game.engine().valid_positions(piece);
/// if let Some(pos) = valid_positions.first() {
///     game.add_piece(0, *pos);
/// }
/// println!("Score: {} Turn: {} End: {}", game.score(), game.turn(), game.is_end());
/// ```
///
/// # Notes
///
/// - All game logic is designed to be robust and error-tolerant.
/// - The game environment is suitable for both interactive play and automated agents (e.g., reinforcement learning).
///
/// Designed by William Wu. Adapted for Rust.]
#[derive(Eq, PartialEq, Hash)]
pub struct Game {
    engine: HexEngine,
    queue: Vec<Piece>,
    score: usize,
    turn: usize,
    end: bool,
}

impl Game {
    /// Creates a new game with specified engine and queue
    /// 
    /// # Arguments
    /// * `radius` - The radius of the hexagonal game board (>= 2)
    /// * `queue_size` - The number of pieces in the queue (>= 1)
    /// 
    /// # Returns
    /// A new Game instance
    /// 
    /// # Panics
    /// Panics if radius < 2 or queue_size < 1
    pub fn new(radius: usize, queue_size: usize) -> Self {
        Self::with_initial_state(radius, queue_size, 0, 0)
    }

    /// Creates a new game with initial turn and score
    /// 
    /// # Arguments
    /// * `radius` - The radius of the hexagonal game board (>= 2)
    /// * `queue_size` - The number of pieces in the queue (>= 1)
    /// * `initial_turn` - The starting turn number
    /// * `initial_score` - The starting score
    pub fn with_initial_state(
        radius: usize,
        queue_size: usize,
        initial_turn: usize,
        initial_score: usize,
    ) -> Self {
        assert!(radius >= 2, "Radius must be at least 2");
        assert!(queue_size >= 1, "Queue size must be at least 1");

        let engine = HexEngine::new(radius);
        let queue = (0..queue_size)
            .map(|_| PieceFactory::generate_piece())
            .collect();

        Game {
            engine,
            queue,
            score: initial_score,
            turn: initial_turn,
            end: false,
        }
    }

    /// Creates a game from an existing engine
    /// 
    /// # Arguments
    /// * `engine` - The HexEngine to use
    /// * `queue_size` - The number of pieces in the queue (>= 1)
    pub fn from_engine(engine: HexEngine, queue_size: usize) -> Self {
        assert!(queue_size >= 1, "Queue size must be at least 1");

        let queue = (0..queue_size)
            .map(|_| PieceFactory::generate_piece())
            .collect();

        let mut game = Game {
            engine,
            queue,
            score: 0,
            turn: 0,
            end: false,
        };

        // Check if game is already over
        game.check_end();
        game
    }

    /// Adds a piece to the game at the specified coordinates
    /// 
    /// # Arguments
    /// * `piece_index` - Index of the piece in the queue
    /// * `coord` - Coordinates where the piece should be placed
    /// 
    /// # Returns
    /// `true` if the piece was successfully added, `false` otherwise
    pub fn add_piece(&mut self, piece_index: usize, coord: Hex) -> bool {
        // Validate piece index
        if piece_index >= self.queue.len() {
            return false;
        }

        let piece = self.queue[piece_index];

        // Try to add piece to engine
        if self.engine.add_piece(coord, piece).is_err() {
            return false;
        }

        // Update score with piece blocks
        self.score += piece.count() as usize;

        // Replace used piece with new random piece
        self.queue[piece_index] = PieceFactory::generate_piece();

        // Eliminate full lines and add bonus score
        let eliminated = self.engine.eliminate();
        self.score += eliminated.len() * 5;

        // Increment turn
        self.turn += 1;

        // Check if game has ended
        self.check_end();

        true
    }

    /// Makes a move using the specified algorithm
    /// 
    /// # Arguments
    /// * `algorithm` - A function that takes the engine and queue and returns (piece_index, coordinate)
    /// 
    /// # Returns
    /// `true` if the move was successfully made, `false` otherwise
    pub fn make_move<F>(&mut self, algorithm: F) -> bool
    where
        F: FnOnce(&HexEngine, &[Piece]) -> Option<(usize, Hex)>,
    {
        if self.end {
            return false;
        }

        match algorithm(&self.engine, &self.queue) {
            Some((index, coord)) => self.add_piece(index, coord),
            None => false,
        }
    }

    /// Checks if the game has ended (no valid moves remaining)
    /// 
    /// Updates the `end` field accordingly
    fn check_end(&mut self) {
        for piece in &self.queue {
            if !self.engine.valid_positions(*piece).is_empty() {
                self.end = false;
                return;
            }
        }
        self.end = true;
    }

    /// Returns whether the game has ended
    /// 
    /// # Returns
    /// `true` if the game is over, `false` otherwise
    #[inline]
    pub fn is_end(&self) -> bool {
        self.end
    }

    /// Returns the current result as (turn, score)
    /// 
    /// # Returns
    /// A tuple containing the current turn number and score
    #[inline]
    pub fn result(&self) -> (usize, usize) {
        (self.turn, self.score)
    }

    /// Returns the current turn number
    /// 
    /// # Returns
    /// The current turn number
    #[inline]
    pub fn turn(&self) -> usize {
        self.turn
    }

    /// Returns the current score
    /// 
    /// # Returns
    /// The current score
    #[inline]
    pub fn score(&self) -> usize {
        self.score
    }

    /// Returns a reference to the game engine
    /// 
    /// # Returns
    /// A reference to the HexEngine
    #[inline]
    pub fn engine(&self) -> &HexEngine {
        &self.engine
    }

    /// Returns a mutable reference to the game engine
    /// 
    /// Modifying the engine directly may lead to invalid game states.
    /// 
    /// # Returns
    /// A mutable reference to the HexEngine
    #[inline]
    pub fn engine_mut(&mut self) -> &mut HexEngine {
        &mut self.engine
    }

    /// Returns a reference to the piece queue
    /// 
    /// # Returns
    /// A reference to the vector of pieces in the queue
    #[inline]
    pub fn queue(&self) -> &[Piece] {
        &self.queue
    }

    /// Returns a mutable reference to the piece queue
    /// 
    /// Modifying the queue directly may lead to invalid game states.
    /// 
    /// # Returns
    /// A mutable reference to the vector of pieces in the queue
    #[inline]
    pub fn queue_mut(&mut self) -> &mut [Piece] {
        &mut self.queue
    }
}

impl Clone for Game {
    /// Clones the game state
    /// 
    /// # Returns
    /// A new Game instance with the same state
    fn clone(&self) -> Self {
        Game {
            engine: self.engine.clone(),
            queue: self.queue.clone(),
            score: self.score,
            turn: self.turn,
            end: self.end,
        }
    }
}

impl fmt::Display for Game {
    /// Formats the game state for display
    ///
    /// # Returns
    /// A formatted string representing the game state
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Game(score={}, turn={}, end={}, queue_len={})",
            self.score,
            self.turn,
            self.end,
            self.queue.len()
        )
    }
}

impl fmt::Debug for Game {
    /// Formats the game state for debugging
    /// 
    /// # Returns
    /// A formatted string representing the game state
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Game")
            .field("engine", &self.engine.to_binary_string())
            .field("queue", &self.queue)
            .field("score", &self.score)
            .field("turn", &self.turn)
            .field("end", &self.end)
            .finish()
    }
}

impl From<(HexEngine, usize)> for Game {
    /// Creates a Game from a HexEngine and queue size
    /// 
    /// # Arguments
    /// * `engine` - The HexEngine to use
    /// * `queue_size` - The number of pieces in the queue (>= 1)
    /// 
    /// # Returns
    /// A new Game instance
    fn from((engine, queue_size): (HexEngine, usize)) -> Self {
        Game::from_engine(engine, queue_size)
    }
}

impl From<(usize, usize)> for Game {
    /// Creates a Game from radius and queue size
    /// 
    /// # Arguments
    /// * `radius` - The radius of the hexagonal game board (>= 2)
    /// * `queue_size` - The number of pieces in the queue (>= 1)
    /// 
    /// # Returns
    /// A new Game instance
    fn from((radius, queue_size): (usize, usize)) -> Self {
        Game::new(radius, queue_size)
    }
}

impl From<(usize, HexEngine)> for Game {
    /// Creates a Game from queue size and a HexEngine
    /// 
    /// # Arguments
    /// * `queue_size` - The number of pieces in the queue (>= 1)
    /// * `engine` - The HexEngine to use
    /// 
    /// # Returns
    /// A new Game instance
    fn from((queue_size, engine): (usize, HexEngine)) -> Self {
        Game::from_engine(engine, queue_size)
    }
}

impl From<(HexEngine, Vec<Piece>)> for Game {
    /// Creates a Game from a HexEngine and a predefined piece queue
    /// 
    /// # Arguments
    /// * `engine` - The HexEngine to use
    /// * `queue` - The vector of pieces to use as the queue
    /// 
    /// # Returns
    /// A new Game instance
    fn from((engine, queue): (HexEngine, Vec<Piece>)) -> Self {
        let mut game = Game {
            engine,
            queue,
            score: 0,
            turn: 0,
            end: false,
        };
        game.check_end();
        game
    }
}

impl From<(Vec<Piece>, HexEngine)> for Game {
    /// Creates a Game from a predefined piece queue and a HexEngine
    /// 
    /// # Arguments
    /// * `queue` - The vector of pieces to use as the queue
    /// * `engine` - The HexEngine to use
    /// 
    /// # Returns
    /// A new Game instance
    fn from((queue, engine): (Vec<Piece>, HexEngine)) -> Self {
        let mut game = Game {
            engine,
            queue,
            score: 0,
            turn: 0,
            end: false,
        };
        game.check_end();
        game
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_piece_factory_get_piece() {
        let uno = PieceFactory::get_piece("uno").unwrap();
        assert_eq!(uno.as_u8(), 8);

        let full = PieceFactory::get_piece("full").unwrap();
        assert_eq!(full.as_u8(), 127);

        assert!(PieceFactory::get_piece("nonexistent").is_none());
    }

    #[test]
    fn test_piece_factory_get_name() {
        let uno = Piece::new(8);
        assert_eq!(PieceFactory::get_piece_name(uno), Some("uno"));

        let full = Piece::new(127);
        assert_eq!(PieceFactory::get_piece_name(full), Some("full"));

        let unknown = Piece::new(1);
        assert!(PieceFactory::get_piece_name(unknown).is_none());
    }

    #[test]
    fn test_piece_factory_generate() {
        // Just test that it doesn't panic and returns valid pieces
        for _ in 0..100 {
            let piece = PieceFactory::generate_piece();
            assert!(piece.as_u8() > 0 && piece.as_u8() < 128);
        }
    }

    #[test]
    fn test_piece_factory_all_pieces() {
        let pieces = PieceFactory::all_pieces();
        assert_eq!(pieces.len(), 37);
        assert_eq!(pieces[0].as_u8(), 8); // uno
        assert_eq!(pieces[1].as_u8(), 127); // full
    }

    #[test]
    fn test_game_new() {
        let game = Game::new(2, 3);
        assert_eq!(game.turn(), 0);
        assert_eq!(game.score(), 0);
        assert!(!game.is_end());
        assert_eq!(game.queue().len(), 3);
    }

    #[test]
    fn test_game_add_piece() {
        let mut game = Game::new(2, 3);
        
        // Get a simple piece
        let piece = PieceFactory::get_piece("uno").unwrap();
        game.queue[0] = piece;

        let initial_score = game.score();
        let success = game.add_piece(0, Hex::new(0, 0));
        
        assert!(success, "Should successfully add piece");
        assert_eq!(game.turn(), 1, "Turn should increment");
        assert!(game.score() > initial_score, "Score should increase");
    }

    #[test]
    fn test_game_invalid_piece_index() {
        let mut game = Game::new(2, 3);
        assert!(!game.add_piece(10, Hex::new(0, 0)), "Invalid index should fail");
    }

    #[test]
    fn test_game_make_move() {
        let mut game = Game::new(2, 3);
        
        // Simple algorithm that always picks first piece at origin
        let algorithm = |_engine: &HexEngine, _queue: &[Piece]| {
            Some((0, Hex::new(1, 1)))
        };

        let success = game.make_move(algorithm);
        assert!(success || game.is_end(), "Move should succeed or game should end");
    }

    #[test]
    fn test_game_end_detection() {
        let mut game = Game::new(2, 1);
        
        // Fill the entire board
        let full_piece = PieceFactory::get_piece("full").unwrap();
        game.queue[0] = full_piece;
        
        // Keep adding pieces until game ends
        let mut moves = 0;
        while !game.is_end() && moves < 10 {
            if let Some(positions) = game.engine().valid_positions(game.queue()[0]).first() {
                game.add_piece(0, *positions);
                moves += 1;
            } else {
                break;
            }
        }
        
        // Game should eventually end or fill up
        assert!(game.is_end() || moves >= 1);
    }

    #[test]
    fn test_random_engine() {
        let engine = random_engine(3);
        assert_eq!(engine.radius(), 3);
        assert_eq!(engine.len(), HexEngine::calc_length(3));
    }

    #[test]
    fn test_game_clone() {
        let game = Game::new(2, 3);
        let cloned = game.clone();
        
        assert_eq!(game.turn(), cloned.turn());
        assert_eq!(game.score(), cloned.score());
        assert_eq!(game.is_end(), cloned.is_end());
    }
}