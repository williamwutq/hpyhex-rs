
//! # hpyhex-rs: High-Performance Hexagonal Grid Library
//!
//! This crate provides a fast, robust, and extensible environment for hexagonal grid-based games and machine learning applications, inspired by the HappyHex projects in Python and Java.
//!
//! ## Overview
//!
//! - **Hex**: Efficient coordinate system and operations for hexagonal grids, supporting both raw and line-based coordinates.
//! - **Piece**: Compact, immutable representation of game pieces using bitfields, with standard positions and occupancy queries.
//! - **HexEngine**: High-performance grid engine for managing block placement, elimination, and analysis, with O(1) coordinate-index conversion.
//! - **PieceFactory**: Utility for creating, naming, and randomly generating game pieces, with weighted probability distributions and static mappings.
//! - **Game**: Integrated game environment managing engine, piece queue, score, turn tracking, and end-state detection. Designed for both interactive play and automated agents.
//! - **random_engine**: Utility for generating randomized grid states for game initialization.
//!
//! ## Core Functionality
//!
//! - Create and manipulate hexagonal grids with [`HexEngine`].
//! - Define and query game pieces with [`Piece`] and [`PieceFactory`].
//! - Simulate full game environments with [`Game`], including piece placement, move execution, scoring, and elimination.
//! - Generate random game states and pieces for testing and reinforcement learning.
//!
//! ## Usage Example
//!
//! ```rust
//! use hpyhex_rs::{Game, PieceFactory, Hex};
//!
//! // Initialize a new game with radius 2 and a queue of 3 pieces
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
//! ## Design Notes
//!
//! - All core types are designed for performance, safety, and ease of use, following Rust conventions for immutability and error handling.
//! - The crate is suitable for both interactive games and AI/ML research, with robust error tolerance and clear APIs.
//! - Thread safety: All read operations are safe; mutable operations should be externally synchronized if used concurrently.
//! - See module-level docs for details on each component.
//!
//! ## References
//!
//! - Python: [HappyHex ML](https://github.com/williamwutq/hpyhexml)
//! - Java: [HappyHex](https://github.com/williamwutq/game_HappyHex)
//!
//! Designed by William Wu. Adapted for Rust.
mod hex;
mod game;
mod games;
mod meta;

#[cfg(any(feature = "default", feature = "core"))]
pub use hex::*;
#[cfg(any(feature = "default", feature = "game"))]
pub use game::*;
//#[cfg(feature = "extended")]
pub use meta::*;
//#[cfg(feature = "extended")]
pub use games::*;