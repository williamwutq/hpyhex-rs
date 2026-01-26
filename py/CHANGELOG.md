# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-20
### Added
- Initial Rust implementation of hpyhex python package.

## [0.1.1] - 2026-01-21
### Added
- Added comprehensive documentation and project description.

## [0.2.0] - 2026-01-23
### Added
- Provided further documentation on various classes and methods.
- Provide comprehensive benchmark suite and results to demonstrate performance improvements over original hpyhex.
- Added NumPy integration for seamless conversion between Piece instances and NumPy arrays.
- Added NumPy integration for seamless conversion between HexEngine instances and NumPy arrays.
- Support serialization to `hpyhex-rs` crate compatible binary format for Piece, HexEngine, and Game classes via `hpyhex_rs_serialize` and `hpyhex_rs_deserialize` methods.
- Added NumPy integration for seamless conversion between a vector of Piece instances and stacked or flat NumPy arrays.
- Added NumPy integration for seamless conversion between Game instances and NumPy arrays.
- Added NumPy integration for seamless conversion between Game queues and stacked or flat NumPy arrays.
- Added special `hpyhex_rs_add_piece_with_index` method to Game class for adding pieces using indices directly.
- Add NumPy integration for game moves using various ndarray masks and maximum value selection

### Changed
- Refactored codebase with macros to reduce redundancy and improve maintainability.
- Improved build script for better automation and efficiency.
- Slightly improved performance for HexEngine by removing the need to call python functions during piece placement.

## [0.2.1] - 2026-01-25
### Fixed
- Fix missing feature flag of `HexEngine.to_numpy_raw_view` causing compilation errors when compiled without `numpy` feature.
- Fix `Hex`, `Piece`, and `HexEngine` class `__iter__` method returning tuples instead of iterators.
- Fix incorrect indentation in Makefile causing build issues on some systems.
- Fix string representation of Game instances showing reference address of Engine instead of engine instances themselves.
- Fix equality comparison of Game instances not comparing engine instances correctly.

### Added
- Add NumPy ndarray conversion methods for valid piece positions in HexEngine class.
- Add comprehensive examples demonstrating usage of this library in /examples directory.
- Add build option to Makefile for building Rust extension module without NumPy support.
- Add examples section to README.md.
- Implement serialization and deserialization for Game class to/from `hpyhex-rs` crate compatible binary format.

## [UNRELEASED]
### Fixed
- Fixed `HexEngine.index_block` method to return `-1` directly for out-of-range coordinates instead of returning a `PyResult`, adhering to the hpyhex API.

### Added
- Added static method `hpyhex_rs_coordinate_block` and `hpyhex_rs_index_block` to `HexEngine` class for direct coordinate/index conversions without needing an instance.
- Added static method `hpyhex_rs_adjacency_list` to `HexEngine` class for retrieving adjacency list for batch processing.
- Added adjacency matrix methods for HexEngine supporting all numeric types.
- Added adjacency list methods for HexEngine supporting integer types.
- Added comprehensive documentation for adjacency structures in README.md, including usage guides for convolution operations, graph algorithms, and advanced board state evaluation techniques.

### Changed
- Refactor `HexEngine.index_block` for performance by avoiding creation of temporary Python references.
- Optimize `add_piece_checked` method of `Game` class.
