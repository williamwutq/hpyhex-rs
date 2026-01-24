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

## [Unreleased]
### Fixed
- Fix `Hex`, `Piece`, and `HexEngine` class `__iter__` method returning tuples instead of iterators.
