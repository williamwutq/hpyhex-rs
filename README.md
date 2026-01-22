# hpyhex-rs
Simplified implementations of the HappyHex game components and hexagonal system in Rust.

## Overview
This repository contains a Rust implementation of the core components of the HappyHex game, including a hexagonal grid system and basic game mechanics. The project aims to provide a foundation for building more complex games based on hexagonal grids.

The original HappyHex game, in Java, can be found at [HappyHex on GitHub](https://github.com/williamwutq/game_HappyHex). An implementation in Python for machine learning purposes is available at [hpyhexml](https://github.com/williamwutq/hpyhexml/tree/main/hpyhex).

The Rust implementation adapts the convenient API design from the Python version while leveraging Rust's performance and safety features. The API is rustified to follow Rust conventions and idioms, and enhanced to deal with advanced application needs such as GUI applications and highly concurrent simulations with syncronized game states.

## Features
- Hexagonal grid representation
- Basic game mechanics for HappyHex
- Utility functions for hexagonal calculations

## Feature Flags
- `core`: Enables core functionalities of the HappyHex engine, including hexagonal grid management and basic game mechanics.
- `game`: Enables game-specific features, such as engine stage management, piece queue, and scoring system.
- `default`: Enable `core` and `game` features by default.
- `extended`: Enables extended functionalities, including extended engine with metadata, advanced piece queue, and syncronized game state management. Use this feature for extra security or for GUI applications. Historically GUI applications require more syncronized state management.

## Author
Developed by William Wu.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## See Also

- [hpyhex Python Implementation](https://github.com/williamwutq/hpyhexml/tree/main/hpyhex/README.md)
  This is the original Python implementation of the HappyHex game, which inspired this Rust version.
  It is originally designed both for machine learning and general gameplay purposes, and features a convenient API.
  Unlike the Java version, it does not concern itself with coloring of the hexagonal grid, focusing instead on game mechanics and piece management. Using optimization techniques such as caching and using integers for piece representation, it achieves good performance for Python.

  PYPI: https://pypi.org/project/hpyhex/

- [hpyhex-rs Python Implementation](./py/README.md)
  This a better, more performant Python implementation of the HappyHex game using Rust via PyO3.
  It provides significant performance improvements over the original hpyhex Python package, especially in scenarios involving extensive game simulations or AI training.
  The API mostly matches the original hpyhex Python package, making it a drop in replacement for existing codebases. In addition, it includes extra features such as NumPy integration that speeds up the needs specifically for high-performance computing and machine learning applications.

  PYPI: https://pypi.org/project/hpyhex-rs/