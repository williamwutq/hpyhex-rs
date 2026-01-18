# hpyhex-rs
Simplified implementations of the HappyHex game components and hexagonal system in Rust.

## Overview
This repository contains a Rust implementation of the core components of the HappyHex game, including a hexagonal grid system and basic game mechanics. The project aims to provide a foundation for building more complex games based on hexagonal grids.

The original HappyHex game, in Java, can be found at [HappyHex on GitHub](https://github.com/williamwutq/game_HappyHex). An implementation in Python for machine learning purposes is available at [hpyhexml](https://github.com/williamwutq/hpyhexml/tree/main/hpyhex).

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