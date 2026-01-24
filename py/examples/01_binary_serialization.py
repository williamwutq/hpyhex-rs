"""
Example 1: Binary Serialization for Game State Persistence

This example demonstrates how to use hpyhex_rs_serialize and hpyhex_rs_deserialize
to save and load game states efficiently. This is useful for:
- Saving game progress
- Network transmission of game states
- Creating game replay systems
- Implementing undo/redo functionality
"""

from hpyhex import Hex, Piece, HexEngine, Game, PieceFactory


def serialize_deserialize_hex():
    """Demonstrate serialization of Hex coordinates."""
    print("=" * 60)
    print("Hex Serialization Example")
    print("=" * 60)
    
    # Create a Hex coordinate
    original_hex = Hex(5, 10)
    print(f"Original Hex: {original_hex}")
    print(f"  Coordinates: i={original_hex.i}, j={original_hex.j}, k={original_hex.k}")
    
    # Serialize to binary format
    serialized = original_hex.hpyhex_rs_serialize()
    print(f"Serialized (bytes): {serialized.hex()}")
    print(f"  Size: {len(serialized)} bytes")
    
    # Deserialize back to Hex
    restored_hex = Hex.hpyhex_rs_deserialize(serialized)
    print(f"Restored Hex: {restored_hex}")
    print(f"  Equal to original: {original_hex == restored_hex}")
    print()


def serialize_deserialize_piece():
    """Demonstrate serialization of Piece objects."""
    print("=" * 60)
    print("Piece Serialization Example")
    print("=" * 60)
    
    # Create a Piece from a named piece
    original_piece = PieceFactory.get_piece("full")
    print(f"Original Piece: {PieceFactory.get_piece_name(original_piece)}")
    print(f"  State value: {int(original_piece)}")
    print(f"  Length: {len(original_piece)}")
    
    # Serialize to binary format
    serialized = original_piece.hpyhex_rs_serialize()
    print(f"Serialized (bytes): {serialized.hex()}")
    print(f"  Size: {len(serialized)} bytes")
    
    # Deserialize back to Piece
    restored_piece = Piece.hpyhex_rs_deserialize(serialized)
    print(f"Restored Piece: {PieceFactory.get_piece_name(restored_piece)}")
    print(f"  Equal to original: {original_piece == restored_piece}")
    print()


def serialize_deserialize_hexengine():
    """Demonstrate serialization of HexEngine (game board state)."""
    print("=" * 60)
    print("HexEngine Serialization Example")
    print("=" * 60)
    
    # Create and modify a HexEngine
    original_engine = HexEngine(5)
    
    # Add some pieces to create a game state
    piece1 = PieceFactory.get_piece("triangle_3_a")
    piece2 = PieceFactory.get_piece("rhombus_4_i")
    
    original_engine.add_piece(Hex(0, 0), piece1)
    original_engine.add_piece(Hex(2, 2), piece2)
    
    print(f"Original Engine (radius={original_engine.radius}):")
    print(f"  Total blocks: {len(original_engine.states)}")
    print(f"  Occupied blocks: {sum(original_engine.states)}")
    print(f"  Hash: {hash(original_engine)}")
    
    # Serialize to binary format
    serialized = original_engine.hpyhex_rs_serialize()
    print(f"Serialized (bytes): {len(serialized)} bytes")
    print(f"  First 32 bytes (hex): {serialized[:32].hex()}")
    
    # Deserialize back to HexEngine
    restored_engine = HexEngine.hpyhex_rs_deserialize(serialized)
    print(f"Restored Engine (radius={restored_engine.radius}):")
    print(f"  Total blocks: {len(restored_engine.states)}")
    print(f"  Occupied blocks: {sum(restored_engine.states)}")
    print(f"  Hash: {hash(restored_engine)}")
    print(f"  Equal to original: {original_engine == restored_engine}")
    print()


def save_and_load_game_state():
    """Practical example: Save and load a complete game state."""
    print("=" * 60)
    print("Complete Game State Persistence")
    print("=" * 60)
    
    # Create a game and play some moves
    game = Game(5, 5)
    print(f"Initial game state:")
    print(f"  Turn: {game.turn}, Score: {game.score}")
    print(f"  Queue length: {len(game.queue)}")
    
    # Play a few moves
    for turn in range(3):
        positions = game.engine.check_positions(game.queue[0])
        if positions:
            # Pick the first valid position
            game.add_piece(0, positions[0])
            print(f"  Turn {turn + 1}: Added piece at {positions[0]}, Score: {game.score}")
    
    # Save game state components to binary
    engine_data = game.engine.hpyhex_rs_serialize()
    queue_data = [piece.hpyhex_rs_serialize() for piece in game.queue]
    score = game.score
    turn = game.turn
    
    print(f"\nGame state serialized:")
    print(f"  Engine data size: {len(engine_data)} bytes")
    print(f"  Queue pieces: {len(queue_data)}")
    print(f"  Score: {score}, Turn: {turn}")
    
    # Simulate loading the game state (e.g., from a file or network)
    restored_engine = HexEngine.hpyhex_rs_deserialize(engine_data)
    restored_queue = [Piece.hpyhex_rs_deserialize(data) for data in queue_data]
    
    # Create a new game with the restored state
    restored_game = Game(restored_engine, restored_queue)
    # Note: score and turn tracking would need to be stored separately
    # as they're not part of the engine or queue state
    
    print(f"\nRestored game state:")
    print(f"  Engine match: {game.engine == restored_game.engine}")
    print(f"  Queue length match: {len(game.queue) == len(restored_game.queue)}")
    print(f"  Turn: {restored_game.turn}")
    
    # Verify we can continue playing
    positions = restored_game.engine.check_positions(restored_game.queue[0])
    if positions:
        restored_game.add_piece(0, positions[0])
        print(f"  Successfully continued game, new score: {restored_game.score}")
    print()


def demonstrate_file_persistence():
    """Example of saving and loading from files."""
    print("=" * 60)
    print("File Persistence Example")
    print("=" * 60)
    
    import tempfile
    import os
    
    # Create a game state
    engine = HexEngine(4)
    piece = PieceFactory.get_piece("full")
    engine.add_piece(Hex(1, 1), piece)
    
    # Create a temporary file to save the state
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.hpyhex') as f:
        temp_filename = f.name
        # Write the serialized data
        f.write(engine.hpyhex_rs_serialize())
        print(f"Saved game state to: {temp_filename}")
        print(f"  File size: {os.path.getsize(temp_filename)} bytes")
    
    # Load the state from file
    with open(temp_filename, 'rb') as f:
        loaded_data = f.read()
        loaded_engine = HexEngine.hpyhex_rs_deserialize(loaded_data)
        print(f"Loaded game state from file")
        print(f"  States match: {engine == loaded_engine}")
        print(f"  Occupied blocks: {sum(loaded_engine.states)}")
    
    # Clean up
    os.unlink(temp_filename)
    print(f"  Temporary file cleaned up")
    print()


def main():
    """Run all serialization examples."""
    print("\n" + "=" * 60)
    print("HpyHex-RS Binary Serialization Examples")
    print("=" * 60)
    print("This demonstrates efficient binary serialization for")
    print("saving and loading game states.\n")
    
    # Run all examples
    serialize_deserialize_hex()
    serialize_deserialize_piece()
    serialize_deserialize_hexengine()
    save_and_load_game_state()
    demonstrate_file_persistence()
    
    print("=" * 60)
    print("All serialization examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
