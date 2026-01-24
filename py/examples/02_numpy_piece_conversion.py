"""
Example 2: NumPy Integration - Piece Conversion and Batch Processing

This example demonstrates how to convert Piece objects to and from NumPy arrays
for efficient numerical computation and machine learning applications. This is
particularly useful for:
- Preparing data for neural networks
- Batch processing multiple pieces
- Efficient storage and manipulation of game states
- Statistical analysis of piece configurations
"""

import numpy as np
from hpyhex import Piece, PieceFactory


def basic_piece_to_numpy():
    """Demonstrate basic Piece to NumPy conversion."""
    print("=" * 60)
    print("Basic Piece to NumPy Conversion")
    print("=" * 60)
    
    # Create a piece
    piece = PieceFactory.get_piece("triangle_3_a")
    print(f"Piece: {PieceFactory.get_piece_name(piece)}")
    print(f"  Integer value: {int(piece)}")
    print(f"  Binary: {bin(int(piece))}")
    
    # Convert to different NumPy dtypes
    bool_array = piece.to_numpy_bool()
    print(f"\nBoolean array: {bool_array}")
    print(f"  dtype: {bool_array.dtype}, shape: {bool_array.shape}")
    
    int8_array = piece.to_numpy_int8()
    print(f"\nInt8 array: {int8_array}")
    print(f"  dtype: {int8_array.dtype}, shape: {int8_array.shape}")
    
    float32_array = piece.to_numpy_float32()
    print(f"\nFloat32 array: {float32_array}")
    print(f"  dtype: {float32_array.dtype}, shape: {float32_array.shape}")
    print()


def batch_piece_conversion():
    """Demonstrate batch conversion of multiple pieces."""
    print("=" * 60)
    print("Batch Piece Conversion")
    print("=" * 60)
    
    # Generate multiple pieces
    pieces = [PieceFactory.generate_piece() for _ in range(5)]
    print(f"Generated {len(pieces)} random pieces")
    
    # Convert to flat NumPy array (concatenated)
    flat_array = Piece.vec_to_numpy_bool_flat(pieces)
    print(f"\nFlat array shape: {flat_array.shape}")
    print(f"  Total elements: {flat_array.size}")
    print(f"  First piece (7 elements): {flat_array[:7]}")
    
    # Convert to stacked 2D array (each row is a piece)
    stacked_array = Piece.vec_to_numpy_bool_stacked(pieces)
    print(f"\nStacked array shape: {stacked_array.shape}")
    print(f"  Each row represents one piece")
    print(f"  First piece: {stacked_array[0]}")
    print(f"  Second piece: {stacked_array[1]}")
    
    # Use different dtypes for different use cases
    float_stacked = Piece.vec_to_numpy_float32_stacked(pieces)
    print(f"\nFloat32 stacked array:")
    print(f"  dtype: {float_stacked.dtype}, shape: {float_stacked.shape}")
    print(f"  Memory size: {float_stacked.nbytes} bytes")
    print()


def numpy_to_piece_conversion():
    """Demonstrate creating Pieces from NumPy arrays."""
    print("=" * 60)
    print("NumPy to Piece Conversion")
    print("=" * 60)
    
    # Create a NumPy array representing a piece
    bool_array = np.array([True, False, True, True, False, True, False], dtype=bool)
    print(f"Input boolean array: {bool_array}")
    
    # Convert to Piece
    piece = Piece.from_numpy_bool(bool_array)
    print(f"Created Piece: {piece}")
    print(f"  Integer value: {int(piece)}")
    
    # Round-trip conversion
    converted_back = piece.to_numpy_bool()
    print(f"Round-trip array: {converted_back}")
    print(f"  Arrays equal: {np.array_equal(bool_array, converted_back)}")
    
    # Create from different dtypes
    int_array = np.array([1, 0, 1, 1, 0, 1, 0], dtype=np.int32)
    piece_from_int = Piece.from_numpy_int32(int_array)
    print(f"\nCreated from int32: {piece_from_int}")
    print(f"  Same as original: {piece == piece_from_int}")
    
    float_array = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    piece_from_float = Piece.from_numpy_float32(float_array)
    print(f"Created from float32: {piece_from_float}")
    print(f"  Same as original: {piece == piece_from_float}")
    print()


def batch_numpy_to_pieces():
    """Demonstrate batch conversion from NumPy to Pieces."""
    print("=" * 60)
    print("Batch NumPy to Pieces Conversion")
    print("=" * 60)
    
    # Create a 2D array representing multiple pieces
    array_2d = np.random.randint(0, 2, size=(4, 7), dtype=np.int8)
    print(f"Input 2D array shape: {array_2d.shape}")
    print(f"Array:\n{array_2d}")
    
    # Convert stacked array to pieces
    pieces = Piece.vec_from_numpy_int8_stacked(array_2d)
    print(f"\nCreated {len(pieces)} pieces")
    for i, piece in enumerate(pieces):
        print(f"  Piece {i}: {int(piece):7b} (decimal: {int(piece)})")
    
    # Verify round-trip conversion
    reconstructed = Piece.vec_to_numpy_int8_stacked(pieces)
    print(f"\nRound-trip successful: {np.array_equal(array_2d, reconstructed)}")
    print()


def efficient_storage_example():
    """Demonstrate efficient storage using different dtypes."""
    print("=" * 60)
    print("Efficient Storage with Different Dtypes")
    print("=" * 60)
    
    # Create a large batch of pieces
    n_pieces = 1000
    pieces = [PieceFactory.generate_piece() for _ in range(n_pieces)]
    print(f"Created {n_pieces} pieces")
    
    # Compare storage requirements for different dtypes
    dtypes_to_test = [
        ("bool", Piece.vec_to_numpy_bool_stacked),
        ("uint8", Piece.vec_to_numpy_uint8_stacked),
        ("int32", Piece.vec_to_numpy_int32_stacked),
        ("float32", Piece.vec_to_numpy_float32_stacked),
        ("float64", Piece.vec_to_numpy_float64_stacked),
    ]
    
    print(f"\nStorage requirements:")
    for dtype_name, conversion_func in dtypes_to_test:
        array = conversion_func(pieces)
        size_kb = array.nbytes / 1024
        print(f"  {dtype_name:8s}: {size_kb:7.2f} KB ({array.itemsize} bytes per element)")
    
    # Boolean is most efficient for binary data!
    bool_array = Piece.vec_to_numpy_bool_stacked(pieces)
    float64_array = Piece.vec_to_numpy_float64_stacked(pieces)
    savings = (1 - bool_array.nbytes / float64_array.nbytes) * 100
    print(f"\nUsing bool instead of float64 saves {savings:.1f}% space!")
    print()


def main():
    """Run all NumPy integration examples."""
    print("\n" + "=" * 60)
    print("HpyHex-RS NumPy Integration Examples - Piece Conversion")
    print("=" * 60)
    print("This demonstrates efficient conversion between Piece objects")
    print("and NumPy arrays for numerical computation and ML.\n")
    
    # Run all examples
    basic_piece_to_numpy()
    batch_piece_conversion()
    numpy_to_piece_conversion()
    batch_numpy_to_pieces()
    efficient_storage_example()
    
    print("=" * 60)
    print("All NumPy integration examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
