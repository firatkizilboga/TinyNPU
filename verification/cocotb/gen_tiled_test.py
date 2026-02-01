#!/usr/bin/env python3
"""
Generate tiled test data for large matrix multiplication on 4x4 systolic array.

Example: 13x17 × 17x24 matrix multiplication
- Tiles A into 4×4 blocks (4 row tiles, 5 K-tiles)
- Tiles B into 4×4 blocks (5 K-tiles, 6 column tiles)
- Generates hex file with all tiles
- Outputs golden reference result and tile metadata
"""

import argparse
import numpy as np
import json


def pack_column(col, width=16):
    """Pack a column of 4 values into a 64-bit hex string."""
    assert len(col) == 4, f"Column must have 4 elements, got {len(col)}"
    packed = 0
    for i, val in enumerate(col):
        val = int(val) & ((1 << width) - 1)
        packed |= val << (i * width)
    return f"{packed:016X}"


def pad_matrix(mat, target_rows, target_cols):
    """Pad matrix with zeros to target dimensions."""
    rows, cols = mat.shape
    if rows >= target_rows and cols >= target_cols:
        return mat[:target_rows, :target_cols]
    
    padded = np.zeros((target_rows, target_cols), dtype=mat.dtype)
    padded[:rows, :cols] = mat
    return padded


def tile_matrix(mat, tile_rows, tile_cols):
    """
    Tile a matrix into blocks of size tile_rows × tile_cols.
    Returns list of tiles and the grid dimensions.
    """
    rows, cols = mat.shape
    n_row_tiles = (rows + tile_rows - 1) // tile_rows
    n_col_tiles = (cols + tile_cols - 1) // tile_cols
    
    tiles = []
    for i in range(n_row_tiles):
        row_tiles = []
        for j in range(n_col_tiles):
            r_start = i * tile_rows
            r_end = min(r_start + tile_rows, rows)
            c_start = j * tile_cols
            c_end = min(c_start + tile_cols, cols)
            
            # Extract tile and pad to tile_rows × tile_cols
            tile = mat[r_start:r_end, c_start:c_end]
            tile_padded = pad_matrix(tile, tile_rows, tile_cols)
            row_tiles.append(tile_padded)
        tiles.append(row_tiles)
    
    return tiles, (n_row_tiles, n_col_tiles)


def matrix_to_hex_cols(mat):
    """Convert 4×K matrix to hex strings, extracting columns.
    
    For A tiles: we feed columns of A to the systolic array.
    Each cycle, we feed one column (4 elements).
    """
    rows, cols = mat.shape
    assert rows == 4, f"Matrix must have 4 rows, got {rows}"
    
    hex_lines = []
    for k in range(cols):
        col = [mat[i, k] for i in range(4)]
        hex_lines.append(pack_column(col))
    return hex_lines


def matrix_to_hex_rows(mat):
    """Convert K×4 matrix to hex strings, extracting rows.
    
    For B tiles: we feed rows of B to the systolic array.
    Each cycle, we feed one row (4 elements).
    """
    rows, cols = mat.shape
    assert cols == 4, f"Matrix must have 4 cols, got {cols}"
    
    hex_lines = []
    for k in range(rows):
        row = [mat[k, j] for j in range(4)]
        hex_lines.append(pack_column(row))
    return hex_lines


def generate_tiled_hex(m, k, n, output_path, seed=42):
    """
    Generate tiled matrix multiplication test data.
    
    Args:
        m: Rows of A
        k: Cols of A, Rows of B (inner dimension)
        n: Cols of B
        output_path: Output hex file path
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    
    # Generate random matrices with small values to avoid overflow
    A = np.random.randint(1, 8, size=(m, k), dtype=np.int32)
    B = np.random.randint(1, 8, size=(k, n), dtype=np.int32)
    C_golden = A @ B
    
    print(f"Matrix A ({m}×{k}):")
    print(A)
    print(f"\nMatrix B ({k}×{n}):")
    print(B)
    print(f"\nGolden C = A × B ({m}×{n}):")
    print(C_golden)
    
    # Tile matrices into 4×4 blocks
    TILE_SIZE = 4
    
    # For A: tile by rows and K dimension
    # A is M×K, we need tiles of 4×4
    # Row tiles: ceil(M/4), K tiles: ceil(K/4)
    a_tiles, (n_m_tiles, n_k_tiles) = tile_matrix(A, TILE_SIZE, TILE_SIZE)
    
    # For B: tile by K dimension and columns
    # B is K×N, we need tiles of 4×4
    # K tiles: ceil(K/4), Col tiles: ceil(N/4)
    b_tiles, (n_k_tiles_b, n_n_tiles) = tile_matrix(B, TILE_SIZE, TILE_SIZE)
    
    assert n_k_tiles == n_k_tiles_b, "K dimension tiles must match"
    
    print(f"\nTiling info:")
    print(f"  M tiles: {n_m_tiles} (rows of A)")
    print(f"  K tiles: {n_k_tiles} (inner dimension)")
    print(f"  N tiles: {n_n_tiles} (cols of B)")
    print(f"  Total passes: {n_m_tiles} × {n_n_tiles} × {n_k_tiles} = {n_m_tiles * n_n_tiles * n_k_tiles}")
    
    # Build hex file
    lines = []
    lines.append("// Auto-generated tiled matmul test data")
    lines.append(f"// A({m}×{k}) × B({k}×{n}) = C({m}×{n})")
    lines.append(f"// Tiles: {n_m_tiles}×{n_k_tiles} (A), {n_k_tiles}×{n_n_tiles} (B)")
    lines.append("")
    
    # Row 0: Sentinel
    lines.append("// Row 0: Sentinel")
    lines.append("DEADDEADDEADDEAD")
    lines.append("")
    
    # Store tiles in memory
    # Layout: All A tiles first, then all B tiles
    # A tiles: [m_tile][k_tile] stored sequentially
    # B tiles: [k_tile][n_tile] stored sequentially
    
    addr = 1
    tile_metadata = {
        "m": m, "k": k, "n": n,
        "n_m_tiles": n_m_tiles,
        "n_k_tiles": n_k_tiles,
        "n_n_tiles": n_n_tiles,
        "a_tiles": {},
        "b_tiles": {},
        "golden": C_golden.tolist()
    }
    
    # Write A tiles
    lines.append(f"// A tiles (M={n_m_tiles}, K={n_k_tiles})")
    for i in range(n_m_tiles):
        for k_tile in range(n_k_tiles):
            tile = a_tiles[i][k_tile]
            lines.append(f"// A[{i}][{k_tile}] at addr {addr}-{addr+3}")
            
            # Store tile address
            tile_key = f"{i},{k_tile}"
            tile_metadata["a_tiles"][tile_key] = addr
            
            # Convert tile to hex (extract columns for A input)
            hex_data = matrix_to_hex_cols(tile)
            for h in hex_data:
                lines.append(h)
            
            addr += TILE_SIZE
            lines.append("")
    
    # Write B tiles
    lines.append(f"// B tiles (K={n_k_tiles}, N={n_n_tiles})")
    for k_tile in range(n_k_tiles):
        for j in range(n_n_tiles):
            tile = b_tiles[k_tile][j]
            lines.append(f"// B[{k_tile}][{j}] at addr {addr}-{addr+3}")
            
            # Store tile address
            tile_key = f"{k_tile},{j}"
            tile_metadata["b_tiles"][tile_key] = addr
            
            # Convert tile to hex (row-major for B weights)
            hex_data = matrix_to_hex_rows(tile)
            for h in hex_data:
                lines.append(h)
            
            addr += TILE_SIZE
            lines.append("")
    
    # Write hex file
    with open(output_path, 'w') as f:
        for line in lines:
            f.write(line + '\n')
    
    # Write metadata JSON
    metadata_path = output_path.replace('.hex', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(tile_metadata, f, indent=2)
    
    print(f"\nGenerated files:")
    print(f"  Hex data: {output_path}")
    print(f"  Metadata: {metadata_path}")
    
    return tile_metadata


def main():
    parser = argparse.ArgumentParser(description='Generate tiled matmul test data')
    parser.add_argument('-m', '--m', type=int, default=13, help='Rows of A')
    parser.add_argument('-k', '--k', type=int, default=17, help='Cols of A / Rows of B')
    parser.add_argument('-n', '--n', type=int, default=24, help='Cols of B')
    parser.add_argument('-o', '--output', type=str, default='buffer_init_tiled.hex',
                        help='Output hex file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    generate_tiled_hex(args.m, args.k, args.n, args.output, args.seed)


if __name__ == '__main__':
    main()
