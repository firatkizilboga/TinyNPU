# Archived Tests

This directory contains older isolated component tests that were used during development but are now superseded by the integrated tests in the parent directory.

## Archived Files

### Isolated Component Tests
- `test_skewer.py` + `Makefile.skewer` - Skewer module test (superseded by `test_ubss.py`)
- `test_ub_skewer.py` + `Makefile.ub_skewer` - UB+Skewer test (superseded by `test_ubss.py`)
- `test_valid_chain.py` + `Makefile.valid` - Valid chain test (superseded by integrated tests)
- `test_numeric.py` - Early numeric verification (superseded by `test_ubss_k12.py`)
- `test_top_markers.py` + `Makefile.top` - Top-level markers test (superseded by `test_ubss_k12.py`)

## Current Active Tests (in parent directory)

Use these instead:
- `test_ubss.py` - Basic 4×4 matrix multiplication
- `test_ubss_k12.py` - K=12 matrix multiplication (4×12 × 12×4)
- `test_tiled_matmul.py` - Tiled matrix multiplication (13×17 × 17×24)
- `test_unified_buffer.py` - Unified buffer standalone test

## Why Archived?

These tests were useful during development to verify individual components, but the integrated tests provide better coverage and are easier to maintain. They're kept here for reference in case we need to debug specific components in isolation.
