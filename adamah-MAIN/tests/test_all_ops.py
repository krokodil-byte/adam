#!/usr/bin/env python3
"""
Test script for all ADAMAH operations
Tests all unary, binary, and special ops
"""

import numpy as np
import sys
import os

# Add src to path
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import adamah

def test_unary_ops():
    """Test all unary operations"""
    print("\n=== Testing Unary Operations ===")
    
    gpu = adamah.Adamah(cache_mb=256)
    
    # Create a simple map
    MAP_ID = 0
    SIZE = 1000
    gpu.map_create(MAP_ID, word_size=4, pack_size=1, n_packs=SIZE * 3)
    
    # Prepare test data
    test_data = np.linspace(-2.0, 2.0, SIZE, dtype=np.float32)
    locs_in = np.arange(0, SIZE, dtype=np.uint32)
    locs_out = np.arange(SIZE, SIZE * 2, dtype=np.uint32)
    
    # Upload input
    gpu.scatter(MAP_ID, locs_in, test_data)
    
    # Test each unary op
    unary_ops = [
        ("NEG", adamah.OP_NEG),
        ("ABS", adamah.OP_ABS),
        ("SQRT", adamah.OP_SQRT),
        ("EXP", adamah.OP_EXP),
        ("LOG", adamah.OP_LOG),
        ("TANH", adamah.OP_TANH),
        ("RELU", adamah.OP_RELU),
        ("GELU", adamah.OP_GELU),
        ("SIN", adamah.OP_SIN),
        ("COS", adamah.OP_COS),
        ("TAN", adamah.OP_TAN),
        ("SIGMOID", adamah.OP_SIGMOID),
        ("SWISH", adamah.OP_SWISH),
        ("MISH", adamah.OP_MISH),
        ("SELU", adamah.OP_SELU),
        ("ELU", adamah.OP_ELU),
        ("LEAKY_RELU", adamah.OP_LEAKY_RELU),
        ("SOFTPLUS", adamah.OP_SOFTPLUS),
        ("HARDSIGMOID", adamah.OP_HARDSIGMOID),
        ("HARDSWISH", adamah.OP_HARDSWISH),
        ("RECIPROCAL", adamah.OP_RECIPROCAL),
        ("SQUARE", adamah.OP_SQUARE),
        ("CUBE", adamah.OP_CUBE),
        ("SIGN", adamah.OP_SIGN),
        ("CEIL", adamah.OP_CEIL),
        ("FLOOR", adamah.OP_FLOOR),
        ("ROUND", adamah.OP_ROUND),
    ]
    
    print(f"Input range: [{test_data.min():.2f}, {test_data.max():.2f}]")
    
    for name, op_code in unary_ops:
        try:
            # Create handles for device buffers
            locs_in_h, _ = gpu.upload_dev(locs_in)
            locs_out_h, _ = gpu.upload_dev(locs_out)
            
            # Execute op
            gpu.map_op1_dev(MAP_ID, op_code, locs_in_h, locs_out_h, SIZE)
            
            # Download result
            result = gpu.gather(MAP_ID, locs_out, n_locs=SIZE)
            
            # Basic validation
            if np.all(np.isfinite(result)):
                status = "✓ PASS"
            else:
                status = f"✗ FAIL (NaN/Inf count: {np.sum(~np.isfinite(result))})"
            
            print(f"  {name:15s} → [{result.min():8.4f}, {result.max():8.4f}] {status}")
            
        except Exception as e:
            print(f"  {name:15s} → ERROR: {e}")
    
    gpu.map_destroy(MAP_ID)
    print("\n✓ Unary ops test complete")


def test_binary_ops():
    """Test all binary operations"""
    print("\n=== Testing Binary Operations ===")
    
    gpu = adamah.Adamah(cache_mb=256)
    
    # Create map
    MAP_ID = 1
    SIZE = 1000
    gpu.map_create(MAP_ID, word_size=4, pack_size=1, n_packs=SIZE * 4)
    
    # Prepare test data
    data_a = np.linspace(-1.0, 1.0, SIZE, dtype=np.float32)
    data_b = np.linspace(0.5, 2.5, SIZE, dtype=np.float32)
    
    locs_a = np.arange(0, SIZE, dtype=np.uint32)
    locs_b = np.arange(SIZE, SIZE * 2, dtype=np.uint32)
    locs_out = np.arange(SIZE * 2, SIZE * 3, dtype=np.uint32)
    
    # Upload inputs
    gpu.scatter(MAP_ID, locs_a, data_a)
    gpu.scatter(MAP_ID, locs_b, data_b)
    
    # Test each binary op
    binary_ops = [
        ("ADD", adamah.OP_ADD),
        ("SUB", adamah.OP_SUB),
        ("MUL", adamah.OP_MUL),
        ("DIV", adamah.OP_DIV),
        ("POW", adamah.OP_POW),
        ("MIN", adamah.OP_MIN),
        ("MAX", adamah.OP_MAX),
        ("MOD", adamah.OP_MOD),
        ("EQ", adamah.OP_EQ),
        ("NE", adamah.OP_NE),
        ("LT", adamah.OP_LT),
        ("LE", adamah.OP_LE),
        ("GT", adamah.OP_GT),
        ("GE", adamah.OP_GE),
        ("AND", adamah.OP_AND),
        ("OR", adamah.OP_OR),
        ("XOR", adamah.OP_XOR),
    ]
    
    print(f"Input A: [{data_a.min():.2f}, {data_a.max():.2f}]")
    print(f"Input B: [{data_b.min():.2f}, {data_b.max():.2f}]")
    
    for name, op_code in binary_ops:
        try:
            # Create handles
            locs_a_h, _ = gpu.upload_dev(locs_a)
            locs_b_h, _ = gpu.upload_dev(locs_b)
            locs_out_h, _ = gpu.upload_dev(locs_out)
            
            # Execute op
            gpu.map_op2_dev(MAP_ID, op_code, locs_a_h, locs_b_h, locs_out_h, SIZE)
            
            # Download result
            result = gpu.gather(MAP_ID, locs_out, n_locs=SIZE)
            
            # Basic validation
            finite_count = np.sum(np.isfinite(result))
            if finite_count == SIZE:
                status = "✓ PASS"
            else:
                status = f"✗ FAIL ({SIZE - finite_count} NaN/Inf)"
            
            print(f"  {name:10s} → [{result.min():8.4f}, {result.max():8.4f}] {status}")
            
        except Exception as e:
            print(f"  {name:10s} → ERROR: {e}")
    
    gpu.map_destroy(MAP_ID)
    print("\n✓ Binary ops test complete")


def test_high_level_api():
    """Test high-level wrapper functions"""
    print("\n=== Testing High-Level API ===")
    
    gpu = adamah.Adamah(cache_mb=256)
    
    MAP_ID = 2
    SIZE = 1000
    gpu.map_create(MAP_ID, word_size=4, pack_size=1, n_packs=SIZE * 3)
    
    # Test data
    test_data = np.random.randn(SIZE).astype(np.float32)
    locs_in = np.arange(0, SIZE, dtype=np.uint32)
    locs_out = np.arange(SIZE, SIZE * 2, dtype=np.uint32)
    
    gpu.scatter(MAP_ID, locs_in, test_data)
    
    # Test high-level activation functions
    activations = [
        ("sigmoid", gpu.sigmoid),
        ("swish", gpu.swish),
        ("mish", gpu.mish),
        ("selu", gpu.selu),
        ("elu", gpu.elu),
        ("leaky_relu", gpu.leaky_relu),
    ]
    
    for name, func in activations:
        try:
            locs_in_h, _ = gpu.upload_dev(locs_in)
            locs_out_h, _ = gpu.upload_dev(locs_out)
            
            func(MAP_ID, locs_in_h, locs_out_h, SIZE)
            
            result = gpu.gather(MAP_ID, locs_out, n_locs=SIZE)
            
            if np.all(np.isfinite(result)):
                print(f"  {name:15s} → ✓ PASS (range: [{result.min():.4f}, {result.max():.4f}])")
            else:
                print(f"  {name:15s} → ✗ FAIL (contains NaN/Inf)")
                
        except Exception as e:
            print(f"  {name:15s} → ERROR: {e}")
    
    gpu.map_destroy(MAP_ID)
    print("\n✓ High-level API test complete")


def test_comparison_ops():
    """Test comparison operations output 0.0 or 1.0"""
    print("\n=== Testing Comparison Ops (should output 0.0 or 1.0) ===")
    
    gpu = adamah.Adamah(cache_mb=256)
    
    MAP_ID = 3
    SIZE = 100
    gpu.map_create(MAP_ID, word_size=4, pack_size=1, n_packs=SIZE * 3)
    
    # Test with known values
    data_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20, dtype=np.float32)
    data_b = np.array([1.0, 3.0, 3.0, 2.0, 6.0] * 20, dtype=np.float32)
    
    locs_a = np.arange(0, SIZE, dtype=np.uint32)
    locs_b = np.arange(SIZE, SIZE * 2, dtype=np.uint32)
    locs_out = np.arange(SIZE * 2, SIZE * 3, dtype=np.uint32)
    
    gpu.scatter(MAP_ID, locs_a, data_a)
    gpu.scatter(MAP_ID, locs_b, data_b)
    
    # Test EQ
    locs_a_h, _ = gpu.upload_dev(locs_a)
    locs_b_h, _ = gpu.upload_dev(locs_b)
    locs_out_h, _ = gpu.upload_dev(locs_out)
    
    gpu.equal(MAP_ID, locs_a_h, locs_b_h, locs_out_h, SIZE)
    result = gpu.gather(MAP_ID, locs_out, n_locs=SIZE)
    
    # Check first 5 results
    expected_eq = np.array([1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    matches = np.allclose(result[:5], expected_eq)
    
    print(f"  EQ test: A={data_a[:5]}, B={data_b[:5]}")
    print(f"  Result: {result[:5]}")
    print(f"  Expected: {expected_eq}")
    print(f"  {'✓ PASS' if matches else '✗ FAIL'}")
    
    # Test that all comparison results are only 0.0 or 1.0
    unique_vals = np.unique(result)
    all_binary = np.all(np.isin(unique_vals, [0.0, 1.0]))
    print(f"\n  All results are 0.0 or 1.0: {'✓ PASS' if all_binary else '✗ FAIL'}")
    print(f"  Unique values: {unique_vals}")
    
    gpu.map_destroy(MAP_ID)
    print("\n✓ Comparison ops test complete")


if __name__ == "__main__":
    print("=" * 60)
    print("ADAMAH Complete Operations Test Suite")
    print("=" * 60)
    
    try:
        test_unary_ops()
        test_binary_ops()
        test_high_level_api()
        test_comparison_ops()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
