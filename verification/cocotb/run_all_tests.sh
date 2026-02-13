#!/bin/bash
set -e

# --- Configuration ---
ROOT_DIR=$(pwd)/../..
WORKLOAD_DIR="../../software/workload"
COMPILER_DIR="../../software/compiler"
COCOTB_DIR=$(pwd)

echo "🚀 Starting TinyNPU All-Tests Suite..."

# Ensure clean start
rm -f results.xml

# Test 1: Simple Chain
echo "📂 Running: simple_chain.npu"
cd $WORKLOAD_DIR
PYTHONPATH=../compiler python3 -c "from npu_test_gen import generate_sample_test; generate_sample_test()"
mv simple_chain.npu $COCOTB_DIR/
cd $COCOTB_DIR
NPU_FILE=simple_chain.npu make -f Makefile.npu > /dev/null 2>&1
echo "✅ simple_chain.npu: PASSED"

# Test 2: MOVE Instruction
echo "📂 Running: move_test.npu"
cd $WORKLOAD_DIR
PYTHONPATH=../compiler python3 -c "from npu_test_gen import generate_move_test; generate_move_test()"
mv move_test.npu $COCOTB_DIR/
cd $COCOTB_DIR
NPU_FILE=move_test.npu make -f Makefile.npu > /dev/null 2>&1
echo "✅ move_test.npu: PASSED"

# Test 3: Bias Support
echo "📂 Running: bias_test.npu"
cd $WORKLOAD_DIR
PYTHONPATH=../compiler python3 bias_test_gen.py
mv bias_test.npu $COCOTB_DIR/
cd $COCOTB_DIR
NPU_FILE=bias_test.npu make -f Makefile.npu > /dev/null 2>&1
echo "✅ bias_test.npu: PASSED"

# Test 4: 3-Layer DNN (Complex Chaining + Reuse)
echo "📂 Running: dnn_example.npu"
cd $WORKLOAD_DIR
PYTHONPATH=../compiler python3 dnn_example_gen.py
mv dnn_example.npu $COCOTB_DIR/
cd $COCOTB_DIR
NPU_FILE=dnn_example.npu make -f Makefile.npu > /dev/null 2>&1
echo "✅ dnn_example.npu: PASSED"

# Test 5: Mixed Precision (INT8 Packing + Masked Write)
echo "📂 Running: mixed_test.npu"
cd $WORKLOAD_DIR
PYTHONPATH=../compiler python3 mixed_precision_gen.py
mv mixed_test.npu $COCOTB_DIR/
cd $COCOTB_DIR
NPU_FILE=mixed_test.npu make -f Makefile.npu > /dev/null 2>&1
echo "✅ mixed_test.npu: PASSED"

echo "--------------------------------------------------------"
echo "🎉 ALL TESTS PASSED SUCCESSFULLY!"
echo "--------------------------------------------------------"