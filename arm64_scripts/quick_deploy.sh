#!/bin/bash
# Quick deployment script for ARM64
# Automates the entire MLLB deployment process

set -e

MLLB_HOME="/home/runner/work/MLLB/MLLB"
TAG="${1:-arm64_baseline}"
TRAINING_TAG="${2:-arm64_model}"

echo "=========================================="
echo "MLLB ARM64 Quick Deployment Script"
echo "=========================================="
echo "Data collection tag: $TAG"
echo "Training model tag: $TRAINING_TAG"
echo ""

# Step 1: Verify environment
echo "Step 1: Verifying environment..."
if ! bash $MLLB_HOME/arm64_scripts/verify_installation.sh > /dev/null 2>&1; then
    echo "✗ Environment verification failed!"
    echo "Please run: sudo ./arm64_scripts/setup_arm64_environment.sh"
    exit 1
fi
echo "✓ Environment verified"

# Step 2: Data collection
echo ""
echo "Step 2: Starting data collection (30 minutes)..."
echo "Press Ctrl+C to stop early, or wait for automatic timeout"
cd $MLLB_HOME

# Start data collection in background
timeout 1800 sudo python3 dump_lb.py -t $TAG --old &
DUMP_PID=$!

# Start workload generation
echo "Generating test workload..."
if [ -f pthread_fibo ]; then
    ./pthread_fibo &
    WORKLOAD_PID=$!
else
    if gcc -o pthread_fibo pthread_fibo_create.c -lpthread 2>/dev/null; then
        ./pthread_fibo &
        WORKLOAD_PID=$!
    else
        echo "⚠ Could not compile test workload"
        WORKLOAD_PID=0
    fi
fi

# Wait for data collection
wait $DUMP_PID 2>/dev/null || true

# Stop workload if running
if [ $WORKLOAD_PID -ne 0 ]; then
    kill $WORKLOAD_PID 2>/dev/null || true
fi

echo "✓ Data collection completed"

# Verify data file
if [ ! -f "raw_${TAG}.csv" ]; then
    echo "✗ Data file not found: raw_${TAG}.csv"
    exit 1
fi

LINES=$(wc -l < "raw_${TAG}.csv")
echo "✓ Collected $LINES samples"

# Step 3: Model training
echo ""
echo "Step 3: Training model..."
cd $MLLB_HOME/training
python3 automate.py -t $TAG -o $TRAINING_TAG

if [ ! -f "models/${TRAINING_TAG}.h5" ]; then
    echo "✗ Model file not found after training"
    exit 1
fi
echo "✓ Model training completed"

# Step 4: Export weights
echo ""
echo "Step 4: Exporting model weights..."
python3 dump_weights.py --model models/${TRAINING_TAG}.h5 --output ../kmod/c_mlp.h
echo "✓ Weights exported"

# Step 5: Compile kernel module
echo ""
echo "Step 5: Compiling kernel module..."
cd $MLLB_HOME/kmod

# Generate optimized Makefile if not exists
if [ ! -f "Makefile.arm64" ]; then
    bash $MLLB_HOME/arm64_scripts/generate_optimized_makefile.sh
fi

make clean
if [ -f "Makefile.arm64" ]; then
    make -f Makefile.arm64
else
    make
fi

if [ ! -f "jc_kmod.ko" ]; then
    echo "✗ Kernel module compilation failed"
    exit 1
fi
echo "✓ Kernel module compiled"

# Step 6: Test kernel module
echo ""
echo "Step 6: Testing kernel module..."
sudo make test
echo "✓ Kernel module test completed"

# Summary
echo ""
echo "=========================================="
echo "Deployment Summary"
echo "=========================================="
echo "✓ Data collected: raw_${TAG}.csv ($LINES samples)"
echo "✓ Model trained: models/${TRAINING_TAG}.h5"
echo "✓ Weights exported: kmod/c_mlp.h"
echo "✓ Kernel module: kmod/jc_kmod.ko"
echo ""
echo "Next steps:"
echo "1. Review training results in training/models/"
echo "2. Install kernel module: cd kmod && sudo make install"
echo "3. Run benchmarks: cd benchmarks && ./run_all_benchmarks.sh"
echo "=========================================="
