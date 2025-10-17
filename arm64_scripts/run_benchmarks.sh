#!/bin/bash
# Benchmark runner script for ARM64
# Runs comprehensive performance tests comparing native vs MLLB scheduler

set -e

RESULTS_DIR="benchmark_results_$(date +%Y%m%d_%H%M%S)"
MLLB_HOME="/home/runner/work/MLLB/MLLB"
KMOD_PATH="$MLLB_HOME/kmod/jc_kmod.ko"

echo "=========================================="
echo "MLLB ARM64 Benchmark Suite"
echo "=========================================="
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Check if kernel module exists
if [ ! -f "$KMOD_PATH" ]; then
    echo "✗ Kernel module not found: $KMOD_PATH"
    echo "Please compile it first: cd kmod && make"
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"
cd "$RESULTS_DIR"

# Record system information
echo "Recording system information..."
{
    echo "=== System Information ==="
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
    echo "Architecture: $(uname -m)"
    echo "Kernel: $(uname -r)"
    echo ""
    echo "=== CPU Information ==="
    lscpu
    echo ""
    echo "=== Memory Information ==="
    free -h
    echo ""
    echo "=== CPU Details ==="
    cat /proc/cpuinfo | grep -E "processor|model name|cpu MHz" | head -20
} > system_info.txt

echo "✓ System information recorded"

# Test 1: CPU Intensive Benchmark
echo ""
echo "Test 1: CPU Intensive Workload (10 minutes total)..."
echo "  Running with native scheduler..."
stress-ng --cpu $(nproc) --cpu-method all --metrics --timeout 300s > results_native_cpu.txt 2>&1
echo "  Waiting for system to stabilize..."
sleep 30

echo "  Running with MLLB scheduler..."
sudo insmod "$KMOD_PATH"
stress-ng --cpu $(nproc) --cpu-method all --metrics --timeout 300s > results_mllb_cpu.txt 2>&1
sudo rmmod jc_kmod
echo "✓ CPU intensive test completed"

# Test 2: Context Switch Benchmark
echo ""
echo "Test 2: Context Switch Performance..."
echo "  Running with native scheduler..."
perf stat -e context-switches,cpu-migrations,page-faults \
    stress-ng --cpu $(nproc) --timeout 60s > results_native_ctxsw.txt 2>&1

sleep 10

echo "  Running with MLLB scheduler..."
sudo insmod "$KMOD_PATH"
perf stat -e context-switches,cpu-migrations,page-faults \
    stress-ng --cpu $(nproc) --timeout 60s > results_mllb_ctxsw.txt 2>&1
sudo rmmod jc_kmod
echo "✓ Context switch test completed"

# Test 3: Mixed Workload
echo ""
echo "Test 3: Mixed Workload (CPU + I/O + Memory)..."
echo "  Running with native scheduler..."
stress-ng --cpu 2 --io 2 --vm 1 --vm-bytes 512M \
    --metrics --timeout 300s > results_native_mixed.txt 2>&1

sleep 30

echo "  Running with MLLB scheduler..."
sudo insmod "$KMOD_PATH"
stress-ng --cpu 2 --io 2 --vm 1 --vm-bytes 512M \
    --metrics --timeout 300s > results_mllb_mixed.txt 2>&1
sudo rmmod jc_kmod
echo "✓ Mixed workload test completed"

# Test 4: Thread Migration Test (if pthread_fibo exists)
if [ -f "$MLLB_HOME/pthread_fibo" ]; then
    echo ""
    echo "Test 4: Thread Migration Test..."
    echo "  Running with native scheduler..."
    perf stat -e migrations,cycles,instructions \
        $MLLB_HOME/pthread_fibo > results_native_migration.txt 2>&1
    
    sleep 10
    
    echo "  Running with MLLB scheduler..."
    sudo insmod "$KMOD_PATH"
    perf stat -e migrations,cycles,instructions \
        $MLLB_HOME/pthread_fibo > results_mllb_migration.txt 2>&1
    sudo rmmod jc_kmod
    echo "✓ Migration test completed"
fi

# Generate summary report
echo ""
echo "Generating summary report..."

cat > benchmark_summary.txt << EOF
MLLB ARM64 Benchmark Summary Report
Generated: $(date)
System: $(uname -a)
========================================

=== CPU Intensive Test ===
Native Scheduler:
$(grep "bogo ops/s" results_native_cpu.txt | head -3 || echo "No data")

MLLB Scheduler:
$(grep "bogo ops/s" results_mllb_cpu.txt | head -3 || echo "No data")

=== Context Switch Test ===
Native Scheduler:
$(grep -E "context-switches|cpu-migrations" results_native_ctxsw.txt || echo "No data")

MLLB Scheduler:
$(grep -E "context-switches|cpu-migrations" results_mllb_ctxsw.txt || echo "No data")

=== Mixed Workload Test ===
Native Scheduler:
$(grep "bogo ops/s" results_native_mixed.txt | head -5 || echo "No data")

MLLB Scheduler:
$(grep "bogo ops/s" results_mllb_mixed.txt | head -5 || echo "No data")

========================================
EOF

cat benchmark_summary.txt

echo ""
echo "=========================================="
echo "Benchmark Completed!"
echo "=========================================="
echo "Results saved in: $(pwd)"
echo "Key files:"
echo "  - system_info.txt: System configuration"
echo "  - benchmark_summary.txt: Test summary"
echo "  - results_*.txt: Detailed test results"
echo ""
echo "To analyze results further, you can:"
echo "  1. View benchmark_summary.txt"
echo "  2. Compare individual test files"
echo "  3. Generate charts (if Python/matplotlib available)"
echo "=========================================="
