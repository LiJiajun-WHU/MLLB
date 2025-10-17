#!/bin/bash
# Verification script for MLLB ARM64 environment
# Checks if all dependencies are correctly installed

echo "=========================================="
echo "MLLB ARM64 Environment Verification"
echo "=========================================="
echo ""

FAILED=0

# Check architecture
echo -n "1. Checking architecture... "
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    echo "✓ ARM64 ($ARCH)"
else
    echo "⚠ Not ARM64 ($ARCH)"
    FAILED=$((FAILED + 1))
fi

# Check kernel version
echo -n "2. Checking kernel version... "
KERNEL_VERSION=$(uname -r | cut -d. -f1,2)
KERNEL_MAJOR=$(echo $KERNEL_VERSION | cut -d. -f1)
KERNEL_MINOR=$(echo $KERNEL_VERSION | cut -d. -f2)
if [ $KERNEL_MAJOR -gt 4 ] || ([ $KERNEL_MAJOR -eq 4 ] && [ $KERNEL_MINOR -ge 15 ]); then
    echo "✓ $KERNEL_VERSION (>= 4.15)"
else
    echo "✗ $KERNEL_VERSION (< 4.15 - upgrade recommended)"
    FAILED=$((FAILED + 1))
fi

# Check kernel headers
echo -n "3. Checking kernel headers... "
if [ -d "/lib/modules/$(uname -r)/build" ]; then
    echo "✓ Installed"
else
    echo "✗ Not found"
    FAILED=$((FAILED + 1))
fi

# Check Python version
echo -n "4. Checking Python version... "
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo "✓ $PYTHON_VERSION"
else
    echo "✗ Python3 not found"
    FAILED=$((FAILED + 1))
fi

# Check BCC
echo -n "5. Checking BCC installation... "
if python3 -c "from bcc import BPF" 2>/dev/null; then
    echo "✓ BCC module available"
else
    echo "✗ BCC not available"
    FAILED=$((FAILED + 1))
fi

# Check TensorFlow
echo -n "6. Checking TensorFlow installation... "
TF_CHECK=$(python3 -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "✓ TensorFlow $TF_CHECK"
else
    echo "✗ TensorFlow not available"
    FAILED=$((FAILED + 1))
fi

# Check NumPy
echo -n "7. Checking NumPy... "
if python3 -c "import numpy" 2>/dev/null; then
    echo "✓ Installed"
else
    echo "✗ Not installed"
    FAILED=$((FAILED + 1))
fi

# Check Pandas
echo -n "8. Checking Pandas... "
if python3 -c "import pandas" 2>/dev/null; then
    echo "✓ Installed"
else
    echo "✗ Not installed"
    FAILED=$((FAILED + 1))
fi

# Check build tools
echo -n "9. Checking GCC compiler... "
if command -v gcc &> /dev/null; then
    GCC_VERSION=$(gcc --version | head -n1)
    echo "✓ $GCC_VERSION"
else
    echo "✗ GCC not found"
    FAILED=$((FAILED + 1))
fi

# Check make
echo -n "10. Checking make... "
if command -v make &> /dev/null; then
    echo "✓ Installed"
else
    echo "✗ Not installed"
    FAILED=$((FAILED + 1))
fi

# Check eBPF support
echo -n "11. Checking eBPF support... "
if [ -d "/sys/kernel/debug/tracing" ]; then
    echo "✓ Enabled"
else
    echo "⚠ Debugfs not mounted or eBPF not supported"
    FAILED=$((FAILED + 1))
fi

# Check JIT compiler
echo -n "12. Checking BPF JIT compiler... "
JIT_STATUS=$(cat /proc/sys/net/core/bpf_jit_enable 2>/dev/null)
if [ "$JIT_STATUS" = "1" ]; then
    echo "✓ Enabled"
elif [ "$JIT_STATUS" = "0" ]; then
    echo "⚠ Disabled (performance will be reduced)"
else
    echo "⚠ Cannot determine status"
fi

# Check stress-ng
echo -n "13. Checking stress-ng... "
if command -v stress-ng &> /dev/null; then
    echo "✓ Installed"
else
    echo "⚠ Not installed (optional for benchmarking)"
fi

# Check sysbench
echo -n "14. Checking sysbench... "
if command -v sysbench &> /dev/null; then
    echo "✓ Installed"
else
    echo "⚠ Not installed (optional for benchmarking)"
fi

# Check perf
echo -n "15. Checking perf tools... "
if command -v perf &> /dev/null; then
    echo "✓ Installed"
else
    echo "⚠ Not installed (optional for benchmarking)"
fi

echo ""
echo "=========================================="
if [ $FAILED -eq 0 ]; then
    echo "Result: ✓ All critical checks passed!"
    echo "Your ARM64 environment is ready for MLLB."
    exit 0
else
    echo "Result: ✗ $FAILED check(s) failed"
    echo "Please install missing dependencies before proceeding."
    echo "Run: sudo ./arm64_scripts/setup_arm64_environment.sh"
    exit 1
fi
echo "=========================================="
