#!/bin/bash
# ARM64 Environment Setup Script for MLLB
# This script automates the installation of dependencies for ARM64 architecture

set -e  # Exit on error

echo "=========================================="
echo "MLLB ARM64 Environment Setup Script"
echo "=========================================="
echo ""

# Check if running on ARM64
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "arm64" ]; then
    echo "Warning: This script is designed for ARM64 architecture."
    echo "Current architecture: $ARCH"
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "This script requires root privileges. Please run with sudo."
    exit 1
fi

echo "Step 1: Updating system packages..."
apt update
apt upgrade -y

echo ""
echo "Step 2: Installing basic development tools..."
apt install -y build-essential linux-headers-$(uname -r) git python3 python3-pip cmake

echo ""
echo "Step 3: Installing BCC dependencies..."
apt install -y bpfcc-tools python3-bpfcc libbpfcc-dev

# Verify BCC installation
echo "Verifying BCC installation..."
if python3 -c "from bcc import BPF" 2>/dev/null; then
    echo "✓ BCC installed successfully"
else
    echo "⚠ BCC installation may have issues. Consider manual installation."
fi

echo ""
echo "Step 4: Installing TensorFlow for ARM64..."
pip3 install --upgrade pip

# Try to install TensorFlow
if pip3 install tensorflow 2>/dev/null; then
    echo "✓ TensorFlow installed successfully"
else
    echo "⚠ Standard TensorFlow installation failed, trying ARM64-specific version..."
    pip3 install tensorflow-aarch64 || echo "⚠ TensorFlow installation failed. May need manual installation."
fi

# Verify TensorFlow installation
echo "Verifying TensorFlow installation..."
if python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)" 2>/dev/null; then
    echo "✓ TensorFlow verified"
else
    echo "⚠ TensorFlow verification failed"
fi

echo ""
echo "Step 5: Installing additional Python packages..."
pip3 install numpy pandas scikit-learn matplotlib

echo ""
echo "Step 6: Installing benchmark tools..."
apt install -y sysbench stress-ng

echo ""
echo "Step 7: Configuring kernel parameters..."
cat > /etc/sysctl.d/99-mllb.conf << EOF
# MLLB ARM64 optimizations
net.core.bpf_jit_enable = 1
net.core.bpf_jit_limit = 264241152
kernel.perf_event_paranoid = -1
kernel.kptr_restrict = 0
kernel.sched_migration_cost_ns = 500000
kernel.sched_nr_migrate = 32
EOF

sysctl -p /etc/sysctl.d/99-mllb.conf

echo ""
echo "Step 8: Setting up environment variables..."
# Add to current user's bashrc (not root's)
SUDO_USER_HOME=$(eval echo ~$SUDO_USER)
if [ -n "$SUDO_USER_HOME" ] && [ -d "$SUDO_USER_HOME" ]; then
    cat >> $SUDO_USER_HOME/.bashrc << EOF

# MLLB environment variables
export MLLB_HOME="$PWD"
export PYTHONPATH="\${MLLB_HOME}:\${PYTHONPATH}"
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0
EOF
    echo "Environment variables added to $SUDO_USER_HOME/.bashrc"
fi

echo ""
echo "=========================================="
echo "Installation Summary"
echo "=========================================="
echo "✓ System packages updated"
echo "✓ Development tools installed"
echo "✓ BCC installed"
echo "✓ TensorFlow installed (verify above)"
echo "✓ Additional Python packages installed"
echo "✓ Benchmark tools installed"
echo "✓ Kernel parameters configured"
echo "✓ Environment variables set"
echo ""
echo "Please log out and log back in, or run:"
echo "  source ~/.bashrc"
echo ""
echo "Then verify installation with:"
echo "  ./arm64_scripts/verify_installation.sh"
echo "=========================================="
