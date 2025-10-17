# ARM64 Scripts for MLLB Deployment

This directory contains helper scripts for deploying MLLB on ARM64 architecture.

## Scripts Overview

### 1. setup_arm64_environment.sh
Automated installation script for all dependencies required for MLLB on ARM64.

**Usage:**
```bash
sudo ./setup_arm64_environment.sh
```

**What it does:**
- Updates system packages
- Installs development tools (gcc, make, etc.)
- Installs BCC (BPF Compiler Collection)
- Installs TensorFlow for ARM64
- Installs Python dependencies (numpy, pandas, etc.)
- Installs benchmark tools (stress-ng, sysbench)
- Configures kernel parameters
- Sets up environment variables

### 2. verify_installation.sh
Verification script to check if all dependencies are correctly installed.

**Usage:**
```bash
./verify_installation.sh
```

**What it checks:**
- Architecture (ARM64)
- Kernel version (>= 4.15)
- Kernel headers
- Python version
- BCC installation
- TensorFlow installation
- NumPy, Pandas
- GCC compiler
- eBPF support
- BPF JIT compiler
- Benchmark tools

### 3. generate_optimized_makefile.sh
Generates an optimized Makefile for kernel module compilation with ARM64-specific flags.

**Usage:**
```bash
./generate_optimized_makefile.sh
```

**What it does:**
- Detects CPU model (Cortex-A53, A72, A76, etc.)
- Generates optimized compilation flags
- Creates Makefile.arm64 in kmod directory
- Enables aggressive optimizations (-O3, -ffast-math, etc.)

### 4. quick_deploy.sh
Complete automated deployment script that runs all steps from data collection to kernel module compilation.

**Usage:**
```bash
./quick_deploy.sh [data_tag] [model_tag]
```

**Arguments:**
- `data_tag`: Tag for data collection output (default: arm64_baseline)
- `model_tag`: Tag for trained model (default: arm64_model)

**What it does:**
1. Verifies environment
2. Collects load balancing data (30 minutes)
3. Trains machine learning model
4. Exports model weights to C code
5. Compiles kernel module
6. Tests kernel module

**Example:**
```bash
./quick_deploy.sh mydata mymodel
```

### 5. run_benchmarks.sh
Comprehensive benchmark suite that compares native scheduler vs MLLB scheduler.

**Usage:**
```bash
./run_benchmarks.sh
```

**Test scenarios:**
1. CPU intensive workload
2. Context switch performance
3. Mixed workload (CPU + I/O + Memory)
4. Thread migration test (if available)

**Output:**
- Creates timestamped results directory
- Generates system_info.txt
- Generates benchmark_summary.txt
- Saves detailed test results

## Quick Start Guide

### First Time Setup

1. **Install dependencies:**
```bash
cd /path/to/MLLB/arm64_scripts
chmod +x *.sh
sudo ./setup_arm64_environment.sh
```

2. **Verify installation:**
```bash
./verify_installation.sh
```

3. **Log out and log back in** to apply environment variables, or:
```bash
source ~/.bashrc
```

### Deploy MLLB

Option A - Quick deployment (automated):
```bash
./quick_deploy.sh
```

Option B - Manual deployment (see main ARM64_DEPLOYMENT.md guide)

### Run Benchmarks

After successful deployment:
```bash
./run_benchmarks.sh
```

## ARM64-Specific Optimizations

### CPU-Specific Tuning

The scripts automatically detect your ARM64 CPU and apply appropriate optimizations:

- **Cortex-A53**: `-mtune=cortex-a53`
- **Cortex-A72**: `-mtune=cortex-a72`
- **Cortex-A76**: `-mtune=cortex-a76`
- **Other**: `-mtune=native`

### Compiler Flags

Optimized builds use:
- `-O3`: Aggressive optimization
- `-march=armv8-a`: Target ARMv8 architecture
- `-mhard-float`: Hardware floating-point
- `-ffast-math`: Fast math operations
- `-funroll-loops`: Loop unrolling
- `-ftree-vectorize`: Auto-vectorization

### Kernel Parameters

Configured for ARM64:
- BPF JIT compilation enabled
- Optimized scheduler migration parameters
- Performance monitoring enabled

## Troubleshooting

### Permission Denied
Make sure scripts are executable:
```bash
chmod +x *.sh
```

### Module Loading Failed
Check kernel module signature requirements:
```bash
mokutil --sb-state
```

If Secure Boot is enabled, you may need to disable it or sign the module.

### BCC Installation Failed
Try installing from source:
```bash
# See ARM64_DEPLOYMENT.md for detailed instructions
```

### TensorFlow Issues
For ARM64-specific TensorFlow builds:
```bash
pip3 install tensorflow-aarch64
# or
pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v50 tensorflow
```

## Performance Tips

1. **CPU Governor**: Set to performance mode
```bash
sudo cpupower frequency-set -g performance
```

2. **Disable CPU Idle**: Reduce latency
```bash
sudo cpupower idle-set -D 0
```

3. **NUMA Binding**: Pin to specific NUMA node
```bash
numactl --cpunodebind=0 --membind=0 <command>
```

4. **Reduce Sampling**: Lower CPU overhead
- Edit `dump_config.py` to reduce sampling rate

## Additional Resources

- Main deployment guide: `../ARM64_DEPLOYMENT.md` (Chinese)
- English deployment guide: `../ARM64_DEPLOYMENT_EN.md`
- Project README: `../README.md`
- Paper: [Machine Learning for Load Balancing in the Linux Kernel](https://doi.org/10.1145/3409963.3410492)

## Support

For issues specific to ARM64 deployment, please check:
1. The troubleshooting sections in the deployment guides
2. System logs: `dmesg | tail -50`
3. Kernel module logs: `journalctl -xe`
4. GitHub issues: https://github.com/LiJiajun-WHU/MLLB/issues

## Version

- Version: 1.0
- Last Updated: 2024
- Tested on: Ubuntu 20.04/22.04 ARM64, Debian 11/12 ARM64
