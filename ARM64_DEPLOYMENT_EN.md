# MLLB ARM64 Processor Deployment Guide

This document provides a complete operational guide for deploying the Machine Learning for Load Balancing (MLLB) system on ARM64 processor architecture.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Prerequisites Installation](#prerequisites-installation)
3. [Parameter Configuration](#parameter-configuration)
4. [File Download and Preparation](#file-download-and-preparation)
5. [Deployment Steps](#deployment-steps)
6. [Performance Optimization](#performance-optimization)
7. [Experimental Comparison Testing](#experimental-comparison-testing)
8. [Troubleshooting](#troubleshooting)

## System Requirements

### Hardware Requirements
- ARM64 architecture processor (ARMv8 or higher)
- Recommended 4 cores or more
- At least 4GB RAM (8GB+ recommended)
- At least 20GB available disk space

### Software Requirements
- Linux kernel version: 4.15 or higher (5.x series recommended)
- Operating system: Ubuntu 20.04/22.04 ARM64, Debian 11/12 ARM64, or other ARM64 Linux distributions
- Python version: 3.7 or higher
- GCC compiler: ARM64 architecture support (usually pre-installed)

## Prerequisites Installation

### 1. Update System Package Manager

```bash
sudo apt update
sudo apt upgrade -y
```

### 2. Install Basic Development Tools

```bash
sudo apt install -y build-essential
sudo apt install -y linux-headers-$(uname -r)
sudo apt install -y git
sudo apt install -y python3 python3-pip
sudo apt install -y cmake
```

### 3. Install BCC (BPF Compiler Collection)

BCC is a toolset for eBPF program development and is a core dependency of this project.

#### Method 1: Install from Repository (Recommended for Ubuntu)

```bash
sudo apt install -y bpfcc-tools linux-headers-$(uname -r)
sudo apt install -y python3-bpfcc
```

#### Method 2: Compile from Source (for latest version or other distributions)

```bash
# Install dependencies
sudo apt install -y bison build-essential cmake flex git libedit-dev \
  libllvm12 llvm-12-dev libclang-12-dev python3 zlib1g-dev libelf-dev \
  libfl-dev python3-setuptools liblzma-dev arping netperf iperf

# Clone BCC repository
git clone https://github.com/iovisor/bcc.git
cd bcc

# Create build directory
mkdir build
cd build

# Configure and compile (for ARM64 architecture)
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install

# Install Python bindings
cd ../
sudo pip3 install .
```

**Note**: Compiling BCC on ARM64 platform may take 20-40 minutes depending on processor performance.

### 4. Install TensorFlow

TensorFlow is used for machine learning model training and inference.

#### Method 1: Install using pip (Recommended)

```bash
# Install TensorFlow (ARM64 optimized version)
sudo pip3 install tensorflow-aarch64

# Or install standard version
sudo pip3 install tensorflow
```

#### Method 2: Optimized for Specific ARM64 Chips (e.g., Raspberry Pi 4 or Jetson series)

For NVIDIA Jetson platform:
```bash
# Install NVIDIA provided optimized TensorFlow
sudo pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v50 tensorflow
```

For Raspberry Pi and other ARM64 devices:
```bash
# Use precompiled wheel files
wget https://github.com/PINTO0309/Tensorflow-bin/releases/download/v2.9.0/tensorflow-2.9.0-cp39-none-linux_aarch64.whl
sudo pip3 install tensorflow-2.9.0-cp39-none-linux_aarch64.whl
```

### 5. Install Other Python Dependencies

```bash
sudo pip3 install numpy pandas scikit-learn matplotlib
```

### 6. Verify Installation

```bash
# Verify BCC installation
python3 -c "from bcc import BPF; print('BCC installed successfully')"

# Verify TensorFlow installation
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"

# Check kernel version
uname -r

# Verify eBPF support
ls /sys/kernel/debug/tracing/
```

## Parameter Configuration

### 1. Kernel Parameter Optimization

Create or edit `/etc/sysctl.d/99-mllb.conf`:

```bash
sudo nano /etc/sysctl.d/99-mllb.conf
```

Add the following content:

```conf
# Enable eBPF JIT compiler (ARM64 architecture)
net.core.bpf_jit_enable = 1

# Increase eBPF complexity limit
net.core.bpf_jit_limit = 264241152

# Performance monitoring related
kernel.perf_event_paranoid = -1
kernel.kptr_restrict = 0

# Scheduler parameters
kernel.sched_migration_cost_ns = 500000
kernel.sched_nr_migrate = 32
```

Apply configuration:
```bash
sudo sysctl -p /etc/sysctl.d/99-mllb.conf
```

### 2. Environment Variable Setup

Add to `~/.bashrc` or `~/.profile`:

```bash
# MLLB environment variables
export MLLB_HOME="/path/to/MLLB"
export PYTHONPATH="${MLLB_HOME}:${PYTHONPATH}"

# TensorFlow ARM64 optimization
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0
```

Apply changes:
```bash
source ~/.bashrc
```

### 3. Data Collection Parameters

Edit `dump_config.py` file to configure data collection parameters:

```python
# Sampling rate settings (reduce CPU usage)
SAMPLE_RATE = 1000  # Sample every 1000 events (ARM64 recommended value)

# Buffer size (adjust according to ARM64 memory)
BUFFER_SIZE = 128  # Reduce to 128 pages (default 256)

# Write batch size
WRITE_BATCH_SIZE = 10000  # ARM64 recommended value
```

### 4. Training Parameter Configuration

Edit `training/training_config.py`:

```python
# ARM64 optimized training parameters
BATCH_SIZE = 256  # Reduce batch size to fit ARM64 memory limitations
EPOCHS = 50  # Number of training epochs
LEARNING_RATE = 0.001  # Learning rate

# Model architecture (lightweight for ARM64)
HIDDEN_LAYERS = [64, 32]  # Reduce hidden layer size
DROPOUT_RATE = 0.2  # Dropout rate
```

## File Download and Preparation

### 1. Clone MLLB Repository

```bash
cd ~
git clone https://github.com/LiJiajun-WHU/MLLB.git
cd MLLB
```

### 2. Verify File Integrity

```bash
# List main files
ls -lh
# Should see: dump_lb.py, dump_lb.c, training/, eval/, kmod/ directories and files

# Check key files
test -f dump_lb.py && echo "dump_lb.py exists"
test -f dump_lb.c && echo "dump_lb.c exists"
test -d training && echo "training/ exists"
test -d kmod && echo "kmod/ exists"
```

### 3. Prepare Working Directories

```bash
# Create data directories
mkdir -p ~/MLLB/data
mkdir -p ~/MLLB/models
mkdir -p ~/MLLB/logs

# Set permissions
chmod +x dump_lb.py
chmod +x training/automate.py
chmod +x numa_map.py
```

## Deployment Steps

### Phase 1: Data Collection

#### 1. Verify System Environment

```bash
# Check if running with root privileges
sudo whoami

# Check kernel modules
lsmod | grep bpf

# Check debugfs mount
mount | grep debugfs
```

#### 2. Start Load Balancing Data Collection

```bash
cd ~/MLLB

# Basic collection (using default parameters)
sudo python3 dump_lb.py -t baseline

# Or use custom output file
sudo python3 dump_lb.py -o data/arm64_data.csv

# If using unmodified kernel, add --old flag
sudo python3 dump_lb.py -t baseline --old
```

**Important Notes**:
- Data collection process needs to run for a while to gather sufficient samples (recommended at least 15-30 minutes)
- Run various workloads during collection to obtain diverse data
- Use `Ctrl+C` to stop collection

#### 3. Run Test Workloads

During data collection, run the following commands in another terminal to generate system load:

```bash
# Compile test program
cd ~/MLLB
gcc -o pthread_fibo pthread_fibo_create.c -lpthread

# Run test load
./pthread_fibo

# Or use system tools to generate load
stress-ng --cpu 4 --timeout 600s  # Need to install first: apt install stress-ng
```

#### 4. Verify Data Collection

```bash
# Check generated data files
ls -lh raw_*.csv

# View first few lines of data
head -n 20 raw_baseline.csv

# Count lines (number of samples)
wc -l raw_baseline.csv
```

### Phase 2: Model Training

#### 1. Prepare Training Data

```bash
cd ~/MLLB/training

# Use automated script for training
# -t specifies data tags (can be multiple)
# -o specifies output model name
python3 automate.py -t baseline -o arm64_model

# If you have multiple datasets, you can merge and train
# python3 automate.py -t baseline test1 test2 -o arm64_model -b
```

The training script will automatically:
1. Preprocess data (`prep.py`)
2. Train Keras model (`keras_lb.py`)
3. Save model weights

#### 2. Monitor Training Process

Training process will output:
- Loss value and accuracy for each epoch
- Training time
- Model save location

```
Epoch 1/50
loss: 0.5234 - accuracy: 0.7456
Epoch 2/50
loss: 0.4123 - accuracy: 0.8012
...
Model saved to: models/arm64_model.h5
```

#### 3. Evaluate Model Performance

```bash
cd ~/MLLB/eval

# Evaluate accuracy
python3 eval_acc.py --model ../training/models/arm64_model.h5 --data ../raw_baseline.csv

# Evaluate prediction time
python3 eval_time.py --model ../training/models/arm64_model.h5
```

### Phase 3: Kernel Module Deployment

#### 1. Export Model Weights

```bash
cd ~/MLLB/training

# Export weights as C code
python3 dump_weights.py --model models/arm64_model.h5 --output ../kmod/c_mlp.h
```

This will generate a C header file containing model weights.

#### 2. Compile Kernel Module

```bash
cd ~/MLLB/kmod

# ARM64 specific compilation options are already configured in Makefile
# Compile module
make clean
make

# Verify compilation result
ls -lh jc_kmod.ko
file jc_kmod.ko  # Should show ARM aarch64 architecture
```

**ARM64 Notes**:
- The `-mhard-float` flag in Makefile ensures hardware floating-point arithmetic is used
- Ensure kernel headers match current kernel version

#### 3. Test Kernel Module

```bash
# Clear dmesg logs
sudo dmesg -C

# Load module
sudo insmod jc_kmod.ko

# Check if module is loaded
lsmod | grep jc_kmod

# View kernel logs
dmesg | tail -20

# Unload module
sudo rmmod jc_kmod

# View logs again
dmesg | tail -20
```

#### 4. Run Complete Test

```bash
# Use make test command
sudo make test
```

### Phase 4: Integration Deployment

#### 1. Persist Kernel Module

Create module configuration file:

```bash
sudo nano /etc/modules-load.d/mllb.conf
```

Add:
```
jc_kmod
```

Copy module to system directory:
```bash
sudo cp jc_kmod.ko /lib/modules/$(uname -r)/kernel/drivers/misc/
sudo depmod -a
```

#### 2. Create systemd Service (Optional)

Create service file:

```bash
sudo nano /etc/systemd/system/mllb-collector.service
```

Content:
```ini
[Unit]
Description=MLLB Data Collector Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/home/user/MLLB
ExecStart=/usr/bin/python3 /home/user/MLLB/dump_lb.py -t production -a
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable mllb-collector.service
sudo systemctl start mllb-collector.service
sudo systemctl status mllb-collector.service
```

## Performance Optimization

### 1. ARM64 Specific Optimizations

#### CPU Affinity Settings
```bash
# Pin BPF program to specific CPU cores
sudo taskset -c 0-3 python3 dump_lb.py -t optimized
```

#### NUMA Optimization (if applicable)
```bash
# Check NUMA topology
numactl --hardware

# Run on specific NUMA node
numactl --cpunodebind=0 --membind=0 python3 dump_lb.py -t numa_opt
```

### 2. Model Optimization

#### Model Quantization
Add quantization support in `training/keras_lb.py` to reduce model size and inference time:

```python
import tensorflow as tf

# Post-training quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save quantized model
with open('models/arm64_model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)
```

#### Reduce Model Complexity
```python
# Adjust in training_config.py
HIDDEN_LAYERS = [32, 16]  # Smaller network
FEATURES = features[:10]   # Reduce number of features
```

### 3. Data Collection Optimization

#### Reduce Sampling Frequency
```python
# Add sampling logic in dump_lb.py
sample_counter = 0
SAMPLE_RATE = 10  # Sample 1 out of every 10 events

def can_migrate_handler(cpu, data, size):
    global sample_counter
    sample_counter += 1
    if sample_counter % SAMPLE_RATE != 0:
        return
    event = b['can_migrate_events'].event(data)
    cm_events.append(event)
```

#### Use Ring Buffer
```python
# Limit number of events in memory
MAX_EVENTS = 1000
if len(cm_events) > MAX_EVENTS:
    cm_events = cm_events[-MAX_EVENTS:]
```

### 4. System Level Optimization

#### Enable Performance Governor
```bash
# Set CPU scheduling policy to performance mode
sudo cpupower frequency-set -g performance

# Disable CPU idle states (reduce latency)
sudo cpupower idle-set -D 0
```

#### Adjust I/O Scheduler
```bash
# Use none or mq-deadline for SSD devices
echo none | sudo tee /sys/block/sda/queue/scheduler
```

### 5. Compiler Optimization

Modify `kmod/Makefile` to enable ARM64 specific optimizations:

```makefile
obj-m += jc_kmod.o
CFLAGS_jc_kmod.o := -mhard-float -O3 -march=armv8-a -mtune=cortex-a72
all:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
test:
	dmesg -C
	insmod jc_kmod.ko
	rmmod jc_kmod
	dmesg
```

Adjust `-mtune` parameter based on specific ARM64 processor:
- Cortex-A53: `-mtune=cortex-a53`
- Cortex-A72: `-mtune=cortex-a72`
- Cortex-A76: `-mtune=cortex-a76`
- Apple M1: `-mtune=apple-a14` (requires newer GCC)

## Experimental Comparison Testing

### 1. Benchmark Setup

#### Test Environment Preparation
```bash
# Create test script directory
mkdir -p ~/MLLB/benchmarks
cd ~/MLLB/benchmarks
```

#### Benchmark Tool Installation
```bash
sudo apt install -y sysbench stress-ng perf-tools-unstable
```

### 2. Test Scenarios

#### Scenario 1: CPU Intensive Workload

Create test script `cpu_intensive_test.sh`:

```bash
#!/bin/bash

echo "Starting CPU intensive benchmark..."

# Test 1: Native kernel scheduler
echo "Test 1: Native kernel scheduler"
stress-ng --cpu $(nproc) --cpu-method all --metrics --timeout 300s > results_native_cpu.txt 2>&1

# Wait for system to stabilize
sleep 30

# Test 2: MLLB scheduler
echo "Test 2: MLLB scheduler"
# Ensure MLLB module is loaded
sudo insmod ../kmod/jc_kmod.ko
stress-ng --cpu $(nproc) --cpu-method all --metrics --timeout 300s > results_mllb_cpu.txt 2>&1
sudo rmmod jc_kmod

echo "CPU intensive benchmark completed"
```

#### Scenario 2: Multithreaded Workload

Create test script `multithread_test.sh`:

```bash
#!/bin/bash

echo "Starting multithread benchmark..."

# Compile test program
gcc -o ../pthread_fibo ../pthread_fibo_create.c -lpthread

# Test 1: Native scheduler
echo "Test 1: Native scheduler - multithread"
perf stat -e cycles,instructions,cache-misses,branch-misses \
    ../pthread_fibo > results_native_thread.txt 2>&1

sleep 10

# Test 2: MLLB scheduler
echo "Test 2: MLLB scheduler - multithread"
sudo insmod ../kmod/jc_kmod.ko
perf stat -e cycles,instructions,cache-misses,branch-misses \
    ../pthread_fibo > results_mllb_thread.txt 2>&1
sudo rmmod jc_kmod

echo "Multithread benchmark completed"
```

### 3. Run Tests and Generate Reports

```bash
cd ~/MLLB/benchmarks

# Make scripts executable
chmod +x *.sh

# Run complete test suite
./run_all_benchmarks.sh

# Enter results directory
cd test_results_*

# View summary report
cat benchmark_summary.txt
```

### 4. Key Performance Metrics

Tests should focus on the following metrics:

1. **Throughput Metrics**
   - Operations per second (bogo ops/s)
   - I/O throughput (MB/s)
   - Transaction processing rate (TPS)

2. **Latency Metrics**
   - Average response time
   - 95/99 percentile latency
   - Maximum latency

3. **Resource Utilization**
   - CPU usage
   - Memory footprint
   - Context switch count
   - Process migration count

4. **Energy Metrics** (if device supports)
   - Power consumption (Watts)
   - Operations per Joule (energy efficiency)

5. **Scheduling Efficiency**
   - Load balance degree
   - CPU idle time
   - Task migration overhead

## Troubleshooting

### Common Issues and Solutions

#### 1. BCC Installation Failed

**Problem**: `bcc/BPF.h: No such file or directory`

**Solution**:
```bash
# Ensure correct development packages are installed
sudo apt install -y libbpfcc-dev python3-bpfcc

# Check BCC installation path
dpkg -L python3-bpfcc | grep -i bpf
```

#### 2. Kernel Headers Mismatch

**Problem**: `linux/kernel.h: No such file or directory`

**Solution**:
```bash
# Install matching kernel headers
sudo apt install -y linux-headers-$(uname -r)

# Verify installation
ls /lib/modules/$(uname -r)/build
```

#### 3. TensorFlow Performance on ARM64 is Low

**Problem**: Training speed extremely slow

**Solution**:
```bash
# Use lightweight model
# Reduce model size in training_config.py

# Or use TensorFlow Lite
pip3 install tflite-runtime

# Use XNNPACK acceleration
export TF_ENABLE_XNNPACK=1
```

#### 4. eBPF Program Loading Failed

**Problem**: `Failed to load BPF program`

**Solution**:
```bash
# Check kernel configuration
zcat /proc/config.gz | grep -i bpf

# Ensure following options are enabled:
# CONFIG_BPF=y
# CONFIG_BPF_SYSCALL=y
# CONFIG_BPF_JIT=y

# Check permissions
sudo sysctl kernel.unprivileged_bpf_disabled
# If 1, set to 0
sudo sysctl kernel.unprivileged_bpf_disabled=0
```

#### 5. Kernel Module Loading Failed

**Problem**: `insmod: ERROR: could not insert module`

**Solution**:
```bash
# View detailed error
dmesg | tail -20

# Check module signature (if Secure Boot is enabled)
mokutil --sb-state

# If needed, sign module
sudo /usr/src/linux-headers-$(uname -r)/scripts/sign-file \
    sha256 signing_key.priv signing_key.x509 jc_kmod.ko

# Or disable Secure Boot (not recommended for production)
```

## References

### Official Documentation
- [BCC GitHub Repository](https://github.com/iovisor/bcc)
- [TensorFlow Official Documentation](https://www.tensorflow.org/install)
- [Linux Kernel Documentation - Scheduler](https://www.kernel.org/doc/html/latest/scheduler/)
- [eBPF Documentation](https://ebpf.io/what-is-ebpf)

### ARM64 Specific Resources
- [ARM Developer Documentation](https://developer.arm.com/documentation)
- [Linux ARM64 Porting Guide](https://www.kernel.org/doc/html/latest/arm64/)

### Papers and Research
- [Machine Learning for Load Balancing in the Linux Kernel](https://doi.org/10.1145/3409963.3410492)

## Summary

This guide covers the complete process of deploying MLLB system on ARM64 processors, including:

1. ✅ **Environment Preparation**: Install all required dependencies
2. ✅ **Parameter Configuration**: ARM64 architecture optimized parameter settings
3. ✅ **Data Collection**: Collect scheduling data using eBPF
4. ✅ **Model Training**: Train machine learning models
5. ✅ **Kernel Integration**: Deploy model into kernel module
6. ✅ **Performance Optimization**: ARM64 specific optimization techniques
7. ✅ **Comprehensive Testing**: Multi-scenario performance comparison testing

By following this guide, you can successfully deploy and optimize the MLLB system on ARM64 platform to achieve intelligent load balancing scheduling.

---

**Version**: 1.0  
**Last Updated**: 2024  
**Maintainer**: MLLB Project Team
