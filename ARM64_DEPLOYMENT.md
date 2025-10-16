# MLLB ARM64处理器部署指南

本文档提供了将Machine Learning for Load Balancing (MLLB)系统部署到ARM64处理器架构的完整操作流程。

## 目录
1. [系统要求](#系统要求)
2. [前置依赖安装](#前置依赖安装)
3. [参数设置](#参数设置)
4. [文件下载与准备](#文件下载与准备)
5. [实际部署步骤](#实际部署步骤)
6. [性能优化建议](#性能优化建议)
7. [实验对比测试](#实验对比测试)
8. [故障排除](#故障排除)

## 系统要求

### 硬件要求
- ARM64架构处理器（ARMv8或更高版本）
- 推荐4核心或以上
- 至少4GB RAM（推荐8GB以上）
- 至少20GB可用磁盘空间

### 软件要求
- Linux内核版本：4.15或更高（推荐5.x系列）
- 操作系统：Ubuntu 20.04/22.04 ARM64、Debian 11/12 ARM64或其他ARM64 Linux发行版
- Python版本：3.7或更高
- GCC编译器：支持ARM64架构（通常预装）

## 前置依赖安装

### 1. 更新系统包管理器

```bash
sudo apt update
sudo apt upgrade -y
```

### 2. 安装基础开发工具

```bash
sudo apt install -y build-essential
sudo apt install -y linux-headers-$(uname -r)
sudo apt install -y git
sudo apt install -y python3 python3-pip
sudo apt install -y cmake
```

### 3. 安装BCC (BPF Compiler Collection)

BCC是用于eBPF程序开发的工具集，是本项目的核心依赖。

#### 方法1：从软件仓库安装（推荐用于Ubuntu）

```bash
sudo apt install -y bpfcc-tools linux-headers-$(uname -r)
sudo apt install -y python3-bpfcc
```

#### 方法2：从源码编译（适用于需要最新版本或其他发行版）

```bash
# 安装依赖
sudo apt install -y bison build-essential cmake flex git libedit-dev \
  libllvm12 llvm-12-dev libclang-12-dev python3 zlib1g-dev libelf-dev \
  libfl-dev python3-setuptools liblzma-dev arping netperf iperf

# 克隆BCC仓库
git clone https://github.com/iovisor/bcc.git
cd bcc

# 创建构建目录
mkdir build
cd build

# 配置并编译（针对ARM64架构）
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install

# 安装Python绑定
cd ../
sudo pip3 install .
```

**注意**：在ARM64平台上编译BCC可能需要20-40分钟，取决于处理器性能。

### 4. 安装TensorFlow

TensorFlow用于机器学习模型的训练和推理。

#### 方法1：使用pip安装（推荐）

```bash
# 安装TensorFlow（ARM64优化版本）
sudo pip3 install tensorflow-aarch64

# 或者安装标准版本
sudo pip3 install tensorflow
```

#### 方法2：针对特定ARM64芯片优化（如树莓派4或Jetson系列）

对于NVIDIA Jetson平台：
```bash
# 安装NVIDIA提供的优化版TensorFlow
sudo pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v50 tensorflow
```

对于树莓派和其他ARM64设备：
```bash
# 使用预编译的wheel文件
wget https://github.com/PINTO0309/Tensorflow-bin/releases/download/v2.9.0/tensorflow-2.9.0-cp39-none-linux_aarch64.whl
sudo pip3 install tensorflow-2.9.0-cp39-none-linux_aarch64.whl
```

### 5. 安装其他Python依赖

```bash
sudo pip3 install numpy pandas scikit-learn matplotlib
```

### 6. 验证安装

```bash
# 验证BCC安装
python3 -c "from bcc import BPF; print('BCC installed successfully')"

# 验证TensorFlow安装
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"

# 检查内核版本
uname -r

# 验证eBPF支持
ls /sys/kernel/debug/tracing/
```

## 参数设置

### 1. 内核参数优化

创建或编辑`/etc/sysctl.d/99-mllb.conf`：

```bash
sudo nano /etc/sysctl.d/99-mllb.conf
```

添加以下内容：

```conf
# 启用eBPF JIT编译器（ARM64架构）
net.core.bpf_jit_enable = 1

# 增加eBPF复杂度限制
net.core.bpf_jit_limit = 264241152

# 性能监控相关
kernel.perf_event_paranoid = -1
kernel.kptr_restrict = 0

# 调度器参数
kernel.sched_migration_cost_ns = 500000
kernel.sched_nr_migrate = 32
```

应用配置：
```bash
sudo sysctl -p /etc/sysctl.d/99-mllb.conf
```

### 2. 环境变量设置

在`~/.bashrc`或`~/.profile`中添加：

```bash
# MLLB环境变量
export MLLB_HOME="/path/to/MLLB"
export PYTHONPATH="${MLLB_HOME}:${PYTHONPATH}"

# TensorFlow ARM64优化
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0
```

应用更改：
```bash
source ~/.bashrc
```

### 3. 数据采集参数

编辑`dump_config.py`文件以配置数据采集参数：

```python
# 采样率设置（降低CPU占用）
SAMPLE_RATE = 1000  # 每1000个事件采样一次（ARM64建议值）

# 缓冲区大小（根据ARM64内存调整）
BUFFER_SIZE = 128  # 降低至128页（默认256）

# 写入批次大小
WRITE_BATCH_SIZE = 10000  # ARM64建议值
```

### 4. 训练参数配置

编辑`training/training_config.py`：

```python
# ARM64优化的训练参数
BATCH_SIZE = 256  # 减小批次大小以适应ARM64内存限制
EPOCHS = 50  # 训练轮数
LEARNING_RATE = 0.001  # 学习率

# 模型架构（针对ARM64轻量化）
HIDDEN_LAYERS = [64, 32]  # 减小隐藏层大小
DROPOUT_RATE = 0.2  # Dropout率
```

## 文件下载与准备

### 1. 克隆MLLB仓库

```bash
cd ~
git clone https://github.com/LiJiajun-WHU/MLLB.git
cd MLLB
```

### 2. 检查文件完整性

```bash
# 列出主要文件
ls -lh
# 应该看到：dump_lb.py, dump_lb.c, training/, eval/, kmod/等目录和文件

# 检查关键文件
test -f dump_lb.py && echo "dump_lb.py exists"
test -f dump_lb.c && echo "dump_lb.c exists"
test -d training && echo "training/ exists"
test -d kmod && echo "kmod/ exists"
```

### 3. 准备工作目录

```bash
# 创建数据目录
mkdir -p ~/MLLB/data
mkdir -p ~/MLLB/models
mkdir -p ~/MLLB/logs

# 设置权限
chmod +x dump_lb.py
chmod +x training/automate.py
chmod +x numa_map.py
```

### 4. 下载测试数据集（可选）

如果有预训练模型或示例数据集：

```bash
# 示例：下载预训练模型（如果可用）
# wget https://example.com/pretrained_model_arm64.h5 -O models/pretrained.h5
```

## 实际部署步骤

### 第一阶段：数据采集

#### 1. 验证系统环境

```bash
# 检查是否以root权限运行
sudo whoami

# 检查内核模块
lsmod | grep bpf

# 检查debugfs挂载
mount | grep debugfs
```

#### 2. 启动负载均衡数据采集

```bash
cd ~/MLLB

# 基础采集（使用默认参数）
sudo python3 dump_lb.py -t baseline

# 或者使用自定义输出文件
sudo python3 dump_lb.py -o data/arm64_data.csv

# 如果使用未修改的内核，添加--old标志
sudo python3 dump_lb.py -t baseline --old
```

**重要提示**：
- 数据采集过程需要运行一段时间以收集足够的样本（建议至少15-30分钟）
- 在采集期间运行各种工作负载以获得多样化的数据
- 使用`Ctrl+C`停止采集

#### 3. 运行测试工作负载

在数据采集期间，在另一个终端运行以下命令生成系统负载：

```bash
# 编译测试程序
cd ~/MLLB
gcc -o pthread_fibo pthread_fibo_create.c -lpthread

# 运行测试负载
./pthread_fibo

# 或者使用系统工具生成负载
stress-ng --cpu 4 --timeout 600s  # 需要先安装：apt install stress-ng
```

#### 4. 验证数据收集

```bash
# 检查生成的数据文件
ls -lh raw_*.csv

# 查看数据前几行
head -n 20 raw_baseline.csv

# 统计行数（样本数量）
wc -l raw_baseline.csv
```

### 第二阶段：模型训练

#### 1. 准备训练数据

```bash
cd ~/MLLB/training

# 使用自动化脚本进行训练
# -t 指定数据标签（可以是多个）
# -o 指定输出模型名称
python3 automate.py -t baseline -o arm64_model

# 如果有多个数据集，可以合并训练
# python3 automate.py -t baseline test1 test2 -o arm64_model -b
```

训练脚本会自动：
1. 预处理数据（`prep.py`）
2. 训练Keras模型（`keras_lb.py`）
3. 保存模型权重

#### 2. 监控训练过程

训练过程会输出：
- 每轮(Epoch)的损失值和准确率
- 训练时间
- 模型保存位置

```
Epoch 1/50
loss: 0.5234 - accuracy: 0.7456
Epoch 2/50
loss: 0.4123 - accuracy: 0.8012
...
Model saved to: models/arm64_model.h5
```

#### 3. 评估模型性能

```bash
cd ~/MLLB/eval

# 评估准确率
python3 eval_acc.py --model ../training/models/arm64_model.h5 --data ../raw_baseline.csv

# 评估预测时间
python3 eval_time.py --model ../training/models/arm64_model.h5
```

### 第三阶段：内核模块部署

#### 1. 导出模型权重

```bash
cd ~/MLLB/training

# 导出权重为C代码
python3 dump_weights.py --model models/arm64_model.h5 --output ../kmod/c_mlp.h
```

这将生成包含模型权重的C头文件。

#### 2. 编译内核模块

```bash
cd ~/MLLB/kmod

# ARM64特定编译选项已在Makefile中配置
# 编译模块
make clean
make

# 验证编译结果
ls -lh jc_kmod.ko
file jc_kmod.ko  # 应显示ARM aarch64架构
```

**ARM64注意事项**：
- Makefile中的`-mhard-float`标志确保使用硬件浮点运算
- 确保使用与当前内核版本匹配的内核头文件

#### 3. 测试内核模块

```bash
# 清除dmesg日志
sudo dmesg -C

# 加载模块
sudo insmod jc_kmod.ko

# 检查模块是否加载
lsmod | grep jc_kmod

# 查看内核日志
dmesg | tail -20

# 卸载模块
sudo rmmod jc_kmod

# 再次查看日志
dmesg | tail -20
```

#### 4. 运行完整测试

```bash
# 使用make test命令
sudo make test
```

### 第四阶段：集成部署

#### 1. 持久化内核模块

创建模块配置文件：

```bash
sudo nano /etc/modules-load.d/mllb.conf
```

添加：
```
jc_kmod
```

复制模块到系统目录：
```bash
sudo cp jc_kmod.ko /lib/modules/$(uname -r)/kernel/drivers/misc/
sudo depmod -a
```

#### 2. 创建systemd服务（可选）

创建服务文件：

```bash
sudo nano /etc/systemd/system/mllb-collector.service
```

内容：
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

启用服务：
```bash
sudo systemctl daemon-reload
sudo systemctl enable mllb-collector.service
sudo systemctl start mllb-collector.service
sudo systemctl status mllb-collector.service
```

## 性能优化建议

### 1. ARM64特定优化

#### CPU亲和性设置
```bash
# 将BPF程序固定到特定CPU核心
sudo taskset -c 0-3 python3 dump_lb.py -t optimized
```

#### NUMA优化（如果适用）
```bash
# 检查NUMA拓扑
numactl --hardware

# 在特定NUMA节点运行
numactl --cpunodebind=0 --membind=0 python3 dump_lb.py -t numa_opt
```

### 2. 模型优化

#### 模型量化
在`training/keras_lb.py`中添加量化支持以减少模型大小和推理时间：

```python
import tensorflow as tf

# 训练后量化
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# 保存量化模型
with open('models/arm64_model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)
```

#### 减少模型复杂度
```python
# 在training_config.py中调整
HIDDEN_LAYERS = [32, 16]  # 更小的网络
FEATURES = features[:10]   # 减少特征数量
```

### 3. 数据采集优化

#### 降低采样频率
```python
# 在dump_lb.py中添加采样逻辑
sample_counter = 0
SAMPLE_RATE = 10  # 每10个事件采样1个

def can_migrate_handler(cpu, data, size):
    global sample_counter
    sample_counter += 1
    if sample_counter % SAMPLE_RATE != 0:
        return
    event = b['can_migrate_events'].event(data)
    cm_events.append(event)
```

#### 使用环形缓冲区
```python
# 限制内存中的事件数量
MAX_EVENTS = 1000
if len(cm_events) > MAX_EVENTS:
    cm_events = cm_events[-MAX_EVENTS:]
```

### 4. 系统级优化

#### 启用性能调度器
```bash
# 设置CPU调度策略为性能模式
sudo cpupower frequency-set -g performance

# 禁用CPU空闲状态（减少延迟）
sudo cpupower idle-set -D 0
```

#### 调整I/O调度器
```bash
# 对于SSD设备使用none或mq-deadline
echo none | sudo tee /sys/block/sda/queue/scheduler
```

### 5. 编译器优化

修改`kmod/Makefile`以启用ARM64特定优化：

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

根据具体ARM64处理器调整`-mtune`参数：
- Cortex-A53: `-mtune=cortex-a53`
- Cortex-A72: `-mtune=cortex-a72`
- Cortex-A76: `-mtune=cortex-a76`
- Apple M1: `-mtune=apple-a14` (需要较新的GCC)

## 实验对比测试

### 1. 基准测试设置

#### 测试环境准备
```bash
# 创建测试脚本目录
mkdir -p ~/MLLB/benchmarks
cd ~/MLLB/benchmarks
```

#### 基准测试工具安装
```bash
sudo apt install -y sysbench stress-ng perf-tools-unstable
```

### 2. 测试场景

#### 场景1：CPU密集型工作负载

创建测试脚本`cpu_intensive_test.sh`：

```bash
#!/bin/bash

echo "Starting CPU intensive benchmark..."

# 测试1：原生内核调度器
echo "Test 1: Native kernel scheduler"
stress-ng --cpu $(nproc) --cpu-method all --metrics --timeout 300s > results_native_cpu.txt 2>&1

# 等待系统稳定
sleep 30

# 测试2：MLLB调度器
echo "Test 2: MLLB scheduler"
# 确保MLLB模块已加载
sudo insmod ../kmod/jc_kmod.ko
stress-ng --cpu $(nproc) --cpu-method all --metrics --timeout 300s > results_mllb_cpu.txt 2>&1
sudo rmmod jc_kmod

echo "CPU intensive benchmark completed"
```

#### 场景2：多线程工作负载

创建测试脚本`multithread_test.sh`：

```bash
#!/bin/bash

echo "Starting multithread benchmark..."

# 编译测试程序
gcc -o ../pthread_fibo ../pthread_fibo_create.c -lpthread

# 测试1：原生调度器
echo "Test 1: Native scheduler - multithread"
perf stat -e cycles,instructions,cache-misses,branch-misses \
    ../pthread_fibo > results_native_thread.txt 2>&1

sleep 10

# 测试2：MLLB调度器
echo "Test 2: MLLB scheduler - multithread"
sudo insmod ../kmod/jc_kmod.ko
perf stat -e cycles,instructions,cache-misses,branch-misses \
    ../pthread_fibo > results_mllb_thread.txt 2>&1
sudo rmmod jc_kmod

echo "Multithread benchmark completed"
```

#### 场景3：I/O密集型工作负载

创建测试脚本`io_intensive_test.sh`：

```bash
#!/bin/bash

echo "Starting I/O intensive benchmark..."

# 测试1：原生调度器
echo "Test 1: Native scheduler - I/O"
sysbench fileio --file-total-size=4G --file-test-mode=rndrw \
    --time=300 --max-requests=0 prepare
sysbench fileio --file-total-size=4G --file-test-mode=rndrw \
    --time=300 --max-requests=0 run > results_native_io.txt
sysbench fileio --file-total-size=4G cleanup

sleep 30

# 测试2：MLLB调度器
echo "Test 2: MLLB scheduler - I/O"
sudo insmod ../kmod/jc_kmod.ko
sysbench fileio --file-total-size=4G --file-test-mode=rndrw \
    --time=300 --max-requests=0 prepare
sysbench fileio --file-total-size=4G --file-test-mode=rndrw \
    --time=300 --max-requests=0 run > results_mllb_io.txt
sysbench fileio --file-total-size=4G cleanup
sudo rmmod jc_kmod

echo "I/O intensive benchmark completed"
```

#### 场景4：混合工作负载

创建测试脚本`mixed_workload_test.sh`：

```bash
#!/bin/bash

echo "Starting mixed workload benchmark..."

# 测试1：原生调度器
echo "Test 1: Native scheduler - mixed"
stress-ng --cpu 2 --io 2 --vm 1 --vm-bytes 1G \
    --metrics --timeout 300s > results_native_mixed.txt 2>&1

sleep 30

# 测试2：MLLB调度器
echo "Test 2: MLLB scheduler - mixed"
sudo insmod ../kmod/jc_kmod.ko
stress-ng --cpu 2 --io 2 --vm 1 --vm-bytes 1G \
    --metrics --timeout 300s > results_mllb_mixed.txt 2>&1
sudo rmmod jc_kmod

echo "Mixed workload benchmark completed"
```

### 3. 性能指标收集

创建监控脚本`monitor_performance.sh`：

```bash
#!/bin/bash

DURATION=300  # 5分钟
INTERVAL=1    # 1秒采样间隔
OUTPUT_FILE="performance_metrics.csv"

echo "timestamp,cpu_usage,load_avg_1m,context_switches,migrations,memory_used" > $OUTPUT_FILE

for i in $(seq 1 $DURATION); do
    TIMESTAMP=$(date +%s)
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
    LOAD_AVG=$(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}' | xargs)
    CONTEXT_SW=$(cat /proc/stat | grep ctxt | awk '{print $2}')
    MIGRATIONS=$(cat /proc/schedstat | head -1 | awk '{print $4}')
    MEM_USED=$(free -m | awk 'NR==2{print $3}')
    
    echo "$TIMESTAMP,$CPU_USAGE,$LOAD_AVG,$CONTEXT_SW,$MIGRATIONS,$MEM_USED" >> $OUTPUT_FILE
    sleep $INTERVAL
done

echo "Performance monitoring completed"
```

### 4. 运行完整测试套件

创建主测试脚本`run_all_benchmarks.sh`：

```bash
#!/bin/bash

RESULTS_DIR="test_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR
cd $RESULTS_DIR

echo "=========================================="
echo "MLLB ARM64 Comprehensive Benchmark Suite"
echo "Started at: $(date)"
echo "=========================================="

# 记录系统信息
echo "=== System Information ===" | tee system_info.txt
uname -a | tee -a system_info.txt
lscpu | tee -a system_info.txt
free -h | tee -a system_info.txt
cat /proc/cpuinfo | grep "model name" | head -1 | tee -a system_info.txt

# 运行各个测试场景
bash ../cpu_intensive_test.sh
bash ../multithread_test.sh
bash ../io_intensive_test.sh
bash ../mixed_workload_test.sh

echo "=========================================="
echo "All benchmarks completed"
echo "Results saved in: $RESULTS_DIR"
echo "=========================================="

# 生成摘要报告
bash ../generate_report.sh
```

### 5. 结果分析

创建分析脚本`generate_report.sh`：

```bash
#!/bin/bash

REPORT_FILE="benchmark_summary.txt"

echo "MLLB ARM64 Benchmark Summary Report" > $REPORT_FILE
echo "Generated at: $(date)" >> $REPORT_FILE
echo "========================================" >> $REPORT_FILE
echo "" >> $REPORT_FILE

echo "=== CPU Intensive Test ===" >> $REPORT_FILE
if [ -f results_native_cpu.txt ] && [ -f results_mllb_cpu.txt ]; then
    echo "Native scheduler:" >> $REPORT_FILE
    grep "bogo ops/s" results_native_cpu.txt >> $REPORT_FILE
    echo "MLLB scheduler:" >> $REPORT_FILE
    grep "bogo ops/s" results_mllb_cpu.txt >> $REPORT_FILE
fi
echo "" >> $REPORT_FILE

echo "=== Multithread Test ===" >> $REPORT_FILE
if [ -f results_native_thread.txt ] && [ -f results_mllb_thread.txt ]; then
    echo "Native scheduler:" >> $REPORT_FILE
    grep "seconds time elapsed" results_native_thread.txt >> $REPORT_FILE
    echo "MLLB scheduler:" >> $REPORT_FILE
    grep "seconds time elapsed" results_mllb_thread.txt >> $REPORT_FILE
fi
echo "" >> $REPORT_FILE

echo "=== I/O Intensive Test ===" >> $REPORT_FILE
if [ -f results_native_io.txt ] && [ -f results_mllb_io.txt ]; then
    echo "Native scheduler:" >> $REPORT_FILE
    grep "throughput" results_native_io.txt >> $REPORT_FILE
    echo "MLLB scheduler:" >> $REPORT_FILE
    grep "throughput" results_mllb_io.txt >> $REPORT_FILE
fi
echo "" >> $REPORT_FILE

echo "=== Mixed Workload Test ===" >> $REPORT_FILE
if [ -f results_native_mixed.txt ] && [ -f results_mllb_mixed.txt ]; then
    echo "Native scheduler:" >> $REPORT_FILE
    grep "bogo ops/s" results_native_mixed.txt | head -5 >> $REPORT_FILE
    echo "MLLB scheduler:" >> $REPORT_FILE
    grep "bogo ops/s" results_mllb_mixed.txt | head -5 >> $REPORT_FILE
fi
echo "" >> $REPORT_FILE

cat $REPORT_FILE
```

### 6. 数据可视化

创建Python分析脚本`analyze_results.py`：

```python
#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def parse_stress_ng_output(filename):
    """解析stress-ng输出文件"""
    metrics = {}
    try:
        with open(filename, 'r') as f:
            for line in f:
                if 'bogo ops/s' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'bogo':
                            metrics['bogo_ops'] = float(parts[i-1])
                            break
    except FileNotFoundError:
        print(f"Warning: {filename} not found")
        return None
    return metrics

def create_comparison_chart():
    """创建性能对比图表"""
    # 测试场景
    scenarios = ['CPU Intensive', 'Multithread', 'I/O Intensive', 'Mixed Workload']
    
    # 示例数据（需要从实际测试结果中提取）
    native_performance = []
    mllb_performance = []
    
    # 解析每个测试场景的结果
    test_files = [
        ('results_native_cpu.txt', 'results_mllb_cpu.txt'),
        ('results_native_thread.txt', 'results_mllb_thread.txt'),
        ('results_native_io.txt', 'results_mllb_io.txt'),
        ('results_native_mixed.txt', 'results_mllb_mixed.txt'),
    ]
    
    for native_file, mllb_file in test_files:
        native_data = parse_stress_ng_output(native_file)
        mllb_data = parse_stress_ng_output(mllb_file)
        
        if native_data and mllb_data:
            native_performance.append(native_data.get('bogo_ops', 0))
            mllb_performance.append(mllb_data.get('bogo_ops', 0))
        else:
            native_performance.append(0)
            mllb_performance.append(0)
    
    # 创建柱状图
    x = np.arange(len(scenarios))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, native_performance, width, label='Native Scheduler')
    rects2 = ax.bar(x + width/2, mllb_performance, width, label='MLLB Scheduler')
    
    ax.set_xlabel('Test Scenarios')
    ax.set_ylabel('Performance (bogo ops/s)')
    ax.set_title('MLLB vs Native Scheduler Performance Comparison on ARM64')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300)
    print("Chart saved as performance_comparison.png")
    
    # 计算改进百分比
    print("\n=== Performance Improvement ===")
    for i, scenario in enumerate(scenarios):
        if native_performance[i] > 0:
            improvement = ((mllb_performance[i] - native_performance[i]) / native_performance[i]) * 100
            print(f"{scenario}: {improvement:+.2f}%")

if __name__ == "__main__":
    create_comparison_chart()
```

### 7. 运行测试并生成报告

```bash
cd ~/MLLB/benchmarks

# 给脚本添加执行权限
chmod +x *.sh
chmod +x analyze_results.py

# 运行完整测试套件
./run_all_benchmarks.sh

# 进入结果目录
cd test_results_*

# 生成可视化报告
python3 ../analyze_results.py

# 查看摘要报告
cat benchmark_summary.txt
```

### 8. 关键性能指标

测试应关注以下指标：

1. **吞吐量指标**
   - 每秒操作数（bogo ops/s）
   - I/O吞吐量（MB/s）
   - 事务处理速率（TPS）

2. **延迟指标**
   - 平均响应时间
   - 95/99百分位延迟
   - 最大延迟

3. **资源利用率**
   - CPU使用率
   - 内存占用
   - 上下文切换次数
   - 进程迁移次数

4. **能耗指标**（如果设备支持）
   - 功耗（瓦特）
   - 每焦耳操作数（energy efficiency）

5. **调度效率**
   - 负载均衡度
   - CPU空闲时间
   - 任务迁移开销

## 故障排除

### 常见问题与解决方案

#### 1. BCC安装失败

**问题**: `bcc/BPF.h: No such file or directory`

**解决方案**:
```bash
# 确保安装了正确的开发包
sudo apt install -y libbpfcc-dev python3-bpfcc

# 检查BCC安装路径
dpkg -L python3-bpfcc | grep -i bpf
```

#### 2. 内核头文件不匹配

**问题**: `linux/kernel.h: No such file or directory`

**解决方案**:
```bash
# 安装匹配的内核头文件
sudo apt install -y linux-headers-$(uname -r)

# 验证安装
ls /lib/modules/$(uname -r)/build
```

#### 3. TensorFlow在ARM64上性能低

**问题**: 训练速度极慢

**解决方案**:
```bash
# 使用轻量级模型
# 在training_config.py中减小模型大小

# 或者使用TensorFlow Lite
pip3 install tflite-runtime

# 使用XNNPACK加速
export TF_ENABLE_XNNPACK=1
```

#### 4. eBPF程序加载失败

**问题**: `Failed to load BPF program`

**解决方案**:
```bash
# 检查内核配置
zcat /proc/config.gz | grep -i bpf

# 确保以下选项启用：
# CONFIG_BPF=y
# CONFIG_BPF_SYSCALL=y
# CONFIG_BPF_JIT=y

# 检查权限
sudo sysctl kernel.unprivileged_bpf_disabled
# 如果为1，设置为0
sudo sysctl kernel.unprivileged_bpf_disabled=0
```

#### 5. 内核模块加载失败

**问题**: `insmod: ERROR: could not insert module`

**解决方案**:
```bash
# 查看详细错误
dmesg | tail -20

# 检查模块签名（如果启用了Secure Boot）
mokutil --sb-state

# 如果需要，签名模块
sudo /usr/src/linux-headers-$(uname -r)/scripts/sign-file \
    sha256 signing_key.priv signing_key.x509 jc_kmod.ko

# 或者禁用Secure Boot（不推荐用于生产环境）
```

#### 6. 数据采集无输出

**问题**: dump_lb.py运行但不生成数据

**解决方案**:
```bash
# 检查eBPF程序是否正确附加
sudo bpftool prog list

# 检查perf事件
sudo perf list | grep sched

# 增加调试输出
# 在dump_lb.py中添加：
# print(f"Events collected: {len(cm_events)}")

# 确保系统有负载
stress-ng --cpu 2 --timeout 60s
```

#### 7. ARM64架构特定问题

**问题**: 浮点运算错误

**解决方案**:
```bash
# 检查CPU是否支持硬件浮点
lscpu | grep -i fp

# 在kmod/Makefile中确保使用正确的标志
# CFLAGS_jc_kmod.o := -mhard-float
```

**问题**: 内存对齐错误

**解决方案**:
```c
// 在C代码中添加对齐属性
struct __attribute__((aligned(8))) my_struct {
    // ...
};
```

### 日志和调试

#### 启用详细日志

```bash
# dump_lb.py日志级别
export PYTHONLOG=DEBUG
python3 dump_lb.py -t debug

# 内核模块调试
sudo dmesg -w  # 实时查看内核日志

# eBPF调试
sudo bpftool prog dump xlated id <prog_id>
```

#### 性能分析

```bash
# 使用perf分析
sudo perf record -g python3 dump_lb.py -t perf_test
sudo perf report

# 分析内核模块性能
sudo perf top -K

# 跟踪系统调用
sudo strace -c python3 dump_lb.py -t trace_test
```

## 参考资源

### 官方文档
- [BCC GitHub仓库](https://github.com/iovisor/bcc)
- [TensorFlow官方文档](https://www.tensorflow.org/install)
- [Linux内核文档 - 调度器](https://www.kernel.org/doc/html/latest/scheduler/)
- [eBPF文档](https://ebpf.io/what-is-ebpf)

### ARM64特定资源
- [ARM开发者文档](https://developer.arm.com/documentation)
- [Linux ARM64移植指南](https://www.kernel.org/doc/html/latest/arm64/)

### 论文和研究
- [Machine Learning for Load Balancing in the Linux Kernel](https://doi.org/10.1145/3409963.3410492)

## 总结

本指南涵盖了在ARM64处理器上部署MLLB系统的完整流程，包括：

1. ✅ **环境准备**: 安装所有必需的依赖项
2. ✅ **参数配置**: 针对ARM64架构优化的参数设置
3. ✅ **数据采集**: 使用eBPF收集调度数据
4. ✅ **模型训练**: 训练机器学习模型
5. ✅ **内核集成**: 将模型部署到内核模块
6. ✅ **性能优化**: ARM64特定的优化技巧
7. ✅ **全面测试**: 多场景性能对比测试

通过遵循本指南，您可以在ARM64平台上成功部署并优化MLLB系统，实现智能的负载均衡调度。

---

**版本**: 1.0  
**最后更新**: 2024  
**维护者**: MLLB项目团队
