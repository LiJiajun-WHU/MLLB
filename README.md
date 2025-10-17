### Source code of paper [*Machine Learning for Load Balancing in the Linux Kernel*](https://doi.org/10.1145/3409963.3410492)

## Prerequisites

- [BCC](https://github.com/iovisor/bcc)
- [Tensorflow](https://www.tensorflow.org/)

## ARM64 Deployment

For complete ARM64 processor deployment instructions, see:
- **[ARM64 Deployment Guide (中文)](ARM64_DEPLOYMENT.md)** - 完整的ARM64部署指南
- **[ARM64 Deployment Guide (English)](ARM64_DEPLOYMENT_EN.md)** - Complete ARM64 deployment guide
- **[ARM64 Helper Scripts](arm64_scripts/)** - Automated deployment scripts

### Quick Start on ARM64

```bash
# 1. Setup environment (first time only)
cd arm64_scripts
sudo ./setup_arm64_environment.sh

# 2. Verify installation
./verify_installation.sh

# 3. Quick deployment
./quick_deploy.sh

# 4. Run benchmarks
./run_benchmarks.sh
```

## Usage

### Dump load balance data:
``` bash
sudo ./dump_lb.py -t tag --old
```
> use `--old` with original kernel without test flag


### Automated training and evaluation:
```bash 
cd training
./automate.py -t tag1 tag2 tag3... -o model_name
```

Preprocessing: `training/prep.py`

Training: `training/keras_lb.py`
