#!/bin/bash
# Optimized Makefile for ARM64 architecture
# This script generates an optimized Makefile for kernel module compilation

KMOD_DIR="/home/runner/work/MLLB/MLLB/kmod"
MAKEFILE_PATH="$KMOD_DIR/Makefile.arm64"

echo "Generating optimized ARM64 Makefile..."

# Detect ARM64 processor type
CPU_MODEL=$(lscpu | grep "Model name" | cut -d: -f2 | xargs)
echo "Detected CPU: $CPU_MODEL"

# Determine optimal tuning flags
TUNE_FLAG="-mtune=native"
if echo "$CPU_MODEL" | grep -iq "cortex-a53"; then
    TUNE_FLAG="-mtune=cortex-a53"
elif echo "$CPU_MODEL" | grep -iq "cortex-a72"; then
    TUNE_FLAG="-mtune=cortex-a72"
elif echo "$CPU_MODEL" | grep -iq "cortex-a76"; then
    TUNE_FLAG="-mtune=cortex-a76"
fi

echo "Selected tuning: $TUNE_FLAG"

# Create optimized Makefile
cat > "$MAKEFILE_PATH" << EOF
# ARM64 Optimized Makefile for MLLB Kernel Module
# Generated for: $CPU_MODEL
# Date: $(date)

obj-m += jc_kmod.o

# ARM64-specific compilation flags
# -mhard-float: Use hardware floating-point
# -O3: Aggressive optimization
# -march=armv8-a: Target ARMv8 architecture
# $TUNE_FLAG: CPU-specific tuning
CFLAGS_jc_kmod.o := -mhard-float -O3 -march=armv8-a $TUNE_FLAG

# Enable additional optimizations
CFLAGS_jc_kmod.o += -ffast-math -funroll-loops -ftree-vectorize

all:
	make -C /lib/modules/\$(shell uname -r)/build M=\$(PWD) modules

clean:
	make -C /lib/modules/\$(shell uname -r)/build M=\$(PWD) clean

test:
	dmesg -C
	insmod jc_kmod.ko
	sleep 1
	rmmod jc_kmod
	dmesg | tail -20

install:
	mkdir -p /lib/modules/\$(shell uname -r)/kernel/drivers/misc/
	cp jc_kmod.ko /lib/modules/\$(shell uname -r)/kernel/drivers/misc/
	depmod -a
	@echo "Module installed. Add 'jc_kmod' to /etc/modules-load.d/mllb.conf to load at boot"

uninstall:
	rm -f /lib/modules/\$(shell uname -r)/kernel/drivers/misc/jc_kmod.ko
	depmod -a
	@echo "Module uninstalled"

.PHONY: all clean test install uninstall
EOF

echo "✓ Optimized Makefile created at: $MAKEFILE_PATH"
echo ""
echo "To use this optimized Makefile:"
echo "  cd kmod"
echo "  make -f Makefile.arm64"
echo ""
echo "Or to replace the default Makefile:"
echo "  cd kmod"
echo "  cp Makefile Makefile.original"
echo "  cp Makefile.arm64 Makefile"
