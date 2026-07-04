# Setup Notes

This repository targets Python 3.12, PyTorch 2.7.0, and CUDA 12.8. The commands
below create a repo-local conda environment in `.venv` and install the required
runtime packages.

## Environment

```bash
conda create -y -p ./.venv python=3.12
conda activate /path/to/feed_forward_Gaussian_Splat/.venv
```

Install PyTorch and torchvision with the CUDA 12.8 wheels:

```bash
pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

Install the pinned Python dependencies:

```bash
pip install -r requirements.txt
```

Install `gsplat`:

```bash
pip install --no-build-isolation \
  git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3
```

Install `pointops`:

```bash
cd src/model/encoder/pointops
CUDA_HOME=/usr/local/cuda python setup.py install
cd ../../../..
```

If no GPU is visible while building `pointops`, PyTorch may fail to infer a CUDA
architecture and raise `IndexError: list index out of range`. Set the
architecture list explicitly and rebuild:

```bash
cd src/model/encoder/pointops
CUDA_HOME=/usr/local/cuda \
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
python setup.py install
cd ../../../..
```

For RTX 3090 systems, `8.6` is the required architecture. The wider list above
also covers common Ampere, Ada, and Hopper GPUs.

## Manual Validation

Check package consistency:

```bash
pip check
```

Check imports:

```bash
python -c "import torch, torchvision, gsplat, pointops; print('imports ok')"
```

Check the PyTorch CUDA build:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda built:", torch.backends.cuda.is_built())
print("device count:", torch.cuda.device_count())
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY
```

Check the NVIDIA driver:

```bash
nvidia-smi
cat /proc/driver/nvidia/version
```

Check CUDA toolkit selection:

```bash
/usr/local/cuda/bin/nvcc --version
readlink -f /usr/local/cuda
```

Check which CUDA libraries PyTorch loads:

```bash
ldd .venv/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so | \
  grep -E 'cuda|nvidia|cudnn|cublas|cudart'
```

## Known Failure Modes

### PyTorch reports `cuda available: False`

First distinguish a PyTorch package problem from a driver or device problem:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.backends.cuda.is_built())"
nvidia-smi
ls -l /dev/nvidia*
```

If PyTorch is a CUDA build, the NVIDIA driver is loaded, but `/dev/nvidia*`
nodes are missing, userspace cannot communicate with the driver. Symptoms
include:

```text
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
RuntimeError: No CUDA GPUs are available
```

Confirm that the driver sees the GPUs:

```bash
lspci | grep -Ei 'nvidia|vga|3d|display'
lsmod | grep '^nvidia'
find /proc/driver/nvidia/gpus -name information -print -exec sed -n '1,80p' {} \;
cat /proc/devices | grep nvidia
```

If `/dev/nvidia*` is missing but `/proc/driver/nvidia/gpus/*/information`
lists GPUs, recreate the device nodes as root. Example for two GPUs with minors
0 and 1, control major 195, and UVM major 506:

```bash
sudo mknod -m 666 /dev/nvidiactl c 195 255
sudo mknod -m 666 /dev/nvidia0 c 195 0
sudo mknod -m 666 /dev/nvidia1 c 195 1
sudo mknod -m 666 /dev/nvidia-uvm c 506 0
sudo mknod -m 666 /dev/nvidia-uvm-tools c 506 1
sudo mknod -m 666 /dev/nvidia-modeset c 195 254
```

Then re-run:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

If `nvidia-modprobe` is available, prefer it over manual `mknod`:

```bash
sudo nvidia-modprobe -u -c=0
```

Device nodes can disappear again after driver reloads, container restarts, or
system reboots if udev or `nvidia-modprobe` is not configured correctly.

### CUDA version mismatch warnings

The PyTorch wheel includes its own CUDA runtime libraries. A local CUDA toolkit
is still used to compile extensions such as `pointops`.

Warnings like this during extension builds are not automatically fatal:

```text
The detected CUDA version (12.1) has a minor version mismatch with the version
that was used to compile PyTorch (12.8).
```

If builds or imports fail, check:

```bash
readlink -f /usr/local/cuda
/usr/local/cuda/bin/nvcc --version
echo "$LD_LIBRARY_PATH"
```

Avoid putting old CUDA runtime paths, such as CUDA 11.8, ahead of the active
toolkit or PyTorch-provided libraries unless a specific dependency requires it.

### Sandbox or container cannot see GPUs

In restricted shells, `nvidia-smi` and PyTorch may fail even when the host GPU
setup works. Compare results from the normal user terminal and the restricted
environment:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

If only the restricted environment fails, pass through `/dev/nvidia*` devices
and NVIDIA driver libraries to that environment instead of changing Python
packages.

