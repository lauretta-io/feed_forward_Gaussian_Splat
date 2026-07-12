# Real VIO backends

ARIADNE evaluates OpenVINS and ORB-SLAM3 out of process so their ROS/C++ dependencies do not
enter the Python package. The setup script clones pinned GPLv3 sources into the ignored
`.cache/ariadne/backends` directory and builds them with Docker.

```bash
applications/ariadne/scripts/setup_vio_backends.sh --build openvins
applications/ariadne/scripts/setup_vio_backends.sh --build orbslam3
PYTHONPATH=applications/ariadne/src .venv/bin/python \
  applications/ariadne/scripts/prepare_openvins_d2_config.py \
  --openvins-root .cache/ariadne/backends/openvins_ws/src/open_vins \
  --output .cache/ariadne/backends/openvins_d2_config
```

Run either backend against D2SLAM sequence 1 and log the evaluation to W&B:

```bash
PYTHONPATH=applications/ariadne/src .venv/bin/python \
  applications/ariadne/scripts/run_real_vio.py --backend openvins --sequence 1 --wandb-mode online
PYTHONPATH=applications/ariadne/src .venv/bin/python \
  applications/ariadne/scripts/run_real_vio.py --backend orbslam3 --sequence 1 --wandb-mode online
```

The report records aligned ATE, relative position RMSE, final drift, elapsed time, command, logs,
and trajectory path. Dataset payloads and backend source/build trees remain clone-local. OpenVINS
runs inside its ROS Noetic image; ORB-SLAM3 consumes an exported EuRoC-style stereo/IMU window.
The current production baseline is D2SLAM because its TUM-VI camera/IMU model is supported by both
backends. Both production binaries run inside their pinned build images. MILUV and S3E replay
adapters are available to Python consumers, but require validated
backend-specific calibration before claiming production VIO metrics.
