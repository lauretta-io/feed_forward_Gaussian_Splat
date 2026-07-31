# Common Types and Frames

This module defines immutable identifiers, nanosecond timestamps, SE(3) transforms, pose
covariance and estimates, sensor calibration/health, and model version types. Import public types
from `ariadne.common`.

```python
import numpy as np
from ariadne.common import FrameId, TransformSE3

camera_to_body = TransformSE3.from_translation_quaternion(
    FrameId("camera_front"),
    FrameId("body"),
    translation_m=[0.1, 0.0, 0.0],
    quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
)
assert np.allclose(camera_to_body.then(camera_to_body.inverse()).matrix, np.eye(4))
```

Run `PYTHONPATH=src python examples/transform_roundtrip.py` for an executable example.
