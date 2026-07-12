import numpy as np

from ariadne.common import FrameId, TransformSE3


def main() -> None:
    transform = TransformSE3.from_translation_quaternion(
        FrameId("camera_front"),
        FrameId("body"),
        [0.1, -0.2, 0.3],
        [0.0, 0.0, 0.0, 2.0],
    )
    round_trip = transform.then(transform.inverse())
    print(f"round-trip error: {np.max(np.abs(round_trip.matrix - np.eye(4))):.3e}")


if __name__ == "__main__":
    main()
