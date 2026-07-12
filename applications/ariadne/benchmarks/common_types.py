from time import perf_counter_ns

from ariadne.common import FrameId, TransformSE3


def main(iterations: int = 10_000) -> None:
    transform = TransformSE3.from_translation_quaternion(
        FrameId("camera_front"), FrameId("body"), [0.1, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]
    )
    inverse = transform.inverse()
    start = perf_counter_ns()
    for _ in range(iterations):
        transform.then(inverse)
    elapsed_ns = perf_counter_ns() - start
    print(f"transform round trip: {elapsed_ns / iterations:.1f} ns/op")


if __name__ == "__main__":
    main()
