# Coordinate Frames

`TransformSE3(source, destination, matrix)` maps homogeneous coordinates from `source` into
`destination`. Composition is explicit: `first.then(second)` requires
`first.destination == second.source` and returns `second.matrix @ first.matrix`.

Allowed frame names are `camera_<id>`, `imu`, `body`, `local_<wingman_id>`, `global`, and
`object_<uuid>`. Quaternions use `(x, y, z, w)`, translations use meters, and timestamps use
integer monotonic nanoseconds. The ReSplat/OpenCV adapter treats camera-to-world matrices as
camera-to-destination transforms and validates both frame names.
