# Neighbourhood 105

Populate `source_train/image/` from Drive folder `1U2EAt98P8jpB0YPRMH1e27oyYw-SVeU0`.
Populate `source_train/colmap_output/` from Drive folder `1IuKip_-pxCD6uW8WVl-7EsvBjc3GkIO6`.

- `resplat/input/`: use `source_train/image/` and `source_train/colmap_output/`.
- `mvsplat/input/`: reserved for converted `.torch` chunks; none were observed in the Drive source.
- `anysplat/input/`: use the flat PNGs from `source_train/image/`.
- `*/outputs/`: place local run products here; Drive output locations are recorded in `../drive_manifest.json`.

