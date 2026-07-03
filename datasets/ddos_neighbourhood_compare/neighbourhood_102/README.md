# Neighbourhood 102

Populate `source_train/image/` from Drive folder `1QdVUf5wDhoXykCuUZwv1P9ABk4cjCd4r`.
Populate `source_train/colmap_output/` from Drive folder `1zrIcr5HkiToznEv6wCycOB_Ulz6KJuxM`.

- `resplat/input/`: use `source_train/image/` and `source_train/colmap_output/`.
- `mvsplat/input/`: reserved for converted `.torch` chunks; none were observed in the Drive source.
- `anysplat/input/`: use the flat PNGs from `source_train/image/`.
- `*/outputs/`: place local run products here; Drive output locations are recorded in `../drive_manifest.json`.

