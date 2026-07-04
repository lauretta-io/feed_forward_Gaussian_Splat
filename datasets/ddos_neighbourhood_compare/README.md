# DDOS Neighbourhood Comparison Data

This scaffold groups the DDOS_Pipeline train data for neighbourhoods 102 and 105 by model, with matching places for inputs and run outputs.

Source Drive folder:
https://drive.google.com/drive/u/1/folders/1xRuZfOD99s78QusuCfxy04DmO0i6zXFV

## Layout

```text
datasets/ddos_neighbourhood_compare/
  drive_manifest.json
  neighbourhood_102/
    source_train/
      image/
      colmap_output/
    resplat/
      input/
      outputs/
    mvsplat/
      input/
      outputs/
    anysplat/
      input/
      outputs/
  neighbourhood_105/
    ...
```

## Model Data Mapping

- ReSplat uses the source `image/` frames plus `colmap_output/` reconstruction. Its Drive outputs are under `resplat/102` and `resplat/105`, with older named copies under `resplat/resplat_102` and `resplat/resplat_105`.
- MVSplat does not consume the raw COLMAP train folder directly in this repo. It expects chunked `.torch` datasets plus an index. The Drive folder contains MVSplat run outputs under `mvsplat/mvsplat_102` and `mvsplat/mvsplat_105`; no prebuilt `.torch` input chunks were observed in the shared folder.
- AnySplat can use the flat `image/` frame folder directly. No AnySplat-specific run outputs were observed in the shared Drive folder, so its `outputs/` folders are placeholders for local runs.

## Local Download

The DDOS subset from the shared folder has been downloaded to:

```text
datasets/DDOS_Pipeline/ddos_pipeline_testing/
```

The comparison folders use symlinks into that downloaded tree:

- `source_train/downloaded_image`
- `source_train/downloaded_colmap_output`
- `resplat/input/images`
- `resplat/input/colmap_output`
- `resplat/outputs/downloaded`
- `mvsplat/outputs/downloaded`
- `mvsplat/outputs/gaussian.ply`
- `anysplat/input/images`

`gdown --folder` could list the shared DDOS folder, but failed to resolve some nested binary file URLs. The local files were downloaded from the `gdown --json` listing with `curl -L` against each Drive file ID.

The broader shared folder also started downloading unrelated AnySplat sample outputs before it was stopped. That partial data is under `datasets/DDOS_Pipeline/AnySplat Outputs/` and is not used by this comparison scaffold.
