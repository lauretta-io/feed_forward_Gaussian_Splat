#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
from typing import Any

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
FRAME_TESTS = ROOT / "outputs" / "frame_tests"
DEFAULT_OUTPUT = ROOT / "reports" / "model_capability_dashboard.html"
DEFAULT_NOTES_OUTPUT = ROOT / "reports" / "model_capability_notes.html"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def image_data_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def first_png(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def collect_runs() -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}

    resplat_5 = read_json(FRAME_TESTS / "resplat_colmap_5f" / "aggregate_metrics.json")
    resplat_10 = read_json(FRAME_TESTS / "resplat_colmap_10f" / "aggregate_metrics.json")
    mvsplat_5 = read_json(FRAME_TESTS / "mvsplat_re10k_5f" / "metrics" / "scores_all_avg.json")
    mvsplat_10 = read_json(FRAME_TESTS / "mvsplat_re10k_10f" / "metrics" / "scores_all_avg.json")
    anysplat_5 = read_json(FRAME_TESTS / "anysplat_5f" / "manifest.json")
    anysplat_10 = read_json(FRAME_TESTS / "anysplat_10f" / "manifest.json")

    scene_dirs_5 = sorted((FRAME_TESTS / "resplat_colmap_5f").glob("*/"))
    scene_dirs_10 = sorted((FRAME_TESTS / "resplat_colmap_10f").glob("*/"))
    render_5 = first_png([p for d in scene_dirs_5 for p in sorted((d / "rendered").glob("*.png"))])
    render_10 = first_png([p for d in scene_dirs_10 for p in sorted((d / "rendered").glob("*.png"))])
    input_5 = first_png([p for d in scene_dirs_5 for p in sorted((d / "input").glob("*.png"))])
    input_10 = first_png([p for d in scene_dirs_10 for p in sorted((d / "input").glob("*.png"))])

    runs["resplat"] = {
        "title": "ReSplat COLMAP",
        "family": "feed-forward Gaussian splatting",
        "inputs": "COLMAP scene, sparse cameras, images_4/images_8 frames",
        "outputs": "Rendered target views, Gaussian scene representation, PSNR/SSIM/LPIPS",
        "capabilities": ["COLMAP ingestion", "multi-view context selection", "target-view rendering", "image metrics"],
        "metrics": {
            "5f": {
                "frame_count": 5,
                "psnr": resplat_5.get("mean_psnr"),
                "ssim": resplat_5.get("mean_ssim"),
                "lpips": resplat_5.get("mean_lpips"),
            },
            "10f": {
                "frame_count": 10,
                "psnr": resplat_10.get("mean_psnr"),
                "ssim": resplat_10.get("mean_ssim"),
                "lpips": resplat_10.get("mean_lpips"),
            },
        },
        "artifacts": [
            {"label": "5f rendered target", "path": render_5},
            {"label": "10f rendered target", "path": render_10},
            {"label": "5f context input", "path": input_5},
            {"label": "10f context input", "path": input_10},
        ],
    }

    runs["mvsplat"] = {
        "title": "MVSplat RE10K",
        "family": "cost-volume Gaussian splatting",
        "inputs": "RE10K torch chunks and sampled context/target views",
        "outputs": "Rendered target views, scores JSON, benchmark JSON, W&B result artifact",
        "capabilities": ["chunk dataset evaluation", "cost-volume depth candidates", "target-view rendering", "image metrics"],
        "metrics": {
            "5f": {
                "frame_count": mvsplat_5.get("frame_count"),
                "psnr": mvsplat_5.get("psnr"),
                "ssim": mvsplat_5.get("ssim"),
                "lpips": mvsplat_5.get("lpips"),
            },
            "10f": {
                "frame_count": mvsplat_10.get("frame_count"),
                "psnr": mvsplat_10.get("psnr"),
                "ssim": mvsplat_10.get("ssim"),
                "lpips": mvsplat_10.get("lpips"),
            },
        },
        "artifacts": [
            {"label": "5f scores", "path": FRAME_TESTS / "mvsplat_re10k_5f" / "metrics" / "scores_all_avg.json"},
            {"label": "10f scores", "path": FRAME_TESTS / "mvsplat_re10k_10f" / "metrics" / "scores_all_avg.json"},
        ],
    }

    runs["anysplat"] = {
        "title": "AnySplat Image Folder",
        "family": "image-folder Gaussian reconstruction",
        "inputs": "Flat directory of 5 or 10 source images",
        "outputs": "Predicted camera poses, Gaussian statistics, manifest artifact",
        "capabilities": ["unordered image-folder inference", "pose prediction", "Gaussian attribute export", "source preview logging"],
        "metrics": {
            "5f": {
                "frame_count": len(anysplat_5.get("source_images", [])) or None,
                **anysplat_5.get("evaluation", {}),
            },
            "10f": {
                "frame_count": len(anysplat_10.get("source_images", [])) or None,
                **anysplat_10.get("evaluation", {}),
            },
        },
        "artifacts": [
            {"label": "5f manifest", "path": FRAME_TESTS / "anysplat_5f" / "manifest.json"},
            {"label": "10f manifest", "path": FRAME_TESTS / "anysplat_10f" / "manifest.json"},
            {"label": "5f first source", "path": Path(anysplat_5.get("source_images", [""])[0]) if anysplat_5.get("source_images") else None},
            {"label": "10f first source", "path": Path(anysplat_10.get("source_images", [""])[0]) if anysplat_10.get("source_images") else None},
        ],
    }
    return runs


def collect_wandb_runs() -> dict[str, str]:
    out: dict[str, str] = {}
    roots = [
        ROOT / "wandb",
        FRAME_TESTS / "mvsplat_re10k_5f" / "wandb",
        FRAME_TESTS / "mvsplat_re10k_10f" / "wandb",
    ]
    for root in roots:
        if not root.exists():
            continue
        for metadata in sorted(root.glob("run-*/files/wandb-metadata.json")):
            try:
                data = json.loads(metadata.read_text())
            except json.JSONDecodeError:
                continue
            args = [str(arg) for arg in data.get("args", [])]
            run_id = metadata.parts[-3].split("-")[-1]
            joined = " ".join(args)
            for key in (
                "resplat-colmap-5f",
                "resplat-colmap-10f",
                "mvsplat-re10k-5f",
                "mvsplat-re10k-10f",
                "anysplat-5f",
                "anysplat-10f",
            ):
                if key in joined:
                    out[key] = run_id
    return out


def page_nav(active: str) -> str:
    links = [
        ("dashboard", "Dashboard", "model_capability_dashboard.html"),
        ("notes", "Notes", "model_capability_notes.html"),
    ]
    return "<nav>" + "".join(
        f'<a class="{"active" if key == active else ""}" href="{href}">{label}</a>'
        for key, label, href in links
    ) + "</nav>"


def shared_css() -> str:
    return """
    :root {
      color-scheme: light;
      --ink: #172018;
      --muted: #5b665e;
      --line: #d8ded7;
      --panel: #ffffff;
      --soft: #f5f7f4;
      --green: #2f6f4e;
      --blue: #315f8d;
      --rose: #9a4253;
      --gold: #9f761a;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--soft); }
    header { padding: 28px 36px 20px; border-bottom: 1px solid var(--line); background: #fff; }
    nav { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 18px; }
    nav a { color: var(--ink); text-decoration: none; border: 1px solid var(--line); border-radius: 999px; padding: 7px 12px; font-size: 13px; background: #fafbf9; }
    nav a.active { color: #fff; border-color: var(--green); background: var(--green); }
    h1 { margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }
    header p { max-width: 980px; margin: 0; color: var(--muted); line-height: 1.45; }
    main { padding: 24px 36px 40px; display: grid; gap: 24px; }
    h2 { margin: 0 0 12px; font-size: 18px; letter-spacing: 0; }
    h3 { margin: 0; font-size: 16px; letter-spacing: 0; }
    h4 { margin: 14px 0 8px; font-size: 13px; color: var(--muted); }
    p { line-height: 1.55; }
    .band { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 18px; }
    .models { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }
    .model-card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; display: grid; gap: 14px; min-height: 320px; }
    .model-card p { margin: 4px 0 0; color: var(--muted); }
    dl { margin: 0; display: grid; gap: 8px; }
    dt { font-size: 12px; color: var(--muted); text-transform: uppercase; }
    dd { margin: 0; line-height: 1.35; }
    ul { margin: 0; padding-left: 18px; line-height: 1.55; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; color: #28352c; }
    .run-ids { display: flex; flex-wrap: wrap; gap: 8px; color: var(--muted); font-size: 12px; }
    .run-ids span { border: 1px solid var(--line); border-radius: 999px; padding: 4px 8px; background: #fafbf9; }
    .capability-graph { width: 100%; max-height: 620px; }
    .edge { stroke: #aeb9b0; stroke-width: 1.4; marker-end: url(#arrow); }
    marker path { fill: #aeb9b0; }
    .node circle { stroke: #fff; stroke-width: 3; }
    .node text { text-anchor: middle; font-size: 12px; fill: var(--ink); paint-order: stroke; stroke: #fff; stroke-width: 4px; }
    .node-source circle { fill: var(--gold); }
    .node-model circle { fill: var(--green); }
    .node-capability circle { fill: var(--blue); }
    .node-output circle { fill: var(--rose); }
    .metric-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }
    .metric-card { border: 1px solid var(--line); border-radius: 8px; padding: 16px; background: #fff; }
    .bar-row { display: grid; grid-template-columns: 95px 1fr 74px; gap: 10px; align-items: center; min-height: 28px; font-size: 13px; }
    .bar-track { height: 9px; border-radius: 999px; background: #e5ebe5; overflow: hidden; }
    .bar-track i { display: block; height: 100%; background: var(--green); }
    .bar-row strong { text-align: right; font-size: 12px; }
    .gallery { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
    figure { margin: 0; background: #fff; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
    figure img { width: 100%; display: block; aspect-ratio: 16 / 9; object-fit: cover; background: #e8ece7; }
    figcaption { padding: 10px; color: var(--muted); line-height: 1.35; font-size: 12px; }
    .file-icon { height: 140px; display: grid; place-items: center; background: #e9eee9; color: var(--green); font-weight: 700; letter-spacing: 0.08em; }
    .doc-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
    .doc-card { background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 16px; }
    .doc-card p { margin: 8px 0 0; color: var(--muted); }
    .doc-card ul { margin-top: 10px; }
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; background: #fff; }
    th, td { border-bottom: 1px solid var(--line); padding: 10px; text-align: left; vertical-align: top; }
    th { color: var(--muted); font-size: 12px; text-transform: uppercase; }
    .callout { border-left: 4px solid var(--green); padding: 12px 14px; background: #f8faf7; color: var(--muted); }
    @media (max-width: 1000px) { .models, .metric-grid, .gallery, .doc-grid { grid-template-columns: 1fr 1fr; } }
    @media (max-width: 700px) { header, main { padding-left: 18px; padding-right: 18px; } .models, .metric-grid, .gallery, .doc-grid { grid-template-columns: 1fr; } .bar-row { grid-template-columns: 86px 1fr 62px; } }
  """


def capability_graph(runs: dict[str, dict[str, Any]]) -> str:
    graph = nx.DiGraph()
    graph.add_node("Frame Tests", kind="source")
    graph.add_node("W&B Artifacts", kind="output")
    graph.add_node("Weave Trace", kind="output")

    for key, run in runs.items():
        model = run["title"]
        graph.add_node(model, kind="model")
        graph.add_edge("Frame Tests", model)
        graph.add_edge(model, "W&B Artifacts")
        graph.add_edge(model, "Weave Trace")
        for cap in run["capabilities"]:
            graph.add_node(cap, kind="capability")
            graph.add_edge(model, cap)

    positions = nx.spring_layout(graph, seed=11, k=0.9)
    min_x = min(x for x, _ in positions.values())
    max_x = max(x for x, _ in positions.values())
    min_y = min(y for _, y in positions.values())
    max_y = max(y for _, y in positions.values())

    def scale(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        sx = 70 + (x - min_x) / max(max_x - min_x, 1e-6) * 860
        sy = 55 + (y - min_y) / max(max_y - min_y, 1e-6) * 490
        return sx, sy

    classes = {
        "source": "node-source",
        "model": "node-model",
        "capability": "node-capability",
        "output": "node-output",
    }
    edges = []
    for src, dst in graph.edges:
        x1, y1 = scale(positions[src])
        x2, y2 = scale(positions[dst])
        edges.append(
            f'<line class="edge" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" />'
        )

    nodes = []
    for node, attrs in graph.nodes(data=True):
        x, y = scale(positions[node])
        cls = classes[attrs["kind"]]
        label = html.escape(node)
        nodes.append(
            f'<g class="node {cls}" transform="translate({x:.1f} {y:.1f})">'
            f'<circle r="22"></circle><text y="38">{label}</text></g>'
        )

    return (
        '<svg class="capability-graph" viewBox="0 0 1000 610" role="img" '
        'aria-label="NetworkX capability graph">'
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z"></path></marker></defs>'
        + "".join(edges)
        + "".join(nodes)
        + "</svg>"
    )


def metric_bar(label: str, value: float | int | None, max_value: float, suffix: str = "") -> str:
    if value is None:
        width = 0
        shown = "n/a"
    else:
        width = max(0, min(100, float(value) / max_value * 100))
        shown = f"{value:.4g}{suffix}" if isinstance(value, float) else f"{value}{suffix}"
    return (
        f'<div class="bar-row"><span>{html.escape(label)}</span>'
        f'<div class="bar-track"><i style="width:{width:.2f}%"></i></div>'
        f'<strong>{shown}</strong></div>'
    )


def metric_panel(runs: dict[str, dict[str, Any]]) -> str:
    panels = []
    for key, run in runs.items():
        rows = []
        for frame_key, metrics in run["metrics"].items():
            rows.append(f'<h4>{frame_key}</h4>')
            if key in ("resplat", "mvsplat"):
                rows.append(metric_bar("PSNR", metrics.get("psnr"), 40))
                rows.append(metric_bar("SSIM", metrics.get("ssim"), 1))
                lpips = metrics.get("lpips")
                inv_lpips = None if lpips is None else max(0, 1 - float(lpips))
                rows.append(metric_bar("1 - LPIPS", inv_lpips, 1))
            else:
                rows.append(metric_bar("Gaussians", metrics.get("num_gaussians"), 1_200_000))
                rows.append(metric_bar("Opacity mean", metrics.get("opacity_mean"), 1))
                rows.append(metric_bar("Scale mean x1000", None if metrics.get("scale_mean") is None else metrics["scale_mean"] * 1000, 1))
        panels.append(
            f'<section class="metric-card"><h3>{html.escape(run["title"])}</h3>'
            + "".join(rows)
            + "</section>"
        )
    return '<div class="metric-grid">' + "".join(panels) + "</div>"


def model_cards(runs: dict[str, dict[str, Any]], wandb_runs: dict[str, str]) -> str:
    cards = []
    slug = {"resplat": "resplat-colmap", "mvsplat": "mvsplat-re10k", "anysplat": "anysplat"}
    for key, run in runs.items():
        caps = "".join(f"<li>{html.escape(cap)}</li>" for cap in run["capabilities"])
        run_links = []
        for frame in ("5f", "10f"):
            run_id = wandb_runs.get(f"{slug[key]}-{frame}", "not found")
            run_links.append(f"<span>{frame}: <code>{html.escape(run_id)}</code></span>")
        cards.append(
            f'<article class="model-card"><div><h3>{html.escape(run["title"])}</h3>'
            f'<p>{html.escape(run["family"])}</p></div>'
            f'<dl><dt>Inputs</dt><dd>{html.escape(run["inputs"])}</dd>'
            f'<dt>Outputs</dt><dd>{html.escape(run["outputs"])}</dd></dl>'
            f'<ul>{caps}</ul><div class="run-ids">{"".join(run_links)}</div></article>'
        )
    return '<section class="models">' + "".join(cards) + "</section>"


def gallery(runs: dict[str, dict[str, Any]]) -> str:
    items = []
    for run in runs.values():
        for artifact in run["artifacts"]:
            path = artifact.get("path")
            if path is None:
                continue
            path = Path(path)
            uri = image_data_uri(path) if path.suffix.lower() == ".png" else None
            if uri:
                media = f'<img src="{uri}" alt="{html.escape(artifact["label"])}">'
            else:
                media = '<div class="file-icon">JSON</div>' if path.suffix == ".json" else '<div class="file-icon">PT</div>'
            items.append(
                f'<figure>{media}<figcaption>{html.escape(artifact["label"])}<br><code>{html.escape(rel(path))}</code></figcaption></figure>'
            )
    return '<section class="gallery">' + "".join(items) + "</section>"


def model_requirements_table(runs: dict[str, dict[str, Any]], wandb_runs: dict[str, str]) -> str:
    slugs = {"resplat": "resplat-colmap", "mvsplat": "mvsplat-re10k", "anysplat": "anysplat"}
    rows = []
    for key, run in runs.items():
        links = []
        for frame in ("5f", "10f"):
            run_id = wandb_runs.get(f"{slugs[key]}-{frame}", "not found")
            links.append(f"{frame}: {html.escape(run_id)}")
        rows.append(
            "<tr>"
            f"<td><strong>{html.escape(run['title'])}</strong><br>{html.escape(run['family'])}</td>"
            f"<td>{html.escape(run['inputs'])}</td>"
            f"<td>{html.escape(run['outputs'])}</td>"
            f"<td><code>{'</code><br><code>'.join(links)}</code></td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Model</th><th>Data Requirements</th><th>Generated Evidence</th><th>W&B Run IDs</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def notes_cards() -> str:
    cards = [
        (
            "How This Is Run Locally",
            "The dashboard is a static local report. It does not need a web server, database, or live API calls after generation.",
            [
                "Open reports/model_capability_dashboard.html or reports/model_capability_notes.html directly in a browser.",
                "Regenerate both pages with python scripts/build_model_capability_page.py.",
                "The dashboard embeds local PNG previews as data URIs and links JSON/PT artifacts by path.",
            ],
        ),
        (
            "W&B And Weave Linkage",
            "Runs are logged to the resplat-tests W&B project, while Weave is initialized with galvin/gaussiansplat test.",
            [
                "Run metadata includes tags, frame counts, result metrics, and saved artifacts.",
                "The local API key is loaded from .env and is intentionally not embedded into generated reports.",
                "Local wandb/ directories are ignored by git and only run IDs are shown in the HTML.",
            ],
        ),
        (
            "Data Requirements",
            "The three model paths need different input formats, so the frame tests use small adapters around shared visual content where possible.",
            [
                "ReSplat expects a COLMAP scene with images_4 or images_8 plus sparse/0 camera data.",
                "MVSplat expects RealEstate10K-style .torch chunks with index.json.",
                "AnySplat expects a flat folder of PNG, JPG, or JPEG source images.",
            ],
        ),
        (
            "Weights And Runtime",
            "All model checks require GPU-oriented dependencies and local checkpoints or downloaded model weights.",
            [
                "ReSplat uses the DL3DV checkpoint plus the gmdepth dependency link.",
                "MVSplat uses checkpoints/re10k.ckpt and the legacy CUDA Gaussian rasterizer.",
                "AnySplat pulls lhjiang/anysplat and facebook/VGGT-1B from Hugging Face on first run.",
            ],
        ),
        (
            "Limitations",
            "These are smoke-scale 5-frame and 10-frame checks, not a full benchmark or scientific comparison.",
            [
                "ReSplat and AnySplat can share selected DL3DV frames, but MVSplat currently uses the RE10K two-scene subset.",
                "AnySplat reports reconstruction statistics instead of PSNR/SSIM/LPIPS because no target-view ground truth path is wired here.",
                "The static HTML reflects the most recent local outputs at generation time; rerun the harness after code or data changes.",
            ],
        ),
        (
            "Troubleshooting",
            "Most failures are caused by missing checkpoints, missing CUDA extensions, or dataset shape mismatches.",
            [
                "Load .env before online W&B runs: set -a; source .env; set +a.",
                "Use dataset.highres=true for the downloaded 720p RE10K subset.",
                "Review SMOKE_TESTS.md for the exact dependency and download commands used in this workspace.",
            ],
        ),
    ]
    out = []
    for title, body, items in cards:
        bullets = "".join(f"<li>{html.escape(item)}</li>" for item in items)
        out.append(
            f'<article class="doc-card"><h3>{html.escape(title)}</h3>'
            f"<p>{html.escape(body)}</p><ul>{bullets}</ul></article>"
        )
    return '<section class="doc-grid">' + "".join(out) + "</section>"


def architecture_cards() -> str:
    cards = [
        (
            "ReSplat COLMAP Architecture",
            [
                ("Input contract", "COLMAP images and sparse cameras are converted into the shared batched context/target format used by the model wrapper."),
                ("Depth and matching", "EncoderReSplat runs MultiViewUniMatch with log-depth candidates, camera-aware matching, and raw mono/CNN/multi-view features."),
                ("Gaussian prediction", "Image, latent depth, match probability, and fused features are projected into latent points. A KNN PlainPointTransformer predicts scale, rotation, spherical harmonics, offsets, and opacity."),
                ("Refinement path", "When enabled, recurrent refinement renders the current Gaussian set, extracts render-error features, applies multi-view attention, and predicts per-Gaussian updates."),
                ("Rendering", "The default decoder uses the gsplat CUDA splatting path; the COLMAP script can also select the OpenSplat CPU decoder for smoke rendering."),
            ],
        ),
        (
            "MVSplat RE10K Architecture",
            [
                ("Input contract", "RealEstate10K-style .torch chunks with index.json are loaded through the Hydra test pipeline and dataset shims."),
                ("Feature backbone", "EncoderCostVolume uses a BackboneMultiview UniMatch-style CNN/transformer stack with optional epipolar transformer support."),
                ("Depth volume", "DepthPredictorMultiView builds a multi-view cost volume over 32 depth candidates, refines it with U-Net blocks, and predicts depths, densities, and raw Gaussian attributes."),
                ("Gaussian conversion", "A GaussianAdapter unprojects per-pixel depth samples with camera intrinsics/extrinsics, applies learned pixel offsets, maps densities to opacity, and packs the scene into the shared Gaussians dataclass."),
                ("Rendering", "The experiment uses model/decoder=mvsplat_splatting_cuda, backed by the legacy diff_gaussian_rasterization CUDA extension."),
            ],
        ),
        (
            "AnySplat Image-Folder Architecture",
            [
                ("Input contract", "A flat image folder is normalized into an image batch; no COLMAP cameras or chunk metadata are required."),
                ("Geometry backbone", "EncoderAnySplat loads VGGT-1B components, using the aggregator plus camera and depth or point heads to predict poses, intrinsics, depth/points, and confidence."),
                ("Gaussian head", "A VGGT_DPT_GS_Head fuses transformer tokens with dense geometry to produce opacity and Gaussian feature channels."),
                ("Gaussian conversion", "The integration optionally filters or voxelizes confident points, maps densities to opacities, and uses the AnySplat GaussianAdapter to produce 3D Gaussian attributes."),
                ("Outputs", "The wired test logs predicted poses, manifest metadata, Gaussian counts, opacity/scale statistics, source previews, and artifacts to W&B."),
            ],
        ),
    ]
    out = []
    for title, rows in cards:
        terms = "".join(
            f"<dt>{html.escape(term)}</dt><dd>{html.escape(body)}</dd>"
            for term, body in rows
        )
        out.append(f'<article class="doc-card"><h3>{html.escape(title)}</h3><dl>{terms}</dl></article>')
    return '<section class="doc-grid">' + "".join(out) + "</section>"


def write_dashboard_page(output_path: Path, runs: dict[str, dict[str, Any]], wandb_runs: dict[str, str]) -> None:
    graph_svg = capability_graph(runs)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Gaussian Splat Model Capability Dashboard</title>
  <style>
{shared_css()}
  </style>
</head>
<body>
  <header>
    {page_nav("dashboard")}
    <h1>Gaussian Splat Model Capability Dashboard</h1>
    <p>Static report generated from local 5-frame and 10-frame test outputs. NetworkX lays out the model capability graph; W&B run IDs and result artifacts come from the frame-test runs.</p>
  </header>
  <main>
    <section>
      {model_cards(runs, wandb_runs)}
    </section>
    <section class="band">
      <h2>Capability Graph</h2>
      {graph_svg}
    </section>
    <section class="band">
      <h2>Evaluation Graphs</h2>
      {metric_panel(runs)}
    </section>
    <section class="band">
      <h2>Outputs And Artifacts</h2>
      {gallery(runs)}
    </section>
  </main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")


def write_notes_page(output_path: Path, runs: dict[str, dict[str, Any]], wandb_runs: dict[str, str]) -> None:
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Gaussian Splat Dashboard Notes</title>
  <style>
{shared_css()}
  </style>
</head>
<body>
  <header>
    {page_nav("notes")}
    <h1>Dashboard Notes And Data Requirements</h1>
    <p>Local documentation for how the Gaussian splat dashboard is produced, what each model needs, what the 5-frame and 10-frame tests prove, and what the current limitations are.</p>
  </header>
  <main>
    <section class="band">
      <h2>Operating Notes</h2>
      {notes_cards()}
    </section>
    <section class="band">
      <h2>Model Inputs, Outputs, And W&B Runs</h2>
      {model_requirements_table(runs, wandb_runs)}
    </section>
    <section class="band">
      <h2>Technical Architecture</h2>
      {architecture_cards()}
    </section>
    <section class="band">
      <h2>Reproduce The Evidence</h2>
      <div class="callout">
        <p>Load local secrets with <code>set -a; source .env; set +a</code>, then run <code>scripts/run_wandb_frame_tests.sh</code>. Rebuild the static pages afterward with <code>python scripts/build_model_capability_page.py</code>.</p>
        <p>Generated evidence is read from <code>outputs/frame_tests</code>. The report files are written to <code>reports/model_capability_dashboard.html</code> and <code>reports/model_capability_notes.html</code>.</p>
      </div>
    </section>
    <section class="band">
      <h2>Local Documentation Sources</h2>
      <ul>
        <li><code>SMOKE_TESTS.md</code> has the verified commands, W&B behavior, checkpoint notes, and smoke-test outcomes.</li>
        <li><code>DATASETS.md</code> describes RealEstate10K, DL3DV, ACID, and chunk-format expectations.</li>
        <li><code>scripts/run_wandb_frame_tests.sh</code> is the repeatable 5-frame and 10-frame test harness.</li>
        <li><code>scripts/build_model_capability_page.py</code> generates this static dashboard and notes page using NetworkX.</li>
      </ul>
    </section>
  </main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")


def write_pages(output_path: Path, notes_path: Path | None = None) -> tuple[Path, Path]:
    resolved_output = output_path.resolve()
    resolved_notes = (notes_path or resolved_output.with_name(DEFAULT_NOTES_OUTPUT.name)).resolve()
    runs = collect_runs()
    wandb_runs = collect_wandb_runs()
    write_dashboard_page(resolved_output, runs, wandb_runs)
    write_notes_page(resolved_notes, runs, wandb_runs)
    return resolved_output, resolved_notes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--notes-output", type=Path, default=None)
    args = parser.parse_args()
    dashboard, notes = write_pages(args.output, args.notes_output)
    print(dashboard)
    print(notes)


if __name__ == "__main__":
    main()
