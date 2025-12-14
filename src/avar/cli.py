# src/avar/cli.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from loguru import logger

# Scripts that are lightweight and should be safe to import at module load
from avar.scripts.build_master_foul_dataset import build_master
from avar.scripts.split_foul_dataset import run as split_foul_dataset_run
from avar.scripts.train_foul_model_splits import run as train_foul_model_splits_run
from avar.scripts.batch_detect_json import batch_detect


try:
    from SoccerNet.Downloader import SoccerNetDownloader
except Exception:  
    SoccerNetDownloader = None


# ---------------------------------------------------------------------
# download: wrapper around SoccerNetDownloader
# ---------------------------------------------------------------------
def cmd_download(args: argparse.Namespace) -> None:
    if SoccerNetDownloader is None:
        raise RuntimeError("SoccerNet package is not installed. Run: pip install SoccerNet")

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    downloader = SoccerNetDownloader(LocalDirectory=str(local_dir))
    if args.password:
        downloader.password = args.password

    splits = None
    if args.splits:
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    if args.files:
        files = [f.strip() for f in args.files.split(",") if f.strip()]
        logger.info(f"Downloading SoccerNet games: files={files}, split={splits}, local_dir={local_dir}")
        downloader.downloadGames(files=files, split=splits)

    if args.task:
        logger.info(f"Downloading SoccerNet task={args.task}, split={splits}, version={args.version}, local_dir={local_dir}")
        kwargs = {"task": args.task, "split": splits}
        if args.version:
            kwargs["version"] = args.version
        downloader.downloadDataTask(**kwargs)


# ---------------------------------------------------------------------
# track: detection + simple tracking (JSON output)
# ---------------------------------------------------------------------
def cmd_track(args: argparse.Namespace) -> None:
    # Lazy imports to avoid breaking the whole CLI if a module is missing
    from avar.io.video_io import iter_video_frames
    from avar.detection.detector_yolov8 import PlayerDetector
    from avar.tracking.simple_tracker import MultiObjectTracker

    video_path = Path(args.video)
    out_json = Path(args.out)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Tracking players in {video_path}")

    detector = PlayerDetector(weights_path=args.weights, device=args.device)
    tracker = MultiObjectTracker(iou_threshold=0.3, max_age=30)

    batch_frames = []
    batch_ids = []
    all_tracks = []

    for fid, frame in iter_video_frames(
        str(video_path),
        step=args.step,
        target_size=(args.width, args.height),
    ):
        batch_frames.append(frame)
        batch_ids.append(fid)

        if len(batch_frames) == args.batch_size:
            detections_list = detector.detect_batch(batch_frames)
            for dets, frame_idx in zip(detections_list, batch_ids):
                tracks = tracker.update(dets)
                for tr in tracks:
                    all_tracks.append(
                        {
                            "frame": int(frame_idx),
                            "track_id": int(tr["track_id"]),
                            "bbox": tr["bbox"],
                            "score": float(tr["score"]),
                            "class_id": int(tr["class_id"]),
                        }
                    )
            batch_frames.clear()
            batch_ids.clear()

    if batch_frames:
        detections_list = detector.detect_batch(batch_frames)
        for dets, frame_idx in zip(detections_list, batch_ids):
            tracks = tracker.update(dets)
            for tr in tracks:
                all_tracks.append(
                    {
                        "frame": int(frame_idx),
                        "track_id": int(tr["track_id"]),
                        "bbox": tr["bbox"],
                        "score": float(tr["score"]),
                        "class_id": int(tr["class_id"]),
                    }
                )

    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"video": str(video_path), "tracks": all_tracks}, f)

    logger.info(f"Saved tracks JSON to {out_json}")


# ---------------------------------------------------------------------
# project-2d: project tracks to pitch coordinates
# ---------------------------------------------------------------------
def cmd_project_2d(args: argparse.Namespace) -> None:
    from avar.projection.project_tracks_2d import project_tracks_to_2d

    tracks = Path(args.tracks)
    calib = Path(args.calib)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Projecting {tracks} to 2D pitch using {calib}")
    project_tracks_to_2d(tracks, calib, out)
    logger.info(f"Saved 2D tracks JSON to {out}")


# ---------------------------------------------------------------------
# heatmap: build heatmap from 2D tracks
# ---------------------------------------------------------------------
def cmd_heatmap(args: argparse.Namespace) -> None:
    # Lazy import; also avoids your previous "cannot import compute_heatmap_grid" crash at module import time
    heatmap_mod = __import__("avar.analytics.heatmap", fromlist=["save_heatmap"])
    if not hasattr(heatmap_mod, "save_heatmap"):
        raise ImportError("avar.analytics.heatmap.save_heatmap not found. Check avar/analytics/heatmap.py")

    # Support either function name (in case you refactored it previously)
    compute_fn = None
    for name in ("compute_heatmap_grid", "compute_heatmap", "compute_heatmap_grid_clipped"):
        if hasattr(heatmap_mod, name):
            compute_fn = getattr(heatmap_mod, name)
            break
    if compute_fn is None:
        raise ImportError(
            "No heatmap compute function found. Expected one of: "
            "compute_heatmap_grid / compute_heatmap / compute_heatmap_grid_clipped"
        )

    tracks_2d = Path(args.tracks_2d)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building heatmap from {tracks_2d}")
    H = compute_fn(tracks_2d)
    heatmap_mod.save_heatmap(H, out)
    logger.info(f"Saved heatmap to {out}")


# ---------------------------------------------------------------------
# vis-tracks: render bounding boxes + track ids from tracks JSON
# ---------------------------------------------------------------------
def cmd_vis_tracks(args: argparse.Namespace) -> None:
    from avar.visualization.vis_tracking import export_tracking_video

    video_path = Path(args.video)
    tracks_json = Path(args.tracks)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Rendering tracking visualization for {video_path} using {tracks_json}")
    export_tracking_video(
        video_path=video_path,
        tracks_json=tracks_json,
        out_path=out_path,
        fps=args.fps,
    )
    logger.info(f"Saved tracking video to {out_path}")


# ---------------------------------------------------------------------
# detect-video: pure YOLO boxes on original video
# ---------------------------------------------------------------------
def cmd_detect_video(args: argparse.Namespace) -> None:
    from avar.visualization.detect_video import export_detection_video

    video_path = Path(args.video)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running YOLO detection-only video for {video_path} with weights={args.weights}, conf={args.conf}, step={args.step}")
    export_detection_video(
        video_path=video_path,
        out_path=out_path,
        weights=args.weights,
        conf=args.conf,
        step=args.step,
        device=args.device,
    )
    logger.info(f"Saved detection video to {out_path}")


# ---------------------------------------------------------------------
# minimap-video: 2D pitch minimap video (runs YOLO internally)
# ---------------------------------------------------------------------
def cmd_minimap_video(args: argparse.Namespace) -> None:
    from avar.visualization.minimap_video import export_minimap_video

    video_path = Path(args.video)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running minimap video for {video_path} with weights={args.weights}, conf={args.conf}, step={args.step}")
    export_minimap_video(
        video_path=video_path,
        out_path=out_path,
        weights=args.weights,
        conf=args.conf,
        step=args.step,
        device=args.device,
    )
    logger.info(f"Saved minimap video to {out_path}")


# ---------------------------------------------------------------------
# detect-json: YOLO detections -> JSON (single source of truth)
# ---------------------------------------------------------------------
def cmd_detect_json(args: argparse.Namespace) -> None:
    from avar.detection.detect_to_json import export_detections_json

    video_path = Path(args.video)
    out_json = Path(args.out)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running YOLO detect-json for {video_path} weights={args.weights}, conf={args.conf}, step={args.step}")
    export_detections_json(
        video_path=video_path,
        out_json=out_json,
        weights=args.weights,
        conf=args.conf,
        step=args.step,
        device=args.device,
    )
    logger.info(f"Saved detection JSON to {out_json}")


# ---------------------------------------------------------------------
# minimap-json: 2D minimap video from detection JSON only
# ---------------------------------------------------------------------
def cmd_minimap_json(args: argparse.Namespace) -> None:
    from avar.visualization.minimap_from_json import export_minimap_from_json

    det_json = Path(args.detections)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building minimap from detection JSON {det_json}")
    export_minimap_from_json(det_json=det_json, out_path=out_path)
    logger.info(f"Saved minimap video to {out_path}")


# ---------------------------------------------------------------------
# build-foul-dataset: per-half foul/nonfoul feature dataset CSV
# ---------------------------------------------------------------------
def cmd_build_foul_dataset(args: argparse.Namespace) -> None:
    from avar.fouls.build_dataset import build_foul_dataset_for_half

    build_foul_dataset_for_half(
        labels_path=Path(args.labels),
        detections_json=Path(args.detections),
        half=int(args.half),
        out_csv=Path(args.out),
        window_sec=float(args.window_sec),
        contact_radius=float(args.contact_radius),
        negatives_per_positive=int(args.negatives_per_positive),
        negative_margin_sec=float(args.negative_margin_sec),
        random_seed=int(args.seed),
    )


# ---------------------------------------------------------------------
# build-master-foul-dataset: aggregate many halves into one master CSV
# ---------------------------------------------------------------------
def cmd_build_master_foul_dataset(args: argparse.Namespace) -> None:
    build_master(
        raw_root=Path(args.raw_root),
        det_root=Path(args.det_root),
        out_csv=Path(args.out),
        per_half_out_dir=Path(args.per_half_out_dir) if args.per_half_out_dir else None,
        league=args.league if args.league else None,
        window_sec=float(args.window_sec),
        contact_radius=float(args.contact_radius),
        negatives_per_positive=int(args.negatives_per_positive),
        negative_margin_sec=float(args.negative_margin_sec),
        random_seed=int(args.random_seed),
    )


# ---------------------------------------------------------------------
# split-foul-dataset: leak-safe split into train/valid/test by GAME
# ---------------------------------------------------------------------
def cmd_split_foul_dataset(args: argparse.Namespace) -> None:
    split_foul_dataset_run(
        master_csv=Path(args.master_csv),
        out_dir=Path(args.out_dir),
        train_ratio=float(args.train_ratio),
        valid_ratio=float(args.valid_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
        stratify=(int(args.stratify) != 0),
    )


# ---------------------------------------------------------------------
# train-foul-model: train sklearn model from (one or more) CSV datasets
# ---------------------------------------------------------------------
def cmd_train_foul_model(args: argparse.Namespace) -> None:
    # IMPORTANT: lazy import fixes your ImportError at CLI import time
    from avar.fouls.train_model import train_foul_model

    csv_paths = [Path(p.strip()) for p in args.datasets.split(",") if p.strip()]
    train_foul_model(csv_paths, Path(args.out))


# ---------------------------------------------------------------------
# train-foul-model-splits: train using train/valid/test splits (recommended)
# ---------------------------------------------------------------------
def cmd_train_foul_model_splits(args: argparse.Namespace) -> None:
    train_foul_model_splits_run(
        splits_dir=Path(args.splits_dir),
        out_model=Path(args.out_model),
        out_metrics=Path(args.out_metrics),
        features=args.features,
        seed=int(args.seed),
    )


# ---------------------------------------------------------------------
# fouls-predict: run model over detections and output foul candidates
# ---------------------------------------------------------------------
def cmd_fouls_predict(args: argparse.Namespace) -> None:
    from avar.fouls.predict_fouls import predict_fouls_for_half

    predict_fouls_for_half(
        detections_json=Path(args.detections),
        model_path=Path(args.model),
        out_json=Path(args.out),
        half=int(args.half),
        window_sec=float(args.window_sec),
        contact_radius=float(args.contact_radius),
        step_sec=float(args.step_sec),
        score_threshold=float(args.score_threshold),
    )


# ---------------------------------------------------------------------
# batch-detect: run detect-json on many videos under data/raw/soccernet
# ---------------------------------------------------------------------
def cmd_batch_detect(args: argparse.Namespace) -> None:
    batch_detect(
        weights=args.weights,
        conf=float(args.conf),
        step=int(args.step),
        device=args.device,
    )

def cmd_fouls_postprocess(args) -> None:
    from avar.scripts.fouls_postprocess import run
    run(
        preds_json=args.preds,
        out_events_json=args.out,
        threshold=args.threshold,
        gap_sec=args.gap_sec,
        min_duration_sec=args.min_duration_sec,
        topk=args.topk,
    )

# ---------------------------------------------------------------------
# Argument parser / main entrypoint
# ---------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="avar", description="Automated Video Assistant Referee CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ---- download ----
    p_dl = sub.add_parser("download", help="Download SoccerNet data")
    p_dl.add_argument("--files", type=str, default=None, help="Comma-separated list of files to download")
    p_dl.add_argument("--task", type=str, default=None, help="SoccerNet task name (e.g. 'tracking-2023')")
    p_dl.add_argument("--splits", type=str, default="train,valid,test", help="Comma-separated splits (e.g. 'train,valid')")
    p_dl.add_argument("--local-dir", type=str, required=True, help="Local root directory to store downloaded data")
    p_dl.add_argument("--password", type=str, default=None, help="Password for videos/challenge if required by NDA")
    p_dl.add_argument("--version", type=str, default=None, help="Optional version argument for downloadDataTask")
    p_dl.set_defaults(func=cmd_download)

    # ---- track ----
    p_tr = sub.add_parser("track", help="Run player tracking on a video")
    p_tr.add_argument("--video", type=str, required=True, help="Input video path")
    p_tr.add_argument("--weights", type=str, default="yolov8n.pt", help="YOLO weights (default: yolov8n.pt)")
    p_tr.add_argument("--out", type=str, required=True, help="Output JSON for tracks")
    p_tr.add_argument("--batch-size", type=int, default=8)
    p_tr.add_argument("--step", type=int, default=2, help="Frame step (2 => half FPS)")
    p_tr.add_argument("--width", type=int, default=960)
    p_tr.add_argument("--height", type=int, default=540)
    p_tr.add_argument("--device", type=str, default=None, help="Device for YOLO (e.g. 'cpu', 'cuda')")
    p_tr.set_defaults(func=cmd_track)

    # ---- project-2d ----
    p_p2 = sub.add_parser("project-2d", help="Project image tracks to 2D pitch")
    p_p2.add_argument("--tracks", type=str, required=True, help="Input tracks JSON")
    p_p2.add_argument("--calib", type=str, required=True, help="Field calib JSON")
    p_p2.add_argument("--out", type=str, required=True, help="Output 2D tracks JSON")
    p_p2.set_defaults(func=cmd_project_2d)

    # ---- heatmap ----
    p_hm = sub.add_parser("heatmap", help="Generate heatmap from 2D tracks")
    p_hm.add_argument("--tracks-2d", type=str, required=True, help="2D tracks JSON")
    p_hm.add_argument("--out", type=str, required=True, help="Output PNG path")
    p_hm.set_defaults(func=cmd_heatmap)

    # ---- vis-tracks ----
    p_vt = sub.add_parser("vis-tracks", help="Render tracking video with bounding boxes and IDs from JSON")
    p_vt.add_argument("--video", type=str, required=True, help="Input video path")
    p_vt.add_argument("--tracks", type=str, required=True, help="Tracks JSON path")
    p_vt.add_argument("--out", type=str, required=True, help="Output MP4 path")
    p_vt.add_argument("--fps", type=int, default=None, help="Output FPS (default: same as input video)")
    p_vt.set_defaults(func=cmd_vis_tracks)

    # ---- detect-video ----
    p_dv = sub.add_parser("detect-video", help="Run YOLO on video and output boxes-only MP4 (no tracking)")
    p_dv.add_argument("--video", type=str, required=True, help="Input video path")
    p_dv.add_argument("--out", type=str, required=True, help="Output MP4 path")
    p_dv.add_argument("--weights", type=str, default="yolov8n.pt", help="YOLO weights (default: yolov8n.pt)")
    p_dv.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p_dv.add_argument("--step", type=int, default=1, help="Process every Nth frame")
    p_dv.add_argument("--device", type=str, default=None, help="Device for YOLO (e.g. 'cpu', 'cuda')")
    p_dv.set_defaults(func=cmd_detect_video)

    # ---- minimap-video ----
    p_mm = sub.add_parser("minimap-video", help="Generate 2D pitch minimap video with player dots (runs YOLO)")
    p_mm.add_argument("--video", type=str, required=True, help="Input video path")
    p_mm.add_argument("--out", type=str, required=True, help="Output MP4 path")
    p_mm.add_argument("--weights", type=str, default="yolov8n.pt", help="YOLO weights (default: yolov8n.pt)")
    p_mm.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p_mm.add_argument("--step", type=int, default=1, help="Process every Nth frame")
    p_mm.add_argument("--device", type=str, default=None, help="Device for YOLO (e.g. 'cpu', 'cuda')")
    p_mm.set_defaults(func=cmd_minimap_video)

    # ---- detect-json ----
    p_dj = sub.add_parser("detect-json", help="Run YOLO and save detections to JSON")
    p_dj.add_argument("--video", type=str, required=True, help="Input video path")
    p_dj.add_argument("--out", type=str, required=True, help="Output JSON path")
    p_dj.add_argument("--weights", type=str, default="yolov8n.pt", help="YOLO weights (default: yolov8n.pt)")
    p_dj.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p_dj.add_argument("--step", type=int, default=1, help="Process every Nth frame")
    p_dj.add_argument("--device", type=str, default=None, help="Device for YOLO (e.g. 'cpu', 'cuda')")
    p_dj.set_defaults(func=cmd_detect_json)

    # ---- minimap-json ----
    p_mj = sub.add_parser("minimap-json", help="Generate 2D pitch minimap from detection JSON")
    p_mj.add_argument("--detections", type=str, required=True, help="Detection JSON path (from detect-json)")
    p_mj.add_argument("--out", type=str, required=True, help="Output MP4 minimap path")
    p_mj.set_defaults(func=cmd_minimap_json)

    # ---- build-foul-dataset ----
    p_fd = sub.add_parser("build-foul-dataset", help="Build baseline foul/non-foul dataset for one half")
    p_fd.add_argument("--labels", type=str, required=True, help="Path to Labels-v2.json for the game")
    p_fd.add_argument("--detections", type=str, required=True, help="Detection JSON for the half (from detect-json)")
    p_fd.add_argument("--half", type=int, required=True, help="Half index (1 or 2)")
    p_fd.add_argument("--out", type=str, required=True, help="Output CSV path")
    p_fd.add_argument("--window-sec", type=float, default=1.0)
    p_fd.add_argument("--contact-radius", type=float, default=0.05)
    p_fd.add_argument("--negatives-per-positive", type=int, default=3)
    p_fd.add_argument("--negative-margin-sec", type=float, default=5.0)
    p_fd.add_argument("--seed", type=int, default=42)
    p_fd.set_defaults(func=cmd_build_foul_dataset)

    # ---- build-master-foul-dataset ----
    p_mfd = sub.add_parser("build-master-foul-dataset", help="Aggregate per-half foul datasets into one master CSV")
    p_mfd.add_argument("--raw-root", type=str, default="data/raw/soccernet")
    p_mfd.add_argument("--det-root", type=str, default="data/processed/detections")
    p_mfd.add_argument("--out", type=str, required=True)
    p_mfd.add_argument("--per-half-out-dir", type=str, default=None)
    p_mfd.add_argument("--league", type=str, default="england_epl", help="Set empty to disable filtering")
    p_mfd.add_argument("--window-sec", type=float, default=1.0)
    p_mfd.add_argument("--contact-radius", type=float, default=0.05)
    p_mfd.add_argument("--negatives-per-positive", type=int, default=3)
    p_mfd.add_argument("--negative-margin-sec", type=float, default=5.0)
    p_mfd.add_argument("--random-seed", type=int, default=42)
    p_mfd.set_defaults(func=cmd_build_master_foul_dataset)

    # ---- split-foul-dataset ----
    p_split = sub.add_parser("split-foul-dataset", help="Leakage-safe split of master dataset by GAME")
    p_split.add_argument("--master-csv", type=str, required=True)
    p_split.add_argument("--out-dir", type=str, required=True)
    p_split.add_argument("--train-ratio", type=float, default=0.70)
    p_split.add_argument("--valid-ratio", type=float, default=0.15)
    p_split.add_argument("--test-ratio", type=float, default=0.15)
    p_split.add_argument("--seed", type=int, default=42)
    p_split.add_argument("--stratify", type=int, default=1, help="1=stratify by game foul-rate bins, 0=no stratify")
    p_split.set_defaults(func=cmd_split_foul_dataset)

    # ---- train-foul-model (single/multi-csv) ----
    p_tfm = sub.add_parser("train-foul-model", help="Train foul classifier from one or more CSV datasets")
    p_tfm.add_argument("--datasets", type=str, required=True, help="Comma-separated list of foul dataset CSVs")
    p_tfm.add_argument("--out", type=str, required=True, help="Output model path (e.g. models/foul_baseline.pkl)")
    p_tfm.set_defaults(func=cmd_train_foul_model)

    # ---- train-foul-model-splits (recommended) ----
    p_tfms = sub.add_parser("train-foul-model-splits", help="Train foul classifier using train/valid/test splits")
    p_tfms.add_argument("--splits-dir", type=str, required=True, help="Directory containing train.csv/valid.csv/test.csv")
    p_tfms.add_argument("--out-model", type=str, required=True, help="Path to save model artifact (.pkl)")
    p_tfms.add_argument("--out-metrics", type=str, required=True, help="Path to save metrics JSON")
    p_tfms.add_argument("--features", type=str, default="", help="Comma-separated feature columns (optional)")
    p_tfms.add_argument("--seed", type=int, default=42)
    p_tfms.set_defaults(func=cmd_train_foul_model_splits)

    # -------------------------
    # fouls-postprocess
    # -------------------------
    p_fp = sub.add_parser(
        "fouls-postprocess",
        help="Convert per-frame foul predictions into consolidated foul events",
    )
    p_fp.add_argument("--preds", required=True, help="Predictions JSON from fouls-predict")
    p_fp.add_argument("--out", required=True, help="Output events JSON")
    p_fp.add_argument("--threshold", type=float, default=0.60, help="Probability threshold")
    p_fp.add_argument("--gap-sec", type=float, default=1.0, help="Merge events if gaps <= this (seconds)")
    p_fp.add_argument("--min-duration-sec", type=float, default=0.6, help="Discard events shorter than this (seconds)")
    p_fp.add_argument("--topk", type=int, default=None, help="Keep only top-K events by peak probability")
    p_fp.set_defaults(func=cmd_fouls_postprocess)

    # ---- fouls-predict ----
    p_fp = sub.add_parser("fouls-predict", help="Predict foul windows from detections JSON using trained model")
    p_fp.add_argument("--detections", type=str, required=True, help="Detection JSON for the half (from detect-json)")
    p_fp.add_argument("--half", type=int, required=True, help="Half index (1 or 2)")
    p_fp.add_argument("--model", type=str, required=True, help="Path to trained foul model (joblib)")
    p_fp.add_argument("--out", type=str, required=True, help="Output JSON with foul predictions")
    p_fp.add_argument("--window-sec", type=float, default=1.0)
    p_fp.add_argument("--contact-radius", type=float, default=0.05)
    p_fp.add_argument("--step-sec", type=float, default=0.5)
    p_fp.add_argument("--score-threshold", type=float, default=0.5)
    p_fp.set_defaults(func=cmd_fouls_predict)

    # ---- batch-detect ----
    p_batch = sub.add_parser("batch-detect", help="Run detect-json on ALL SoccerNet games")
    p_batch.add_argument("--weights", type=str, default="yolov8n.pt")
    p_batch.add_argument("--conf", type=float, default=0.25)
    p_batch.add_argument("--step", type=int, default=1)
    p_batch.add_argument("--device", type=str, default=None)
    p_batch.set_defaults(func=cmd_batch_detect)

    return ap


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
