"""Resumable multi-model control and expansion campaign runner.

Follows the exact Google Drive storage, checkpoint recovery, and incremental CSV
synchronization pattern from the reference notebook.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from kaggle_drive import DriveStore, account_email, atomic_json, checksum, find_token_file, get_drive

ROW_FIELDS = (
    "dataset",
    "model",
    "seed",
    "MAE",
    "RMSE",
    "WAPE",
    "sMAPE",
    "R2",
    "MAPE_masked",
    "MAPE_valid_fraction",
    "MSE",
    "epochs_completed",
    "selected_epoch",
    "parameter_count",
    "training_seconds",
)

ARTIFACT_FILES = (
    "config.yaml",
    "runtime.json",
    "processed_manifest.json",
    "training.log",
    "checkpoints/best.pt",
    "checkpoints/latest.pt",
    "predictions/validation.npz",
    "metrics/validation.json",
    "logs/history.json",
)


def digest(path, algorithm="sha256"):
    hasher = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def manifest(root, names):
    result = {}
    for name in names:
        p = Path(root) / name
        if p.exists():
            result[name] = {
                "size": p.stat().st_size,
                "sha256": digest(p),
                "md5Checksum": digest(p, "md5"),
            }
    return result


class Tee:
    def __init__(self, console, file):
        self.console, self.file = console, file

    def write(self, text):
        self.console.write(text)
        self.file.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.file.flush()


class Persistence:
    """Manages incremental per-epoch checkpoints and final artifact sync to Google Drive."""

    def __init__(self, store: DriveStore, root: Path, run_id: str, signature: str, on_epoch):
        self.store, self.root, self.run_id = store, Path(root), run_id
        self.signature, self.on_epoch = signature, on_epoch

    def upload(self, relative: str):
        local_file = self.root / relative
        if local_file.exists():
            return self.store.upload(local_file, f"{self.run_id}/{relative}")
        return None

    def epoch(self, epoch: int):
        checkpoint = self.root / "checkpoints/latest.pt"
        if not checkpoint.exists():
            return
        sha = digest(checkpoint)
        relative = f"{self.run_id}/checkpoints/epoch-{epoch:04d}-{sha[:12]}.pt"
        self.upload("checkpoints/best.pt")
        self.upload("logs/history.json")
        self.upload("training.log")
        remote = self.store.upload(checkpoint, relative)
        pointer = {
            "experiment_id": self.run_id,
            "signature": self.signature,
            "epoch": epoch,
            "checkpoint": relative,
            "sha256": sha,
            "md5Checksum": remote.get("md5Checksum"),
        }
        self.store.put_json(f"{self.run_id}/recovery.json", pointer)
        self.on_epoch(epoch)

    def finish(self):
        for relative in ARTIFACT_FILES:
            self.upload(relative)
        complete = {
            "experiment_id": self.run_id,
            "signature": self.signature,
            "files": manifest(self.root, ARTIFACT_FILES),
        }
        self.store.put_json(f"{self.run_id}/complete.json", complete)


def hook_save_json(trainer, persistence: Persistence):
    original = trainer.save_json

    def saved(payload, path):
        result = original(payload, path)
        try:
            if Path(path).resolve() == (persistence.root / "logs/history.json").resolve() and payload.get("history"):
                # Preserve RNG state during upload
                rng = trainer.capture_rng_state()
                try:
                    last_epoch = int(payload["history"][-1]["epoch"])
                    persistence.epoch(last_epoch)
                finally:
                    trainer.restore_rng_state(rng)
        except Exception as e:
            print(f"[Persistence Warning] Non-critical error during epoch upload: {e}", flush=True)
        return result

    trainer.save_json = saved
    return original


def restore(store: DriveStore, run_id: str, signature: str, local_root: Path) -> Path | None:
    pointer = store.get_json(f"{run_id}/recovery.json")
    if not pointer:
        return None
    if pointer["experiment_id"] != run_id or pointer["signature"] != signature:
        print(f"[Warning] Recovery pointer signature mismatch for {run_id}. Ignoring stale checkpoint.", flush=True)
        return None
    local = store.download(pointer["checkpoint"], Path(local_root) / "checkpoints/latest.pt")
    if digest(local) != pointer["sha256"]:
        raise IOError(f"Recovery checkpoint sha256 mismatch for {run_id}")
    return local


def load_registry(store: DriveStore, name: str, campaign: str) -> dict:
    return store.get_json(name) or {"schema_version": 1, "campaign": campaign, "runs": {}, "updated_at": None}


def save_registry(store: DriveStore, work: Path, name: str, state: dict):
    state["updated_at"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    local = Path(work) / name
    atomic_json(local, state)
    store.upload(local, name)


def append_row(path: Path, row: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in ROW_FIELDS})
        handle.flush()


def run(args):
    import yaml

    root = args.project_root.resolve()
    work = args.work_root.resolve()
    processed = args.processed_root.resolve()
    work.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / "src"))
    import train as trainer

    base = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base.setdefault("data", {})["dataset_name"] = args.dataset
    if args.max_time_steps is not None:
        base["data"]["max_time_steps"] = int(args.max_time_steps)
    if args.max_job_seconds is not None:
        base.setdefault("experiment", {})["max_job_seconds"] = int(args.max_job_seconds)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        commit = "local"

    token_path = find_token_file(args.token_file)
    drive = get_drive(token_path)
    print(f"Authenticated Drive account: {account_email(drive)}", flush=True)

    root_store = DriveStore(drive, args.drive_root_folder_id)
    campaign_id = root_store.folder(f"experiments/{args.campaign}")
    store = DriveStore(drive, campaign_id)

    registry_name = f"{args.campaign}.json"
    state = load_registry(store, registry_name, args.campaign)

    csv_name = args.results_name or f"{args.campaign}_results.csv"
    rows_path = work / csv_name
    if rows_path.exists():
        rows_path.unlink()
    remote_rows = store.entry(csv_name)
    if remote_rows:
        store.download(csv_name, rows_path)
        print(f"Downloaded existing results CSV from Drive: {csv_name}", flush=True)

    processed_manifest = manifest(
        processed,
        ["nodes.csv", "edges.csv", "sales.csv", "target_definition.json", "data_quality.json"],
    )
    for name in processed_manifest:
        store.upload(processed / name, f"provenance/processed/{name}")
    store.upload(args.config, "provenance/config.yaml")
    save_registry(store, work, registry_name, state)

    models_to_run = args.models if args.models else [args.model]

    for model in models_to_run:
        for seed in args.seeds:
            run_id = f"{args.dataset}_{model}_seed{seed}"
            payload = {
                "dataset": args.dataset,
                "model": model,
                "seed": seed,
                "max_time_steps": base["data"].get("max_time_steps"),
                "config": base,
                "processed": processed_manifest,
            }
            signature = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
            record = state["runs"].get(run_id)

            if record and record.get("status") == "completed":
                print(f"Skipping recorded completed run: {run_id}", flush=True)
                continue

            artifact = work / run_id
            artifact.mkdir(parents=True, exist_ok=True)
            resume = restore(store, run_id, signature, artifact)
            if record and resume is None and not args.restart_uncommitted:
                print(f"[Notice] {run_id} marked as running/interrupted without checkpoint, restarting...", flush=True)

            if record is None:
                record = {
                    "experiment_id": run_id,
                    "signature": signature,
                    "status": "reserved",
                    "seed": seed,
                    "model": model,
                    "created_at": str(__import__("datetime").datetime.now()),
                }
                state["runs"][run_id] = record

            config = copy.deepcopy(base)
            config["seed"] = seed
            config["data"]["processed_dir"] = str(processed)
            config["outputs"] = {
                "checkpoint_dir": str(artifact / "checkpoints"),
                "metrics_dir": str(artifact / "metrics"),
                "predictions_dir": str(artifact / "predictions"),
                "logs_dir": str(artifact / "logs"),
            }
            config["run"] = {"experiment_id": run_id, "signature": signature, "git_commit": commit}

            # Handle permuted model configuration
            if model == "stgt_permuted":
                config["model"] = "stgt_permuted"
                config["data"]["graph_mode"] = "permuted"
                config["data"]["graph_seed"] = 0
                if "stgt_permuted" in config.get("models", {}):
                    config["models"]["stgt_permuted"].pop("graph_seed", None)
            else:
                config["model"] = model

            config_path = artifact / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            atomic_json(artifact / "processed_manifest.json", processed_manifest)
            atomic_json(
                artifact / "runtime.json",
                {
                    "git_commit": commit,
                    "packages": {
                        pkg: importlib.metadata.version(pkg)
                        for pkg in ("torch", "torch-geometric", "numpy", "pandas")
                        if importlib.util.find_spec(pkg.replace("-", "_")) is not None
                    },
                },
            )

            record.setdefault("sessions", []).append({"seed": seed, "resume": bool(resume)})
            record["status"] = "running"
            save_registry(store, work, registry_name, state)

            persistence = None
            original_save_json = None
            try:
                # Validation check on window counts
                data = trainer.ExperimentData(config, root)
                is_t221 = base["data"].get("max_time_steps") == 221 or args.dataset == "supplygraph"
                if is_t221:
                    expected_val_windows = 27
                    expected_test_windows = 28
                else:
                    expected_val_windows = 7878
                    expected_test_windows = 7878

                val_len = len(data.datasets["validation"])
                test_len = len(data.datasets["test"])
                if val_len != expected_val_windows or test_len != expected_test_windows:
                    raise RuntimeError(
                        f"Window count mismatch for {args.dataset}: "
                        f"validation={val_len} (expected {expected_val_windows}), "
                        f"test={test_len} (expected {expected_test_windows})"
                    )

                def record_epoch(epoch):
                    record["last_drive_epoch"] = epoch
                    save_registry(store, work, registry_name, state)

                persistence = Persistence(store, artifact, run_id, signature, record_epoch)
                for rel in ("config.yaml", "runtime.json", "processed_manifest.json"):
                    persistence.upload(rel)

                original_save_json = hook_save_json(trainer, persistence)
                log_path = artifact / "training.log"

                print(f"\n================ Starting Run: {run_id} ================", flush=True)
                with log_path.open("a", encoding="utf-8", buffering=1) as log:
                    with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                        report = trainer.train_model(model, config, root, resume)

                persistence.finish()

                row = {
                    "dataset": args.dataset,
                    "model": model,
                    "seed": seed,
                    **report["validation"],
                    "epochs_completed": report["epochs_completed"],
                    "selected_epoch": report["selected_epoch"],
                    "parameter_count": report["parameter_count"],
                    "training_seconds": report["training_seconds"],
                }
                append_row(rows_path, row)
                store.upload(rows_path, csv_name)
                record.update(status="completed", report=row)
                save_registry(store, work, registry_name, state)
                print(f"Successfully completed {run_id}; result appended and synced to {csv_name}", flush=True)

            except TimeoutError as exc:
                print(
                    f"\n[Interrupted/Timeout] {run_id}: {type(exc).__name__}: {exc}. "
                    f"Preserving committed epoch {record.get('last_drive_epoch')} and returning code 124 for resumption.",
                    flush=True,
                )
                record.update(
                    status="interrupted",
                    failure=f"{type(exc).__name__}: {exc}",
                    last_drive_epoch=record.get("last_drive_epoch"),
                )
                try:
                    if persistence:
                        persistence.upload("training.log")
                    save_registry(store, work, registry_name, state)
                except Exception as sync_err:
                    print(f"[Warning] Sync on interruption encountered error: {sync_err}", flush=True)
                return 124

            except BaseException as exc:
                print(f"\n[Error] {run_id} failed with {type(exc).__name__}: {exc}", flush=True)
                record.update(status="failed", failure={"type": type(exc).__name__, "reason": str(exc)})
                try:
                    if persistence and (artifact / "training.log").exists():
                        persistence.upload("training.log")
                    save_registry(store, work, registry_name, state)
                except Exception:
                    pass
                raise

            finally:
                if original_save_json is not None:
                    trainer.save_json = original_save_json

    print(f"\nCompleted campaign: {args.campaign}", flush=True)
    return 0


def parser():
    p = argparse.ArgumentParser(description="Kaggle Control and Expansion Campaign Runner")
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--processed-root", type=Path, required=True)
    p.add_argument("--token-file", type=Path, default=None)
    p.add_argument("--drive-root-folder-id", default="1jMUTXWHeee5LVuRralPSP1dFOLHXith-")
    p.add_argument("--work-root", type=Path, required=True)
    p.add_argument("--campaign", required=True)
    p.add_argument("--dataset", required=True, choices=["isomorph", "supplygraph"])
    p.add_argument(
        "--model",
        default="stgt",
        choices=["stgt", "stgt_permuted", "temporal_only", "spatial_only", "adaptive_stgt"],
    )
    p.add_argument(
        "--models",
        nargs="+",
        choices=["stgt", "stgt_permuted", "temporal_only", "spatial_only", "adaptive_stgt"],
        default=None,
    )
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--max-time-steps", "--max-steps", dest="max_time_steps", type=int, default=None)
    p.add_argument("--results-name", default=None)
    p.add_argument("--max-job-seconds", type=int, default=7200)
    p.add_argument("--restart-uncommitted", action="store_true")
    p.add_argument("--allow-execution", action="store_true")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if not args.allow_execution:
        raise SystemExit("Add --allow-execution to run the campaign")
    raise SystemExit(run(args))
