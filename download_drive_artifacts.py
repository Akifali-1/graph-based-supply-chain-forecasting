from __future__ import annotations

import io
import argparse
from pathlib import Path

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


class Drive:
    def __init__(self, token_file: Path, root_id: str):
        credentials = Credentials.from_authorized_user_file(
            str(token_file), ["https://www.googleapis.com/auth/drive"]
        )
        self.api = build("drive", "v3", credentials=credentials, cache_discovery=False)
        self.folder_cache = {"": root_id}

    def children(self, parent_id: str):
        query = f"'{parent_id}' in parents and trashed=false"
        result = []
        token = None
        while True:
            response = self.api.files().list(
                q=query,
                pageToken=token,
                pageSize=1000,
                fields="nextPageToken,files(id,name,mimeType,size)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute(num_retries=3)
            result.extend(response.get("files", []))
            token = response.get("nextPageToken")
            if not token:
                return result

    def resolve(self, relative: str):
        if relative in self.folder_cache:
            return self.folder_cache[relative]
        parent = ROOT_ID
        parts = Path(relative).parts
        walked = []
        for part in parts:
            walked.append(part)
            key = "/".join(walked)
            matches = [f for f in self.children(parent) if f["name"] == part]
            if len(matches) != 1:
                raise FileNotFoundError(f"Drive path component not unique: {key}")
            parent = matches[0]["id"]
            self.folder_cache[key] = parent
        return parent

    def file(self, parent_relative: str, name: str):
        parent = self.resolve(parent_relative)
        matches = [f for f in self.children(parent) if f["name"] == name]
        if len(matches) != 1:
            raise FileNotFoundError(f"Drive file not unique: {parent_relative}/{name}")
        return matches[0]

    def download(self, parent_relative: str, name: str, destination: Path):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size > 0:
            return
        item = self.file(parent_relative, name)
        request = self.api.files().get_media(fileId=item["id"], supportsAllDrives=True)
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request, chunksize=8 * 1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk(num_retries=3)
        destination.write_bytes(buffer.getvalue())
        if item.get("size") and destination.stat().st_size != int(item["size"]):
            raise IOError(f"Size mismatch for {parent_relative}/{name}")


def main(token_file: Path, drive_root_folder_id: str, output: Path):
    drive = Drive(token_file, drive_root_folder_id)
    datasets = {
        "isomorph": "experiments/isomorph_sample_kaggle_campaign/processed_inventory",
        "supplygraph": "experiments/supplygraph_sales_order_plant_campaign/processed_inventory",
    }
    names = ["nodes.csv", "edges.csv", "sales.csv", "target_definition.json", "data_quality.json"]
    for dataset, source in datasets.items():
        for name in names:
            drive.download(source, name, output / dataset / name)

    runs = []
    for dataset, campaign, prefix in [
        ("isomorph", "experiments/isomorph_sample_kaggle_campaign", "isomorph_sample"),
        ("supplygraph", "experiments/supplygraph_sales_order_plant_campaign", "supplygraph_sales_order_plant"),
    ]:
        for model in ["temporal_only", "spatial_only", "stgt", "adaptive_stgt"]:
            for seed in [42, 123, 2026]:
                runs.append((dataset, f"{campaign}/{prefix}_{model}_seed{seed}", f"{model}_seed{seed}"))
    for dataset, campaign, prefix in [
        ("isomorph", "experiments/job_b_isomorph_stgt_permuted", "stgt_permuted"),
        ("supplygraph", "experiments/job_b_supplygraph_stgt_permuted", "stgt_permuted"),
    ]:
        for seed in [42, 123, 2026]:
            runs.append((dataset, f"{campaign}/{dataset}_stgt_permuted_seed{seed}", f"stgt_permuted_seed{seed}"))

    files = [
        "checkpoints/best.pt",
        "metrics/validation.json",
        "logs/history.json",
        "complete.json",
        "recovery.json",
    ]
    for dataset, source, name in runs:
        for relative in files:
            if "/" in relative:
                parent, filename = relative.rsplit("/", 1)
            else:
                parent, filename = "", relative
            try:
                drive.download(source + "/" + parent, filename, output / dataset / name / relative)
            except FileNotFoundError:
                print("MISSING", source + "/" + relative, flush=True)
    print("Downloaded artifacts to", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-file", type=Path, required=True,
                        help="Path to a local OAuth token JSON; never commit it")
    parser.add_argument("--drive-root-folder-id", required=True,
                        help="Shared Drive/research root folder ID")
    parser.add_argument("--output", type=Path, default=Path("drive_artifacts"),
                        help="Local directory for downloaded artifacts")
    args = parser.parse_args()
    main(args.token_file, args.drive_root_folder_id, args.output)
