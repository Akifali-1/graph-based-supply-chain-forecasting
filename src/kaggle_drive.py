"""Google Drive API storage using the notebook's authorized-user OAuth token."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import tempfile
import time
from pathlib import Path, PurePosixPath


DEFAULT_TOKEN_FILE = Path("/kaggle/input/datasets/mohdakifaliparvez/google-drive-oauth-token-1/drive_token.json")
DEFAULT_ROOT_ID = "1jMUTXWHeee5LVuRralPSP1dFOLHXith-"


def checksum(path, algorithm="md5"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".json-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def find_token_file(preferred_path: Path | str | None = None) -> Path:
    """Find the drive token file across standard Kaggle and local locations."""
    candidates = []
    if preferred_path:
        candidates.append(Path(preferred_path))
    candidates.extend([
        DEFAULT_TOKEN_FILE,
        Path("/kaggle/input/google-drive-oauth-token-1/drive_token.json"),
        Path("/kaggle/input/datasets/mohdakifaliparvez/google-drive-oauth-token-1/token.json"),
        Path("/kaggle/input/google-drive-oauth-token-1/token.json"),
        Path("/kaggle/working/drive_token.json"),
        Path("/kaggle/working/token.json"),
        Path("drive_token.json"),
        Path("token.json"),
        Path.home() / ".gdrive" / "drive_token.json",
        Path.home() / ".gdrive" / "token.json",
    ])
    for c in candidates:
        if c.is_file():
            return c.resolve()

    # Dynamic glob search in /kaggle/input if mounted under custom name
    if Path("/kaggle/input").exists():
        found = (
            list(Path("/kaggle/input").glob("**/drive_token*.json"))
            + list(Path("/kaggle/input").glob("**/token*.json"))
        )
        if found:
            return found[0].resolve()

    # Fallback to Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        for key in ["GDRIVE_TOKEN", "gdrive_token", "GOOGLE_DRIVE_TOKEN", "drive_token"]:
            try:
                val = secrets.get_secret(key)
                if val:
                    tmp = Path(tempfile.gettempdir()) / "drive_token.json"
                    tmp.write_text(val, encoding="utf-8")
                    return tmp.resolve()
            except Exception:
                pass
    except Exception:
        pass

    return Path(preferred_path) if preferred_path else DEFAULT_TOKEN_FILE


def get_drive(token_file: Path | str | None = None):
    # These are the exact OAuth credentials/build calls used in the reference notebook.
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    actual_file = find_token_file(token_file)
    if not actual_file.exists():
        raise FileNotFoundError(f"Google Drive token file not found: {actual_file}")

    credentials = Credentials.from_authorized_user_file(
        str(actual_file), ["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def account_email(drive):
    return drive.about().get(fields="user(emailAddress)").execute(num_retries=3)["user"]["emailAddress"]


class DriveStore:
    """Preserve relative directories, replace named files, and verify downloads."""

    def __init__(self, drive, root_id=DEFAULT_ROOT_ID):
        self.drive = drive
        self.root_id = root_id
        self.folders = {"": root_id}

    @staticmethod
    def _quote(value):
        return str(value).replace("\\", "\\\\").replace("'", "\\'")

    @staticmethod
    def _relative(name):
        path = PurePosixPath(str(name))
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("Drive artifact paths must be relative")
        return path

    def _find(self, name, parent_id):
        query = f"name='{self._quote(name)}' and '{self._quote(parent_id)}' in parents and trashed=false"
        files = []
        token = None
        while True:
            response = self.drive.files().list(
                q=query,
                pageToken=token,
                pageSize=1000,
                fields="nextPageToken,files(id,name,mimeType,size,md5Checksum)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute(num_retries=3)
            files.extend(response.get("files", []))
            token = response.get("nextPageToken")
            if not token:
                break
        if len(files) > 1:
            # Prefer non-folder if multiple entries, or pick the first
            return files[0]
        return files[0] if files else None

    def folder(self, relative):
        if str(relative) in ("", "."):
            return self.root_id
        path = self._relative(relative)
        parent_id = self.root_id
        for index, name in enumerate(path.parts, 1):
            key = str(PurePosixPath(*path.parts[:index]))
            if key not in self.folders:
                found = self._find(name, parent_id)
                if found:
                    if found["mimeType"] != "application/vnd.google-apps.folder":
                        raise ValueError(f"Drive path is not a folder: {key}")
                    folder_id = found["id"]
                else:
                    folder_id = self.drive.files().create(
                        body={"name": name, "parents": [parent_id], "mimeType": "application/vnd.google-apps.folder"},
                        fields="id",
                        supportsAllDrives=True,
                    ).execute(num_retries=3)["id"]
                self.folders[key] = folder_id
            parent_id = self.folders[key]
        return parent_id

    def entry(self, relative, create_parent=False):
        path = self._relative(relative)
        parent = str(path.parent)
        if create_parent:
            parent_id = self.folder(parent)
        else:
            parent_id = self.root_id
            for name in path.parts[:-1]:
                found = self._find(name, parent_id)
                if not found:
                    return None
                parent_id = found["id"]
        return self._find(path.name, parent_id)

    def upload(self, local_path, relative):
        from googleapiclient.http import MediaFileUpload

        local_path = Path(local_path)
        path = self._relative(relative)
        parent_id = self.folder(str(path.parent))
        found = self._find(path.name, parent_id)
        md5 = checksum(local_path)
        if found and found.get("md5Checksum") == md5:
            return found
        media = MediaFileUpload(str(local_path), resumable=True, chunksize=4 * 1024 * 1024)
        fields = "id,name,size,md5Checksum"
        for attempt in range(3):
            try:
                if found:
                    result = self.drive.files().update(
                        fileId=found["id"], media_body=media, fields=fields, supportsAllDrives=True,
                    ).execute(num_retries=3)
                else:
                    result = self.drive.files().create(
                        body={"name": path.name, "parents": [parent_id]},
                        media_body=media,
                        fields=fields,
                        supportsAllDrives=True,
                    ).execute(num_retries=3)
                break
            except (TimeoutError, socket.timeout):
                if attempt == 2:
                    raise
                time.sleep(1.0 * (attempt + 1))
        if result.get("md5Checksum") != md5:
            raise IOError(f"Drive upload checksum mismatch: {relative}")
        return result

    def put_json(self, relative, payload):
        with tempfile.TemporaryDirectory(prefix="drive-json-") as temporary:
            path = Path(temporary) / "payload.json"
            atomic_json(path, payload)
            return self.upload(path, relative)

    def get_json(self, relative):
        found = self.entry(relative)
        if not found:
            return None
        content = self.drive.files().get_media(fileId=found["id"], supportsAllDrives=True).execute(num_retries=3)
        return json.loads(content)

    def download(self, relative, local_path):
        from googleapiclient.http import MediaIoBaseDownload

        found = self.entry(relative)
        if not found:
            raise FileNotFoundError(f"Missing Drive artifact: {relative}")
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".download-", suffix=".tmp", dir=local_path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                request = self.drive.files().get_media(fileId=found["id"], supportsAllDrives=True)
                downloader = MediaIoBaseDownload(handle, request, chunksize=4 * 1024 * 1024)
                done = False
                while not done:
                    _, done = downloader.next_chunk(num_retries=3)
                handle.flush()
                os.fsync(handle.fileno())
            if checksum(temporary) != found.get("md5Checksum"):
                raise IOError(f"Drive download checksum mismatch: {relative}")
            temporary.replace(local_path)
        finally:
            temporary.unlink(missing_ok=True)
        return local_path

    def delete(self, relative):
        found = self.entry(relative)
        if found:
            self.drive.files().delete(fileId=found["id"], supportsAllDrives=True).execute(num_retries=3)
