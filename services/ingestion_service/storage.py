"""Private local original-PDF storage with usage accounting."""

import hashlib
import os
from pathlib import Path

from services.ingestion_service.models import StorageUsage


class LocalOriginalStore:
    def __init__(self, root: str, quota_bytes: int | None = None, warning_threshold: float = 0.8):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.quota_bytes = quota_bytes
        self.warning_threshold = warning_threshold

    def path_for(self, content_hash: str) -> Path:
        if len(content_hash) != 64 or any(char not in "0123456789abcdef" for char in content_hash):
            raise ValueError("Invalid content hash")
        return self.root / f"{content_hash}.pdf"

    def save(self, content: bytes, content_hash: str) -> Path:
        actual_hash = hashlib.sha256(content).hexdigest()
        if actual_hash != content_hash:
            raise ValueError("Content hash changed before storage")
        path = self.path_for(content_hash)
        if self.quota_bytes is not None and not path.exists():
            usage = self.usage()
            if usage.bytes_used + len(content) > self.quota_bytes:
                raise ValueError("Configured original-PDF storage quota would be exceeded")
        if not path.exists():
            temporary = path.with_suffix(".uploading")
            temporary.write_bytes(content)
            os.replace(temporary, path)
        return path

    def usage(self) -> StorageUsage:
        files = list(self.root.glob("*.pdf"))
        bytes_used = sum(path.stat().st_size for path in files)
        remaining = max(0, self.quota_bytes - bytes_used) if self.quota_bytes is not None else None
        return StorageUsage(
            file_count=len(files),
            bytes_used=bytes_used,
            failed_uploads=0,
            quota_bytes=self.quota_bytes,
            warning_threshold=self.warning_threshold,
            estimated_remaining_bytes=remaining,
        )
