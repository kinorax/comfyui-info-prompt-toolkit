from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import io
import json
import math
import os
from pathlib import Path
import queue
import sqlite3
import threading
from typing import Any, Mapping
from urllib import error as url_error
from urllib import request as url_request

from PIL import Image

from .model_lora_metadata_pipeline import ModelLoraMetadataPipeline, get_shared_metadata_pipeline

try:
    import folder_paths  # type: ignore
except Exception:
    folder_paths = None  # type: ignore[assignment]

try:
    import av  # type: ignore
except Exception:
    av = None  # type: ignore[assignment]


_CACHE_DIR_NAME = "info_prompt_toolkit"
_THUMBNAIL_DB_FILE_NAME = "model_preview_thumbnail.sqlite3"
_THUMBNAIL_WORKER_NAME = "IPT-ThumbnailWorker"
_DOWNLOAD_TIMEOUT_SECONDS = 20
_DOWNLOAD_MAX_BYTES = 32 * 1024 * 1024
_THUMBNAIL_MAX_HEIGHT = 500
_THUMBNAIL_MAX_PIXELS = 512_000
_THUMBNAIL_WEBP_QUALITY = 75
_VIDEO_FRAME_POSITION_RATIO = 0.3
_RETRY_BASE_SECONDS = 120
_RETRY_MAX_SECONDS = 1800
_RETRY_NO_SOURCE_SECONDS = 3600
_MEDIA_TYPE_IMAGE = "image"
_MEDIA_TYPE_VIDEO = "video"
_STATE_READY = "ready"
_STATE_QUEUED = "queued"
_STATE_NO_SOURCE = "no_source"
_STATE_DOWNLOAD_ERROR = "download_error"
_STATE_DECODE_ERROR = "decode_error"
_VALID_STATES = {
    _STATE_READY,
    _STATE_QUEUED,
    _STATE_NO_SOURCE,
    _STATE_DOWNLOAD_ERROR,
    _STATE_DECODE_ERROR,
}

_DDL_SQL = """
CREATE TABLE IF NOT EXISTS model_thumbnail_cache (
    content_id INTEGER PRIMARY KEY,
    source_url TEXT NOT NULL,
    image_mime TEXT NOT NULL,
    image_width INTEGER NOT NULL,
    image_height INTEGER NOT NULL,
    image_size_bytes INTEGER NOT NULL,
    image_blob BLOB NOT NULL,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_thumbnail_state (
    content_id INTEGER PRIMARY KEY,
    source_url TEXT,
    status TEXT NOT NULL CHECK (
        status IN ('ready', 'queued', 'no_source', 'download_error', 'decode_error')
    ),
    fail_count INTEGER NOT NULL DEFAULT 0 CHECK (fail_count >= 0),
    last_error TEXT,
    checked_at TEXT NOT NULL,
    next_retry_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_thumbnail_state_retry
    ON model_thumbnail_state (status, next_retry_at);
"""

try:
    _RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9
    _RESAMPLE_LANCZOS = Image.LANCZOS


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso_utc(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
    return default


def _normalize_image_url(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered.startswith("https://") or lowered.startswith("http://"):
        return text
    return ""


def _parse_civitai_payload(raw_json_text: Any) -> Mapping[str, Any] | None:
    if not isinstance(raw_json_text, str):
        return None
    text = raw_json_text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    if isinstance(parsed, Mapping):
        return parsed
    return None


@dataclass(frozen=True)
class _RepresentativeMedia:
    url: str
    media_type: str


def _rank_media_candidate(item: Mapping[str, Any], index: int) -> tuple[int, int, int]:
    item_type = str(item.get("type") or "").strip().lower()
    availability = str(item.get("availability") or "").strip().lower()
    minor = _as_bool(item.get("minor"), default=False)
    nsfw_level = _as_int(item.get("nsfwLevel"))
    nsfw_value = nsfw_level if nsfw_level is not None else 99

    score = 0
    if item_type == _MEDIA_TYPE_IMAGE:
        score += 0
    elif item_type == _MEDIA_TYPE_VIDEO:
        score += 30
    else:
        score += 100
    if availability != "public":
        score += 40
    if minor:
        score += 20
    if nsfw_level is None:
        score += 10
    elif nsfw_level > 2:
        score += 10 + min(nsfw_level, 10)

    return score, nsfw_value, index


def pick_representative_media_source(raw_json_text: Any) -> _RepresentativeMedia | None:
    payload = _parse_civitai_payload(raw_json_text)
    if payload is None:
        return None
    raw_images = payload.get("images")
    if not isinstance(raw_images, list):
        return None

    candidates: list[tuple[tuple[int, int, int], _RepresentativeMedia]] = []
    for index, raw_item in enumerate(raw_images):
        if not isinstance(raw_item, Mapping):
            continue
        media_url = _normalize_image_url(raw_item.get("url"))
        if not media_url:
            continue
        media_type = str(raw_item.get("type") or "").strip().lower()
        if media_type not in (_MEDIA_TYPE_IMAGE, _MEDIA_TYPE_VIDEO):
            continue
        rank = _rank_media_candidate(raw_item, index)
        candidates.append((rank, _RepresentativeMedia(url=media_url, media_type=media_type)))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def pick_representative_image_url(raw_json_text: Any) -> str:
    selected = pick_representative_media_source(raw_json_text)
    if selected is None:
        return ""
    return selected.url


def resolve_thumbnail_db_path() -> str:
    if folder_paths is not None:
        getter = getattr(folder_paths, "get_user_directory", None)
        if callable(getter):
            try:
                user_dir = getter()
            except Exception:
                user_dir = None
            if isinstance(user_dir, str) and user_dir:
                return str(Path(user_dir) / _CACHE_DIR_NAME / _THUMBNAIL_DB_FILE_NAME)
    return str(Path(__file__).resolve().parents[1] / ".cache" / _CACHE_DIR_NAME / _THUMBNAIL_DB_FILE_NAME)


class ThumbnailCacheDatabase:
    def __init__(self, db_path: str | os.PathLike[str]) -> None:
        self._db_path = Path(db_path)
        self._init_lock = threading.Lock()
        self._initialized = False

    @property
    def db_path(self) -> Path:
        return self._db_path

    def initialize(self) -> None:
        with self._init_lock:
            if self._initialized:
                return
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            con = self._connect()
            try:
                con.executescript(_DDL_SQL)
            finally:
                con.close()
            self._initialized = True

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self._db_path), timeout=5.0, isolation_level=None)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=5000")
        return con

    def get_thumbnail(self, content_id: int) -> dict[str, Any] | None:
        cid = _as_int(content_id)
        if cid is None:
            return None
        self.initialize()
        con = self._connect()
        try:
            row = con.execute(
                """
                SELECT
                    content_id,
                    source_url,
                    image_mime,
                    image_width,
                    image_height,
                    image_size_bytes,
                    image_blob,
                    fetched_at
                FROM model_thumbnail_cache
                WHERE content_id = ?
                LIMIT 1
                """,
                (cid,),
            ).fetchone()
        finally:
            con.close()
        if row is None:
            return None
        blob = row["image_blob"]
        if not isinstance(blob, (bytes, bytearray)):
            return None
        return {
            "content_id": int(row["content_id"]),
            "source_url": str(row["source_url"] or ""),
            "image_mime": str(row["image_mime"] or "image/webp"),
            "image_width": int(row["image_width"] or 0),
            "image_height": int(row["image_height"] or 0),
            "image_size_bytes": int(row["image_size_bytes"] or len(blob)),
            "image_blob": bytes(blob),
            "fetched_at": str(row["fetched_at"] or ""),
        }

    def get_state(self, content_id: int) -> dict[str, Any] | None:
        cid = _as_int(content_id)
        if cid is None:
            return None
        self.initialize()
        con = self._connect()
        try:
            row = con.execute(
                """
                SELECT
                    content_id,
                    source_url,
                    status,
                    fail_count,
                    last_error,
                    checked_at,
                    next_retry_at
                FROM model_thumbnail_state
                WHERE content_id = ?
                LIMIT 1
                """,
                (cid,),
            ).fetchone()
        finally:
            con.close()
        if row is None:
            return None
        status = str(row["status"] or "")
        return {
            "content_id": int(row["content_id"]),
            "source_url": str(row["source_url"] or ""),
            "status": status if status in _VALID_STATES else _STATE_DOWNLOAD_ERROR,
            "fail_count": max(0, int(row["fail_count"] or 0)),
            "last_error": str(row["last_error"] or ""),
            "checked_at": str(row["checked_at"] or ""),
            "next_retry_at": str(row["next_retry_at"] or ""),
        }

    def upsert_state(
        self,
        *,
        content_id: int,
        source_url: str,
        status: str,
        fail_count: int,
        last_error: str,
        checked_at: str,
        next_retry_at: str,
    ) -> None:
        cid = _as_int(content_id)
        status_text = str(status or "").strip()
        if cid is None or status_text not in _VALID_STATES:
            return

        self.initialize()
        con = self._connect()
        try:
            con.execute("BEGIN")
            con.execute(
                """
                INSERT INTO model_thumbnail_state (
                    content_id,
                    source_url,
                    status,
                    fail_count,
                    last_error,
                    checked_at,
                    next_retry_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(content_id) DO UPDATE SET
                    source_url = excluded.source_url,
                    status = excluded.status,
                    fail_count = excluded.fail_count,
                    last_error = excluded.last_error,
                    checked_at = excluded.checked_at,
                    next_retry_at = excluded.next_retry_at
                """,
                (
                    cid,
                    str(source_url or ""),
                    status_text,
                    max(0, int(fail_count)),
                    str(last_error or ""),
                    str(checked_at or ""),
                    str(next_retry_at or ""),
                ),
            )
            con.execute("COMMIT")
        except Exception:
            try:
                con.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            con.close()

    def upsert_thumbnail(
        self,
        *,
        content_id: int,
        source_url: str,
        image_mime: str,
        image_width: int,
        image_height: int,
        image_blob: bytes,
        fetched_at: str,
    ) -> None:
        cid = _as_int(content_id)
        if cid is None:
            return
        blob = bytes(image_blob or b"")
        if not blob:
            return

        self.initialize()
        con = self._connect()
        try:
            con.execute("BEGIN")
            con.execute(
                """
                INSERT INTO model_thumbnail_cache (
                    content_id,
                    source_url,
                    image_mime,
                    image_width,
                    image_height,
                    image_size_bytes,
                    image_blob,
                    fetched_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(content_id) DO UPDATE SET
                    source_url = excluded.source_url,
                    image_mime = excluded.image_mime,
                    image_width = excluded.image_width,
                    image_height = excluded.image_height,
                    image_size_bytes = excluded.image_size_bytes,
                    image_blob = excluded.image_blob,
                    fetched_at = excluded.fetched_at
                """,
                (
                    cid,
                    str(source_url or ""),
                    str(image_mime or "image/webp"),
                    max(1, int(image_width)),
                    max(1, int(image_height)),
                    len(blob),
                    sqlite3.Binary(blob),
                    str(fetched_at or ""),
                ),
            )
            con.execute(
                """
                INSERT INTO model_thumbnail_state (
                    content_id,
                    source_url,
                    status,
                    fail_count,
                    last_error,
                    checked_at,
                    next_retry_at
                )
                VALUES (?, ?, ?, 0, '', ?, '')
                ON CONFLICT(content_id) DO UPDATE SET
                    source_url = excluded.source_url,
                    status = excluded.status,
                    fail_count = excluded.fail_count,
                    last_error = excluded.last_error,
                    checked_at = excluded.checked_at,
                    next_retry_at = excluded.next_retry_at
                """,
                (cid, str(source_url or ""), _STATE_READY, str(fetched_at or "")),
            )
            con.execute("COMMIT")
        except Exception:
            try:
                con.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            con.close()

    def should_queue_download(self, *, content_id: int, source_url: str, now_iso: str) -> bool:
        state = self.get_state(content_id)
        if state is None:
            return True

        current_url = str(state.get("source_url") or "")
        if current_url != str(source_url or ""):
            return True

        status = str(state.get("status") or "")
        if status == _STATE_READY:
            return False
        if status == _STATE_QUEUED:
            return False

        next_retry_at = _parse_iso_utc(state.get("next_retry_at"))
        if next_retry_at is None:
            return True
        now_dt = _parse_iso_utc(now_iso)
        if now_dt is None:
            return True
        return next_retry_at <= now_dt


def _compute_thumbnail_size(width: int, height: int) -> tuple[int, int]:
    src_w = max(1, int(width))
    src_h = max(1, int(height))

    scale_height = min(1.0, _THUMBNAIL_MAX_HEIGHT / float(src_h))
    current_pixels = float(src_w * src_h)
    scale_area = 1.0
    if current_pixels > float(_THUMBNAIL_MAX_PIXELS):
        scale_area = math.sqrt(float(_THUMBNAIL_MAX_PIXELS) / current_pixels)
    scale = min(1.0, scale_height, scale_area)

    dst_w = max(1, int(round(src_w * scale)))
    dst_h = max(1, int(round(src_h * scale)))
    return dst_w, dst_h


@dataclass(frozen=True)
class _ThumbnailTask:
    content_id: int
    source_url: str
    media_type: str


class ModelPreviewThumbnailPipeline:
    def __init__(
        self,
        *,
        metadata_pipeline: ModelLoraMetadataPipeline | None = None,
        db: ThumbnailCacheDatabase | None = None,
    ) -> None:
        self._metadata_pipeline = metadata_pipeline or get_shared_metadata_pipeline(start=False)
        self._db = db or ThumbnailCacheDatabase(resolve_thumbnail_db_path())
        self._queue: queue.Queue[_ThumbnailTask] = queue.Queue()
        self._queued_lock = threading.Lock()
        self._queued_content_ids: set[int] = set()
        self._start_lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        with self._start_lock:
            if self._started:
                return
            self._metadata_pipeline.start()
            self._db.initialize()
            worker = threading.Thread(target=self._worker_loop, name=_THUMBNAIL_WORKER_NAME, daemon=True)
            worker.start()
            self._started = True

    def _snapshot_state(self, content_id: int | None) -> dict[str, Any] | None:
        if content_id is None:
            return None
        state = self._db.get_state(content_id)
        if state is None:
            return None
        return {
            "status": str(state.get("status") or ""),
            "fail_count": max(0, int(state.get("fail_count") or 0)),
            "last_error": str(state.get("last_error") or ""),
            "checked_at": str(state.get("checked_at") or ""),
            "next_retry_at": str(state.get("next_retry_at") or ""),
        }

    def _build_result(
        self,
        *,
        state: str,
        content_id: int | None = None,
        source_url: str = "",
        media_type: str = "",
        detail: str = "",
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"state": str(state or "")}
        if content_id is not None:
            result["content_id"] = content_id
        if source_url:
            result["source_url"] = str(source_url)
        if media_type:
            result["media_type"] = str(media_type)
        if detail:
            result["detail"] = str(detail)
        snapshot = self._snapshot_state(content_id)
        if snapshot is not None:
            result["thumbnail_state"] = snapshot
        return result

    def get_thumbnail_by_relative_path(self, *, folder_name: str, relative_path: str) -> dict[str, Any]:
        info = self._metadata_pipeline.get_civitai_version_payload_by_relative_path(
            folder_name=folder_name,
            relative_path=relative_path,
        )
        if info is None:
            return self._build_result(state="metadata_missing", detail="metadata_missing")

        content_id = _as_int(info.get("content_id"))
        if content_id is None:
            return self._build_result(state="metadata_missing", detail="content_id_missing")

        selected_media = pick_representative_media_source(info.get("civitai_model_version_raw_json"))
        if selected_media is None:
            detail = "no_eligible_media_in_civitai_payload"
            self._mark_no_source(content_id=content_id, reason=detail)
            return self._build_result(
                state="no_source",
                content_id=content_id,
                detail=detail,
            )

        source_url = selected_media.url
        media_type = selected_media.media_type
        if media_type == _MEDIA_TYPE_VIDEO and av is None:
            detail = "video_preview_requires_pyav"
            self._mark_no_source(content_id=content_id, reason=detail)
            return self._build_result(
                state="no_source",
                content_id=content_id,
                source_url=source_url,
                media_type=media_type,
                detail=detail,
            )

        cached = self._db.get_thumbnail(content_id)
        if cached is not None and str(cached.get("source_url") or "") == source_url:
            return {
                "state": "ready",
                "content_id": content_id,
                "source_url": source_url,
                "media_type": media_type,
                "image_mime": cached["image_mime"],
                "image_width": cached["image_width"],
                "image_height": cached["image_height"],
                "image_blob": cached["image_blob"],
                "is_stale": False,
            }

        enqueued = self._enqueue_if_allowed(content_id=content_id, source_url=source_url, media_type=media_type)
        if cached is not None and cached.get("image_blob"):
            return {
                "state": "ready",
                "content_id": content_id,
                "source_url": str(cached.get("source_url") or source_url),
                "media_type": media_type,
                "image_mime": str(cached.get("image_mime") or "image/webp"),
                "image_width": int(cached.get("image_width") or 0),
                "image_height": int(cached.get("image_height") or 0),
                "image_blob": bytes(cached["image_blob"]),
                "is_stale": True,
                "refresh_queued": bool(enqueued),
            }

        return self._build_result(
            state="queued" if enqueued else "pending",
            content_id=content_id,
            source_url=source_url,
            media_type=media_type,
            detail="thumbnail_queued" if enqueued else "thumbnail_pending",
        )

    def _mark_no_source(self, *, content_id: int, reason: str) -> None:
        now = _utc_now_iso()
        retry_at = (
            datetime.now(timezone.utc) + timedelta(seconds=_RETRY_NO_SOURCE_SECONDS)
        ).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        self._db.upsert_state(
            content_id=content_id,
            source_url="",
            status=_STATE_NO_SOURCE,
            fail_count=0,
            last_error=str(reason or "no_source"),
            checked_at=now,
            next_retry_at=retry_at,
        )

    def _enqueue_if_allowed(self, *, content_id: int, source_url: str, media_type: str) -> bool:
        now_iso = _utc_now_iso()
        if not self._db.should_queue_download(content_id=content_id, source_url=source_url, now_iso=now_iso):
            return False

        with self._queued_lock:
            if content_id in self._queued_content_ids:
                return False
            self._queued_content_ids.add(content_id)

        state = self._db.get_state(content_id)
        fail_count = 0
        if state is not None and str(state.get("source_url") or "") == source_url:
            fail_count = max(0, int(state.get("fail_count") or 0))

        self._db.upsert_state(
            content_id=content_id,
            source_url=source_url,
            status=_STATE_QUEUED,
            fail_count=fail_count,
            last_error="",
            checked_at=now_iso,
            next_retry_at="",
        )
        self._queue.put(_ThumbnailTask(content_id=content_id, source_url=source_url, media_type=media_type))
        return True

    def _worker_loop(self) -> None:
        while True:
            try:
                task = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self._process_task(task)
            except Exception as exc:
                self._record_failure(task, _STATE_DOWNLOAD_ERROR, f"unexpected:{exc}")
            finally:
                with self._queued_lock:
                    self._queued_content_ids.discard(task.content_id)
                self._queue.task_done()

    def _process_task(self, task: _ThumbnailTask) -> None:
        try:
            media_bytes = self._download_media_bytes(task.source_url, task.media_type)
        except Exception as exc:
            self._record_failure(task, _STATE_DOWNLOAD_ERROR, str(exc))
            return

        try:
            if task.media_type == _MEDIA_TYPE_VIDEO:
                thumb_bytes, width, height = self._build_video_thumbnail_webp(media_bytes)
            else:
                thumb_bytes, width, height = self._build_image_thumbnail_webp(media_bytes)
        except Exception as exc:
            self._record_failure(task, _STATE_DECODE_ERROR, str(exc))
            return

        fetched_at = _utc_now_iso()
        self._db.upsert_thumbnail(
            content_id=task.content_id,
            source_url=task.source_url,
            image_mime="image/webp",
            image_width=width,
            image_height=height,
            image_blob=thumb_bytes,
            fetched_at=fetched_at,
        )

    def _download_media_bytes(self, source_url: str, media_type: str) -> bytes:
        accept_header = "image/*,video/*"
        if media_type == _MEDIA_TYPE_VIDEO:
            accept_header = "video/*,image/*"
        request = url_request.Request(
            str(source_url),
            headers={
                "accept": accept_header,
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
            },
            method="GET",
        )
        try:
            with url_request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
                status = int(getattr(response, "status", 200))
                if status != 200:
                    raise RuntimeError(f"http_{status}")
                data = response.read(_DOWNLOAD_MAX_BYTES + 1)
        except url_error.HTTPError as exc:
            raise RuntimeError(f"http_{int(exc.code)}") from exc
        except url_error.URLError as exc:
            raise RuntimeError(f"url_error:{exc.reason}") from exc
        except Exception as exc:
            raise RuntimeError(f"download_exception:{exc}") from exc

        if len(data) > _DOWNLOAD_MAX_BYTES:
            raise RuntimeError("download_too_large")
        if not data:
            raise RuntimeError("download_empty")
        return data

    def _build_image_thumbnail_webp(self, image_bytes: bytes) -> tuple[bytes, int, int]:
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.load()
                working = image.copy()
        except Exception as exc:
            raise RuntimeError(f"decode_failed:{exc}") from exc
        return self._encode_thumbnail_webp(working)

    def _resolve_video_stream(self, container: Any) -> Any | None:
        streams = getattr(container, "streams", None)
        if streams is None:
            return None
        for stream in streams:
            if str(getattr(stream, "type", "")).strip().lower() == _MEDIA_TYPE_VIDEO:
                return stream
        return None

    def _estimate_video_frame_count(self, stream: Any) -> int:
        frames = _as_int(getattr(stream, "frames", None))
        if frames is not None and frames > 0:
            return frames

        duration = getattr(stream, "duration", None)
        time_base = getattr(stream, "time_base", None)
        average_rate = getattr(stream, "average_rate", None)
        if duration in (None, 0) or time_base in (None, 0) or average_rate in (None, 0):
            return 0

        try:
            duration_seconds = float(duration * time_base)
            fps = float(average_rate)
        except Exception:
            return 0
        if not math.isfinite(duration_seconds) or not math.isfinite(fps):
            return 0

        estimated = int(duration_seconds * fps)
        return estimated if estimated > 0 else 0

    def _decode_video_frame(self, *, container: Any, stream: Any, target_index: int) -> Any | None:
        stream_index = _as_int(getattr(stream, "index", None))
        if stream_index is None:
            stream_index = 0
        index = 0
        for frame in container.decode(video=stream_index):
            if index >= target_index:
                return frame
            index += 1
        return None

    def _build_video_thumbnail_webp(self, video_bytes: bytes) -> tuple[bytes, int, int]:
        if av is None:
            raise RuntimeError("pyav_not_available")

        try:
            with av.open(io.BytesIO(video_bytes), mode="r") as container:
                stream = self._resolve_video_stream(container)
                if stream is None:
                    raise RuntimeError("video_stream_missing")

                frame_count = self._estimate_video_frame_count(stream)
                target_index = 0
                if frame_count > 1:
                    target_index = max(0, min(frame_count - 1, int(frame_count * _VIDEO_FRAME_POSITION_RATIO)))

                frame = self._decode_video_frame(container=container, stream=stream, target_index=target_index)
                if frame is None and target_index > 0:
                    try:
                        container.seek(0)
                    except Exception:
                        pass
                    frame = self._decode_video_frame(container=container, stream=stream, target_index=0)
                if frame is None:
                    raise RuntimeError("video_read_failed")

                image = frame.to_image()
        except Exception as exc:
            raise RuntimeError(f"video_decode_failed:{exc}") from exc

        return self._encode_thumbnail_webp(image)

    def _encode_thumbnail_webp(self, image: Image.Image) -> tuple[bytes, int, int]:
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")
        else:
            image = image.copy()

        dst_w, dst_h = _compute_thumbnail_size(image.width, image.height)
        if image.size != (dst_w, dst_h):
            image = image.resize((dst_w, dst_h), _RESAMPLE_LANCZOS)

        buffer = io.BytesIO()
        try:
            image.save(buffer, format="WEBP", quality=_THUMBNAIL_WEBP_QUALITY, method=4)
        except Exception as exc:
            raise RuntimeError(f"encode_failed:{exc}") from exc
        payload = buffer.getvalue()
        if not payload:
            raise RuntimeError("encode_empty")
        return payload, dst_w, dst_h

    def _record_failure(self, task: _ThumbnailTask, status: str, message: str) -> None:
        safe_status = status if status in {_STATE_DOWNLOAD_ERROR, _STATE_DECODE_ERROR} else _STATE_DOWNLOAD_ERROR
        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")

        previous = self._db.get_state(task.content_id)
        prev_fail_count = 0
        if previous is not None and str(previous.get("source_url") or "") == task.source_url:
            prev_fail_count = max(0, int(previous.get("fail_count") or 0))
        fail_count = prev_fail_count + 1

        capped_power = max(0, min(fail_count - 1, 4))
        delay_seconds = min(_RETRY_MAX_SECONDS, _RETRY_BASE_SECONDS * (2 ** capped_power))
        next_retry_iso = (
            now_dt + timedelta(seconds=delay_seconds)
        ).replace(microsecond=0).isoformat().replace("+00:00", "Z")

        self._db.upsert_state(
            content_id=task.content_id,
            source_url=task.source_url,
            status=safe_status,
            fail_count=fail_count,
            last_error=str(message or ""),
            checked_at=now_iso,
            next_retry_at=next_retry_iso,
        )


_SHARED_THUMBNAIL_PIPELINE: ModelPreviewThumbnailPipeline | None = None
_SHARED_THUMBNAIL_PIPELINE_LOCK = threading.Lock()


def get_shared_thumbnail_pipeline(
    *,
    start: bool = True,
    metadata_pipeline: ModelLoraMetadataPipeline | None = None,
) -> ModelPreviewThumbnailPipeline:
    global _SHARED_THUMBNAIL_PIPELINE
    with _SHARED_THUMBNAIL_PIPELINE_LOCK:
        if _SHARED_THUMBNAIL_PIPELINE is None:
            base_metadata_pipeline = metadata_pipeline or get_shared_metadata_pipeline(start=start)
            _SHARED_THUMBNAIL_PIPELINE = ModelPreviewThumbnailPipeline(
                metadata_pipeline=base_metadata_pipeline
            )
        pipeline = _SHARED_THUMBNAIL_PIPELINE
    if start:
        pipeline.start()
    return pipeline
