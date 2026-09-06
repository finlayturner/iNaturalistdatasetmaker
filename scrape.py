#!/usr/bin/env python3
from __future__ import annotations

"""
iNaturalist image scraper, YOLO review tool, and dataset builder.

Run:
    python3 scrape.py

The normal entry point opens the GUI. The hidden --review-session mode is used
by the GUI to launch the OpenCV reviewer in a separate Python process.
"""

import argparse
import csv
import json
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import cv2
import requests
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

# ---------------- Storage / session management ----------------
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")
META_FILE = ".inat_yolo_session.json"
STATE_FILE = ".inat_yolo_state.json"
WORKSPACE_FILE = ".inat_yolo_workspace.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("._-") or "taxon"


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def ensure_session_dirs(path: Path) -> Dict[str, Path]:
    dirs = {
        "base": path,
        "images": path / "images",
        "labels": path / "labels",
        "details": path / "details",
        "rejected": path / "rejected",
        "used_images": path / "used_images",  # legacy compatibility
    }
    path.mkdir(parents=True, exist_ok=True)
    for key in ("images", "labels", "details", "rejected"):
        dirs[key].mkdir(parents=True, exist_ok=True)
    return dirs


@dataclass
class SessionInfo:
    path: Path
    taxon_id: int
    species_name: str
    scientific_name: str = ""
    rank: str = ""
    model_class_name: str = ""

    @property
    def display_name(self) -> str:
        sci = f" — {self.scientific_name}" if self.scientific_name and self.scientific_name != self.species_name else ""
        return f"{self.species_name}{sci} [taxon {self.taxon_id}]"

    def to_meta(self) -> dict:
        existing = read_json(self.path / META_FILE, {})
        created = existing.get("created_at") or utc_now()
        return {
            "version": 1,
            "taxon_id": self.taxon_id,
            "species_name": self.species_name,
            "scientific_name": self.scientific_name,
            "rank": self.rank,
            "model_class_name": self.model_class_name,
            "created_at": created,
            "updated_at": utc_now(),
        }


class Workspace:
    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "datasets").mkdir(exist_ok=True)

    @property
    def config_path(self) -> Path:
        return self.root / WORKSPACE_FILE

    def _config(self) -> dict:
        return read_json(self.config_path, {"external_sessions": []})

    def _save_config(self, config: dict) -> None:
        write_json(self.config_path, config)

    def register_external(self, path: Path) -> None:
        path = path.expanduser().resolve()
        config = self._config()
        values = [str(Path(p).expanduser().resolve()) for p in config.get("external_sessions", [])]
        if str(path) not in values:
            values.append(str(path))
        config["external_sessions"] = values
        self._save_config(config)

    def create_or_update_session(
        self,
        taxon_id: int,
        species_name: str,
        scientific_name: str = "",
        rank: str = "",
        model_class_name: str = "",
    ) -> SessionInfo:
        # Prefer an existing session with the same taxon ID so repeated downloads
        # resume into the same folder even if the display name changes.
        for session in self.discover_sessions():
            if session.taxon_id == int(taxon_id):
                session.species_name = species_name or session.species_name
                session.scientific_name = scientific_name or session.scientific_name
                session.rank = rank or session.rank
                if model_class_name:
                    session.model_class_name = model_class_name
                ensure_session_dirs(session.path)
                write_json(session.path / META_FILE, session.to_meta())
                return session

        folder = self.root / f"{int(taxon_id)}_{slugify(species_name)}"
        ensure_session_dirs(folder)
        session = SessionInfo(
            path=folder,
            taxon_id=int(taxon_id),
            species_name=species_name or f"Taxon {taxon_id}",
            scientific_name=scientific_name,
            rank=rank,
            model_class_name=model_class_name,
        )
        write_json(folder / META_FILE, session.to_meta())
        return session

    def register_existing(
        self,
        folder: Path,
        taxon_id: int,
        species_name: str,
        scientific_name: str = "",
        rank: str = "",
    ) -> SessionInfo:
        folder = folder.expanduser().resolve()
        ensure_session_dirs(folder)
        existing = read_json(folder / META_FILE, {})
        session = SessionInfo(
            path=folder,
            taxon_id=int(taxon_id),
            species_name=species_name or existing.get("species_name") or f"Taxon {taxon_id}",
            scientific_name=scientific_name or existing.get("scientific_name", ""),
            rank=rank or existing.get("rank", ""),
            model_class_name=existing.get("model_class_name", ""),
        )
        write_json(folder / META_FILE, session.to_meta())
        try:
            folder.relative_to(self.root)
        except ValueError:
            self.register_external(folder)
        self.bootstrap_legacy_review_state(session)
        return session

    def discover_sessions(self) -> List[SessionInfo]:
        candidates: List[Path] = []
        for p in self.root.iterdir():
            if p.is_dir() and p.name != "datasets":
                if (p / META_FILE).exists() or (p / "images").is_dir():
                    candidates.append(p.resolve())

        for raw in self._config().get("external_sessions", []):
            p = Path(raw).expanduser()
            if p.is_dir():
                candidates.append(p.resolve())

        seen = set()
        sessions: List[SessionInfo] = []
        for p in candidates:
            if p in seen:
                continue
            seen.add(p)
            meta = read_json(p / META_FILE, {})
            taxon_id = meta.get("taxon_id")
            if taxon_id is None:
                m = re.match(r"^(\d+)(?:_|$)", p.name)
                if m:
                    taxon_id = int(m.group(1))
                elif p.name.isdigit():
                    taxon_id = int(p.name)
                else:
                    continue
            try:
                taxon_id = int(taxon_id)
            except (TypeError, ValueError):
                continue
            sessions.append(
                SessionInfo(
                    path=p,
                    taxon_id=taxon_id,
                    species_name=meta.get("species_name") or f"Taxon {taxon_id}",
                    scientific_name=meta.get("scientific_name", ""),
                    rank=meta.get("rank", ""),
                    model_class_name=meta.get("model_class_name", ""),
                )
            )
        sessions.sort(key=lambda s: (s.species_name.lower(), s.taxon_id))
        return sessions

    def save_session(self, session: SessionInfo) -> None:
        ensure_session_dirs(session.path)
        write_json(session.path / META_FILE, session.to_meta())

    @staticmethod
    def load_state(session: SessionInfo) -> dict:
        return read_json(session.path / STATE_FILE, {"review": {}})

    @staticmethod
    def save_state(session: SessionInfo, state: dict) -> None:
        write_json(session.path / STATE_FILE, state)

    def bootstrap_legacy_review_state(self, session: SessionInfo) -> None:
        """Infer review status from labels / rejected copies without overwriting state."""
        dirs = ensure_session_dirs(session.path)
        state = self.load_state(session)
        review = state.setdefault("review", {})
        changed = False
        for image in iter_images(dirs["images"]):
            base = image.stem
            if base in review:
                continue
            label = dirs["labels"] / f"{base}.txt"
            rejected = find_image_by_stem(dirs["rejected"], base)
            if label.exists():
                review[base] = {"status": "approved", "updated_at": utc_now(), "legacy_inferred": True}
                changed = True
            elif rejected:
                review[base] = {"status": "rejected", "updated_at": utc_now(), "legacy_inferred": True}
                changed = True
        if changed:
            self.save_state(session, state)

    def stats(self, session: SessionInfo) -> dict:
        dirs = ensure_session_dirs(session.path)
        self.bootstrap_legacy_review_state(session)
        state = self.load_state(session)
        review = state.get("review", {})
        images = list(iter_images(dirs["images"]))
        counts = {"images": len(images), "approved": 0, "rejected": 0, "excluded": 0, "pending": 0}
        for image in images:
            status = review.get(image.stem, {}).get("status", "pending")
            if status in counts:
                counts[status] += 1
            else:
                counts["pending"] += 1
        return counts


def iter_images(folder: Path) -> Iterable[Path]:
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS), key=lambda p: p.name.lower())


def find_image_by_stem(folder: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def open_in_file_manager(path: Path) -> None:
    path = Path(path).resolve()
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])

# ---------------- iNaturalist API / scraper ----------------
API_ROOT = "https://api.inaturalist.org/v2"
OBSERVATIONS_URL = f"{API_ROOT}/observations"
TAXA_AUTOCOMPLETE_URL = f"{API_ROOT}/taxa/autocomplete"
TAXON_URL = f"{API_ROOT}/taxa"
DEFAULT_PER_PAGE = 200
DEFAULT_IMAGE_SIZE = "large"

ALLOW_LICENSES = {
    "CC0",
    "CC-BY",
    "CC-BY-NC",
    "CC-BY-SA",
    "CC-BY-ND",
    "CC-BY-NC-SA",
    "CC-BY-NC-ND",
}
_COPYRIGHT_PLACEHOLDER_MARKER = "copyright-infringement"


def normalize_license_code(value: Optional[str]) -> str:
    if not value:
        return ""
    return value.strip().upper().replace("_", "-")


def is_license_allowed(license_code: Optional[str], include_all_rights_reserved: bool) -> bool:
    if include_all_rights_reserved:
        return True
    code = normalize_license_code(license_code)
    return bool(code) and code in ALLOW_LICENSES


def resize_photo_url(url: str, size: str = DEFAULT_IMAGE_SIZE) -> str:
    if not url:
        return url
    parsed = urlparse(url)
    path = parsed.path
    path = re.sub(r"/(square|thumb|small|medium|large|original)(?=\.[a-zA-Z0-9]+$)", f"/{size}", path)
    path = re.sub(r"/(square|thumb|small|medium|large|original)(?=/)", f"/{size}", path)
    path = re.sub(r"\.(square|thumb|small|medium|large|original)(?=\.[a-zA-Z0-9]+$)", f".{size}", path)
    return parsed._replace(path=path).geturl()


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name.strip("._-") or "image"


def guess_extension(url: str) -> str:
    ext = os.path.splitext(urlparse(url).path)[1].lower().lstrip(".")
    return ext if ext in {"jpg", "jpeg", "png", "webp", "tif", "tiff"} else "jpg"


def checkpoint_path(session_dir: Path) -> Path:
    return session_dir / "checkpoint.txt"


def load_checkpoint(session_dir: Path) -> int:
    try:
        return int(checkpoint_path(session_dir).read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError, OSError):
        return 0


def save_checkpoint(session_dir: Path, obs_id: int) -> None:
    checkpoint_path(session_dir).write_text(str(int(obs_id)), encoding="utf-8")


def ledger_path(session_dir: Path) -> Path:
    return session_dir / ".downloaded_photos.jsonl"


def load_downloaded_photo_ids(session_dir: Path) -> set[int]:
    values: set[int] = set()
    path = ledger_path(session_dir)
    if not path.exists():
        return values
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                if item.get("photo_id") is not None:
                    values.add(int(item["photo_id"]))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return values


def append_download_ledger(session_dir: Path, record: dict) -> None:
    with ledger_path(session_dir).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_attribution_text(observation: Dict, photo: Dict, source_url: str) -> str:
    creator = (
        photo.get("attribution")
        or observation.get("user", {}).get("login")
        or observation.get("user_login")
        or "unknown"
    )
    license_code = normalize_license_code(
        photo.get("license_code") or observation.get("photo_license_code") or observation.get("license_code")
    )
    lines = [
        f"Source page:           {source_url}",
        f"Observation ID:        {observation.get('id', '')}",
        f"Observation URL:       https://www.inaturalist.org/observations/{observation.get('id', '')}",
        f"Image URL:             {photo.get('url', '')}",
        f"Creator / attribution: {creator}",
        f"License code:          {license_code or 'not provided'}",
    ]
    if license_code in {"CC-BY", "CC-BY-NC", "CC-BY-SA", "CC-BY-ND", "CC-BY-NC-SA", "CC-BY-NC-ND"}:
        lines.append("Credit note: attribution required for reuse under the listed Creative Commons license.")
    elif not license_code:
        lines.append("Credit note: no license code was provided; treat as permission required.")
    else:
        lines.append("Credit note: review the license terms before reuse.")
    return "\n".join(lines) + "\n"


class INaturalistClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "inat-yolo-manager/1.0",
            "Accept": "application/json,text/plain,*/*",
        })

    def search_taxa(self, query: str, per_page: int = 30) -> List[dict]:
        params = {
            "q": query,
            "per_page": max(1, min(int(per_page), 100)),
            "fields": "id,name,preferred_common_name,english_common_name,rank,is_active,matched_term",
        }
        r = self.session.get(TAXA_AUTOCOMPLETE_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", []) if isinstance(data, dict) else data
        out = []
        for taxon in results or []:
            if not isinstance(taxon, dict) or taxon.get("id") is None:
                continue
            scientific = taxon.get("name") or ""
            common = taxon.get("preferred_common_name") or taxon.get("english_common_name") or taxon.get("matched_term") or scientific
            out.append({
                "id": int(taxon["id"]),
                "species_name": common or scientific or f"Taxon {taxon['id']}",
                "scientific_name": scientific,
                "rank": taxon.get("rank") or "",
                "is_active": bool(taxon.get("is_active", True)),
            })
        return out

    def get_taxon(self, taxon_id: int) -> Optional[dict]:
        params = {"fields": "id,name,preferred_common_name,english_common_name,rank,is_active"}
        r = self.session.get(f"{TAXON_URL}/{int(taxon_id)}", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", []) if isinstance(data, dict) else data
        if isinstance(results, list):
            taxon = results[0] if results else None
        elif isinstance(results, dict):
            taxon = results
        else:
            taxon = data if isinstance(data, dict) and data.get("id") else None
        if not taxon:
            return None
        scientific = taxon.get("name") or ""
        common = taxon.get("preferred_common_name") or taxon.get("english_common_name") or scientific
        return {
            "id": int(taxon.get("id", taxon_id)),
            "species_name": common or scientific or f"Taxon {taxon_id}",
            "scientific_name": scientific,
            "rank": taxon.get("rank") or "",
            "is_active": bool(taxon.get("is_active", True)),
        }

    def get_observations(self, taxon_id: int, per_page: int, id_above: int = 0) -> List[dict]:
        params: Dict = {
            "taxon_id": int(taxon_id),
            "has": "photos",
            "per_page": max(1, min(int(per_page), 200)),
            "order_by": "id",
            "order": "asc",
            "fields": (
                "id,uri,user.login,user_login,"
                "photos.url,photos.attribution,photos.license_code,photos.original_url,photos.id"
            ),
        }
        if id_above > 0:
            params["id_above"] = int(id_above)
        r = self.session.get(OBSERVATIONS_URL, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            return data.get("results") or []
        if isinstance(data, list):
            return data
        return []

    def download_file(self, url: str, output_path: Path) -> None:
        tmp = output_path.with_suffix(output_path.suffix + ".part")
        try:
            with self.session.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 128):
                        if chunk:
                            f.write(chunk)
            tmp.replace(output_path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def download_images(
        self,
        session_dir: Path,
        taxon_id: int,
        images_to_add: int,
        image_size: str = DEFAULT_IMAGE_SIZE,
        include_all_rights_reserved: bool = False,
        delay: float = 0.5,
        reset_checkpoint: bool = False,
        progress: Optional[Callable[[dict], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> dict:
        session_dir = Path(session_dir)
        images_dir = session_dir / "images"
        details_dir = session_dir / "details"
        used_dir = session_dir / "used_images"
        images_dir.mkdir(parents=True, exist_ok=True)
        details_dir.mkdir(parents=True, exist_ok=True)

        target = max(0, int(images_to_add))
        if target == 0:
            return {"downloaded": 0, "seen": 0, "checkpoint": load_checkpoint(session_dir), "cancelled": False}

        if reset_checkpoint:
            id_above = 0
            try:
                checkpoint_path(session_dir).unlink()
            except FileNotFoundError:
                pass
        else:
            id_above = load_checkpoint(session_dir)

        downloaded_ids = load_downloaded_photo_ids(session_dir)
        downloaded = 0
        observations_seen = 0
        cancelled = False

        def emit(message: str, **extra):
            if progress:
                progress({
                    "message": message,
                    "downloaded": downloaded,
                    "target": target,
                    "observations_seen": observations_seen,
                    "checkpoint": id_above,
                    **extra,
                })

        emit(f"Starting at observation cursor {id_above}.")

        while downloaded < target:
            if cancel_event and cancel_event.is_set():
                cancelled = True
                break
            observations = self.get_observations(taxon_id, DEFAULT_PER_PAGE, id_above=id_above)
            if not observations:
                emit("No more observations returned by iNaturalist.")
                break

            for observation in observations:
                if cancel_event and cancel_event.is_set():
                    cancelled = True
                    break
                obs_id = int(observation.get("id") or 0)
                observations_seen += 1
                photos = observation.get("photos") or []
                source_url = observation.get("uri") or f"https://www.inaturalist.org/observations/{obs_id}"

                fully_processed_observation = True
                for idx, photo in enumerate(photos, start=1):
                    if downloaded >= target:
                        fully_processed_observation = False
                        break
                    if cancel_event and cancel_event.is_set():
                        cancelled = True
                        fully_processed_observation = False
                        break
                    if not isinstance(photo, dict):
                        continue
                    if not is_license_allowed(photo.get("license_code"), include_all_rights_reserved):
                        continue
                    image_url = photo.get("url") or photo.get("original_url") or ""
                    image_url = resize_photo_url(image_url, image_size) if image_url else ""
                    if not image_url or _COPYRIGHT_PLACEHOLDER_MARKER in image_url:
                        continue

                    photo_id = photo.get("id")
                    try:
                        photo_id_int = int(photo_id) if photo_id is not None else None
                    except (TypeError, ValueError):
                        photo_id_int = None

                    base_name = sanitize_filename(f"obs{obs_id}_photo{idx}")
                    ext = guess_extension(image_url)
                    image_path = images_dir / f"{base_name}.{ext}"
                    details_path = details_dir / f"{base_name}.txt"
                    used_path = used_dir / f"{base_name}.{ext}"

                    already_downloaded = (
                        image_path.exists()
                        or (used_dir.is_dir() and used_path.exists())
                        or (photo_id_int is not None and photo_id_int in downloaded_ids)
                    )
                    if already_downloaded:
                        continue

                    try:
                        self.download_file(image_url, image_path)
                    except requests.RequestException as exc:
                        emit(f"Skipped {base_name}: {exc}", error=True)
                        continue

                    details = build_attribution_text(observation, {**photo, "url": image_url}, source_url)
                    details_path.write_text(details, encoding="utf-8")
                    record = {
                        "photo_id": photo_id_int,
                        "observation_id": obs_id,
                        "filename": image_path.name,
                        "url": image_url,
                    }
                    append_download_ledger(session_dir, record)
                    if photo_id_int is not None:
                        downloaded_ids.add(photo_id_int)
                    downloaded += 1
                    emit(f"Downloaded {image_path.name}")
                    if delay > 0 and downloaded < target:
                        time.sleep(delay)

                # Only advance beyond an observation when every one of its photos
                # has been considered. If target/cancel interrupts mid-observation,
                # keep the previous cursor; a resumed run re-fetches this observation
                # and skips already saved photos.
                if fully_processed_observation:
                    id_above = max(id_above, obs_id)
                    save_checkpoint(session_dir, id_above)
                else:
                    break

            if cancelled or downloaded >= target:
                break

        emit("Download finished." if not cancelled else "Download cancelled.", done=True)
        return {
            "downloaded": downloaded,
            "seen": observations_seen,
            "checkpoint": id_above,
            "cancelled": cancelled,
        }

# ---------------- OpenCV / YOLO reviewer ----------------
CONF_THRESHOLD = 0.4
MIN_BOX_SIZE = 4
WINDOW_NAME = "iNaturalist YOLO Label Review"
SPECIES_LABEL_ID = 0
PERSON_LABEL_ID = 1

BOX_COLORS = {
    SPECIES_LABEL_ID: (0, 255, 0),
    PERSON_LABEL_ID: (0, 165, 255),
}


def load_ultralytics_model(model_path: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed. Run: pip install -r requirements.txt") from exc
    return YOLO(model_path)


def model_class_names(model_path: str) -> List[str]:
    model = load_ultralytics_model(model_path)
    names = model.names
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names)]
    return [str(x) for x in names]


def _session_from_path(path: Path) -> SessionInfo:
    meta = read_json(path / META_FILE, {})
    taxon_id = int(meta.get("taxon_id") or 0)
    return SessionInfo(
        path=path,
        taxon_id=taxon_id,
        species_name=meta.get("species_name") or f"Taxon {taxon_id}",
        scientific_name=meta.get("scientific_name", ""),
        rank=meta.get("rank", ""),
        model_class_name=meta.get("model_class_name", ""),
    )


def make_mouse_callback(state):
    """Left-drag adds a box; right-click removes the smallest box under cursor."""
    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["dragging"] = True
            state["drag_start"] = (x, y)
            state["drag_current"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            state["drag_current"] = (x, y)
            state["redraw"] = True
        elif event == cv2.EVENT_LBUTTONUP and state["dragging"]:
            state["dragging"] = False
            x1, x2 = sorted((state["drag_start"][0], x))
            y1, y2 = sorted((state["drag_start"][1], y))
            if (x2 - x1) >= MIN_BOX_SIZE and (y2 - y1) >= MIN_BOX_SIZE:
                state["boxes"].append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "label_id": state["active_class"],
                    "conf": None,
                    "manual": True,
                    "source": None,
                })
            state["redraw"] = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            best_idx, best_area = None, None
            for i, b in enumerate(state["boxes"]):
                if b["x1"] <= x <= b["x2"] and b["y1"] <= y <= b["y2"]:
                    area = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
                    if best_area is None or area < best_area:
                        best_idx, best_area = i, area
            if best_idx is not None:
                del state["boxes"][best_idx]
                state["redraw"] = True
    return callback


def render(orig_img, state, label_names: Dict[int, str]):
    img = orig_img.copy()
    for b in state["boxes"]:
        color = BOX_COLORS.get(b["label_id"], (255, 255, 255))
        cv2.rectangle(img, (b["x1"], b["y1"]), (b["x2"], b["y2"]), color, 2)
        name = label_names.get(b["label_id"], str(b["label_id"]))
        tag = name if b["manual"] or b.get("conf") is None else f"{name} {b['conf']:.2f}"
        cv2.putText(img, tag, (b["x1"], max(b["y1"] - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if state["dragging"] and state["drag_start"] and state["drag_current"]:
        cv2.rectangle(img, state["drag_start"], state["drag_current"], BOX_COLORS[state["active_class"]], 1)

    active_name = label_names[state["active_class"]]
    info = (
        f"active: {active_name} | 1=species 2=person | L-drag add | R-click delete | "
        "Enter=approve  s=reject  d=exclude  r=reset detections  q=quit"
    )
    cv2.rectangle(img, (0, 0), (img.shape[1], 25), (0, 0, 0), -1)
    cv2.putText(img, info, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return img


def load_existing_label(label_path: Path, w: int, h: int) -> List[dict]:
    boxes = []
    if not label_path.exists():
        return boxes
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) != 5:
            continue
        try:
            label_id = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:])
        except ValueError:
            continue
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        boxes.append({
            "x1": max(0, x1), "y1": max(0, y1), "x2": min(w - 1, x2), "y2": min(h - 1, y2),
            "label_id": label_id,
            "conf": None,
            "manual": True,
            "source": "existing label",
        })
    return boxes


def detections_for_image(model, img, target_model_id: int, person_model_id: Optional[int], conf_threshold: float) -> List[dict]:
    model_id_to_label_id = {target_model_id: SPECIES_LABEL_ID}
    if person_model_id is not None:
        model_id_to_label_id[person_model_id] = PERSON_LABEL_ID
    results = model(img, verbose=False)[0]
    boxes = []
    for box in results.boxes:
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        if cls not in model_id_to_label_id or conf < conf_threshold:
            continue
        x1, y1, x2, y2 = box.xyxy[0]
        boxes.append({
            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
            "label_id": model_id_to_label_id[cls],
            "conf": conf,
            "manual": False,
            "source": str(model.names[cls]),
        })
    return boxes


def write_yolo_label(label_path: Path, boxes: List[dict], w: int, h: int) -> None:
    with label_path.open("w", encoding="utf-8") as f:
        for b in boxes:
            xc = (b["x1"] + b["x2"]) / 2 / w
            yc = (b["y1"] + b["y2"]) / 2 / h
            bw = (b["x2"] - b["x1"]) / w
            bh = (b["y2"] - b["y1"]) / h
            f.write(f"{b['label_id']} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}\n")


def review_session(
    session_path: Path,
    model_path: str,
    model_class_name: str,
    conf_threshold: float = CONF_THRESHOLD,
    include_processed: bool = False,
) -> dict:
    session_path = Path(session_path).resolve()
    session = _session_from_path(session_path)
    dirs = ensure_session_dirs(session_path)
    workspace = Workspace(session_path.parent)
    workspace.bootstrap_legacy_review_state(session)
    state_file = session_path / ".inat_yolo_state.json"
    state = read_json(state_file, {"review": {}})
    review_state = state.setdefault("review", {})

    model = load_ultralytics_model(model_path)
    model_names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}
    by_name = {str(name).strip().lower(): int(cid) for cid, name in model_names.items()}
    target_model_id = by_name.get(model_class_name.strip().lower())
    if target_model_id is None:
        available = ", ".join(str(v) for v in model_names.values())
        raise ValueError(f"Model class '{model_class_name}' was not found. Available classes: {available}")
    person_model_id = by_name.get("person")

    label_names = {SPECIES_LABEL_ID: session.species_name}
    if person_model_id is not None:
        label_names[PERSON_LABEL_ID] = "person"

    images = list(iter_images(dirs["images"]))
    pending = []
    for image_path in images:
        status = review_state.get(image_path.stem, {}).get("status")
        if include_processed or status not in {"approved", "rejected", "excluded"}:
            pending.append(image_path)

    reviewed = approved = rejected = excluded = 0
    if not pending:
        print("No pending images to review.")
        return {"reviewed": 0, "approved": 0, "rejected": 0, "excluded": 0}

    print("Controls:")
    print("  left-click + drag = draw a new box")
    print("  right-click = delete smallest box under cursor")
    print("  1 = species, 2 = person (if available)")
    print("  Enter = save + approve, s = reject, d = exclude, r = reset detections, q = quit")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        for index, img_path in enumerate(pending, start=1):
            img = cv2.imread(str(img_path))
            if img is None:
                print("Could not read image, skipping:", img_path)
                continue
            h, w = img.shape[:2]
            label_path = dirs["labels"] / f"{img_path.stem}.txt"

            def fresh_boxes():
                if label_path.exists() and include_processed:
                    return load_existing_label(label_path, w, h)
                return detections_for_image(model, img, target_model_id, person_model_id, conf_threshold)

            state_ui = {
                "boxes": fresh_boxes(),
                "active_class": SPECIES_LABEL_ID,
                "dragging": False,
                "drag_start": None,
                "drag_current": None,
                "redraw": True,
            }
            cv2.setWindowTitle(WINDOW_NAME, f"{WINDOW_NAME} — {index}/{len(pending)} — {img_path.name}")
            cv2.setMouseCallback(WINDOW_NAME, make_mouse_callback(state_ui))

            key = None
            while True:
                if state_ui["redraw"]:
                    cv2.imshow(WINDOW_NAME, render(img, state_ui, label_names))
                    state_ui["redraw"] = False
                key = cv2.waitKey(20) & 0xFF
                if key == ord("1"):
                    state_ui["active_class"] = SPECIES_LABEL_ID
                    state_ui["redraw"] = True
                elif key == ord("2") and person_model_id is not None:
                    state_ui["active_class"] = PERSON_LABEL_ID
                    state_ui["redraw"] = True
                elif key == ord("r"):
                    state_ui["boxes"] = detections_for_image(model, img, target_model_id, person_model_id, conf_threshold)
                    state_ui["redraw"] = True
                elif key in {ord("q"), ord("s"), ord("d"), 13, 10}:
                    break

            if key == ord("q"):
                print("Review stopped by user.")
                break

            reviewed += 1
            if key == ord("s"):
                rejected_copy = dirs["rejected"] / img_path.name
                shutil.copy2(img_path, rejected_copy)
                review_state[img_path.stem] = {"status": "rejected", "updated_at": utc_now()}
                rejected += 1
            elif key == ord("d"):
                try:
                    label_path.unlink()
                except FileNotFoundError:
                    pass
                review_state[img_path.stem] = {"status": "excluded", "updated_at": utc_now()}
                excluded += 1
            else:
                write_yolo_label(label_path, state_ui["boxes"], w, h)
                review_state[img_path.stem] = {
                    "status": "approved",
                    "updated_at": utc_now(),
                    "boxes": len(state_ui["boxes"]),
                }
                approved += 1
            write_json(state_file, state)
    finally:
        cv2.destroyAllWindows()

    return {"reviewed": reviewed, "approved": approved, "rejected": rejected, "excluded": excluded}



# ---------------- YOLO dataset builder ----------------
def _approved_pairs(session: SessionInfo) -> List[Tuple[Path, Path, Path | None]]:
    dirs = ensure_session_dirs(session.path)
    state = Workspace.load_state(session)
    review = state.get("review", {})
    pairs = []
    for image in iter_images(dirs["images"]):
        label = dirs["labels"] / f"{image.stem}.txt"
        if not label.exists():
            continue
        status = review.get(image.stem, {}).get("status")
        # Existing legacy labels count as approved even if no state file existed.
        if status not in (None, "approved"):
            continue
        details = dirs["details"] / f"{image.stem}.txt"
        pairs.append((image, label, details if details.exists() else None))
    return pairs


def _allocate_splits(items: list, train_pct: int, val_pct: int, rng: random.Random) -> Dict[str, list]:
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    train_n = int(round(n * train_pct / 100))
    val_n = int(round(n * val_pct / 100))
    if train_n + val_n > n:
        val_n = max(0, n - train_n)
    # Ensure training gets at least one item for non-empty classes.
    train_n = max(1, min(train_n, n))
    if train_n + val_n > n:
        val_n = max(0, n - train_n)
    return {
        "train": items[:train_n],
        "val": items[train_n:train_n + val_n],
        "test": items[train_n + val_n:],
    }


def _remap_label(source: Path, dest: Path, species_global_id: int, person_global_id: int | None) -> tuple[int, int]:
    out_lines = []
    species_boxes = 0
    person_boxes = 0
    for raw in source.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) != 5:
            continue
        try:
            local_id = int(float(parts[0]))
        except ValueError:
            continue
        if local_id == 0:
            global_id = species_global_id
            species_boxes += 1
        elif local_id == 1 and person_global_id is not None:
            global_id = person_global_id
            person_boxes += 1
        else:
            continue
        out_lines.append(" ".join([str(global_id)] + parts[1:]))
    dest.write_text(("\n".join(out_lines) + "\n") if out_lines else "", encoding="utf-8")
    return species_boxes, person_boxes


def build_dataset(
    workspace: Workspace,
    sessions: Sequence[SessionInfo],
    dataset_name: str,
    train_pct: int = 80,
    val_pct: int = 20,
    include_person: bool = True,
    seed: int = 42,
    overwrite: bool = False,
) -> dict:
    train_pct = int(train_pct)
    val_pct = int(val_pct)
    if train_pct < 0 or val_pct < 0 or train_pct + val_pct > 100:
        raise ValueError("Train and validation percentages must be >= 0 and total no more than 100.")
    if not sessions:
        raise ValueError("Select at least one session.")

    dataset_slug = slugify(dataset_name)
    out = workspace.root / "datasets" / dataset_slug
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"Dataset already exists: {out}")
        shutil.rmtree(out)

    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        (out / "attribution" / split).mkdir(parents=True, exist_ok=True)

    # Taxon ID is the stable class key. This prevents accidental merging of
    # distinct taxa that happen to have the same common name.
    taxa = []
    for s in sessions:
        if s.taxon_id not in [x.taxon_id for x in taxa]:
            taxa.append(s)
    taxa.sort(key=lambda s: (s.species_name.lower(), s.taxon_id))

    names: List[str] = []
    taxon_to_class: Dict[int, int] = {}
    used_names: set[str] = set()
    for s in taxa:
        display = s.species_name or s.scientific_name or f"Taxon {s.taxon_id}"
        if display in used_names:
            display = f"{display} (taxon {s.taxon_id})"
        used_names.add(display)
        taxon_to_class[s.taxon_id] = len(names)
        names.append(display)
    person_global_id = len(names) if include_person else None
    if include_person:
        names.append("person")

    rng = random.Random(seed)
    source_rows = []
    split_counts = {"train": 0, "val": 0, "test": 0}
    box_counts = defaultdict(int)
    skipped_missing = 0

    # Split within each source class so every species is represented proportionally.
    grouped = []
    for session in sessions:
        pairs = _approved_pairs(session)
        grouped.append((session, _allocate_splits(pairs, train_pct, val_pct, rng)))

    for session, splits in grouped:
        species_global_id = taxon_to_class[session.taxon_id]
        prefix = f"t{session.taxon_id}_{slugify(session.species_name)}"
        for split, pairs in splits.items():
            for image, label, details in pairs:
                new_stem = f"{prefix}_{image.stem}"
                dest_image = out / "images" / split / f"{new_stem}{image.suffix.lower()}"
                dest_label = out / "labels" / split / f"{new_stem}.txt"
                dest_details = out / "attribution" / split / f"{new_stem}.txt"
                if not image.exists() or not label.exists():
                    skipped_missing += 1
                    continue
                shutil.copy2(image, dest_image)
                species_boxes, person_boxes = _remap_label(label, dest_label, species_global_id, person_global_id)
                if details and details.exists():
                    shutil.copy2(details, dest_details)
                split_counts[split] += 1
                box_counts[names[species_global_id]] += species_boxes
                if include_person:
                    box_counts["person"] += person_boxes
                source_rows.append({
                    "split": split,
                    "dataset_image": str(dest_image.relative_to(out)),
                    "source_session": str(session.path),
                    "source_image": str(image),
                    "taxon_id": session.taxon_id,
                    "species_name": session.species_name,
                    "scientific_name": session.scientific_name,
                    "attribution_file": str(dest_details.relative_to(out)) if dest_details.exists() else "",
                })

    if sum(split_counts.values()) == 0:
        shutil.rmtree(out)
        raise ValueError("No approved image/label pairs were found in the selected sessions.")

    yaml_lines = [
        f"path: {out.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for idx, name in enumerate(names):
        safe_name = name.replace("'", "''")
        yaml_lines.append(f"  {idx}: '{safe_name}'")
    (out / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    with (out / "sources.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(source_rows[0].keys()))
        writer.writeheader()
        writer.writerows(source_rows)

    manifest = {
        "dataset_name": dataset_name,
        "created_at": utc_now(),
        "seed": seed,
        "split_percentages": {"train": train_pct, "val": val_pct, "test": 100 - train_pct - val_pct},
        "classes": {i: name for i, name in enumerate(names)},
        "taxon_to_class": taxon_to_class,
        "sessions": [str(s.path) for s in sessions],
        "image_counts": split_counts,
        "box_counts": dict(box_counts),
        "skipped_missing": skipped_missing,
    }
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"path": out, **manifest}

# ---------------- Desktop GUI ----------------
APP_TITLE = "iNaturalist YOLO Dataset Manager"
CONFIG_PATH = Path.home() / ".inat_yolo_manager_config.json"


def load_app_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_app_config(config: dict) -> None:
    try:
        CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    except OSError:
        pass


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1180x760")
        self.minsize(980, 650)

        config = load_app_config()
        default_workspace = Path(config.get("workspace") or (Path.cwd() / "inat_data"))
        self.workspace = Workspace(default_workspace)
        self.client = INaturalistClient()
        self.sessions: list[SessionInfo] = []
        self.session_by_display: dict[str, SessionInfo] = {}
        self.taxon_results: list[dict] = []
        self.download_cancel = threading.Event()
        self.events: queue.Queue = queue.Queue()
        self.review_process: subprocess.Popen | None = None

        self._build_ui()
        self.refresh_sessions()
        self.after(100, self._poll_events)

    def _build_ui(self):
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="Workspace:").pack(side="left")
        self.workspace_var = tk.StringVar(value=str(self.workspace.root))
        ttk.Entry(top, textvariable=self.workspace_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(top, text="Choose…", command=self.choose_workspace).pack(side="left")
        ttk.Button(top, text="Open", command=lambda: open_in_file_manager(self.workspace.root)).pack(side="left", padx=(6, 0))

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.download_tab = ttk.Frame(notebook, padding=10)
        self.review_tab = ttk.Frame(notebook, padding=10)
        self.dataset_tab = ttk.Frame(notebook, padding=10)
        notebook.add(self.download_tab, text="Species & Downloads")
        notebook.add(self.review_tab, text="Review")
        notebook.add(self.dataset_tab, text="Dataset Export")

        self._build_download_tab()
        self._build_review_tab()
        self._build_dataset_tab()

    def _build_download_tab(self):
        tab = self.download_tab
        tab.columnconfigure(0, weight=3)
        tab.columnconfigure(1, weight=2)
        tab.rowconfigure(1, weight=1)

        search_frame = ttk.LabelFrame(tab, text="1. Find a species / taxon", padding=8)
        search_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 8))
        search_frame.columnconfigure(1, weight=1)
        ttk.Label(search_frame, text="Search:").grid(row=0, column=0, sticky="w")
        self.species_query_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.species_query_var)
        search_entry.grid(row=0, column=1, sticky="ew", padx=6)
        search_entry.bind("<Return>", lambda e: self.search_species())
        ttk.Button(search_frame, text="Search iNaturalist", command=self.search_species).grid(row=0, column=2)

        self.taxa_tree = ttk.Treeview(search_frame, columns=("common", "scientific", "rank", "taxon"), show="headings", height=8)
        for col, text, width in (("common", "Common name", 210), ("scientific", "Scientific name", 190), ("rank", "Rank", 80), ("taxon", "Taxon ID", 90)):
            self.taxa_tree.heading(col, text=text)
            self.taxa_tree.column(col, width=width, anchor="w")
        self.taxa_tree.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=(8, 4))
        search_frame.rowconfigure(1, weight=1)
        self.taxa_tree.bind("<<TreeviewSelect>>", self.on_taxon_select)
        self.taxa_tree.bind("<Double-1>", self.on_taxon_select)

        selected = ttk.Frame(search_frame)
        selected.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        selected.columnconfigure(1, weight=1)
        ttk.Label(selected, text="Selected:").grid(row=0, column=0, sticky="w")
        self.selected_species_var = tk.StringVar()
        ttk.Entry(selected, textvariable=self.selected_species_var, state="readonly").grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Label(selected, text="Taxon ID:").grid(row=0, column=2, sticky="e")
        self.selected_taxon_var = tk.StringVar()
        ttk.Entry(selected, textvariable=self.selected_taxon_var, state="readonly", width=12).grid(row=0, column=3, padx=(6, 0))

        download_frame = ttk.LabelFrame(tab, text="2. Download", padding=8)
        download_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        for i in range(4):
            download_frame.columnconfigure(i, weight=1 if i in (1, 3) else 0)

        ttk.Label(download_frame, text="Images to add:").grid(row=0, column=0, sticky="w")
        self.image_count_var = tk.IntVar(value=100)
        ttk.Spinbox(download_frame, from_=1, to=100000, textvariable=self.image_count_var, width=12).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(download_frame, text="Image size:").grid(row=0, column=2, sticky="e")
        self.image_size_var = tk.StringVar(value="large")
        ttk.Combobox(download_frame, textvariable=self.image_size_var, values=["square", "small", "medium", "large", "original"], state="readonly", width=12).grid(row=0, column=3, sticky="w", padx=6)

        self.arr_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(download_frame, text="Include all-rights-reserved / unlicensed photos", variable=self.arr_var).grid(row=1, column=0, columnspan=3, sticky="w", pady=6)
        self.reset_cp_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(download_frame, text="Restart observation cursor (existing files still skipped)", variable=self.reset_cp_var).grid(row=2, column=0, columnspan=3, sticky="w")

        buttons = ttk.Frame(download_frame)
        buttons.grid(row=3, column=0, columnspan=4, sticky="w", pady=(8, 4))
        self.download_btn = ttk.Button(buttons, text="Create / select session and download", command=self.start_download)
        self.download_btn.pack(side="left")
        self.cancel_btn = ttk.Button(buttons, text="Cancel", command=self.cancel_download, state="disabled")
        self.cancel_btn.pack(side="left", padx=6)

        self.download_progress = ttk.Progressbar(download_frame, mode="determinate")
        self.download_progress.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(6, 4))
        self.download_status_var = tk.StringVar(value="Ready.")
        ttk.Label(download_frame, textvariable=self.download_status_var).grid(row=5, column=0, columnspan=4, sticky="w")

        session_frame = ttk.LabelFrame(tab, text="Managed sessions", padding=8)
        session_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(6, 0))
        session_frame.columnconfigure(0, weight=1)
        session_frame.rowconfigure(0, weight=1)
        self.sessions_tree = ttk.Treeview(session_frame, columns=("species", "taxon", "images", "approved", "pending"), show="headings")
        for col, text, width in (("species", "Species", 180), ("taxon", "Taxon", 70), ("images", "Images", 65), ("approved", "Approved", 75), ("pending", "Pending", 70)):
            self.sessions_tree.heading(col, text=text)
            self.sessions_tree.column(col, width=width, anchor="w")
        self.sessions_tree.grid(row=0, column=0, sticky="nsew")
        self.sessions_tree.bind("<Double-1>", lambda e: self.open_selected_session())

        sf_buttons = ttk.Frame(session_frame)
        sf_buttons.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(sf_buttons, text="Refresh", command=self.refresh_sessions).pack(side="left")
        ttk.Button(sf_buttons, text="Import existing folder…", command=self.import_existing).pack(side="left", padx=6)
        ttk.Button(sf_buttons, text="Open folder", command=self.open_selected_session).pack(side="left")

    def _build_review_tab(self):
        tab = self.review_tab
        tab.columnconfigure(1, weight=1)

        ttk.Label(tab, text="Session:").grid(row=0, column=0, sticky="w", pady=4)
        self.review_session_var = tk.StringVar()
        self.review_session_combo = ttk.Combobox(tab, textvariable=self.review_session_var, state="readonly")
        self.review_session_combo.grid(row=0, column=1, sticky="ew", padx=6)
        self.review_session_combo.bind("<<ComboboxSelected>>", lambda e: self.on_review_session_selected())

        ttk.Label(tab, text="YOLO model path:").grid(row=1, column=0, sticky="w", pady=4)
        self.model_path_var = tk.StringVar(value="../standard_fp32_ncnn_model")
        ttk.Entry(tab, textvariable=self.model_path_var).grid(row=1, column=1, sticky="ew", padx=6)
        ttk.Button(tab, text="Browse…", command=self.choose_model).grid(row=1, column=2)
        ttk.Button(tab, text="Load model classes", command=self.load_model_classes).grid(row=2, column=2)

        ttk.Label(tab, text="Detection class for selected species:").grid(row=2, column=0, sticky="w", pady=4)
        self.model_class_var = tk.StringVar()
        self.model_class_combo = ttk.Combobox(tab, textvariable=self.model_class_var)
        self.model_class_combo.grid(row=2, column=1, sticky="ew", padx=6)

        ttk.Label(tab, text="Confidence threshold:").grid(row=3, column=0, sticky="w", pady=4)
        self.conf_var = tk.DoubleVar(value=0.4)
        ttk.Spinbox(tab, from_=0.0, to=1.0, increment=0.05, textvariable=self.conf_var, width=10).grid(row=3, column=1, sticky="w", padx=6)

        self.rereview_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tab, text="Include already approved/rejected/excluded images", variable=self.rereview_var).grid(row=4, column=1, sticky="w", padx=6, pady=4)

        controls = ttk.LabelFrame(tab, text="Reviewer controls", padding=8)
        controls.grid(row=5, column=0, columnspan=3, sticky="ew", pady=10)
        ttk.Label(controls, text="Left-drag: add box    Right-click: delete box    1: species    2: person    Enter: approve/save    S: reject    D: exclude    R: reset detections    Q: quit", wraplength=900).pack(anchor="w")

        self.review_btn = ttk.Button(tab, text="Launch reviewer", command=self.launch_reviewer)
        self.review_btn.grid(row=6, column=1, sticky="w", padx=6)
        self.review_status_var = tk.StringVar(value="Select a session and model class.")
        ttk.Label(tab, textvariable=self.review_status_var).grid(row=7, column=0, columnspan=3, sticky="w", pady=8)

    def _build_dataset_tab(self):
        tab = self.dataset_tab
        tab.columnconfigure(0, weight=2)
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(0, weight=1)

        left = ttk.LabelFrame(tab, text="Approved sessions to include", padding=8)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        self.dataset_list = tk.Listbox(left, selectmode="extended", exportselection=False)
        self.dataset_list.grid(row=0, column=0, sticky="nsew")
        ttk.Button(left, text="Select all", command=lambda: self.dataset_list.select_set(0, tk.END)).grid(row=1, column=0, sticky="w", pady=(6, 0))

        right = ttk.LabelFrame(tab, text="Dataset settings", padding=8)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right.columnconfigure(1, weight=1)
        ttk.Label(right, text="Dataset name:").grid(row=0, column=0, sticky="w", pady=4)
        self.dataset_name_var = tk.StringVar(value="inat_dataset")
        ttk.Entry(right, textvariable=self.dataset_name_var).grid(row=0, column=1, sticky="ew", padx=6)

        ttk.Label(right, text="Train %:").grid(row=1, column=0, sticky="w", pady=4)
        self.train_var = tk.IntVar(value=80)
        ttk.Spinbox(right, from_=0, to=100, textvariable=self.train_var, width=8).grid(row=1, column=1, sticky="w", padx=6)
        ttk.Label(right, text="Validation %:").grid(row=2, column=0, sticky="w", pady=4)
        self.val_var = tk.IntVar(value=20)
        ttk.Spinbox(right, from_=0, to=100, textvariable=self.val_var, width=8).grid(row=2, column=1, sticky="w", padx=6)
        ttk.Label(right, text="Test % = remainder").grid(row=3, column=1, sticky="w", padx=6)

        self.include_person_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(right, text="Include person as a dataset class", variable=self.include_person_var).grid(row=4, column=0, columnspan=2, sticky="w", pady=6)
        self.overwrite_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(right, text="Overwrite dataset if it already exists", variable=self.overwrite_var).grid(row=5, column=0, columnspan=2, sticky="w")
        ttk.Button(right, text="Build YOLO dataset", command=self.create_dataset).grid(row=6, column=0, columnspan=2, sticky="w", pady=(12, 4))
        self.dataset_status_var = tk.StringVar(value="Approved labels are remapped to global class IDs during export.")
        ttk.Label(right, textvariable=self.dataset_status_var, wraplength=330).grid(row=7, column=0, columnspan=2, sticky="w", pady=6)

    def choose_workspace(self):
        folder = filedialog.askdirectory(initialdir=str(self.workspace.root), title="Choose workspace folder")
        if not folder:
            return
        self.workspace = Workspace(Path(folder))
        self.workspace_var.set(str(self.workspace.root))
        save_app_config({"workspace": str(self.workspace.root)})
        self.refresh_sessions()

    def search_species(self):
        query = self.species_query_var.get().strip()
        if len(query) < 2:
            messagebox.showinfo(APP_TITLE, "Enter at least two characters to search.")
            return
        self.download_status_var.set("Searching iNaturalist taxa…")
        def worker():
            try:
                results = self.client.search_taxa(query)
                self.events.put(("taxa_results", results))
            except Exception as exc:
                self.events.put(("error", f"Taxon search failed: {exc}"))
        threading.Thread(target=worker, daemon=True).start()

    def on_taxon_select(self, event=None):
        selected = self.taxa_tree.selection()
        if not selected:
            return
        index = int(selected[0])
        if 0 <= index < len(self.taxon_results):
            item = self.taxon_results[index]
            self.selected_species_var.set(item["species_name"])
            self.selected_taxon_var.set(str(item["id"]))

    def start_download(self):
        if not self.selected_taxon_var.get():
            messagebox.showinfo(APP_TITLE, "Search for and select a taxon first.")
            return
        try:
            taxon_id = int(self.selected_taxon_var.get())
            count = int(self.image_count_var.get())
            if count <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(APP_TITLE, "Images to add must be a positive integer.")
            return
        taxon = next((x for x in self.taxon_results if x["id"] == taxon_id), None) or {
            "species_name": self.selected_species_var.get() or f"Taxon {taxon_id}",
            "scientific_name": "",
            "rank": "",
        }
        session = self.workspace.create_or_update_session(
            taxon_id, taxon["species_name"], taxon.get("scientific_name", ""), taxon.get("rank", "")
        )
        self.download_cancel.clear()
        self.download_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.download_progress.configure(maximum=count, value=0)
        self.download_status_var.set("Starting download…")

        def progress(data):
            self.events.put(("download_progress", data))

        def worker():
            try:
                result = self.client.download_images(
                    session.path, taxon_id, count,
                    image_size=self.image_size_var.get(),
                    include_all_rights_reserved=self.arr_var.get(),
                    reset_checkpoint=self.reset_cp_var.get(),
                    progress=progress,
                    cancel_event=self.download_cancel,
                )
                self.events.put(("download_done", result))
            except Exception as exc:
                self.events.put(("error", f"Download failed: {exc}"))
                self.events.put(("download_done", None))
        threading.Thread(target=worker, daemon=True).start()

    def cancel_download(self):
        self.download_cancel.set()
        self.download_status_var.set("Cancelling after the current request/file…")

    def refresh_sessions(self):
        self.sessions = self.workspace.discover_sessions()
        self.session_by_display = {s.display_name: s for s in self.sessions}
        if hasattr(self, "sessions_tree"):
            self.sessions_tree.delete(*self.sessions_tree.get_children())
            for i, session in enumerate(self.sessions):
                stats = self.workspace.stats(session)
                self.sessions_tree.insert("", "end", iid=str(i), values=(session.species_name, session.taxon_id, stats["images"], stats["approved"], stats["pending"]))
        displays = [s.display_name for s in self.sessions]
        if hasattr(self, "review_session_combo"):
            self.review_session_combo["values"] = displays
            if self.review_session_var.get() not in displays:
                self.review_session_var.set(displays[0] if displays else "")
                self.on_review_session_selected()
        if hasattr(self, "dataset_list"):
            self.dataset_list.delete(0, tk.END)
            for session in self.sessions:
                stats = self.workspace.stats(session)
                self.dataset_list.insert(tk.END, f"{session.display_name} — {stats['approved']} approved")

    def selected_managed_session(self) -> SessionInfo | None:
        selected = self.sessions_tree.selection()
        if not selected:
            return None
        idx = int(selected[0])
        return self.sessions[idx] if 0 <= idx < len(self.sessions) else None

    def open_selected_session(self):
        session = self.selected_managed_session()
        if not session:
            messagebox.showinfo(APP_TITLE, "Select a managed session first.")
            return
        open_in_file_manager(session.path)

    def import_existing(self):
        folder = filedialog.askdirectory(title="Choose an existing scraper/reviewer folder")
        if not folder:
            return
        path = Path(folder)
        meta = read_json(path / META_FILE, {})
        taxon_id = meta.get("taxon_id")
        if taxon_id is None:
            import re
            m = re.match(r"^(\d+)", path.name)
            taxon_id = int(m.group(1)) if m else None
        if taxon_id is None:
            taxon_id = simpledialog.askinteger(APP_TITLE, "Taxon ID for this existing folder:", minvalue=1)
            if not taxon_id:
                return
        try:
            taxon = self.client.get_taxon(int(taxon_id))
        except Exception:
            taxon = None
        taxon = taxon or {"species_name": meta.get("species_name") or f"Taxon {taxon_id}", "scientific_name": meta.get("scientific_name", ""), "rank": meta.get("rank", "")}
        self.workspace.register_existing(path, int(taxon_id), taxon["species_name"], taxon.get("scientific_name", ""), taxon.get("rank", ""))
        self.refresh_sessions()

    def choose_model(self):
        selected = filedialog.askopenfilename(title="Choose YOLO model file (Cancel to choose a model directory)")
        if selected:
            self.model_path_var.set(selected)
            return
        folder = filedialog.askdirectory(title="Choose YOLO / NCNN model directory")
        if folder:
            self.model_path_var.set(folder)

    def load_model_classes(self):
        model_path = self.model_path_var.get().strip()
        if not model_path:
            return
        self.review_status_var.set("Loading model classes…")
        def worker():
            try:
                names = model_class_names(model_path)
                self.events.put(("model_classes", names))
            except Exception as exc:
                self.events.put(("error", f"Could not load model: {exc}"))
        threading.Thread(target=worker, daemon=True).start()

    def on_review_session_selected(self):
        session = self.session_by_display.get(self.review_session_var.get())
        if not session:
            return
        if session.model_class_name:
            self.model_class_var.set(session.model_class_name)
        stats = self.workspace.stats(session)
        self.review_status_var.set(f"{stats['images']} images: {stats['approved']} approved, {stats['rejected']} rejected, {stats['excluded']} excluded, {stats['pending']} pending.")

    def launch_reviewer(self):
        session = self.session_by_display.get(self.review_session_var.get())
        if not session:
            messagebox.showinfo(APP_TITLE, "Select a session first.")
            return
        model_path = self.model_path_var.get().strip()
        model_class = self.model_class_var.get().strip()
        if not model_path or not model_class:
            messagebox.showinfo(APP_TITLE, "Choose a model path and the model class used for this species.")
            return
        session.model_class_name = model_class
        self.workspace.save_session(session)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--review-session", str(session.path),
            "--model", model_path,
            "--model-class", model_class,
            "--conf", str(float(self.conf_var.get())),
        ]
        if self.rereview_var.get():
            cmd.append("--include-processed")
        try:
            self.review_process = subprocess.Popen(cmd, cwd=str(Path(__file__).parent))
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not launch reviewer: {exc}")
            return
        self.review_btn.configure(state="disabled")
        self.review_status_var.set("Reviewer running in the OpenCV window…")
        self.after(500, self._poll_reviewer)

    def _poll_reviewer(self):
        if not self.review_process:
            return
        code = self.review_process.poll()
        if code is None:
            self.after(500, self._poll_reviewer)
            return
        self.review_process = None
        self.review_btn.configure(state="normal")
        self.refresh_sessions()
        if code == 0:
            self.review_status_var.set("Reviewer closed. Session statistics refreshed.")
        else:
            self.review_status_var.set(f"Reviewer exited with code {code}. See the terminal for details.")

    def create_dataset(self):
        selected_indices = list(self.dataset_list.curselection())
        if not selected_indices:
            messagebox.showinfo(APP_TITLE, "Select one or more sessions to export.")
            return
        sessions = [self.sessions[i] for i in selected_indices if i < len(self.sessions)]
        try:
            result = build_dataset(
                self.workspace,
                sessions,
                self.dataset_name_var.get().strip() or "inat_dataset",
                train_pct=int(self.train_var.get()),
                val_pct=int(self.val_var.get()),
                include_person=self.include_person_var.get(),
                overwrite=self.overwrite_var.get(),
            )
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Dataset export failed:\n{exc}")
            return
        counts = result["image_counts"]
        self.dataset_status_var.set(f"Created {result['path']} — train {counts['train']}, val {counts['val']}, test {counts['test']} images.")
        if messagebox.askyesno(APP_TITLE, "Dataset created successfully. Open its folder?"):
            open_in_file_manager(result["path"])

    def _poll_events(self):
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == "taxa_results":
                    self.taxon_results = data
                    self.taxa_tree.delete(*self.taxa_tree.get_children())
                    for i, item in enumerate(data):
                        self.taxa_tree.insert("", "end", iid=str(i), values=(item["species_name"], item["scientific_name"], item["rank"], item["id"]))
                    self.download_status_var.set(f"Found {len(data)} taxa. Select one to fill the taxon ID.")
                elif kind == "download_progress":
                    self.download_progress.configure(value=data.get("downloaded", 0), maximum=max(1, data.get("target", 1)))
                    self.download_status_var.set(data.get("message", ""))
                elif kind == "download_done":
                    self.download_btn.configure(state="normal")
                    self.cancel_btn.configure(state="disabled")
                    if data:
                        self.download_status_var.set(f"Finished: {data['downloaded']} new images saved; cursor {data['checkpoint']}.")
                    self.refresh_sessions()
                elif kind == "model_classes":
                    self.model_class_combo["values"] = data
                    if not self.model_class_var.get() and data:
                        non_person = [x for x in data if x.lower() != "person"]
                        self.model_class_var.set(non_person[0] if non_person else data[0])
                    self.review_status_var.set(f"Loaded {len(data)} model classes. Choose the class matching the selected species.")
                elif kind == "error":
                    messagebox.showerror(APP_TITLE, data)
        except queue.Empty:
            pass
        self.after(100, self._poll_events)


def main() -> int:
    # The GUI calls this same file in a subprocess for OpenCV review.
    if "--review-session" in sys.argv:
        parser = argparse.ArgumentParser(description="Review an iNaturalist image session with a YOLO model.")
        parser.add_argument("--review-session", required=True)
        parser.add_argument("--model", required=True)
        parser.add_argument("--model-class", required=True)
        parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
        parser.add_argument("--include-processed", action="store_true")
        args = parser.parse_args()
        result = review_session(
            Path(args.review_session),
            args.model,
            args.model_class,
            conf_threshold=args.conf,
            include_processed=args.include_processed,
        )
        print(result)
        return 0

    App().mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
