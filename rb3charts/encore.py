"""Chorus Encore reference lookup and chart-only SNG retrieval."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import struct
from urllib.request import Request, urlopen


API_URL = "https://api.enchor.us/search"
FILES_URL = "https://files.enchor.us"
CATALOG_PATH = Path("reference/catalog.json")


def _json_request(url: str, payload: dict, timeout: int = 30):
    request = Request(url, data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json", "User-Agent": "RB3-Charts/0.1"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def search_references(title: str, artist: str, timeout: int = 30, official_only: bool = True) -> list[dict]:
    response = _json_request(API_URL, {"search": f'"{title}" "{artist}"', "page": 1, "per_page": 50, "source": "api"}, timeout)
    results = []
    for row in response.get("data", []):
        if _normalized(row.get("name", "")) != _normalized(title) or _normalized(row.get("artist", "")) != _normalized(artist):
            continue
        notes = row.get("notesData") or {}
        if not any(item.get("instrument") == "guitar" and item.get("difficulty") == "expert" for item in notes.get("noteCounts", [])):
            continue
        if official_only and row.get("charter", "").casefold() not in ("harmonix", "neversoft"):
            continue
        results.append(reference_manifest(row))
    priority = {"harmonix": 0, "neversoft": 1}
    return sorted(results, key=lambda item: (priority.get(item["charter"].casefold(), 2), -item["expert_guitar_notes"], item["chart_id"]))


def reference_manifest(row: dict) -> dict:
    notes = row.get("notesData") or {}
    note_counts = notes.get("noteCounts", [])
    track_hashes = {f"{item.get('instrument')}.{item.get('difficulty')}": item.get("hash") for item in notes.get("trackHashes", [])}
    return {
        "provider": "chorus-encore", "chart_id": row["chartId"], "title": row.get("name", ""),
        "artist": row.get("artist", ""), "album": row.get("album", ""), "charter": row.get("charter", ""),
        "pack": row.get("packName", ""), "icon": row.get("icon", ""), "md5": row["md5"],
        "encoder_version": row.get("encoderVersion", 0), "chart_hash": row.get("chartHash"),
        "tempo_map_hash": notes.get("tempoMapHash"), "track_hashes": track_hashes,
        "expert_guitar_notes": next((item.get("count", 0) for item in note_counts if item.get("instrument") == "guitar" and item.get("difficulty") == "expert"), 0),
        "modified_time": row.get("modifiedTime"),
    }


def sng_url(reference: dict) -> str:
    prefix = "" if int(reference.get("encoder_version", 0)) == 0 else f"charts/v{int(reference['encoder_version'])}/"
    return f"{FILES_URL}/{prefix}{reference['md5']}.sng"


def _range(url: str, start: int, end: int, timeout: int = 30) -> bytes:
    request = Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "RB3-Charts/0.1"})
    with urlopen(request, timeout=timeout) as response:
        if getattr(response, "status", None) != 206:
            raise IOError("Chorus Encore did not honor the chart-only byte-range request")
        data = response.read()
    expected = end - start + 1
    if len(data) == expected:
        return data
    raise IOError(f"Incomplete byte range from Chorus Encore: expected {expected}, received {len(data)}")


def fetch_notes_midi(reference: dict, cache: Path | str = Path("reference/encore-cache"), timeout: int = 30) -> Path:
    """Fetch only notes.mid from an Encore SNG; audio ranges are never requested."""
    cache_folder = Path(cache).expanduser().resolve() / reference["md5"]
    midi_path = cache_folder / "notes.mid"
    manifest_path = cache_folder / "manifest.json"
    if midi_path.is_file() and manifest_path.is_file():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved.get("reference", {}).get("md5") == reference["md5"] and saved.get("notes_sha256") == sha256(midi_path.read_bytes()).hexdigest():
            return midi_path

    url = sng_url(reference)
    header = _range(url, 0, 33, timeout)
    if header[:6] != b"SNGPKG":
        raise ValueError("Encore file is not a supported SNG package")
    xor_mask = header[10:26]
    metadata_length = struct.unpack_from("<Q", header, 26)[0]
    index_start = 26 + 8 + metadata_length
    index_length = struct.unpack("<Q", _range(url, index_start, index_start + 7, timeout))[0]
    index = _range(url, index_start + 8, index_start + 7 + index_length, timeout)
    count = struct.unpack_from("<Q", index, 0)[0]
    position = 8
    selected = None
    for _ in range(count):
        name_length = index[position]; position += 1
        name = index[position:position + name_length].decode("utf-8"); position += name_length
        contents_length, contents_index = struct.unpack_from("<QQ", index, position); position += 16
        if Path(name).name.casefold() == "notes.mid":
            selected = (contents_index, contents_length)
    if selected is None:
        raise FileNotFoundError("The selected Encore chart does not contain notes.mid")
    contents_index, contents_length = selected
    if contents_length > 20 * 1024 * 1024:
        raise ValueError("Reference notes.mid is unexpectedly large")
    masked = _range(url, contents_index, contents_index + contents_length - 1, timeout)
    midi = bytes(value ^ xor_mask[index % 16] ^ (index & 0xFF) for index, value in enumerate(masked))
    if not midi.startswith(b"MThd"):
        raise ValueError("Encore notes.mid failed SNG decoding")
    cache_folder.mkdir(parents=True, exist_ok=True)
    midi_path.write_bytes(midi)
    manifest_path.write_text(json.dumps({"schema": 1, "reference": reference, "sng_url": url,
                                         "notes_sha256": sha256(midi).hexdigest(), "audio_downloaded": False}, indent=2), encoding="utf-8")
    return midi_path


def load_catalog(path: Path | str = CATALOG_PATH) -> dict:
    catalog_path = Path(path)
    return json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.is_file() else {"schema": 1, "references": []}


def save_to_catalog(reference: dict, path: Path | str = CATALOG_PATH) -> Path:
    catalog_path = Path(path)
    catalog = load_catalog(catalog_path)
    catalog["references"] = [item for item in catalog.get("references", []) if item.get("chart_id") != reference["chart_id"]]
    catalog["references"].append(reference)
    catalog["references"].sort(key=lambda item: (item["artist"].casefold(), item["title"].casefold(), item["chart_id"]))
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    return catalog_path


def metadata_from_run(folder: Path | str) -> tuple[str, str]:
    root = Path(folder).expanduser().resolve()
    for path in (root / "metadata/song.json", root / "result.json"):
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            title, artist = data.get("title", "").strip(), data.get("artist", "").strip()
            if title and artist:
                return title, artist
    raise ValueError("The review folder does not contain title and artist metadata")


def resolve_encore_reference(folder: Path | str, catalog_path: Path | str = CATALOG_PATH, refresh: bool = False,
                             cache: Path | str = Path("reference/encore-cache")) -> tuple[dict, Path]:
    title, artist = metadata_from_run(folder)
    catalog = load_catalog(catalog_path)
    matches = [item for item in catalog.get("references", []) if _normalized(item["title"]) == _normalized(title) and _normalized(item["artist"]) == _normalized(artist)]
    if not matches or refresh:
        matches = search_references(title, artist)
        if not matches:
            raise LookupError(f"Chorus Encore has no exact five-lane Guitar reference for {artist} - {title}")
        save_to_catalog(matches[0], catalog_path)
    reference = matches[0]
    return reference, fetch_notes_midi(reference, cache)


def compare_with_encore(folder: Path | str, output: Path | str = Path("comparisons"),
                        catalog_path: Path | str = CATALOG_PATH, refresh: bool = False,
                        instrument: str = "guitar", difficulty: str = "expert",
                        cache: Path | str = Path("reference/encore-cache")) -> dict:
    from .compare import compare_charts
    reference, midi = resolve_encore_reference(folder, catalog_path, refresh, cache)
    report = compare_charts(folder, midi, output, instrument, difficulty)
    provenance_path = Path(report["output"]["folder"]) / "encore-reference.json"
    provenance_path.write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    report["encore_reference"] = reference
    report["output"]["encore_reference"] = str(provenance_path)
    return report
