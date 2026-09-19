from hashlib import sha256
import json
from pathlib import Path
import struct

from rb3charts import encore


def _sng_with_midi(midi: bytes):
    mask = bytes(range(16))
    header = b"SNGPKG" + struct.pack("<I", 1) + mask
    metadata = struct.pack("<QQ", 8, 0)
    name = b"notes.mid"
    index_length = 8 + 1 + len(name) + 16
    contents_index = len(header) + len(metadata) + 8 + index_length + 8
    index = struct.pack("<QQB", index_length, 1, len(name)) + name + struct.pack("<QQ", len(midi), contents_index)
    masked = bytes(value ^ mask[i % 16] ^ (i & 0xFF) for i, value in enumerate(midi))
    return header + metadata + index + struct.pack("<Q", len(masked)) + masked


def test_fetch_notes_midi_reads_only_index_and_midi_ranges(tmp_path, monkeypatch):
    midi = b"MThd" + bytes(range(40))
    package = _sng_with_midi(midi)
    ranges = []

    def read_range(url, start, end, timeout=30):
        ranges.append((start, end))
        return package[start:end + 1]

    monkeypatch.setattr(encore, "_range", read_range)
    reference = {"md5": "a" * 32, "encoder_version": 0, "chart_id": 1}
    result = encore.fetch_notes_midi(reference, tmp_path)
    assert result.read_bytes() == midi
    assert max(end - start + 1 for start, end in ranges) < len(package)
    manifest = json.loads((result.parent / "manifest.json").read_text())
    assert manifest["audio_downloaded"] is False
    assert manifest["notes_sha256"] == sha256(midi).hexdigest()


def test_search_prefers_exact_harmonix_reference(monkeypatch):
    base = {"name": "My Name Is Jonas", "artist": "Weezer", "album": "Blue", "md5": "b" * 32,
            "encoderVersion": 0, "chartHash": "hash", "modifiedTime": "date",
            "notesData": {"noteCounts": [{"instrument": "guitar", "difficulty": "expert", "count": 800}], "trackHashes": []}}
    monkeypatch.setattr(encore, "_json_request", lambda *args, **kwargs: {"data": [
        {**base, "chartId": 2, "charter": "Neversoft"},
        {**base, "chartId": 1, "charter": "Harmonix"},
        {**base, "chartId": 3, "charter": "Other", "name": "Different Song"},
    ]})
    results = encore.search_references("My Name Is Jonas", "Weezer")
    assert [item["chart_id"] for item in results] == [1, 2]


def test_catalog_pins_provider_identity(tmp_path):
    path = tmp_path / "catalog.json"
    reference = {"chart_id": 4, "title": "Song", "artist": "Artist", "md5": "c" * 32}
    encore.save_to_catalog(reference, path)
    assert encore.load_catalog(path)["references"] == [reference]


def test_pinned_catalog_resolves_without_search(tmp_path, monkeypatch):
    run = tmp_path / "run"; (run / "metadata").mkdir(parents=True)
    (run / "metadata" / "song.json").write_text(json.dumps({"title": "Song", "artist": "Artist"}))
    reference = {"chart_id": 4, "title": "Song", "artist": "Artist", "md5": "c" * 32}
    catalog = tmp_path / "catalog.json"; encore.save_to_catalog(reference, catalog)
    expected = tmp_path / "notes.mid"; expected.write_bytes(b"MThd")
    monkeypatch.setattr(encore, "search_references", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network search used")))
    monkeypatch.setattr(encore, "fetch_notes_midi", lambda selected, cache: expected)
    selected, midi = encore.resolve_encore_reference(run, catalog, cache=tmp_path / "cache")
    assert selected == reference
    assert midi == expected
