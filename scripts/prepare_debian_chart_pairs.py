"""Build the local v2 authorized-pair corpus from verified Debian packs."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil

from rb3charts.model import save_json


PACKS = {
    "sectoid": {
        "artist_id": "sectoid",
        "chart_author_id": "carlos-viola-iborra",
        "split": "train",
        "license": "MIT",
        "attribution": "Carlos Viola Iborra (Sectoid)",
        "source_url": (
            "https://deb.debian.org/debian/pool/main/f/"
            "fretsonfire-songs-sectoid/"
        ),
    },
    "muldjord": {
        "artist_id": "muldjord",
        "chart_author_id": "el-senator",
        "split": "validation",
        "license": "CC BY-SA 3.0",
        "attribution": "El Senator (Muldjord)",
        "source_url": (
            "https://deb.debian.org/debian/pool/main/f/"
            "fretsonfire-songs-muldjord/"
        ),
    },
}


def _hash_file(path):
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value):
    return "-".join(
        part for part in "".join(
            character.lower()
            if character.isalnum() else " "
            for character in value
        ).split()
        if part
    )


def _link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _hash_file(destination) != _hash_file(source):
            raise ValueError(f"Existing destination differs: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def run(base_manifest, intake_path, sync_audit_path, output_root):
    base_root = base_manifest.parent
    acquisition_root = intake_path.parent
    base = json.loads(base_manifest.read_text(encoding="utf-8"))
    intake = json.loads(intake_path.read_text(encoding="utf-8"))
    sync = json.loads(sync_audit_path.read_text(encoding="utf-8"))
    sync_by_hash = {
        row["chart_sha256"]: row
        for row in sync["pairs"]
    }
    if len(sync_by_hash) != 8 or not all(
        row["correctable_sync_pass"] for row in sync_by_hash.values()
    ):
        raise ValueError(
            "All eight pairs must pass corrected synchronization audit"
        )

    items = []
    for item in base["items"]:
        copied = json.loads(json.dumps(item))
        for field in ("audio", "annotations"):
            source = base_root / copied[field]
            destination = output_root / copied[field]
            _link_or_copy(source, destination)
        for rights_name in ("audio_rights", "chart_rights"):
            evidence = copied[rights_name]["evidence"]
            _link_or_copy(base_root / evidence, output_root / evidence)
        items.append(copied)

    for pack in intake["packs"]:
        configuration = PACKS[pack["id"]]
        evidence_source = acquisition_root / pack["copyright_file"]
        evidence_relative = (
            Path("permissions") / f"{pack['id']}-copyright.txt"
        )
        _link_or_copy(evidence_source, output_root / evidence_relative)
        evidence_hash = _hash_file(evidence_source)
        if evidence_hash != pack["copyright_sha256"]:
            raise ValueError(f"{pack['id']} rights hash does not match intake")
        for song in pack["songs"]:
            title = song["metadata"]["name"]
            song_id = f"{pack['id']}-{_slug(title)}"
            source_folder = acquisition_root / song["folder"]
            audio_source = acquisition_root / song["listening"]["fullmix"]
            chart_source = source_folder / "notes.mid"
            audio_relative = Path("audio") / f"{song_id}.flac"
            chart_relative = Path("charts") / f"{song_id}.mid"
            _link_or_copy(audio_source, output_root / audio_relative)
            _link_or_copy(chart_source, output_root / chart_relative)
            if _hash_file(audio_source) != song["listening"]["fullmix_sha256"]:
                raise ValueError(f"{song_id} mix hash does not match intake")
            if _hash_file(chart_source) != song["chart_sha256"]:
                raise ValueError(f"{song_id} chart hash does not match intake")
            audit = sync_by_hash[song["chart_sha256"]]
            items.append({
                "song_id": song_id,
                "artist_id": configuration["artist_id"],
                "chart_author_id": configuration["chart_author_id"],
                "split": configuration["split"],
                "audio": str(audio_relative),
                "audio_sha256": song["listening"]["fullmix_sha256"],
                "annotations": str(chart_relative),
                "annotations_sha256": song["chart_sha256"],
                "chart_format": "frets_on_fire_midi",
                "synchronization": {
                    "method": "embedded_audio_offset",
                    "audio_offset_seconds": audit[
                        "recommended_audio_offset_seconds"
                    ],
                    "verified": True,
                    "evidence": {
                        "audit_schema": sync["schema"],
                        "chart_sha256": song["chart_sha256"],
                        "global_best_offset_seconds": (
                            audit["global"]["best_offset_seconds"]
                        ),
                        "window_best_offsets_seconds": [
                            row["best_offset_seconds"]
                            for row in audit["windows"]
                        ],
                        "attacks_within_50ms_of_onset": (
                            audit["corrected_attack_onset_support"][
                                "within_50ms"
                            ]
                        ),
                    },
                },
                "audio_rights": {
                    "basis": "compatible_license",
                    "license": configuration["license"],
                    "attribution": configuration["attribution"],
                    "training_allowed": True,
                    "redistribution_allowed": True,
                    "evidence": str(evidence_relative),
                    "evidence_sha256": evidence_hash,
                },
                "chart_rights": {
                    "basis": "compatible_license",
                    "license": configuration["license"],
                    "attribution": configuration["attribution"],
                    "training_allowed": True,
                    "redistribution_allowed": True,
                    "evidence": str(evidence_relative),
                    "evidence_sha256": evidence_hash,
                },
            })

    dataset = dict(base["dataset"])
    dataset.update({
        "id": "authorized-chart-structure-v2",
        "version": "2026-10-05",
        "source_url": (
            f"{dataset['source_url']}, "
            f"{PACKS['sectoid']['source_url']}, "
            f"{PACKS['muldjord']['source_url']}"
        ),
        "attribution": (
            f"{dataset['attribution']}; Carlos Viola Iborra (Sectoid); "
            "El Senator (Muldjord)"
        ),
    })
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "dataset": dataset,
        "items": items,
    }
    save_json(output_root / "manifest.json", manifest)
    save_json(output_root / "debian-intake-lineage.json", {
        "schema": "rb3charts-debian-chart-pair-lineage",
        "schema_version": 1,
        "base_manifest_sha256": _hash_file(base_manifest),
        "intake_report_sha256": _hash_file(intake_path),
        "sync_audit_sha256": _hash_file(sync_audit_path),
        "pack_assignments": {
            pack: {
                "split": configuration["split"],
                "artist_id": configuration["artist_id"],
                "chart_author_id": configuration["chart_author_id"],
            }
            for pack, configuration in PACKS.items()
        },
    })
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_manifest", type=Path)
    parser.add_argument("intake", type=Path)
    parser.add_argument("sync_audit", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    manifest = run(
        args.base_manifest.resolve(),
        args.intake.resolve(),
        args.sync_audit.resolve(),
        args.output_root.resolve(),
    )
    print(json.dumps({
        "items": len(manifest["items"]),
        "splits": {
            split: sum(item["split"] == split for item in manifest["items"])
            for split in ("train", "validation", "test")
        },
        "chart_authors": sorted({
            item["chart_author_id"] for item in manifest["items"]
        }),
    }, indent=2))


if __name__ == "__main__":
    main()
