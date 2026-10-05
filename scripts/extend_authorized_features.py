"""Extract new authorized features and extend an existing feature index."""

import argparse
import json
from pathlib import Path

from rb3charts.model import save_json
from rb3charts.training import extract_structure_features


def _structure_labels(example, manifest, manifest_item, chart_path):
    import pretty_midi

    targets = {
        row["attack_id"]: row
        for row in example["expert_targets"]
    }
    notes = []
    attacks = []
    for attack in example["conditioning"]:
        target = targets[attack["attack_id"]]
        source_ids = []
        for lane, duration in zip(
            target["lanes"],
            target["lane_durations_seconds"],
        ):
            source_id = f"{attack['attack_id']}:lane:{lane}"
            source_ids.append(source_id)
            notes.append({
                "start": float(attack["time_seconds"]),
                "end": float(attack["time_seconds"]) + float(duration),
                "pitch": 48 + int(lane),
                "source_id": source_id,
            })
        attacks.append({
            "start": float(attack["time_seconds"]),
            "polyphony": len(source_ids),
            "source_ids": source_ids,
        })
    midi = pretty_midi.PrettyMIDI(str(chart_path))
    beats = [
        {"time": float(time), "position": index % 4 + 1}
        for index, time in enumerate(midi.get_beats())
    ]
    return {
        "schema": "rb3charts-structure-labels",
        "schema_version": 1,
        "source_format": "authorized-chart-pair",
        "source_annotation": manifest_item["annotations"],
        "source_annotation_sha256": manifest_item["annotations_sha256"],
        "provenance": {
            "source_url": manifest["dataset"]["source_url"],
            "audio_license": manifest["dataset"]["audio_license"],
            "annotation_license": manifest["dataset"]["annotation_license"],
            "commercial_use": manifest["dataset"]["commercial_use"],
            "attribution": manifest["dataset"]["attribution"],
            "dataset_id": manifest["dataset"]["id"],
            "dataset_version": manifest["dataset"]["version"],
        },
        "group_id": f"authorized:{example['song_id']}",
        "split_keys": example["split_keys"],
        "targets": ["attack", "polyphony", "sustain"],
        "notes": notes,
        "attacks": attacks,
        "beats": beats,
        "summary": {
            "notes": len(notes),
            "attacks": len(attacks),
        },
    }


def run(base_index, manifest_path, converted_index, output_index):
    root = output_index.parent.resolve()
    base = json.loads(base_index.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    converted = json.loads(converted_index.read_text(encoding="utf-8"))
    manifest_items = {
        str(item["song_id"]): item
        for item in manifest["items"]
    }
    existing = {
        str(item["song_id"])
        for item in base["items"]
    }
    output_items = list(base["items"])
    feature_root = manifest_path.parent / "features"
    for example in converted["examples"]:
        song_id = str(example["song_id"])
        if song_id in existing:
            continue
        item = manifest_items[song_id]
        example_path = converted_index.parent / example["artifact"]
        example_payload = json.loads(example_path.read_text(encoding="utf-8"))
        label_path = manifest_path.parent / "labels" / f"{song_id}.json"
        save_json(
            label_path,
            _structure_labels(
                example_payload,
                manifest,
                item,
                manifest_path.parent / item["annotations"],
            ),
        )
        feature_path = feature_root / f"{song_id}.npz"
        extract_structure_features(
            manifest_path.parent / item["audio"],
            label_path,
            feature_path,
        )
        output_items.append({
            "song_id": song_id,
            "group_id": f"authorized:{song_id}",
            "split": item["split"],
            "features": str(feature_path.relative_to(root)),
            "manifest": str(manifest_path.relative_to(root)),
        })
    report = {
        "schema": "rb3charts-structure-feature-index",
        "schema_version": 2,
        "items": output_items,
    }
    save_json(output_index, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_index", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("converted_index", type=Path)
    parser.add_argument("output_index", type=Path)
    args = parser.parse_args()
    report = run(
        args.base_index.resolve(),
        args.manifest.resolve(),
        args.converted_index.resolve(),
        args.output_index.resolve(),
    )
    print(json.dumps({
        "items": len(report["items"]),
        "new_items": len(report["items"]) - len(json.loads(
            args.base_index.read_text(encoding="utf-8")
        )["items"]),
        "output": str(args.output_index),
    }, indent=2))


if __name__ == "__main__":
    main()
