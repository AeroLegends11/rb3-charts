"""Rights-clean conversion into lane-independent structure-model labels."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .model import save_json


STRUCTURE_LABEL_VERSION = 1
AUTHORING_EXAMPLE_VERSION = 1
FEATURE_VERSION = 3
FEATURE_SAMPLE_RATE = 22050
FEATURE_HOP_LENGTH = 256
FEATURE_MELS = 128
GUITARSET_PROVENANCE = {
    "dataset_id": "guitarset",
    "dataset_version": "1.1.0",
    "source_url": "https://doi.org/10.5281/zenodo.3371780",
    "audio_license": "CC BY 4.0",
    "annotation_license": "CC BY 4.0",
    "commercial_use": "allowed",
    "attribution": "Qingyang Xi, Rachel M. Bittner, Johan Pauwels, Xuzhou Ye, and Juan P. Bello",
}
SLAKH_PROVENANCE = {
    "dataset_id": "slakh2100-redux",
    "dataset_version": "2021-03-15",
    "source_url": "https://doi.org/10.5281/zenodo.4599666",
    "audio_license": "CC BY 4.0",
    "annotation_license": "CC BY 4.0",
    "commercial_use": "allowed",
    "attribution": "Ethan Manilow, Gordon Wichern, Prem Seetharaman, and Jonathan Le Roux",
}
GUITAR_TECHS_PROVENANCE = {
    "dataset_id": "guitar-techs",
    "dataset_version": "2025-03-28",
    "source_url": "https://zenodo.org/records/14963133",
    "audio_license": "CC BY 4.0",
    "annotation_license": "CC BY 4.0",
    "commercial_use": "allowed",
    "attribution": (
        "Hegel Pedroza Villalobos, Termeh Taheri, Wallace Abreu, "
        "Ryan M. Corey, and Iran R. Roman"
    ),
}


class TrainingDataError(ValueError):
    pass


def _hash_file(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _annotation_source(annotation):
    metadata = annotation.get("annotation_metadata") or {}
    return str(metadata.get("data_source", ""))


def _group_attacks(notes, simultaneous_tolerance):
    notes.sort(key=lambda row: (row["start"], row["pitch"], row["source_id"]))
    attacks = []
    for note in notes:
        if attacks and note["start"] - attacks[-1]["start"] <= simultaneous_tolerance:
            attacks[-1]["notes"].append(note)
        else:
            attacks.append({"start": note["start"], "notes": [note]})
    for attack in attacks:
        rows = attack["notes"]
        attack["end"] = max(row["end"] for row in rows)
        attack["pitches"] = sorted({row["pitch"] for row in rows})
        attack["note_count"] = len(rows)
        attack["polyphony"] = len(attack["pitches"])
        attack["is_chord"] = attack["polyphony"] > 1
        attack["source_ids"] = [row["source_id"] for row in rows]
        del attack["notes"]
    return attacks


def _label_summary(notes, attacks):
    return {
        "notes": len(notes),
        "attacks": len(attacks),
        "chord_attacks": sum(attack["is_chord"] for attack in attacks),
        "polyphony": dict(sorted(Counter(attack["polyphony"] for attack in attacks).items())),
    }


def convert_authorized_chart_pairs(manifest, output):
    """Create aligned Stage B examples from a validated authorized-pair manifest."""
    from .datasets import DatasetManifestError, validate_chart_pair_manifest

    manifest_path = Path(manifest).resolve()
    output_root = Path(output).resolve()
    try:
        validation = validate_chart_pair_manifest(manifest_path, verify_hashes=True)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (DatasetManifestError, OSError, json.JSONDecodeError) as exc:
        raise TrainingDataError(f"Authorized chart-pair manifest is invalid: {exc}") from exc

    try:
        import pretty_midi
    except ImportError as exc:
        raise TrainingDataError("pretty_midi is required for chart-pair conversion") from exc

    examples = []
    for index, item in enumerate(payload["items"]):
        song_id = str(item["song_id"])
        chart_path = manifest_path.parent / item["annotations"]
        try:
            midi = pretty_midi.PrettyMIDI(str(chart_path))
        except (EOFError, OSError, RuntimeError, ValueError) as exc:
            raise TrainingDataError(f"Cannot read authorized chart for {song_id}: {exc}") from exc
        guitar = next(
            (instrument for instrument in midi.instruments if instrument.name == "PART GUITAR"),
            None,
        )
        if guitar is None:
            raise TrainingDataError(f"Authorized chart for {song_id} has no PART GUITAR")

        offset = float(item["synchronization"]["audio_offset_seconds"])
        grouped = {}
        for note in guitar.notes:
            if not 96 <= note.pitch <= 100:
                continue
            key = round(float(note.start), 6)
            grouped.setdefault(key, []).append(note)

        conditioning = []
        targets = []
        for attack_index, (chart_time, notes) in enumerate(sorted(grouped.items())):
            notes.sort(key=lambda note: note.pitch)
            time_seconds = chart_time + offset
            lane_durations = [
                round(float(note.end - note.start), 6)
                for note in notes
            ]
            lanes = [int(note.pitch - 96) for note in notes]
            duration = max(lane_durations)
            conditioning.append({
                "attack_id": f"{song_id}:attack:{attack_index}",
                "time_seconds": round(time_seconds, 6),
                "beat": round(float(midi.time_to_tick(chart_time) / midi.resolution), 6),
                "polyphony": len(lanes),
                "duration_seconds": duration,
            })
            targets.append({
                "attack_id": f"{song_id}:attack:{attack_index}",
                "lanes": lanes,
                "lane_durations_seconds": lane_durations,
            })

        validated = validation["validated_items"][song_id]
        example = {
            "schema": "rb3charts-authoring-example",
            "schema_version": AUTHORING_EXAMPLE_VERSION,
            "song_id": song_id,
            "artist_id": str(item["artist_id"]),
            "chart_author_id": str(item["chart_author_id"]),
            "split": str(item["split"]),
            "split_keys": [
                f"artist:{item['artist_id']}",
                f"chart-author:{item['chart_author_id']}",
            ],
            "source": {
                "manifest_sha256": validation["manifest_sha256"],
                "audio": str(item["audio"]),
                "audio_sha256": validated["audio_sha256"],
                "chart": str(item["annotations"]),
                "chart_sha256": validated["annotations_sha256"],
                "synchronization": validated["synchronization"],
                "audio_rights": validated["audio_rights"],
                "chart_rights": validated["chart_rights"],
            },
            "conditioning_scope": (
                "chart-derived lane-independent proxy; replace with independently "
                "predicted audio structure for held-out inference"
            ),
            "conditioning": conditioning,
            "expert_targets": targets,
            "summary": {
                "attacks": len(conditioning),
                "chord_attacks": sum(row["polyphony"] > 1 for row in conditioning),
                "held_attacks": sum(row["duration_seconds"] >= 0.35 for row in conditioning),
                "lanes_used": sorted({
                    lane for target in targets for lane in target["lanes"]
                }),
            },
        }
        relative = Path("items") / f"{index:05d}.json"
        destination = output_root / relative
        save_json(destination, example)
        examples.append({
            "song_id": song_id,
            "artist_id": example["artist_id"],
            "chart_author_id": example["chart_author_id"],
            "split": example["split"],
            "split_keys": example["split_keys"],
            "artifact": str(relative),
            "artifact_sha256": _hash_file(destination),
            "attacks": len(conditioning),
        })

    index_payload = {
        "schema": "rb3charts-authoring-index",
        "schema_version": AUTHORING_EXAMPLE_VERSION,
        "dataset_id": validation["dataset_id"],
        "dataset_version": validation["dataset_version"],
        "manifest": str(manifest_path),
        "manifest_sha256": validation["manifest_sha256"],
        "examples": examples,
    }
    save_json(output_root / "index.json", index_payload)
    return index_payload


def convert_guitarset_jams(path, output=None, simultaneous_tolerance=0.03):
    """Convert GuitarSet JAMS notes into lane-independent attack groups."""
    source = Path(path).resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingDataError(f"Cannot read GuitarSet annotation: {exc}") from exc
    annotations = payload.get("annotations")
    if not isinstance(annotations, list):
        raise TrainingDataError("GuitarSet annotation requires an annotations array")

    notes = []
    for annotation in annotations:
        if not isinstance(annotation, dict) or annotation.get("namespace") != "note_midi":
            continue
        string = _annotation_source(annotation)
        data = annotation.get("data")
        if not isinstance(data, list):
            raise TrainingDataError("GuitarSet note_midi annotation requires a data array")
        for index, row in enumerate(data):
            try:
                start = float(row["time"])
                duration = float(row["duration"])
                pitch = int(round(float(row["value"])))
            except (KeyError, TypeError, ValueError) as exc:
                raise TrainingDataError(
                    f"Invalid note_midi row for string {string or 'unknown'} at index {index}"
                ) from exc
            if start < 0 or duration <= 0 or not 0 <= pitch <= 127:
                raise TrainingDataError(
                    f"Invalid note timing or pitch for string {string or 'unknown'} at index {index}"
                )
            notes.append({
                "start": start,
                "end": start + duration,
                "pitch": pitch,
                "string": string,
                "source_id": f"string-{string or 'unknown'}-{index}",
            })
    if not notes:
        raise TrainingDataError("GuitarSet annotation contains no note_midi events")
    attacks = _group_attacks(notes, simultaneous_tolerance)

    beats = []
    for annotation in annotations:
        if not isinstance(annotation, dict) or annotation.get("namespace") != "beat_position":
            continue
        data = annotation.get("data")
        if not isinstance(data, list):
            raise TrainingDataError("GuitarSet beat_position annotation requires a data array")
        for row in data:
            try:
                beats.append({
                    "time": float(row["time"]),
                    "position": int(row["value"]["position"]),
                })
            except (KeyError, TypeError, ValueError) as exc:
                raise TrainingDataError("Invalid GuitarSet beat_position row") from exc
        break

    file_metadata = payload.get("file_metadata") or {}
    name_parts = source.stem.split("_")
    group_id = f"guitarset:{name_parts[1]}" if len(name_parts) >= 3 else f"guitarset:{source.stem}"
    performer_id = name_parts[0] if len(name_parts) >= 3 else source.stem
    result = {
        "schema": "rb3charts-structure-labels",
        "schema_version": STRUCTURE_LABEL_VERSION,
        "source_format": "guitarset-jams",
        "source_annotation": source.name,
        "source_annotation_sha256": _hash_file(source),
        "provenance": GUITARSET_PROVENANCE,
        "group_id": group_id,
        "split_keys": [group_id, f"performer:guitarset:{performer_id}"],
        "simultaneous_tolerance_seconds": simultaneous_tolerance,
        "duration": file_metadata.get("duration"),
        "targets": ["attack", "polyphony", "sustain"],
        "notes": notes,
        "attacks": attacks,
        "beats": beats,
        "summary": _label_summary(notes, attacks),
    }
    if output is not None:
        save_json(output, result)
    return result


def _read_midi_notes(path, source_id, close_unterminated=False, diagnostics=None):
    import mido

    midi = mido.MidiFile(path)
    tempo = 500000
    elapsed = 0.0
    absolute_tick = 0
    tempo_segments = [(0, 0.0, tempo)]
    active = {}
    notes = []
    order = 0
    for message in mido.merge_tracks(midi.tracks):
        absolute_tick += message.time
        elapsed += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        if message.type == "set_tempo":
            tempo = message.tempo
            tempo_segments.append((absolute_tick, elapsed, tempo))
        elif message.type == "note_on" and message.velocity > 0:
            active.setdefault((message.channel, message.note), []).append((elapsed, order))
            order += 1
        elif message.type in ("note_off", "note_on"):
            starts = active.get((message.channel, message.note))
            if starts:
                start, note_order = starts.pop(0)
                if not starts:
                    del active[(message.channel, message.note)]
                if elapsed > start:
                    notes.append({
                        "start": start,
                        "end": elapsed,
                        "pitch": int(message.note),
                        "source_id": f"{source_id}-{note_order}",
                        "source": source_id,
                    })
    if active and not close_unterminated:
        raise TrainingDataError(f"MIDI has unterminated notes: {path}")
    closed = 0
    for (_, pitch), starts in active.items():
        for start, note_order in starts:
            if elapsed <= start:
                continue
            notes.append({
                "start": start,
                "end": elapsed,
                "pitch": int(pitch),
                "source_id": f"{source_id}-{note_order}",
                "source": source_id,
            })
            closed += 1
    if diagnostics is not None:
        diagnostics["unterminated_notes_closed_at_track_end"] = closed
    notes.sort(key=lambda row: (row["start"], row["pitch"], row["source_id"]))
    return notes, midi.ticks_per_beat, absolute_tick, tempo_segments


def _midi_beats(ticks_per_beat, max_tick, tempo_segments):
    import mido

    beats = []
    segment_index = 0
    for index, tick in enumerate(range(0, max_tick + 1, ticks_per_beat)):
        while (
            segment_index + 1 < len(tempo_segments)
            and tempo_segments[segment_index + 1][0] <= tick
        ):
            segment_index += 1
        segment_tick, segment_time, tempo = tempo_segments[segment_index]
        time = segment_time + mido.tick2second(tick - segment_tick, ticks_per_beat, tempo)
        beats.append({"time": time, "position": index % 4 + 1})
    return beats


def convert_slakh_track(track_folder, output=None, simultaneous_tolerance=0.03):
    """Convert rendered Slakh guitar-source MIDI into structure labels."""
    import yaml

    track = Path(track_folder).resolve()
    metadata_path = track / "metadata.yaml"
    try:
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TrainingDataError(f"Cannot read Slakh metadata: {exc}") from exc
    if not isinstance(metadata, dict) or not isinstance(metadata.get("stems"), dict):
        raise TrainingDataError("Slakh metadata requires a stems mapping")
    midi_dir = track / str(metadata.get("midi_dir", "MIDI"))
    selected = []
    for stem_id, details in sorted(metadata["stems"].items()):
        if not isinstance(details, dict):
            continue
        if (
            details.get("inst_class") == "Guitar"
            and not details.get("is_drum", False)
        ):
            midi_path = midi_dir / f"{stem_id}.mid"
            if midi_path.is_file():
                selected.append((str(stem_id), midi_path))
    if not selected:
        raise TrainingDataError("Slakh track has no rendered guitar MIDI sources")

    notes = []
    source_files = [{"path": metadata_path.name, "sha256": _hash_file(metadata_path)}]
    for stem_id, midi_path in selected:
        if not midi_path.is_file():
            raise TrainingDataError(f"Slakh guitar MIDI is missing: {midi_path}")
        rows, ticks_per_beat, max_tick, tempo_segments = _read_midi_notes(midi_path, stem_id)
        notes.extend(rows)
        source_files.append({
            "path": str(midi_path.relative_to(track)),
            "sha256": _hash_file(midi_path),
        })
    if not notes:
        raise TrainingDataError("Slakh guitar MIDI sources contain no notes")

    duration = None
    for name in ("mix.flac", "mix.wav"):
        audio_path = track / name
        if audio_path.is_file():
            import soundfile as sf
            duration = float(sf.info(audio_path).duration)
            break
    all_src = track / "all_src.mid"
    if not all_src.is_file():
        raise TrainingDataError("Slakh track is missing all_src.mid for the authoritative tempo map")
    composition_hash = _hash_file(all_src)
    _, ticks_per_beat, max_tick, tempo_segments = _read_midi_notes(all_src, "all-src")
    source_files.append({"path": all_src.name, "sha256": composition_hash})
    attacks = _group_attacks(notes, simultaneous_tolerance)
    beats = _midi_beats(ticks_per_beat, max_tick, tempo_segments)
    if duration is not None and len(beats) >= 2:
        interval = beats[-1]["time"] - beats[-2]["time"]
        while beats[-1]["time"] + interval <= duration + 1e-6:
            beats.append({
                "time": beats[-1]["time"] + interval,
                "position": len(beats) % 4 + 1,
            })
    result = {
        "schema": "rb3charts-structure-labels",
        "schema_version": STRUCTURE_LABEL_VERSION,
        "source_format": "slakh-midi",
        "source_annotation": track.name,
        "source_files": source_files,
        "provenance": SLAKH_PROVENANCE,
        "composition_id": metadata.get("UUID"),
        "group_id": f"slakh:{metadata.get('UUID') or track.name}",
        "split_keys": [
            f"slakh:{metadata.get('UUID') or track.name}",
            f"slakh-midi:{composition_hash}",
        ],
        "simultaneous_tolerance_seconds": simultaneous_tolerance,
        "duration": duration,
        "targets": ["attack", "polyphony", "sustain"],
        "notes": notes,
        "attacks": attacks,
        "beats": beats,
        "summary": dict(
            _label_summary(notes, attacks),
            guitar_sources=len(selected),
        ),
    }
    if output is not None:
        save_json(output, result)
    return result


def convert_guitar_techs_midi(
    midi_path,
    audio_path,
    output=None,
    performer_id="P3",
    simultaneous_tolerance=0.03,
):
    """Convert one Guitar-TECHS excerpt into lane-independent structure labels."""
    import soundfile as sf

    midi_path = Path(midi_path).resolve()
    audio_path = Path(audio_path).resolve()
    if not midi_path.is_file():
        raise TrainingDataError(f"Guitar-TECHS MIDI does not exist: {midi_path}")
    if not audio_path.is_file():
        raise TrainingDataError(f"Guitar-TECHS audio does not exist: {audio_path}")
    performer_id = str(performer_id).strip()
    if not performer_id:
        raise TrainingDataError("Guitar-TECHS conversion requires performer_id")
    try:
        duration = float(sf.info(audio_path).duration)
    except (OSError, RuntimeError) as exc:
        raise TrainingDataError(f"Cannot read Guitar-TECHS audio: {exc}") from exc
    annotation_repairs = {}
    rows, ticks_per_beat, max_tick, tempo_segments = _read_midi_notes(
        midi_path,
        f"guitar-techs-{performer_id}",
        close_unterminated=True,
        diagnostics=annotation_repairs,
    )
    if not rows:
        raise TrainingDataError("Guitar-TECHS MIDI contains no notes")
    if max(row["end"] for row in rows) > duration + 0.15:
        raise TrainingDataError("Guitar-TECHS MIDI extends beyond the aligned audio")
    composition_hash = _hash_file(midi_path)
    group_id = f"guitar-techs:{performer_id}:{midi_path.stem}"
    attacks = _group_attacks(rows, simultaneous_tolerance)
    result = {
        "schema": "rb3charts-structure-labels",
        "schema_version": STRUCTURE_LABEL_VERSION,
        "source_format": "guitar-techs-midi",
        "source_annotation": midi_path.name,
        "source_annotation_sha256": composition_hash,
        "source_files": [{"path": midi_path.name, "sha256": composition_hash}],
        "provenance": GUITAR_TECHS_PROVENANCE,
        "group_id": group_id,
        "split_keys": [group_id, f"performer:guitar-techs:{performer_id}"],
        "performer_id": performer_id,
        "simultaneous_tolerance_seconds": simultaneous_tolerance,
        "alignment_tolerance_seconds": 0.1,
        "annotation_repairs": annotation_repairs,
        "duration": duration,
        "targets": ["attack", "polyphony", "sustain"],
        "notes": rows,
        "attacks": attacks,
        "beats": _midi_beats(ticks_per_beat, max_tick, tempo_segments),
        "summary": _label_summary(rows, attacks),
    }
    if output is not None:
        save_json(output, result)
    return result


def mix_slakh_guitar_stems(track_folder, output):
    """Sum aligned rendered Slakh Guitar-class stems without normalization."""
    import soundfile as sf
    import yaml

    track = Path(track_folder).resolve()
    output = Path(output).resolve()
    if output.suffix.lower() != ".wav":
        raise TrainingDataError("Slakh guitar mix output must use the .wav suffix")
    metadata_path = track / "metadata.yaml"
    try:
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TrainingDataError(f"Cannot read Slakh metadata: {exc}") from exc
    if not isinstance(metadata, dict) or not isinstance(metadata.get("stems"), dict):
        raise TrainingDataError("Slakh metadata requires a stems mapping")
    audio_dir = track / str(metadata.get("audio_dir", "stems"))
    paths = []
    for stem_id, details in sorted(metadata["stems"].items()):
        if (
            isinstance(details, dict)
            and details.get("inst_class") == "Guitar"
            and not details.get("is_drum", False)
        ):
            path = next(
                (audio_dir / f"{stem_id}{suffix}" for suffix in (".wav", ".flac")
                 if (audio_dir / f"{stem_id}{suffix}").is_file()),
                None,
            )
            if path is not None:
                paths.append(path)
    if not paths:
        raise TrainingDataError("Slakh track has no rendered guitar audio stems")
    sample_rate = None
    mixture = None
    for path in paths:
        audio, current_rate = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)
        if sample_rate is None:
            sample_rate = current_rate
            mixture = np.zeros(len(audio), dtype=np.float32)
        if current_rate != sample_rate or len(audio) != len(mixture):
            raise TrainingDataError("Slakh guitar stems are not sample-aligned")
        mixture += audio
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, mixture, sample_rate, subtype="FLOAT")
    return {
        "output": str(output),
        "sample_rate": sample_rate,
        "samples": len(mixture),
        "sources": [
            {"path": str(path.relative_to(track)), "sha256": _hash_file(path)}
            for path in paths
        ],
        "peak": float(np.max(np.abs(mixture))),
    }


def _extract_audio_features(audio_path, beat_times):
    from audioread.exceptions import NoBackendError
    import librosa
    from soundfile import LibsndfileError

    audio_path = Path(audio_path).resolve()
    try:
        y, _ = librosa.load(audio_path, sr=FEATURE_SAMPLE_RATE, mono=True)
    except (OSError, ValueError, LibsndfileError, NoBackendError) as exc:
        raise TrainingDataError(f"Cannot read structure-model audio: {exc}") from exc
    if not len(y):
        raise TrainingDataError("Structure-model audio is empty")
    audio_duration = len(y) / FEATURE_SAMPLE_RATE
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=FEATURE_SAMPLE_RATE,
        n_fft=2048,
        hop_length=FEATURE_HOP_LENGTH,
        n_mels=FEATURE_MELS,
        fmin=30,
        fmax=FEATURE_SAMPLE_RATE / 2,
        power=2,
    )
    log_mel = np.clip(
        (librosa.power_to_db(mel, ref=1.0, top_db=80) + 80) / 80,
        0,
        1,
    )
    log_mel = log_mel.T.astype(np.float32)
    mel_delta = np.maximum(
        np.diff(log_mel, axis=0, prepend=log_mel[:1]),
        0,
    ).astype(np.float32)
    frame_times = librosa.frames_to_time(
        np.arange(len(log_mel)),
        sr=FEATURE_SAMPLE_RATE,
        hop_length=FEATURE_HOP_LENGTH,
    ).astype(np.float32)
    beat_phase = np.zeros((len(frame_times), 2), dtype=np.float32)
    beat_times = np.asarray(sorted(float(time) for time in beat_times), dtype=float)
    beat_phase_valid = bool(len(beat_times) >= 2 and np.all(np.diff(beat_times) > 0))
    beat_phase_coverage = 0.0
    if beat_phase_valid:
        covered = (frame_times >= beat_times[0]) & (frame_times <= beat_times[-1])
        beat_phase_coverage = float(np.mean(covered))
        beat_index = np.interp(
            frame_times[covered],
            beat_times,
            np.arange(len(beat_times), dtype=float),
        )
        beat_phase[covered, 0] = np.sin(2 * np.pi * beat_index)
        beat_phase[covered, 1] = np.cos(2 * np.pi * beat_index)
    return {
        "duration": audio_duration,
        "log_mel": log_mel,
        "mel_delta": mel_delta,
        "frame_times": frame_times,
        "beat_phase": beat_phase,
        "beat_count": len(beat_times),
        "beat_phase_valid": beat_phase_valid,
        "beat_phase_coverage": beat_phase_coverage,
    }


def extract_structure_features(audio_path, labels_path, output):
    """Write deterministic log-mel features and lane-independent frame targets."""
    audio_path = Path(audio_path).resolve()
    labels_path = Path(labels_path).resolve()
    output = Path(output).resolve()
    if output.suffix.lower() != ".npz":
        raise TrainingDataError("Structure feature output must use the .npz suffix")
    try:
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingDataError(f"Cannot read structure labels: {exc}") from exc
    if (
        labels.get("schema") != "rb3charts-structure-labels"
        or labels.get("schema_version") != STRUCTURE_LABEL_VERSION
    ):
        raise TrainingDataError("Unsupported structure-label schema")
    beat_times = [beat["time"] for beat in labels.get("beats", [])]
    audio = _extract_audio_features(audio_path, beat_times)
    audio_duration = audio["duration"]
    log_mel = audio["log_mel"]
    mel_delta = audio["mel_delta"]
    frame_times = audio["frame_times"]
    beat_phase = audio["beat_phase"]
    duration_tolerance = max(0.1, 2 * FEATURE_HOP_LENGTH / FEATURE_SAMPLE_RATE)
    label_duration = labels.get("duration")
    if label_duration is not None and abs(float(label_duration) - audio_duration) > duration_tolerance:
        raise TrainingDataError(
            f"Audio/label duration mismatch: audio is {audio_duration:.3f}s, labels are {float(label_duration):.3f}s"
        )
    attack_target = np.zeros(len(frame_times), dtype=np.float32)
    polyphony_target = np.zeros(len(frame_times), dtype=np.int8)
    sustain_target = np.zeros(len(frame_times), dtype=np.int8)
    attack_frames = []

    for attack in labels.get("attacks", []):
        start = float(attack["start"])
        if start < 0 or start > audio_duration + duration_tolerance:
            raise TrainingDataError(f"Attack at {start:.3f}s is outside the training audio")
        frame = int(np.searchsorted(frame_times, start, side="left"))
        if frame >= len(frame_times):
            frame = len(frame_times) - 1
        attack_target[frame] = 1
        polyphony_target[frame] = min(6, int(attack["polyphony"]))
        attack_frames.append(frame)
    for note in labels.get("notes", []):
        start = float(note["start"])
        end = float(note["end"])
        if end <= start:
            raise TrainingDataError("Structure labels contain a non-positive note duration")
        if start < 0 or start > audio_duration + duration_tolerance:
            raise TrainingDataError(
                f"Note from {start:.3f}s to {end:.3f}s is outside the training audio"
            )
        active = (frame_times >= start) & (frame_times < min(end, audio_duration))
        sustain_target[active] = np.minimum(6, sustain_target[active] + 1)
    for frame in attack_frames:
        sustain_target[frame] = max(sustain_target[frame], polyphony_target[frame])

    metadata = {
        "schema": "rb3charts-structure-features",
        "schema_version": FEATURE_VERSION,
        "sample_rate": FEATURE_SAMPLE_RATE,
        "hop_length": FEATURE_HOP_LENGTH,
        "mel_bins": FEATURE_MELS,
        "delta_bins": FEATURE_MELS,
        "audio_sha256": _hash_file(audio_path),
        "labels_sha256": _hash_file(labels_path),
        "source_annotation_hashes": sorted({
            str(value)
            for value in (
                [labels.get("source_annotation_sha256")]
                + [
                    row.get("sha256")
                    for row in labels.get("source_files", [])
                    if isinstance(row, dict)
                ]
            )
            if value
        }),
        "audio_name": audio_path.name,
        "labels_name": labels_path.name,
        "frames": len(frame_times),
        "attack_positives": int(np.sum(attack_target)),
        "beat_count": audio["beat_count"],
        "beat_phase_valid": audio["beat_phase_valid"],
        "beat_phase_coverage": audio["beat_phase_coverage"],
        "targets": ["attack", "polyphony", "sustain"],
        "provenance": labels.get("provenance"),
        "group_id": labels.get("group_id"),
        "split_keys": labels.get("split_keys"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        log_mel=log_mel,
        mel_delta=mel_delta,
        frame_times=frame_times,
        beat_phase=beat_phase,
        attack_target=attack_target,
        polyphony_target=polyphony_target,
        sustain_target=sustain_target,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return metadata


def _load_feature_metadata(path):
    try:
        with np.load(path) as artifact:
            return json.loads(str(artifact["metadata"]))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise TrainingDataError(f"Cannot read structure feature metadata {path}: {exc}") from exc


def _load_feature_index(path):
    index_path = Path(path).resolve()
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingDataError(f"Cannot read feature index: {exc}") from exc
    if payload.get("schema") != "rb3charts-structure-feature-index" or payload.get("schema_version") != 2:
        raise TrainingDataError("Unsupported structure feature-index schema")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise TrainingDataError("Structure feature index requires a non-empty items array")
    root = index_path.parent
    groups = {}
    split_keys = {}
    manifests = {}
    resolved = []
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            raise TrainingDataError(f"Feature index item {position} must be an object")
        split = item.get("split")
        if split not in ("train", "validation", "test"):
            raise TrainingDataError(f"Feature index item {position} has an invalid split")
        group_id = str(item.get("group_id") or "")
        if not group_id:
            raise TrainingDataError(f"Feature index item {position} requires group_id")
        previous = groups.setdefault(group_id, split)
        if previous != split:
            raise TrainingDataError(f"Feature group {group_id} crosses {previous}/{split} splits")
        relative = Path(str(item.get("features") or ""))
        if relative.is_absolute():
            raise TrainingDataError(f"Feature index path must be relative: {relative}")
        feature_path = (root / relative).resolve()
        try:
            feature_path.relative_to(root)
        except ValueError as exc:
            raise TrainingDataError(f"Feature index path escapes its root: {relative}") from exc
        if not feature_path.is_file():
            raise TrainingDataError(f"Feature artifact does not exist: {relative}")
        manifest_relative = Path(str(item.get("manifest") or ""))
        if manifest_relative.is_absolute():
            raise TrainingDataError(f"Dataset manifest path must be relative: {manifest_relative}")
        manifest_path = (root / manifest_relative).resolve()
        try:
            manifest_path.relative_to(root)
        except ValueError as exc:
            raise TrainingDataError(
                f"Dataset manifest path escapes the feature-index root: {manifest_relative}"
            ) from exc
        if manifest_path not in manifests:
            from .datasets import DatasetManifestError, validate_dataset_manifest
            try:
                manifests[manifest_path] = validate_dataset_manifest(
                    manifest_path,
                    verify_hashes=True,
                )
            except DatasetManifestError as exc:
                raise TrainingDataError(f"Invalid dataset manifest {manifest_relative}: {exc}") from exc
        manifest = manifests[manifest_path]
        song_id = str(item.get("song_id") or "")
        manifest_item = manifest["validated_items"].get(song_id)
        if manifest_item is None:
            raise TrainingDataError(
                f"Feature index song_id {song_id or '<empty>'} is absent from {manifest_relative}"
            )
        if manifest_item["split"] != split:
            raise TrainingDataError(
                f"Feature split {split} does not match manifest split "
                f"{manifest_item['split']} for {song_id}"
            )
        metadata = _load_feature_metadata(feature_path)
        if metadata.get("schema_version") != FEATURE_VERSION:
            raise TrainingDataError(f"Incompatible feature version in {relative}")
        artifact_group = str(metadata.get("group_id") or "")
        if not artifact_group:
            raise TrainingDataError(f"Feature artifact has no authoritative group_id: {relative}")
        if artifact_group != group_id:
            raise TrainingDataError(
                f"Feature index group_id {group_id} does not match artifact group_id {artifact_group}: {relative}"
            )
        if metadata.get("audio_sha256") != manifest_item["audio_sha256"]:
            raise TrainingDataError(
                f"Feature audio hash does not match dataset manifest for {song_id}: {relative}"
            )
        annotation_hashes = {
            str(metadata.get("labels_sha256") or ""),
            *[str(value) for value in metadata.get("source_annotation_hashes", [])],
        }
        if manifest_item["annotations_sha256"] not in annotation_hashes:
            raise TrainingDataError(
                f"Feature annotation lineage does not match dataset manifest for {song_id}: {relative}"
            )
        provenance = metadata.get("provenance")
        if not isinstance(provenance, dict):
            provenance = {}
        expected_provenance = manifest["dataset"]
        provenance_keys = (
            "id",
            "version",
            "source_url",
            "audio_license",
            "annotation_license",
            "commercial_use",
            "attribution",
        )
        actual_values = {
            "id": (provenance or {}).get("dataset_id"),
            "version": (provenance or {}).get("dataset_version"),
            **{
                key: (provenance or {}).get(key)
                for key in provenance_keys
                if key not in ("id", "version")
            },
        }
        if any(actual_values[key] != expected_provenance.get(key) for key in provenance_keys):
            raise TrainingDataError(
                f"Feature provenance does not match dataset manifest for {song_id}: {relative}"
            )
        artifact_split_keys = metadata.get("split_keys")
        if not isinstance(artifact_split_keys, list) or not artifact_split_keys:
            raise TrainingDataError(f"Feature artifact has no authoritative split_keys: {relative}")
        for key in artifact_split_keys:
            key = str(key)
            previous_split = split_keys.setdefault(key, split)
            if previous_split != split:
                raise TrainingDataError(f"Feature split key {key} crosses {previous_split}/{split} splits")
        resolved.append({
            "song_id": song_id,
            "group_id": group_id,
            "split": split,
            "path": feature_path,
            "sha256": _hash_file(feature_path),
            "metadata": metadata,
            "manifest_sha256": manifest["manifest_sha256"],
            "manifest_path": manifest_path,
            "manifest_relative": str(manifest_relative),
            "dataset": manifest["dataset"],
        })
    if not any(item["split"] == "train" for item in resolved):
        raise TrainingDataError("Feature index requires at least one training item")
    if not any(item["split"] == "validation" for item in resolved):
        raise TrainingDataError("Feature index requires at least one validation item")
    return index_path, resolved


def _load_feature_artifact(path):
    try:
        with np.load(path) as artifact:
            arrays = {
                name: artifact[name].copy()
                for name in (
                    "log_mel",
                    "mel_delta",
                    "beat_phase",
                    "attack_target",
                    "polyphony_target",
                    "sustain_target",
                )
            }
            metadata = json.loads(str(artifact["metadata"]))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise TrainingDataError(f"Cannot read structure feature artifact {path}: {exc}") from exc
    frames = len(arrays["log_mel"])
    if (
        arrays["log_mel"].shape != (frames, FEATURE_MELS)
        or arrays["mel_delta"].shape != (frames, FEATURE_MELS)
        or arrays["beat_phase"].shape != (frames, 2)
        or any(len(arrays[name]) != frames for name in ("attack_target", "polyphony_target", "sustain_target"))
    ):
        raise TrainingDataError(f"Incompatible feature shapes in {path}")
    if metadata.get("schema_version") != FEATURE_VERSION:
        raise TrainingDataError(f"Incompatible feature version in {path}")
    return arrays, metadata


def _attack_peaks(scores, threshold=None):
    from scipy import signal

    distance = max(1, round(0.05 * FEATURE_SAMPLE_RATE / FEATURE_HOP_LENGTH))
    kwargs = {"distance": distance}
    if threshold is not None:
        kwargs["height"] = threshold
    peaks, _ = signal.find_peaks(scores, **kwargs)
    return peaks


def _onset_metrics(records, threshold, tolerance_frames):
    matched = predicted_total = actual_total = 0
    for scores, actual_frames in records:
        predicted_frames = _attack_peaks(scores, threshold).tolist()
        unmatched_actual = set(range(len(actual_frames)))
        for predicted_frame in predicted_frames:
            candidates = [
                index for index in unmatched_actual
                if abs(actual_frames[index] - predicted_frame) <= tolerance_frames
            ]
            if candidates:
                best = min(candidates, key=lambda index: abs(actual_frames[index] - predicted_frame))
                unmatched_actual.remove(best)
                matched += 1
        predicted_total += len(predicted_frames)
        actual_total += len(actual_frames)
    precision = matched / max(1, predicted_total)
    recall = matched / max(1, actual_total)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_attacks": matched,
        "predicted_attacks": predicted_total,
        "positive_attacks": actual_total,
    }


def train_structure_baseline(index_path, output, epochs=1, device="auto", seed=1337):
    """Train a deterministic first-pass attack/polyphony/sustain network."""
    import torch
    import torch.nn.functional as functional

    from .structure_model import build_structure_baseline

    if epochs < 1:
        raise TrainingDataError("Training epochs must be at least one")
    output = Path(output).resolve()
    if output.suffix.lower() != ".pt":
        raise TrainingDataError("Structure model output must use the .pt suffix")
    index_path, items = _load_feature_index(index_path)
    requested_device = device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise TrainingDataError("Training device must be auto, cpu, or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise TrainingDataError("CUDA training was requested but is unavailable")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    model = build_structure_baseline(
        input_features=FEATURE_MELS * 2 + 2,
        hidden_features=64,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_items = [item for item in items if item["split"] == "train"]
    validation_items = [item for item in items if item["split"] == "validation"]
    test_items = [item for item in items if item["split"] == "test"]
    positives = 0.0
    frames = 0
    provenance = {}
    dataset_manifests = {}
    for item in train_items:
        metadata = item["metadata"]
        if item["split"] == "train":
            positives += float(metadata.get("attack_positives", 0))
            frames += int(metadata.get("frames", 0))
        details = item["dataset"]
        provenance[str(details["id"])] = {
            "dataset_id": details["id"],
            "dataset_version": details["version"],
            **{
                key: details[key]
                for key in (
                    "source_url",
                    "audio_license",
                    "annotation_license",
                    "commercial_use",
                    "attribution",
                )
            },
        }
        dataset_manifests[str(item["manifest_path"])] = {
            "path": item["manifest_relative"],
            "sha256": item["manifest_sha256"],
        }
    if positives <= 0:
        raise TrainingDataError("Training features contain no positive attacks")
    positive_weight = min(50.0, max(1.0, (frames - positives) / positives))
    epoch_losses = []

    def tensors(item):
        arrays, _ = _load_feature_artifact(item["path"])
        features = np.concatenate(
            (arrays["log_mel"], arrays["mel_delta"], arrays["beat_phase"]),
            axis=1,
        )
        return (
            torch.from_numpy(features).unsqueeze(0).to(device),
            torch.from_numpy(arrays["attack_target"]).unsqueeze(0).to(device),
            torch.from_numpy(arrays["polyphony_target"].astype(np.int64)).unsqueeze(0).to(device),
            torch.from_numpy(arrays["sustain_target"].astype(np.int64)).unsqueeze(0).to(device),
        )

    for _ in range(epochs):
        model.train()
        total = 0.0
        for item in train_items:
            features, attacks, polyphony, sustain = tensors(item)
            outputs = model(features)
            attack_loss = functional.binary_cross_entropy_with_logits(
                outputs["attack"],
                attacks,
                pos_weight=torch.tensor(positive_weight, device=device),
            )
            attack_mask = attacks > 0.5
            if torch.any(attack_mask):
                polyphony_loss = functional.cross_entropy(
                    outputs["polyphony"][attack_mask],
                    polyphony[attack_mask],
                )
            else:
                polyphony_loss = outputs["polyphony"].sum() * 0
            sustain_loss = functional.cross_entropy(
                outputs["sustain"].reshape(-1, 7),
                sustain.reshape(-1),
            )
            loss = attack_loss + 0.5 * polyphony_loss + 0.25 * sustain_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
        epoch_losses.append(total / len(train_items))

    model.eval()
    onset_records = []
    polyphony_correct = polyphony_total = 0
    tolerance_frames = max(1, round(0.05 * FEATURE_SAMPLE_RATE / FEATURE_HOP_LENGTH))
    with torch.no_grad():
        for item in validation_items:
            features, attacks, polyphony, _ = tensors(item)
            outputs = model(features)
            actual = attacks > 0.5
            scores = torch.sigmoid(outputs["attack"][0]).cpu().numpy()
            actual_frames = torch.nonzero(actual[0], as_tuple=False).flatten().cpu().tolist()
            onset_records.append((scores, actual_frames))
            if torch.any(actual):
                predicted_polyphony = torch.argmax(outputs["polyphony"], dim=-1)
                polyphony_correct += int(torch.sum(predicted_polyphony[actual] == polyphony[actual]).cpu())
                polyphony_total += int(torch.sum(actual).cpu())
    fixed_metrics = _onset_metrics(onset_records, 0.5, tolerance_frames)
    candidates = [
        _onset_metrics(onset_records, threshold, tolerance_frames)
        for threshold in np.linspace(0.05, 0.95, 37)
    ]
    calibrated = max(candidates, key=lambda row: (row["f1"], row["precision"], row["threshold"]))
    test_metrics = None
    test_polyphony_correct = test_polyphony_total = 0
    if test_items:
        test_records = []
        with torch.no_grad():
            for item in test_items:
                features, attacks, polyphony, _ = tensors(item)
                outputs = model(features)
                actual = attacks > 0.5
                scores = torch.sigmoid(outputs["attack"][0]).cpu().numpy()
                actual_frames = torch.nonzero(actual[0], as_tuple=False).flatten().cpu().tolist()
                test_records.append((scores, actual_frames))
                if torch.any(actual):
                    predicted_polyphony = torch.argmax(outputs["polyphony"], dim=-1)
                    test_polyphony_correct += int(
                        torch.sum(predicted_polyphony[actual] == polyphony[actual]).cpu()
                    )
                    test_polyphony_total += int(torch.sum(actual).cpu())
        test_metrics = _onset_metrics(test_records, calibrated["threshold"], tolerance_frames)
        test_metrics["polyphony_accuracy"] = (
            test_polyphony_correct / max(1, test_polyphony_total)
        )
    report = {
        "model": "structure-baseline-conv1d",
        "model_version": 1,
        "feature_version": FEATURE_VERSION,
        "device_requested": requested_device,
        "device_used": device,
        "seed": seed,
        "epochs": epochs,
        "training_items": len(train_items),
        "validation_items": len(validation_items),
        "test_items": len(test_items),
        "positive_weight": positive_weight,
        "epoch_losses": epoch_losses,
        "validation": {
            "attack_threshold": calibrated["threshold"],
            "attack_precision": calibrated["precision"],
            "attack_recall": calibrated["recall"],
            "attack_f1": calibrated["f1"],
            "polyphony_accuracy": polyphony_correct / max(1, polyphony_total),
            "positive_attacks": calibrated["positive_attacks"],
            "onset_tolerance_ms": 50,
            "fixed_threshold_0_5": fixed_metrics,
        },
        "feature_index_sha256": _hash_file(index_path),
        "training_data": [provenance[key] for key in sorted(provenance)],
        "dataset_manifests": [
            dataset_manifests[key]
            for key in sorted(dataset_manifests)
        ],
    }
    if test_metrics is not None:
        report["test"] = test_metrics
    artifact_digest = sha256()
    for item in sorted(items, key=lambda row: (row["split"], row["song_id"])):
        artifact_digest.update(f"{item['split']}:{item['song_id']}:{item['sha256']}\n".encode())
    report["feature_artifacts_sha256"] = artifact_digest.hexdigest()
    try:
        from importlib.metadata import version
        report["runtime_versions"] = {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "librosa": version("librosa"),
        }
    except Exception:
        report["runtime_versions"] = {
            "torch": torch.__version__,
            "numpy": np.__version__,
        }
    checkpoint = {
        "schema": "rb3charts-structure-model",
        "schema_version": 2,
        "model": report["model"],
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "configuration": {
            "input_features": FEATURE_MELS * 2 + 2,
            "hidden_features": 64,
            "sample_rate": FEATURE_SAMPLE_RATE,
            "hop_length": FEATURE_HOP_LENGTH,
            "mel_bins": FEATURE_MELS,
            "delta_bins": FEATURE_MELS,
            "attack_threshold": calibrated["threshold"],
        },
        "training_report": report,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    save_json(output.with_suffix(".json"), report)
    return report


def predict_structure(model_path, audio_path, output, beats_path=None, device="auto"):
    """Run lane-independent structure inference without changing a chart."""
    import torch

    from .structure_model import build_structure_baseline

    model_path = Path(model_path).resolve()
    audio_path = Path(audio_path).resolve()
    output = Path(output).resolve()
    if output.suffix.lower() != ".json":
        raise TrainingDataError("Structure prediction output must use the .json suffix")
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise TrainingDataError(f"Cannot read structure model: {exc}") from exc
    if checkpoint.get("schema") != "rb3charts-structure-model" or checkpoint.get("schema_version") != 2:
        raise TrainingDataError("Unsupported structure-model checkpoint")
    configuration = checkpoint.get("configuration") or {}
    expected = {
        "sample_rate": FEATURE_SAMPLE_RATE,
        "hop_length": FEATURE_HOP_LENGTH,
        "mel_bins": FEATURE_MELS,
        "delta_bins": FEATURE_MELS,
    }
    if (
        checkpoint.get("training_report", {}).get("feature_version") != FEATURE_VERSION
        or any(configuration.get(key) != value for key, value in expected.items())
    ):
        raise TrainingDataError("Structure model uses incompatible feature parameters")

    beat_times = []
    if beats_path is not None:
        try:
            beat_payload = json.loads(Path(beats_path).read_text(encoding="utf-8"))
            beat_times = [float(value) for value in beat_payload["beats"]]
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise TrainingDataError(f"Cannot read beat map for structure inference: {exc}") from exc
    audio = _extract_audio_features(audio_path, beat_times)
    features = np.concatenate(
        (audio["log_mel"], audio["mel_delta"], audio["beat_phase"]),
        axis=1,
    )
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise TrainingDataError("Inference device must be auto, cpu, or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise TrainingDataError("CUDA inference was requested but is unavailable")
    model = build_structure_baseline(
        input_features=int(configuration.get("input_features", FEATURE_MELS * 2 + 2)),
        hidden_features=int(configuration.get("hidden_features", 64)),
    )
    try:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    except (KeyError, RuntimeError) as exc:
        raise TrainingDataError(f"Structure-model weights are incompatible: {exc}") from exc
    model.to(device).eval()
    with torch.no_grad():
        outputs = model(torch.from_numpy(features).unsqueeze(0).to(device))
        attack_scores = torch.sigmoid(outputs["attack"][0]).cpu().numpy()
        polyphony = torch.argmax(outputs["polyphony"][0], dim=-1).cpu().numpy()
        sustain = torch.argmax(outputs["sustain"][0], dim=-1).cpu().numpy()
    threshold = float(configuration.get("attack_threshold", 0.5))
    candidate_peaks = _attack_peaks(attack_scores)
    candidate_heights=attack_scores[candidate_peaks]
    rank_percentiles=np.zeros(len(candidate_peaks),dtype=float)
    if len(candidate_peaks)==1:
        rank_percentiles[0]=1.0
    elif len(candidate_peaks)>1:
        order=np.argsort(candidate_heights,kind="stable")
        rank_percentiles[order]=np.linspace(0.0,1.0,len(candidate_peaks))
    rank_by_frame={
        int(frame):float(percentile)
        for frame,percentile in zip(candidate_peaks,rank_percentiles)
    }

    def event_rows(peaks, heights, prefix):
        rows = []
        for index, frame in enumerate(peaks):
            voices = max(1, int(polyphony[frame]))
            rows.append({
                "source_id": f"{prefix}-{index}",
                "start": float(audio["frame_times"][frame]),
                "attack_probability": float(heights[index]),
                "attack_rank_percentile": rank_by_frame[int(frame)],
                "polyphony": voices,
                "is_chord": voices > 1,
                "active_voices": int(sustain[frame]),
            })
        return rows

    candidates = event_rows(
        candidate_peaks,
        attack_scores[candidate_peaks],
        "structure-candidate",
    )
    accepted = candidate_peaks[attack_scores[candidate_peaks] >= threshold]
    events = event_rows(
        accepted,
        attack_scores[accepted],
        "structure",
    )
    result = {
        "schema": "rb3charts-structure-predictions",
        "schema_version": 1,
        "scope": "lane-independent evaluation; not applied to chart generation",
        "model": {
            "path": model_path.name,
            "sha256": _hash_file(model_path),
            "feature_version": checkpoint["training_report"]["feature_version"],
            "attack_threshold": threshold,
        },
        "audio": {
            "path": audio_path.name,
            "sha256": _hash_file(audio_path),
            "duration": audio["duration"],
        },
        "beat_phase": {
            "count": audio["beat_count"],
            "valid": audio["beat_phase_valid"],
            "coverage": audio["beat_phase_coverage"],
        },
        "summary": {
            "candidate_peaks": len(candidates),
            "attacks": len(events),
            "chord_attacks": sum(event["is_chord"] for event in events),
            "chord_rate": sum(event["is_chord"] for event in events) / max(1, len(events)),
            "candidate_score_quantiles": {
                str(quantile):float(np.quantile(candidate_heights,quantile))
                for quantile in (.5,.75,.9,.95)
            } if len(candidate_heights) else {},
        },
        "candidates": candidates,
        "events": events,
    }
    save_json(output, result)
    return result
