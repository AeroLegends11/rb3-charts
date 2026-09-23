import hashlib
import json

import numpy as np
import pytest
import soundfile as sf
import yaml

from rb3charts.training import (
    FEATURE_HOP_LENGTH,
    FEATURE_MELS,
    FEATURE_SAMPLE_RATE,
    TrainingDataError,
    _onset_metrics,
    convert_guitarset_jams,
    convert_guitar_techs_midi,
    convert_slakh_track,
    extract_structure_features,
    mix_slakh_guitar_stems,
    predict_structure,
    train_structure_baseline,
)


def _annotation(tmp_path, notes=True, name="fixture.jams"):
    annotations = [
        {
            "namespace": "note_midi",
            "annotation_metadata": {"data_source": "0"},
            "data": [
                {"time": 1.0, "duration": 0.4, "value": 48},
                {"time": 2.0, "duration": 0.2, "value": 50},
            ] if notes else [],
        },
        {
            "namespace": "note_midi",
            "annotation_metadata": {"data_source": "1"},
            "data": [
                {"time": 1.012, "duration": 0.3, "value": 55},
                {"time": 2.2, "duration": 0.1, "value": 57},
            ] if notes else [],
        },
        {
            "namespace": "beat_position",
            "data": [
                {"time": 0.5, "duration": 0, "value": {"position": 1}},
                {"time": 1.0, "duration": 0, "value": {"position": 2}},
            ],
        },
    ]
    path = tmp_path / name
    path.write_text(json.dumps({
        "file_metadata": {"duration": 3.0},
        "annotations": annotations,
    }), encoding="utf-8")
    return path


def test_guitarset_conversion_groups_polyphony_without_gameplay_lanes(tmp_path):
    source = _annotation(tmp_path)
    output = tmp_path / "labels.json"
    labels = convert_guitarset_jams(source, output)
    assert labels["schema_version"] == 1
    assert labels["summary"] == {
        "notes": 4,
        "attacks": 3,
        "chord_attacks": 1,
        "polyphony": {1: 2, 2: 1},
    }
    assert labels["attacks"][0]["pitches"] == [48, 55]
    assert labels["attacks"][0]["polyphony"] == 2
    assert labels["attacks"][0]["end"] == pytest.approx(1.4)
    assert labels["beats"][0] == {"time": 0.5, "position": 1}
    assert all("lanes" not in attack for attack in labels["attacks"])
    assert json.loads(output.read_text(encoding="utf-8"))["schema"] == "rb3charts-structure-labels"


def test_guitarset_conversion_rejects_empty_note_annotations(tmp_path):
    with pytest.raises(TrainingDataError, match="no note_midi"):
        convert_guitarset_jams(_annotation(tmp_path, notes=False))


def test_guitarset_conversion_rejects_invalid_note_values(tmp_path):
    source = _annotation(tmp_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["annotations"][0]["data"][0]["duration"] = -0.1
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TrainingDataError, match="Invalid note timing"):
        convert_guitarset_jams(source)


def test_guitarset_conversion_treats_same_pitch_on_two_strings_as_one_voice(tmp_path):
    source = _annotation(tmp_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["annotations"][1]["data"][0]["value"] = 48
    source.write_text(json.dumps(payload), encoding="utf-8")
    labels = convert_guitarset_jams(source)
    assert labels["attacks"][0]["pitches"] == [48]
    assert labels["attacks"][0]["note_count"] == 2
    assert labels["attacks"][0]["polyphony"] == 1
    assert labels["attacks"][0]["is_chord"] is False


def test_guitar_techs_conversion_preserves_performer_split_and_polyphony(tmp_path):
    import mido

    midi_path = tmp_path / "midi_01.mid"
    midi = mido.MidiFile(ticks_per_beat=480)
    tempo = mido.MidiTrack()
    tempo.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    midi.tracks.append(tempo)
    for pitch in (48, 55):
        track = mido.MidiTrack()
        track.append(mido.Message("note_on", note=pitch, velocity=90, time=480))
        track.append(mido.Message("note_off", note=pitch, velocity=0, time=240))
        midi.tracks.append(track)
    midi.save(midi_path)
    audio_path = tmp_path / "micamp_01.wav"
    sf.write(audio_path, np.zeros(FEATURE_SAMPLE_RATE * 2), FEATURE_SAMPLE_RATE)
    labels = convert_guitar_techs_midi(midi_path, audio_path, performer_id="P3")
    assert labels["provenance"]["dataset_id"] == "guitar-techs"
    assert labels["split_keys"][-1] == "performer:guitar-techs:P3"
    assert labels["attacks"][0]["pitches"] == [48, 55]
    assert labels["attacks"][0]["polyphony"] == 2


def test_guitar_techs_conversion_closes_unterminated_note_at_track_end(tmp_path):
    import mido

    midi_path = tmp_path / "unterminated.mid"
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.Message("note_on", note=48, velocity=90, time=0))
    track.append(mido.MetaMessage("end_of_track", time=480))
    midi.tracks.append(track)
    midi.save(midi_path)
    audio_path = tmp_path / "audio.wav"
    sf.write(audio_path, np.zeros(FEATURE_SAMPLE_RATE), FEATURE_SAMPLE_RATE)

    labels = convert_guitar_techs_midi(midi_path, audio_path, performer_id="P1")

    assert labels["notes"][0]["end"] == 0.5
    assert labels["annotation_repairs"]["unterminated_notes_closed_at_track_end"] == 1


def test_structure_features_align_attacks_polyphony_sustains_and_beats(tmp_path):
    annotation = _annotation(tmp_path)
    labels_path = tmp_path / "labels.json"
    convert_guitarset_jams(annotation, labels_path)
    audio_path = tmp_path / "audio.wav"
    sf.write(audio_path, np.zeros(FEATURE_SAMPLE_RATE * 3, dtype=np.float32), FEATURE_SAMPLE_RATE)
    output = tmp_path / "features.npz"
    metadata = extract_structure_features(audio_path, labels_path, output)
    with np.load(output) as features:
        assert features["log_mel"].shape[1] == FEATURE_MELS
        assert features["mel_delta"].shape == features["log_mel"].shape
        assert np.min(features["mel_delta"]) >= 0
        assert features["beat_phase"].shape == (len(features["log_mel"]), 2)
        assert int(np.sum(features["attack_target"])) == 3
        chord_frame = int(np.argmax(features["polyphony_target"]))
        assert features["polyphony_target"][chord_frame] == 2
        assert np.max(features["sustain_target"]) == 2
        frame_duration = FEATURE_HOP_LENGTH / FEATURE_SAMPLE_RATE
        assert 1.0 <= features["frame_times"][chord_frame] < 1.0 + frame_duration
        assert features["sustain_target"][chord_frame] == 2
        stored = json.loads(str(features["metadata"]))
        assert stored["audio_sha256"] == metadata["audio_sha256"]
        assert stored["targets"] == ["attack", "polyphony", "sustain"]
        assert stored["beat_phase_valid"] is True
        assert stored["provenance"]["dataset_id"] == "guitarset"


def test_structure_features_reject_unknown_label_schema(tmp_path):
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"schema": "other", "schema_version": 1}), encoding="utf-8")
    audio = tmp_path / "audio.wav"
    sf.write(audio, np.zeros(100, dtype=np.float32), FEATURE_SAMPLE_RATE)
    with pytest.raises(TrainingDataError, match="Unsupported"):
        extract_structure_features(audio, labels, tmp_path / "features.npz")


def test_structure_features_reject_unreadable_audio(tmp_path):
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({
        "schema": "rb3charts-structure-labels",
        "schema_version": 1,
        "attacks": [],
        "notes": [],
        "beats": [],
    }), encoding="utf-8")
    audio = tmp_path / "broken.mp3"
    audio.write_bytes(b"not an audio file")
    with pytest.raises(TrainingDataError, match="Cannot read structure-model audio"):
        extract_structure_features(audio, labels, tmp_path / "features.npz")


def test_structure_features_reject_labels_outside_audio(tmp_path):
    annotation = _annotation(tmp_path)
    labels_path = tmp_path / "labels.json"
    convert_guitarset_jams(annotation, labels_path)
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    payload["duration"] = None
    payload["attacks"][0]["start"] = 25.0
    labels_path.write_text(json.dumps(payload), encoding="utf-8")
    audio = tmp_path / "audio.wav"
    sf.write(audio, np.zeros(FEATURE_SAMPLE_RATE * 3, dtype=np.float32), FEATURE_SAMPLE_RATE)
    with pytest.raises(TrainingDataError, match="outside"):
        extract_structure_features(audio, labels_path, tmp_path / "features.npz")


def test_structure_features_require_npz_output_suffix(tmp_path):
    annotation = _annotation(tmp_path)
    labels_path = tmp_path / "labels.json"
    convert_guitarset_jams(annotation, labels_path)
    audio = tmp_path / "audio.wav"
    sf.write(audio, np.zeros(FEATURE_SAMPLE_RATE * 3, dtype=np.float32), FEATURE_SAMPLE_RATE)
    with pytest.raises(TrainingDataError, match=r"\.npz"):
        extract_structure_features(audio, labels_path, tmp_path / "features")


def _write_midi(path, pitch):
    import mido

    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    track.append(mido.Message("note_on", note=pitch, velocity=100, time=480))
    track.append(mido.Message("note_off", note=pitch, velocity=0, time=480))
    midi.save(path)


def test_slakh_conversion_selects_rendered_guitars_and_groups_sources(tmp_path):
    track = tmp_path / "Track00001"
    midi_dir = track / "MIDI"
    midi_dir.mkdir(parents=True)
    _write_midi(midi_dir / "S00.mid", 48)
    _write_midi(midi_dir / "S01.mid", 55)
    _write_midi(midi_dir / "S02.mid", 36)
    _write_midi(track / "all_src.mid", 60)
    metadata = {
        "UUID": "composition-one",
        "midi_dir": "MIDI",
        "stems": {
            "S00": {"inst_class": "Guitar", "is_drum": False, "midi_saved": True, "audio_rendered": True},
            "S01": {"inst_class": "Guitar", "is_drum": False, "midi_saved": True, "audio_rendered": True},
            "S02": {"inst_class": "Bass", "is_drum": False, "midi_saved": True, "audio_rendered": True},
        },
    }
    (track / "metadata.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
    sf.write(track / "mix.wav", np.zeros(FEATURE_SAMPLE_RATE * 2, dtype=np.float32), FEATURE_SAMPLE_RATE)
    output = tmp_path / "slakh-labels.json"
    labels = convert_slakh_track(track, output)
    assert labels["summary"]["guitar_sources"] == 2
    assert labels["summary"]["notes"] == 2
    assert labels["summary"]["attacks"] == 1
    assert labels["summary"]["chord_attacks"] == 1
    assert labels["attacks"][0]["pitches"] == [48, 55]
    assert labels["attacks"][0]["start"] == pytest.approx(0.5)
    assert labels["duration"] == pytest.approx(2.0)
    assert labels["composition_id"] == "composition-one"
    assert labels["provenance"]["dataset_id"] == "slakh2100-redux"
    assert labels["group_id"] == "slakh:composition-one"
    assert labels["beats"][-1]["time"] == pytest.approx(2.0)
    assert len(labels["source_files"]) == 4
    assert output.is_file()


def test_slakh_conversion_rejects_track_without_guitar_sources(tmp_path):
    track = tmp_path / "Track00002"
    track.mkdir()
    (track / "metadata.yaml").write_text(yaml.safe_dump({
        "stems": {
            "S00": {"inst_class": "Bass", "midi_saved": True, "audio_rendered": True},
        },
    }), encoding="utf-8")
    with pytest.raises(TrainingDataError, match="no rendered guitar"):
        convert_slakh_track(track)


def test_slakh_guitar_mix_uses_present_files_despite_stale_metadata_flags(tmp_path):
    track = tmp_path / "Track00003"
    stems = track / "stems"
    stems.mkdir(parents=True)
    first = np.full(100, 0.1, dtype=np.float32)
    second = np.full(100, 0.2, dtype=np.float32)
    sf.write(stems / "S00.wav", first, 16000, subtype="FLOAT")
    sf.write(stems / "S01.wav", second, 16000, subtype="FLOAT")
    sf.write(stems / "S02.wav", np.full(100, 0.5, dtype=np.float32), 16000, subtype="FLOAT")
    (track / "metadata.yaml").write_text(yaml.safe_dump({
        "audio_dir": "stems",
        "stems": {
            "S00": {"inst_class": "Guitar", "is_drum": False, "audio_rendered": False},
            "S01": {"inst_class": "Guitar", "is_drum": False, "audio_rendered": False},
            "S02": {"inst_class": "Bass", "is_drum": False, "audio_rendered": True},
        },
    }), encoding="utf-8")
    output = tmp_path / "guitar.wav"
    report = mix_slakh_guitar_stems(track, output)
    audio, sample_rate = sf.read(output)
    assert sample_rate == 16000
    assert audio == pytest.approx(first + second, abs=1e-6)
    assert len(report["sources"]) == 2
    assert report["peak"] == pytest.approx(0.3, abs=1e-6)


def _write_feature_index_fixture(tmp_path, validation_player="01"):
    items = []
    manifest_items = []
    for ordinal, (player, split, token) in enumerate((
        ("00", "train", "BN1-120-C"),
        (validation_player, "validation", "JZ2-140-D"),
        ("02", "test", "RK1-100-E"),
    )):
        folder = tmp_path / split
        folder.mkdir()
        annotation = _annotation(folder, name=f"{player}_{token}_comp.jams")
        annotation_payload = json.loads(annotation.read_text(encoding="utf-8"))
        annotation_payload["annotations"][0]["data"][0]["value"] += ordinal
        annotation.write_text(json.dumps(annotation_payload), encoding="utf-8")
        labels = folder / "labels.json"
        converted = convert_guitarset_jams(annotation, labels)
        audio = folder / "audio.wav"
        tone = np.zeros(FEATURE_SAMPLE_RATE * 3, dtype=np.float32)
        tone[FEATURE_SAMPLE_RATE:FEATURE_SAMPLE_RATE + 100] = (
            np.hanning(100) * (ordinal + 1) / 3
        )
        sf.write(audio, tone, FEATURE_SAMPLE_RATE)
        features = folder / "features.npz"
        extract_structure_features(audio, labels, features)
        items.append({
            "song_id": f"{split}-song",
            "group_id": converted["group_id"],
            "split": split,
            "features": str(features.relative_to(tmp_path)),
            "manifest": "manifest.json",
        })
        manifest_items.append({
            "song_id": f"{split}-song",
            "artist_id": player,
            "split": split,
            "audio": str(audio.relative_to(tmp_path)),
            "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
            "annotations": str(annotation.relative_to(tmp_path)),
            "annotations_sha256": hashlib.sha256(annotation.read_bytes()).hexdigest(),
        })
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "dataset": {
            "id": "guitarset",
            "version": "1.1.0",
            "source_url": "https://doi.org/10.5281/zenodo.3371780",
            "audio_license": "CC BY 4.0",
            "annotation_license": "CC BY 4.0",
            "commercial_use": "allowed",
            "attribution": (
                "Qingyang Xi, Rachel M. Bittner, Johan Pauwels, "
                "Xuzhou Ye, and Juan P. Bello"
            ),
        },
        "items": manifest_items,
    }), encoding="utf-8")
    index = tmp_path / "feature-index.json"
    index.write_text(json.dumps({
        "schema": "rb3charts-structure-feature-index",
        "schema_version": 2,
        "items": items,
    }), encoding="utf-8")
    return index


def test_structure_baseline_trains_and_writes_auditable_checkpoint(tmp_path):
    import torch

    index = _write_feature_index_fixture(tmp_path)
    output = tmp_path / "baseline.pt"
    report = train_structure_baseline(index, output, epochs=1, device="cpu", seed=7)
    assert report["device_used"] == "cpu"
    assert report["training_items"] == 1
    assert report["validation_items"] == 1
    assert report["test_items"] == 1
    assert len(report["epoch_losses"]) == 1
    assert 0 <= report["validation"]["attack_f1"] <= 1
    assert 0 <= report["test"]["f1"] <= 1
    assert report["training_data"][0]["dataset_id"] == "guitarset"
    assert report["dataset_manifests"][0]["path"] == "manifest.json"
    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    assert checkpoint["schema"] == "rb3charts-structure-model"
    assert checkpoint["configuration"]["mel_bins"] == FEATURE_MELS
    assert output.with_suffix(".json").is_file()
    prediction = predict_structure(
        output,
        tmp_path / "validation" / "audio.wav",
        tmp_path / "prediction.json",
        device="cpu",
    )
    assert prediction["schema"] == "rb3charts-structure-predictions"
    assert prediction["scope"].startswith("lane-independent")
    assert prediction["summary"]["attacks"] == len(prediction["events"])
    assert prediction["summary"]["candidate_peaks"] == len(prediction["candidates"])
    assert len(prediction["candidates"]) >= len(prediction["events"])
    ranks=[event["attack_rank_percentile"] for event in prediction["candidates"]]
    assert min(ranks)==pytest.approx(0.0)
    assert max(ranks)==pytest.approx(1.0)
    assert set(prediction["summary"]["candidate_score_quantiles"])=={"0.5","0.75","0.9","0.95"}


def test_structure_baseline_rejects_group_split_leakage(tmp_path):
    index = _write_feature_index_fixture(tmp_path, validation_player="00")
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["items"][1]["artist_id"] = "misidentified-validation-artist"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(TrainingDataError, match="split key .* crosses"):
        train_structure_baseline(index, tmp_path / "baseline.pt", device="cpu")


def test_structure_baseline_rejects_audio_outside_validated_manifest(tmp_path):
    index = _write_feature_index_fixture(tmp_path)
    replacement = tmp_path / "replacement.wav"
    replacement.write_bytes(b"not the feature audio")
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["items"][0]["audio"] = replacement.name
    manifest["items"][0]["audio_sha256"] = hashlib.sha256(replacement.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(TrainingDataError, match="Feature audio hash does not match"):
        train_structure_baseline(index, tmp_path / "baseline.pt", device="cpu")


def test_onset_metrics_use_same_peak_suppression_as_inference():
    metrics = _onset_metrics(
        [(np.asarray([0.1, 0.8, 0.8, 0.1], dtype=np.float32), [1])],
        threshold=0.5,
        tolerance_frames=1,
    )
    assert metrics["predicted_attacks"] == 1
    assert metrics["f1"] == 1


def test_structure_baseline_runs_on_cuda_when_available(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    index = _write_feature_index_fixture(tmp_path)
    report = train_structure_baseline(
        index,
        tmp_path / "baseline.pt",
        epochs=1,
        device="cuda",
        seed=7,
    )
    assert report["device_used"] == "cuda"
