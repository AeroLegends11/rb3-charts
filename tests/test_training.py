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
    _attack_localization_loss,
    _attack_offset_targets,
    _decode_attack_peaks,
    _onset_metrics,
    _peak_width_rank_percentiles,
    _periodic_peak_support,
    convert_egfxset_audio,
    convert_guitarset_jams,
    convert_guitar_techs_midi,
    convert_slakh_track,
    evaluate_source_agreement,
    evaluate_structure_ranking,
    extract_structure_features,
    mix_slakh_guitar_stems,
    pin_source_agreement_manifest,
    predict_structure,
    synthesize_egfxset_phrase,
    synthesize_procedural_guitar_phrase,
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


def test_egfxset_conversion_uses_string_fret_pitch_and_attack_only_targets(tmp_path):
    audio = tmp_path / "Clean" / "Neck" / "6-3.wav"
    audio.parent.mkdir(parents=True)
    sf.write(audio, np.zeros(FEATURE_SAMPLE_RATE, dtype=np.float32), FEATURE_SAMPLE_RATE)
    output = tmp_path / "labels.json"

    labels = convert_egfxset_audio(audio, output)

    assert labels["provenance"]["dataset_id"] == "egfxset"
    assert labels["provenance"]["commercial_use"] == "allowed"
    assert labels["targets"] == ["attack", "polyphony"]
    assert labels["notes"][0]["pitch"] == 43
    assert labels["effect"] == "Clean"
    assert labels["pickup_configuration"] == "Neck"
    assert labels["group_id"] == "egfxset:Neck:6-3"
    assert labels["split_keys"][-1] == "performer:egfxset:unknown"
    assert json.loads(output.read_text(encoding="utf-8"))["summary"]["attacks"] == 1
    features = tmp_path / "features.npz"
    metadata = extract_structure_features(audio, output, features)
    assert metadata["targets"] == ["attack", "polyphony"]


def test_egfxset_conversion_rejects_invalid_published_filename(tmp_path):
    audio = tmp_path / "2-23.wav"
    sf.write(audio, np.zeros(FEATURE_SAMPLE_RATE, dtype=np.float32), FEATURE_SAMPLE_RATE)
    with pytest.raises(TrainingDataError, match="fret must be between"):
        convert_egfxset_audio(audio)


def test_egfxset_phrase_synthesis_is_deterministic_and_attack_only(tmp_path):
    source = tmp_path / "Clean"
    for string in range(1, 5):
        audio = source / "Neck" / f"{string}-0.wav"
        audio.parent.mkdir(parents=True, exist_ok=True)
        tone = np.full(FEATURE_SAMPLE_RATE, string / 10, dtype=np.float32)
        sf.write(audio, tone, FEATURE_SAMPLE_RATE, subtype="FLOAT")

    first_audio = tmp_path / "first.wav"
    first_labels = tmp_path / "first.json"
    second_audio = tmp_path / "second.wav"
    second_labels = tmp_path / "second.json"
    first = synthesize_egfxset_phrase(
        source, first_audio, first_labels, seed=17, duration=2.5
    )
    second = synthesize_egfxset_phrase(
        source, second_audio, second_labels, seed=17, duration=2.5
    )

    assert first["targets"] == ["attack", "polyphony"]
    assert first["attacks"][0]["start"] == pytest.approx(0.5)
    assert len(first["attacks"]) > 1
    assert first["group_id"] == "egfxset-synthetic:seed-17:sequence-00"
    assert first["split_keys"][-1] == "performer:egfxset:unknown"
    assert all(item["sha256"] for item in first["source_files"])
    assert first == second
    assert first_audio.read_bytes() == second_audio.read_bytes()


def test_egfxset_phrase_synthesis_requires_source_diversity(tmp_path):
    source = tmp_path / "Clean" / "Neck"
    source.mkdir(parents=True)
    for string in range(1, 4):
        sf.write(
            source / f"{string}-0.wav",
            np.zeros(FEATURE_SAMPLE_RATE, dtype=np.float32),
            FEATURE_SAMPLE_RATE,
        )
    with pytest.raises(TrainingDataError, match="at least four"):
        synthesize_egfxset_phrase(
            source.parent, tmp_path / "phrase.wav", tmp_path / "phrase.json"
        )


def test_procedural_guitar_synthesis_is_deterministic_and_fully_labeled(tmp_path):
    first_audio = tmp_path / "first.wav"
    first_labels = tmp_path / "first.json"
    second_audio = tmp_path / "second.wav"
    second_labels = tmp_path / "second.json"
    first = synthesize_procedural_guitar_phrase(
        first_audio,
        first_labels,
        seed=23,
        sequence=0,
        duration=4.0,
    )
    second = synthesize_procedural_guitar_phrase(
        second_audio,
        second_labels,
        seed=23,
        sequence=0,
        duration=4.0,
    )

    assert first == second
    assert first_audio.read_bytes() == second_audio.read_bytes()
    assert first["targets"] == ["attack", "polyphony", "sustain"]
    assert first["source_files"] == []
    assert first["provenance"]["commercial_use"] == "allowed"
    assert first["split_keys"][-1] == (
        "generator:rb3charts-procedural-guitar-v1"
    )
    assert first["summary"]["chord_attacks"] > 0
    assert first["beats"]
    features = tmp_path / "features.npz"
    metadata = extract_structure_features(
        first_audio,
        first_labels,
        features,
    )
    assert metadata["targets"] == ["attack", "polyphony", "sustain"]


def test_procedural_guitar_synthesis_rejects_negative_seed(tmp_path):
    with pytest.raises(TrainingDataError, match="seed must be non-negative"):
        synthesize_procedural_guitar_phrase(
            tmp_path / "phrase.wav",
            tmp_path / "phrase.json",
            seed=-1,
        )


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


def _write_feature_index_fixture(tmp_path, validation_player="01", targets=None):
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
        if targets is not None:
            label_payload = json.loads(labels.read_text(encoding="utf-8"))
            label_payload["targets"] = targets
            labels.write_text(json.dumps(label_payload), encoding="utf-8")
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
    assert report["test_evaluated"] is True
    assert report["attack_loss"] == "weighted-bce"
    assert report["architecture"] == "conv1d"
    assert len(report["epoch_losses"]) == 1
    assert 0 <= report["validation"]["attack_f1"] <= 1
    assert 0 <= report["validation"]["macro_attack_f1"] <= 1
    assert report["validation"]["median_timing_error_ms"] is not None
    assert report["validation"]["p95_timing_error_ms"] is not None
    assert 0 <= report["test"]["f1"] <= 1
    assert set(report["validation"]["by_song"]) == {"validation-song"}
    assert set(report["test"]["by_song"]) == {"test-song"}
    assert set(report["validation"]["by_dataset"]) == {"guitarset"}
    assert set(report["test"]["by_dataset"]) == {"guitarset"}
    assert report["test"]["macro_f1"] == report["test"]["by_song"]["test-song"]["f1"]
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
    periodic_ranks = [
        event["periodic_attack_rank_percentile"]
        for event in prediction["candidates"]
    ]
    assert min(periodic_ranks) == pytest.approx(0.0)
    assert max(periodic_ranks) == pytest.approx(1.0)
    assert all(
        0 <= event["periodic_support"] <= 1
        for event in prediction["candidates"]
    )
    assert all(
        event["sustain_seconds"] >= 0
        for event in prediction["candidates"]
    )
    assert all(
        0 <= event["bounded_sustain_seconds"] <= event["sustain_seconds"]
        for event in prediction["candidates"]
    )
    width_ranks = [
        event["peak_width_rank_percentile"]
        for event in prediction["candidates"]
    ]
    assert min(width_ranks) == pytest.approx(0.0)
    assert max(width_ranks) == pytest.approx(1.0)
    width_reranked = [
        event["periodic_width_attack_rank_percentile"]
        for event in prediction["candidates"]
    ]
    assert min(width_reranked) == pytest.approx(0.0)
    assert max(width_reranked) == pytest.approx(1.0)
    assert set(prediction["summary"]["candidate_score_quantiles"])=={"0.5","0.75","0.9","0.95"}


def test_structure_baseline_supports_focal_attack_loss(tmp_path):
    index = _write_feature_index_fixture(tmp_path)
    report = train_structure_baseline(
        index,
        tmp_path / "focal.pt",
        epochs=1,
        device="cpu",
        seed=7,
        attack_loss="focal",
        focal_gamma=2.0,
        calibration_dataset="guitarset",
        evaluate_test=False,
    )
    assert report["attack_loss"] == "focal"
    assert report["focal_gamma"] == 2.0
    assert report["calibration_dataset"] == "guitarset"
    assert report["validation"]["threshold_selection"] == "macro-f1"
    assert report["test_items"] == 1
    assert report["test_evaluated"] is False
    assert "test" not in report


def test_structure_baseline_supports_dilated_tcn(tmp_path):
    index = _write_feature_index_fixture(tmp_path)
    output = tmp_path / "tcn.pt"
    report = train_structure_baseline(
        index,
        output,
        epochs=1,
        device="cpu",
        seed=7,
        architecture="dilated-tcn",
    )
    assert report["architecture"] == "dilated-tcn"
    prediction = predict_structure(
        output,
        tmp_path / "validation" / "audio.wav",
        tmp_path / "tcn-prediction.json",
        device="cpu",
    )
    assert prediction["schema"] == "rb3charts-structure-predictions"


def test_structure_baseline_supports_multi_scale_context(tmp_path):
    index = _write_feature_index_fixture(tmp_path)
    output = tmp_path / "multi-scale.pt"
    report = train_structure_baseline(
        index,
        output,
        epochs=1,
        device="cpu",
        seed=7,
        architecture="multi-scale",
    )
    assert report["architecture"] == "multi-scale"
    assert report["model_version"] == 3
    prediction = predict_structure(
        output,
        tmp_path / "validation" / "audio.wav",
        tmp_path / "multi-scale-prediction.json",
        device="cpu",
    )
    assert prediction["schema"] == "rb3charts-structure-predictions"


def test_structure_baseline_supports_explicit_onset_feature(tmp_path):
    index = _write_feature_index_fixture(tmp_path)
    output = tmp_path / "multi-scale-onset.pt"
    report = train_structure_baseline(
        index,
        output,
        epochs=1,
        device="cpu",
        seed=7,
        architecture="multi-scale",
        onset_feature=True,
    )

    assert report["onset_feature"] is True
    prediction = predict_structure(
        output,
        tmp_path / "validation" / "audio.wav",
        tmp_path / "multi-scale-onset-prediction.json",
        device="cpu",
    )
    assert prediction["schema"] == "rb3charts-structure-predictions"


def test_structure_baseline_supports_local_attack_with_context(tmp_path):
    index = _write_feature_index_fixture(tmp_path)
    output = tmp_path / "multi-scale-local.pt"
    report = train_structure_baseline(
        index,
        output,
        epochs=1,
        device="cpu",
        seed=7,
        architecture="multi-scale-local",
    )
    assert report["architecture"] == "multi-scale-local"
    assert report["model_version"] == 4
    prediction = predict_structure(
        output,
        tmp_path / "validation" / "audio.wav",
        tmp_path / "multi-scale-local-prediction.json",
        device="cpu",
    )
    assert prediction["schema"] == "rb3charts-structure-predictions"


def test_attack_offset_targets_point_neighbor_frames_to_attack():
    import torch

    attacks = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
    targets, mask = _attack_offset_targets(attacks, radius=2)

    assert mask.tolist() == [[True, True, True, True, True]]
    assert targets.tolist() == [[1.0, 0.5, 0.0, -0.5, -1.0]]


def test_attack_offset_decoder_moves_context_peak_without_changing_selection():
    scores = np.asarray([0.0, 0.1, 0.8, 0.1, 0.0], dtype=np.float32)
    offsets = np.asarray([0.0, 0.0, -0.5, 0.0, 0.0], dtype=np.float32)

    assert _decode_attack_peaks(scores, 0.5).tolist() == [2]
    assert _decode_attack_peaks(scores, 0.5, offsets, 2).tolist() == [1]


def test_structure_baseline_supports_joint_attack_offsets(tmp_path):
    index = _write_feature_index_fixture(tmp_path)
    output = tmp_path / "multi-scale-offset.pt"
    report = train_structure_baseline(
        index,
        output,
        epochs=1,
        device="cpu",
        seed=7,
        architecture="multi-scale-offset",
    )

    assert report["architecture"] == "multi-scale-offset"
    assert report["model_version"] == 5
    assert report["attack_offset_radius_frames"] == 4
    prediction = predict_structure(
        output,
        tmp_path / "validation" / "audio.wav",
        tmp_path / "multi-scale-offset-prediction.json",
        device="cpu",
    )
    assert prediction["schema"] == "rb3charts-structure-predictions"


def test_structure_baseline_supports_smooth_context_gate(tmp_path):
    index = _write_feature_index_fixture(tmp_path)
    output = tmp_path / "multi-scale-gated.pt"
    report = train_structure_baseline(
        index,
        output,
        epochs=1,
        device="cpu",
        seed=7,
        architecture="multi-scale-gated",
    )

    assert report["architecture"] == "multi-scale-gated"
    assert report["model_version"] == 6
    prediction = predict_structure(
        output,
        tmp_path / "validation" / "audio.wav",
        tmp_path / "multi-scale-gated-prediction.json",
        device="cpu",
    )
    assert prediction["schema"] == "rb3charts-structure-predictions"


def test_structure_offset_head_can_freeze_initialized_selector(tmp_path):
    index = _write_feature_index_fixture(tmp_path)
    initialized = tmp_path / "initialized.pt"
    train_structure_baseline(
        index,
        initialized,
        epochs=1,
        device="cpu",
        seed=7,
        architecture="multi-scale",
        evaluate_test=False,
    )
    report = train_structure_baseline(
        index,
        tmp_path / "offset-only.pt",
        epochs=1,
        device="cpu",
        seed=7,
        architecture="multi-scale-offset",
        evaluate_test=False,
        initialize_model=initialized,
        offset_only=True,
    )

    assert report["offset_only"] is True
    assert report["initialization_sha256"] == hashlib.sha256(
        initialized.read_bytes()
    ).hexdigest()


def test_attack_localization_loss_prefers_centered_peak():
    import torch

    attacks = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
    centered = torch.tensor([[0.0, 0.0, 2.0, 0.0, 0.0]])
    shifted = torch.tensor([[0.0, 2.0, 0.0, 0.0, 0.0]])

    assert (
        _attack_localization_loss(centered, attacks)
        < _attack_localization_loss(shifted, attacks)
    )


def test_structure_ranking_evaluation_uses_fixed_top_k_and_one_to_one_matches(tmp_path):
    predictions = tmp_path / "predictions.json"
    predictions.write_text(json.dumps({
        "schema": "rb3charts-structure-predictions",
        "candidates": [
            {"start": 0.99, "periodic_width_attack_rank_percentile": 0.9},
            {"start": 2.04, "periodic_width_attack_rank_percentile": 0.8},
            {"start": 4.0, "periodic_width_attack_rank_percentile": 0.7},
            {"start": 3.02, "periodic_width_attack_rank_percentile": 0.6},
        ],
    }), encoding="utf-8")
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({
        "schema": "rb3charts-structure-labels",
        "notes": [
            {"start": 1.0},
            {"start": 1.0},
            {"start": 2.0},
            {"start": 3.0},
        ],
    }), encoding="utf-8")
    output = tmp_path / "evaluation.json"
    report = evaluate_structure_ranking(predictions, labels, output=output)
    assert report["selection_count"] == 3
    assert report["matched_attacks"] == 2
    assert report["selected_candidates"] == 3
    assert report["reference_attacks"] == 3
    assert report["f1"] == pytest.approx(2 / 3)
    assert json.loads(output.read_text(encoding="utf-8")) == report

    invalid = json.loads(predictions.read_text(encoding="utf-8"))
    invalid["candidates"][0]["periodic_width_attack_rank_percentile"] = float("nan")
    predictions.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(TrainingDataError, match="must be finite"):
        evaluate_structure_ranking(predictions, labels)


def test_structure_baseline_trains_with_attack_polyphony_only_targets(tmp_path):
    index = _write_feature_index_fixture(
        tmp_path, targets=["attack", "polyphony"]
    )
    report = train_structure_baseline(
        index, tmp_path / "attack-only.pt", epochs=1, device="cpu", seed=7
    )
    assert len(report["epoch_losses"]) == 1
    assert 0 <= report["validation"]["attack_f1"] <= 1


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
        [(np.asarray([0.1, 0.8, 0.8, 0.1], dtype=np.float32), [2])],
        threshold=0.5,
        tolerance_frames=1,
    )
    assert metrics["predicted_attacks"] == 1
    assert metrics["f1"] == 1
    assert metrics["median_timing_error_ms"] == pytest.approx(
        FEATURE_HOP_LENGTH / FEATURE_SAMPLE_RATE * 1000
    )
    assert metrics["p95_timing_error_ms"] == pytest.approx(
        FEATURE_HOP_LENGTH / FEATURE_SAMPLE_RATE * 1000
    )


def test_periodic_peak_support_requires_three_short_regular_attacks():
    support = _periodic_peak_support(
        [0.0, 0.125, 0.25, 0.6],
        [0.7, 0.8, 0.9, 1.0],
    )
    assert support.tolist() == pytest.approx([0.0, 0.7, 0.0, 0.0])


def test_peak_width_rank_prefers_broader_score_curve():
    scores = np.zeros(32, dtype=float)
    scores[5] = 1
    scores[18:23] = [0.2, 0.6, 1.0, 0.6, 0.2]

    ranks = _peak_width_rank_percentiles(scores, [5, 20])

    assert ranks.tolist() == pytest.approx([0.0, 1.0])


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


def test_source_agreement_preserves_provenance_and_scores_shared_attacks(tmp_path):
    primary=tmp_path/"audio"/"micamp"/"micamp_take.wav"
    alternate=tmp_path/"audio"/"directinput"/"directinput_take.wav"
    primary.parent.mkdir(parents=True)
    alternate.parent.mkdir(parents=True)
    labels_path=tmp_path/"labels"/"take.json"
    labels_path.parent.mkdir()
    sample_rate=FEATURE_SAMPLE_RATE
    primary_audio=np.zeros(sample_rate,dtype=np.float32)
    alternate_audio=np.zeros(sample_rate,dtype=np.float32)
    for seconds in (.2,.5,.8):
        index=int(seconds*sample_rate)
        primary_audio[index:index+100]=1
    for seconds in (.2,.5,.7):
        index=int(seconds*sample_rate)
        alternate_audio[index:index+100]=1
    sf.write(primary,primary_audio,sample_rate)
    sf.write(alternate,alternate_audio,sample_rate)
    dataset={
        "id":"fixture",
        "version":"1",
        "source_url":"https://example.invalid/fixture",
        "audio_license":"CC BY 4.0",
        "annotation_license":"CC BY 4.0",
        "commercial_use":"allowed",
        "attribution":"Fixture Author",
    }
    labels={
        "provenance":dataset,
        "group_id":"fixture:take",
        "split_keys":["fixture:take","performer:fixture:p1"],
        "targets":["attack"],
        "attacks":[{"start":seconds} for seconds in (.2,.5,.8)],
    }
    labels_path.write_text(json.dumps(labels),encoding="utf-8")
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    manifest=tmp_path/"manifest.json"
    manifest.write_text(json.dumps({
        "schema_version":1,
        "dataset":dataset,
        "items":[{
            "song_id":"take",
            "artist_id":"p1",
            "split":"train",
            "audio":"audio/micamp/micamp_take.wav",
            "audio_sha256":digest(primary),
            "annotations":"labels/take.json",
            "annotations_sha256":digest(labels_path),
        }],
    }),encoding="utf-8")
    archive=tmp_path/"release.zip"
    archive.write_bytes(b"official fixture archive")
    pinned=tmp_path/"manifest-source-agreement.json"
    with pytest.raises(
        TrainingDataError,match="hash-pinned alternate recording"
    ):
        evaluate_source_agreement(manifest)
    pin_report=pin_source_agreement_manifest(
        manifest,
        alternate.parent,
        pinned,
        "micamp_take.wav",
        "directinput_take.wav",
        archive,
        hashlib.md5(archive.read_bytes()).hexdigest(),
        "https://example.invalid/fixture/record",
    )
    output=tmp_path/"agreement.json"

    report=evaluate_source_agreement(pinned,output)

    assert pin_report["items"]==1
    assert pin_report["release"]["archive_bytes"]==len(b"official fixture archive")
    assert pin_report["release"]["record_url"].endswith("/record")
    assert report["manifest_sha256"]==digest(pinned)
    assert report["items"][0]["split_keys"]==labels["split_keys"]
    assert report["items"][0]["primary"]["sha256"]==digest(primary)
    assert report["items"][0]["alternate"]["sha256"]==digest(alternate)
    assert report["items"][0]["alternate"]["hash_manifested"] is True
    assert report["splits"]["train"]["shared"]["precision"]==1.0
    assert report["splits"]["train"]["shared"]["label_matches"]==2
    assert report["splits"]["train"]["shared_precision_lift_vs_best_source"]==0
    assert output.is_file()

    alternate.write_bytes(primary.read_bytes())
    with pytest.raises(
        TrainingDataError,match="duplicates primary content"
    ):
        pin_source_agreement_manifest(
            manifest,
            alternate.parent,
            tmp_path/"duplicate-source-manifest.json",
            "micamp_take.wav",
            "directinput_take.wav",
        )
