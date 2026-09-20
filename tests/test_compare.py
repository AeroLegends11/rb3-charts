from pathlib import Path
import json
from urllib.request import Request, urlopen

import mido

from rb3charts.compare import compare_charts, resolve_midi
from rb3charts.video_compare import create_video_comparison, start_comparison_server, youtube_video_id


def _midi(path: Path, attacks, ppq=480):
    midi = mido.MidiFile(ticks_per_beat=ppq)
    track = mido.MidiTrack(); track.name = "PART GUITAR"; midi.tracks.append(track)
    events = []
    for beat, lanes, duration in attacks:
        for lane in lanes:
            events.append((round(beat * ppq), True, 96 + lane))
            events.append((round((beat + duration) * ppq), False, 96 + lane))
    last = 0
    for tick, on, note in sorted(events, key=lambda row: (row[0], not row[1])):
        track.append(mido.Message("note_on" if on else "note_off", note=note,
                                  velocity=100 if on else 0, time=tick - last))
        last = tick
    midi.save(path)


def test_chart_comparison_aligns_and_reports_differences(tmp_path):
    generated = tmp_path / "generated.mid"
    reference = tmp_path / "reference.mid"
    _midi(generated, [(0, [0], .25), (1, [1], .5), (2, [2], .25), (3, [3], .25), (3.5, [4], .25)])
    _midi(reference, [(4, [0], .25), (5, [1], .25), (6, [2], .25), (7, [4], .25), (8, [0], .25)])
    generated_hash = generated.read_bytes()

    result = compare_charts(generated, reference, tmp_path / "comparisons")

    assert result["provenance"]["alignment_offset_seconds"] == 2
    assert result["summary"]["matched_attacks"] == 4
    assert result["summary"]["extra_attacks"] == 1
    assert result["summary"]["missing_attacks"] == 1
    assert result["summary"]["exact_lane_rate"] == .75
    assert Path(result["output"]["html"]).is_file()
    assert Path(result["output"]["json"]).is_file()
    assert generated.read_bytes() == generated_hash
    page = Path(result["output"]["html"]).read_text(encoding="utf-8")
    assert "Human review → Copilot update" in page
    assert "Build and copy Copilot review update" in page
    assert "Most important correction" in page
    assert "Strong variations to preserve" in page
    assert "Preservation check" in page
    assert "Priority 1" in page
    assert result["provenance"]["software_revision"]
    assert str(tmp_path) not in page.split("const REVIEW_CONTEXT=", 1)[1].split(";", 1)[0]


def test_output_folder_resolves_midi(tmp_path):
    folder = tmp_path / "run" / "midi"; folder.mkdir(parents=True)
    _midi(folder / "notes.mid", [(0, [0], .25)])
    assert resolve_midi(tmp_path / "run") == (folder / "notes.mid").resolve()


def test_video_comparison_accepts_review_folder_and_youtube_url(tmp_path):
    review = tmp_path / "song-run"; review.mkdir(); (review / "review.html").write_text("<p>chart</p>")
    (review / "audio").mkdir(); (review / "audio" / "song.wav").write_bytes(b"0123456789")
    result = create_video_comparison(review, "https://www.youtube.com/watch?v=Iorv-Iw13oE", tmp_path / "comparisons")
    page = Path(result["html"]).read_text(encoding="utf-8")
    assert result["youtube_video_id"] == "Iorv-Iw13oE"
    assert "Official gameplay reference" in page
    assert "Generated RB3 Charts review" in page
    assert "Export observations as JSON" in page
    server, url = start_comparison_server(result)
    try:
        assert "Official gameplay reference" in urlopen(url).read().decode()
        request = Request(url.replace("video-comparison.html", "review/audio/song.wav"), headers={"Range": "bytes=2-5"})
        with urlopen(request) as response:
            assert response.status == 206
            assert response.read() == b"2345"
        payload = {"observations": [{"kind": "wrong-lane", "video": 12.5}]}
        save = Request(url.replace("video-comparison.html", "observations.json"),
                       data=json.dumps(payload).encode(), method="POST",
                       headers={"Content-Type": "application/json"})
        with urlopen(save) as response:
            assert response.status == 204
        assert json.loads((Path(result["folder"]) / "observations.json").read_text()) == payload
    finally:
        server.shutdown()


def test_youtube_url_variants():
    assert youtube_video_id("https://youtu.be/Iorv-Iw13oE") == "Iorv-Iw13oE"
    assert youtube_video_id("https://youtube.com/embed/Iorv-Iw13oE") == "Iorv-Iw13oE"
