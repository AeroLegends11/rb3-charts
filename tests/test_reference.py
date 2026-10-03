import json
import mido
import pytest
from rb3charts.reference import analyze_reference_corpus


def test_reference_analyzer_reads_difficulties_and_style(tmp_path):
    midi=mido.MidiFile(ticks_per_beat=480)
    track=mido.MidiTrack();track.name="PART GUITAR";midi.tracks.append(track)
    events=[]
    for base in (96,84,72,60):
        for tick,lanes in ((0,(0,2)),(240,(1,)),(480,(2,)),(720,(3,))):
            for lane in lanes:
                events.append((tick,1,base+lane));events.append((tick+120,0,base+lane))
    last=0
    for tick,on,pitch in sorted(events,key=lambda row:(row[0],-row[1])):
        track.append(mido.Message("note_on" if on else "note_off",note=pitch,velocity=100 if on else 0,time=tick-last))
        last=tick
    midi.save(tmp_path/"reference.mid")
    report=analyze_reference_corpus(
        tmp_path,
        tmp_path/"style.json",
        tmp_path/"full-report.json",
    )
    assert report["analyzed_files"]==1
    assert report["styles"]["guitar.expert"]["chord_rate"]==.25
    assert report["expert_to_hard_retention"]["guitar"]["median"]==1
    assert report["expert_to_difficulty_retention"]["guitar"]["medium"]["median"]==1
    assert report["expert_to_difficulty_retention"]["guitar"]["easy"]["median"]==1
    assert report["schema"]==2
    assert (tmp_path/"style.json").is_file()
    compact=json.loads(
        (tmp_path/"style.json").read_text(encoding="utf-8")
    )
    assert "songs" not in compact
    assert "source_folder" not in compact
    full=json.loads(
        (tmp_path/"full-report.json").read_text(encoding="utf-8")
    )
    assert full["songs"]
    assert full["source_folder"]==str(tmp_path.resolve())
    with pytest.raises(ValueError,match="require different paths"):
        analyze_reference_corpus(
            tmp_path,tmp_path/"same.json",tmp_path/"same.json"
        )
