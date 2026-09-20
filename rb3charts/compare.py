"""Read-only comparison of generated and human-authored five-lane charts."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from hashlib import sha256
from html import escape
import json
import mido
from pathlib import Path
from statistics import median

from .reference import read_five_lane


MIDI_CANDIDATES = (Path("midi/notes.mid"), Path("import/notes.mid"), Path("notes.mid"))


def resolve_midi(value: Path | str) -> Path:
    """Resolve either a MIDI file or an RB3 Charts output folder."""
    path = Path(value).expanduser().resolve()
    if path.is_file() and path.suffix.lower() in (".mid", ".midi"):
        return path
    if path.is_dir():
        for relative in MIDI_CANDIDATES:
            candidate = path / relative
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(f"No five-lane MIDI found at {path}")


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chart(path: Path, instrument: str, difficulty: str) -> list[dict]:
    charts = read_five_lane(path)
    if instrument not in charts:
        raise ValueError(f"{path} has no PART {instrument.upper()} track")
    attacks = charts[instrument].get(difficulty, [])
    if not attacks:
        raise ValueError(f"{path} has no {difficulty.title()} {instrument.title()} notes")
    midi = mido.MidiFile(path)
    tempo_events = [(0, 500000)]
    for track in midi.tracks:
        absolute = 0
        for message in track:
            absolute += message.time
            if message.type == "set_tempo":
                tempo_events.append((absolute, message.tempo))
    tempo_events = sorted(dict(tempo_events).items())

    def seconds(tick):
        elapsed = 0.0; previous_tick = 0; tempo = 500000
        for change_tick, next_tempo in tempo_events:
            if change_tick >= tick: break
            elapsed += mido.tick2second(change_tick - previous_tick, midi.ticks_per_beat, tempo)
            previous_tick, tempo = change_tick, next_tempo
        return elapsed + mido.tick2second(tick - previous_tick, midi.ticks_per_beat, tempo)

    for attack in attacks:
        attack["second"] = seconds(attack["tick"])
        end_tick = attack["tick"] + attack["duration_beats"] * midi.ticks_per_beat
        attack["duration_seconds"] = seconds(end_tick) - attack["second"]
    return attacks


def _match(generated: list[dict], reference: list[dict], offset: float, tolerance: float, coordinate: str = "second"):
    pairs = []
    left = 0
    for gi, attack in enumerate(generated):
        aligned = attack[coordinate] + offset
        while left < len(reference) and reference[left][coordinate] < aligned - tolerance:
            left += 1
        ri = left
        while ri < len(reference) and reference[ri][coordinate] <= aligned + tolerance:
            target = reference[ri]
            error = abs(aligned - target[coordinate])
            pairs.append((error, gi, ri))
            ri += 1
    used_g, used_r, matches = set(), set(), []
    for error, gi, ri in sorted(pairs):
        if gi in used_g or ri in used_r:
            continue
        used_g.add(gi)
        used_r.add(ri)
        matches.append((gi, ri, error))
    return sorted(matches), sorted(set(range(len(generated))) - used_g), sorted(set(range(len(reference))) - used_r)


def _alignment(generated: list[dict], reference: list[dict], tolerance: float, coordinate: str = "second") -> float:
    """Choose a beat offset using onset agreement without inspecting lanes."""
    # Early and late samples make the search stable without an unrestricted
    # all-pairs scan on very long songs. Offsets are quantized finer than the
    # default matching tolerance, then scored against every attack.
    def sample(rows):
        return rows[:96] + (rows[-32:] if len(rows) > 96 else [])

    candidate_counts = Counter({0.0: 1})
    for generated_attack in sample(generated):
        for reference_attack in sample(reference):
            delta = reference_attack[coordinate] - generated_attack[coordinate]
            if abs(delta) <= (90 if coordinate == "second" else 128):
                quantum = 1000 if coordinate == "second" else 96
                candidate_counts[round(delta * quantum) / quantum] += 1
    best = None
    for offset, _ in candidate_counts.most_common(256):
        matches, _, _ = _match(generated, reference, offset, tolerance, coordinate)
        errors = [row[2] for row in matches]
        score = (len(matches), -median(errors) if errors else -999, -abs(offset))
        if best is None or score > best[0]:
            best = (score, offset)
    return round(best[1], 6) if best else 0.0


def _shape(lanes: list[int]) -> tuple[int, ...]:
    low = min(lanes)
    return tuple(lane - low for lane in sorted(lanes))


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _motif_signature(rows: list[dict], coordinate: str = "second") -> tuple:
    origin = min(lane for row in rows for lane in row["lanes"])
    lanes = tuple(tuple(lane - origin for lane in sorted(row["lanes"])) for row in rows)
    quantum = 100 if coordinate == "second" else 24
    gaps = tuple(round((rows[i + 1][coordinate] - rows[i][coordinate]) * quantum) for i in range(len(rows) - 1))
    return lanes, gaps


def _motifs(rows: list[dict], coordinate: str = "second") -> Counter:
    return Counter(_motif_signature(rows[index:index + 4], coordinate) for index in range(max(0, len(rows) - 3)))


def _section_rows(generated, reference, matches, offset, coordinate="second", size=10):
    positions = [row[coordinate] for row in reference] + [row[coordinate] + offset for row in generated]
    first = int(min(positions) // size) * size
    last = int(max(positions) // size) * size
    match_lookup = {(gi, ri) for gi, ri, _ in matches}
    sections = []
    for start in range(first, last + 1, size):
        end = start + size
        g_indices = [i for i, row in enumerate(generated) if start <= row[coordinate] + offset < end]
        r_indices = [i for i, row in enumerate(reference) if start <= row[coordinate] < end]
        local = [(gi, ri) for gi, ri in match_lookup if gi in g_indices and ri in r_indices]
        exact = sum(sorted(generated[gi]["lanes"]) == sorted(reference[ri]["lanes"]) for gi, ri in local)
        sections.append({
            f"start_{coordinate}": start, f"end_{coordinate}": end,
            "generated_attacks": len(g_indices), "reference_attacks": len(r_indices),
            "matched_attacks": len(local), "extra_attacks": len(g_indices) - len(local),
            "missing_attacks": len(r_indices) - len(local),
            "exact_lane_rate": round(exact / len(local), 4) if local else 0.0,
        })
    return sections


def compare_charts(generated_input: Path | str, reference_input: Path | str,
                   output: Path | str = Path("comparisons"), instrument: str = "guitar",
                   difficulty: str = "expert", tolerance_beats: float = 0.125,
                   offset_beats: float | None = None, tolerance_seconds: float = 0.1,
                   offset_seconds: float | None = None) -> dict:
    """Compare charts and write HTML, Markdown, and machine-readable reports."""
    if tolerance_beats <= 0:
        raise ValueError("tolerance_beats must be positive")
    generated_path = resolve_midi(generated_input)
    reference_path = resolve_midi(reference_input)
    song = _song_context(generated_input, generated_path)
    generated = _chart(generated_path, instrument, difficulty)
    reference = _chart(reference_path, instrument, difficulty)
    coordinate = "beat" if offset_beats is not None else "second"
    tolerance = tolerance_beats if coordinate == "beat" else tolerance_seconds
    manual_offset = offset_beats if coordinate == "beat" else offset_seconds
    offset = float(manual_offset) if manual_offset is not None else _alignment(generated, reference, tolerance, coordinate)
    matches, extras, missing = _match(generated, reference, offset, tolerance, coordinate)

    errors = [error for _, _, error in matches]
    exact = sum(sorted(generated[gi]["lanes"]) == sorted(reference[ri]["lanes"]) for gi, ri, _ in matches)
    shapes = sum(_shape(generated[gi]["lanes"]) == _shape(reference[ri]["lanes"]) for gi, ri, _ in matches)
    chord_class = sum((len(generated[gi]["lanes"]) > 1) == (len(reference[ri]["lanes"]) > 1) for gi, ri, _ in matches)
    duration_key = "duration_beats" if coordinate == "beat" else "duration_seconds"
    sustain_errors = [abs(generated[gi][duration_key] - reference[ri][duration_key]) for gi, ri, _ in matches]
    precision = len(matches) / len(generated)
    recall = len(matches) / len(reference)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    generated_motifs, reference_motifs = _motifs(generated, coordinate), _motifs(reference, coordinate)
    recurring = {signature: count for signature, count in reference_motifs.items() if count >= 2}
    recurring_total = sum(recurring.values())
    recurring_matched = sum(min(count, generated_motifs.get(signature, 0)) for signature, count in recurring.items())

    report = {
        "schema": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "song": song,
        "provenance": {
            "generated_midi": str(generated_path), "generated_sha256": _hash(generated_path),
            "reference_midi": str(reference_path), "reference_sha256": _hash(reference_path),
            "instrument": instrument, "difficulty": difficulty,
            f"tolerance_{coordinate}s" if coordinate == "second" else "tolerance_beats": tolerance,
            f"alignment_offset_{coordinate}s" if coordinate == "second" else "alignment_offset_beats": offset,
            "alignment_coordinate": coordinate,
            "alignment": "manual" if manual_offset is not None else "automatic-onsets-only",
        },
        "summary": {
            "generated_attacks": len(generated), "reference_attacks": len(reference),
            "matched_attacks": len(matches), "extra_attacks": len(extras), "missing_attacks": len(missing),
            "attack_precision": round(precision, 4), "attack_recall": round(recall, 4), "attack_f1": round(f1, 4),
            f"median_timing_error_{coordinate}s" if coordinate == "second" else "median_timing_error_beats": round(median(errors), 4) if errors else None,
            f"p95_timing_error_{coordinate}s" if coordinate == "second" else "p95_timing_error_beats": round(_percentile(errors, .95), 4) if errors else None,
            "exact_lane_rate": round(exact / len(matches), 4) if matches else 0.0,
            "relative_shape_rate": round(shapes / len(matches), 4) if matches else 0.0,
            "chord_class_rate": round(chord_class / len(matches), 4) if matches else 0.0,
            f"sustain_mae_{coordinate}s" if coordinate == "second" else "sustain_mae_beats": round(sum(sustain_errors) / len(sustain_errors), 4) if sustain_errors else None,
            "reference_recurring_patterns": len(recurring),
            "recurring_pattern_coverage": round(recurring_matched / recurring_total, 4) if recurring_total else None,
        },
        "sections": _section_rows(generated, reference, matches, offset, coordinate),
        "unmatched": {
            "generated": [{coordinate: round(generated[i][coordinate] + offset, 4), "lanes": generated[i]["lanes"]} for i in extras],
            "reference": [{coordinate: round(reference[i][coordinate], 4), "lanes": reference[i]["lanes"]} for i in missing],
        },
    }
    destination = Path(output).expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = destination / f"{generated_path.stem}_vs_{reference_path.stem}_{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    json_path = run / "comparison.json"
    markdown_path = run / "comparison.md"
    html_path = run / "comparison.html"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    html_path.write_text(_html(report), encoding="utf-8")
    report["output"] = {"folder": str(run), "json": str(json_path), "markdown": str(markdown_path), "html": str(html_path)}
    return report


def _song_context(generated_input: Path | str, generated_path: Path) -> dict:
    value = Path(generated_input).expanduser().resolve()
    roots = [value] if value.is_dir() else [value.parent]
    roots += [generated_path.parent, generated_path.parent.parent]
    for root in roots:
        for relative in (Path("metadata/song.json"), Path("result.json")):
            path = root / relative
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            return {key: data.get(key, "") for key in ("title", "artist", "chart_profile")}
    return {"title": generated_path.stem, "artist": "", "chart_profile": ""}


def _display(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _markdown(report: dict) -> str:
    p, s = report["provenance"], report["summary"]
    coordinate = p.get("alignment_coordinate", "beat")
    offset_key = "alignment_offset_seconds" if coordinate == "second" else "alignment_offset_beats"
    lines = [
        "# Chart comparison", "",
        f"Generated: `{p['generated_midi']}`", "",
        f"Reference: `{p['reference_midi']}`", "",
        f"Track: **{p['difficulty'].title()} {p['instrument'].title()}**  ",
        f"Alignment: **{p[offset_key]:+.4f} {coordinate}s** ({p['alignment']})", "",
        "| Measure | Result |", "|---|---:|",
    ]
    labels = {
        "generated_attacks": "Generated attacks", "reference_attacks": "Reference attacks",
        "matched_attacks": "Matched attacks", "extra_attacks": "Extra generated attacks",
        "missing_attacks": "Missing generated attacks", "attack_precision": "Attack precision",
        "attack_recall": "Attack recall", "attack_f1": "Attack F1",
        "median_timing_error_beats": "Median timing error (beats)", "p95_timing_error_beats": "P95 timing error (beats)",
        "exact_lane_rate": "Exact lane agreement", "relative_shape_rate": "Relative shape agreement",
        "chord_class_rate": "Single/chord agreement", "sustain_mae_beats": "Sustain MAE (beats)",
        "reference_recurring_patterns": "Recurring reference patterns", "recurring_pattern_coverage": "Recurring-pattern coverage",
    }
    lines += [f"| {labels.get(key, key.replace('_', ' ').title())} | {_display(value)} |" for key, value in s.items()]
    unit_label = "Ten-second" if coordinate == "second" else "Sixteen-beat"
    start_key, end_key = f"start_{coordinate}", f"end_{coordinate}"
    lines += ["", f"## {unit_label} regions", "", f"| {coordinate.title()}s | Generated | Reference | Matched | Extra | Missing | Exact lanes |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["sections"]:
        lines.append(f"| {row[start_key]}–{row[end_key]} | {row['generated_attacks']} | {row['reference_attacks']} | {row['matched_attacks']} | {row['extra_attacks']} | {row['missing_attacks']} | {_display(row['exact_lane_rate'])} |")
    return "\n".join(lines) + "\n"


def _html(report: dict) -> str:
    p, s = report["provenance"], report["summary"]
    coordinate = p.get("alignment_coordinate", "beat")
    offset_key = "alignment_offset_seconds" if coordinate == "second" else "alignment_offset_beats"
    tolerance_key = "tolerance_seconds" if coordinate == "second" else "tolerance_beats"
    start_key, end_key = f"start_{coordinate}", f"end_{coordinate}"
    cells = "".join(f"<div class='metric'><b>{escape(key.replace('_', ' ').title())}</b><span>{escape(_display(value))}</span></div>" for key, value in s.items())
    rows = "".join(
        f"<tr><td>{row[start_key]}–{row[end_key]}</td><td>{row['generated_attacks']}</td><td>{row['reference_attacks']}</td><td>{row['matched_attacks']}</td><td>{row['extra_attacks']}</td><td>{row['missing_attacks']}</td><td>{_display(row['exact_lane_rate'])}</td></tr>"
        for row in report["sections"]
    )
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Chart comparison</title><style>
body{{font:15px system-ui;background:#10141d;color:#e9edf5;margin:0;padding:32px}}main{{max-width:1100px;margin:auto}}h1{{margin-bottom:8px}}.muted{{color:#aab3c5;overflow-wrap:anywhere}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin:24px 0}}.metric{{background:#1a2130;border:1px solid #303a50;border-radius:8px;padding:12px;display:flex;flex-direction:column;gap:7px}}.metric span{{font-size:1.45rem;color:#78d8ff}}table{{width:100%;border-collapse:collapse;background:#171d29}}th,td{{padding:9px;border-bottom:1px solid #303a50;text-align:right}}th:first-child,td:first-child{{text-align:left}}code{{color:#b9e7ff}}.worksheet{{margin-top:32px;padding:20px;background:#171d29;border:1px solid #303a50;border-radius:10px}}.questions{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:13px}}label{{display:flex;flex-direction:column;gap:6px;color:#c9d3e5}}select,textarea,button{{font:inherit;color:#eef4ff;background:#222c3e;border:1px solid #52627d;border-radius:7px;padding:9px}}textarea{{min-height:76px;resize:vertical}}button{{cursor:pointer;background:#286c60;font-weight:700}}button:hover{{background:#348878}}#copilot-output{{min-height:280px}}#copy-status{{margin-left:10px;color:#79ddc7}}
</style></head><body><main><h1>Chart comparison</h1><p class='muted'>Generated: {escape(p['generated_midi'])}<br>Reference: {escape(p['reference_midi'])}</p><p><b>{escape(p['difficulty'].title())} {escape(p['instrument'].title())}</b> · alignment {p[offset_key]:+.4f} {coordinate}s ({escape(p['alignment'])}) · tolerance ±{p[tolerance_key]} {coordinate}s</p><div class='grid'>{cells}</div><h2>{'Ten-second' if coordinate == 'second' else 'Sixteen-beat'} regions</h2><table><thead><tr><th>{coordinate.title()}s</th><th>Generated</th><th>Reference</th><th>Matched</th><th>Extra</th><th>Missing</th><th>Exact lanes</th></tr></thead><tbody>{rows}</tbody></table>{_review_panel(report)}</main></body></html>"""


def _review_context(report: dict) -> dict:
    """Return aggregate, path-free evidence safe to paste into a coding task."""
    p, s = report["provenance"], report["summary"]
    coordinate = p.get("alignment_coordinate", "second")
    start_key, end_key = f"start_{coordinate}", f"end_{coordinate}"
    worst = sorted(report.get("sections", []),
                   key=lambda row: row.get("extra_attacks", 0) + row.get("missing_attacks", 0),
                   reverse=True)[:6]
    return {
        "song": report.get("song", {}),
        "track": f"{p.get('difficulty', 'expert').title()} {p.get('instrument', 'guitar').title()}",
        "generated_sha256": p.get("generated_sha256", "unknown"),
        "summary": s,
        "coordinate": coordinate,
        "worst_regions": [
            {"start": row.get(start_key), "end": row.get(end_key),
             "extra": row.get("extra_attacks", 0), "missing": row.get("missing_attacks", 0),
             "exact_lane_rate": row.get("exact_lane_rate", 0)} for row in worst
        ],
    }


def _review_panel(report: dict) -> str:
    context = json.dumps(_review_context(report), separators=(",", ":")).replace("</", "<\\/")
    panel = r"""
<section class='worksheet'><h2>Human review → Copilot update</h2>
<p class='muted'>Answer after watching and listening. Your answers are saved in this browser. Copy the completed update only when it describes the problems you want fixed.</p>
<div class='questions'>
<label>Overall verdict<select id='human-verdict'><option>Not answered</option><option>Close enough</option><option>Review closely</option><option>Needs substantial work</option></select></label>
<label>Correct instrument or musical part?<select id='human-role'><option>Not answered</option><option>Yes</option><option>Mostly</option><option>No — it follows the wrong sound</option><option>Mixed by section</option></select></label>
<label>Note density and playability<select id='human-density'><option>Not answered</option><option>Feels playable</option><option>Too dense</option><option>Too sparse</option><option>Varies by section</option></select></label>
<label>Repeated patterns<select id='human-patterns'><option>Not answered</option><option>Stable and recognizable</option><option>Sometimes lose their shape</option><option>Mostly random or unstable</option><option>No clear repeated pattern</option></select></label>
<label>Attack timing<select id='human-timing'><option>Not answered</option><option>Mostly accurate</option><option>Often early</option><option>Often late</option><option>Inconsistent</option></select></label>
<label>Hold notes<select id='human-sustains'><option>Not answered</option><option>Mostly appropriate</option><option>Too many</option><option>Too few</option><option>Too short</option><option>Too long</option><option>Mixed problems</option></select></label>
<label>Lane/color behavior<select id='human-lanes'><option>Not answered</option><option>Readable and consistent</option><option>Repeated riffs change colors</option><option>Too many random jumps</option><option>Too repetitive or flat</option><option>Mixed problems</option></select></label>
<label>Most important area<select id='human-area'><option>Not answered</option><option>Instrument selection</option><option>Missing or extra attacks</option><option>Density/playability</option><option>Repeated riff stability</option><option>Lane choices</option><option>Timing</option><option>Hold notes</option></select></label>
</div>
<label>What already works well<textarea id='human-good' placeholder='Example: the intro rhythm is recognizable and playable'></textarea></label>
<label>Problem timestamps and what you hear<textarea id='human-times' placeholder='Example: 00:56 — too many attacks; 01:20 — the same sound happens twice but only one note appears'></textarea></label>
<label>Highest-priority correction<textarea id='human-priority' placeholder='Describe the one problem Copilot should target first'></textarea></label>
<p><button id='copy-review'>Build and copy Copilot review update</button><span id='copy-status'></span></p>
<label>Copyable update<textarea id='copilot-output' readonly placeholder='Complete the worksheet, then use the button above'></textarea></label>
</section>
<script>
const REVIEW_CONTEXT=__CONTEXT__;
const REVIEW_FIELDS=['human-verdict','human-role','human-density','human-patterns','human-timing','human-sustains','human-lanes','human-area','human-good','human-times','human-priority'];
const REVIEW_KEY='rb3-human-review:'+REVIEW_CONTEXT.generated_sha256;
function reviewAnswers(){const answer={};REVIEW_FIELDS.forEach(id=>answer[id]=document.getElementById(id).value.trim());return answer}
function persistReview(){localStorage.setItem(REVIEW_KEY,JSON.stringify(reviewAnswers()))}
function restoreReview(){try{const saved=JSON.parse(localStorage.getItem(REVIEW_KEY)||'{}');REVIEW_FIELDS.forEach(id=>{if(saved[id]!==undefined)document.getElementById(id).value=saved[id]})}catch(e){}}
function pct(value){return value===null||value===undefined?'n/a':(100*Number(value)).toFixed(1)+'%'}
function buildReview(){
 const a=reviewAnswers(),s=REVIEW_CONTEXT.summary,song=REVIEW_CONTEXT.song||{};
 const regions=REVIEW_CONTEXT.worst_regions.map(r=>`- ${r.start}-${r.end} ${REVIEW_CONTEXT.coordinate}s: ${r.extra} extra, ${r.missing} missing, ${pct(r.exact_lane_rate)} exact lanes`).join('\n');
 return `RB3 Charts human review update\n\nSong: ${song.artist||'Unknown artist'} - ${song.title||'Unknown title'}\nTrack: ${REVIEW_CONTEXT.track}\nResolved chart profile: ${song.chart_profile||'unknown'}\nGenerated MIDI SHA-256: ${REVIEW_CONTEXT.generated_sha256}\n\nAutomatic comparison\n- Attacks: ${s.generated_attacks} generated / ${s.reference_attacks} reference; ${s.extra_attacks} extra and ${s.missing_attacks} missing\n- Precision / recall / F1: ${pct(s.attack_precision)} / ${pct(s.attack_recall)} / ${pct(s.attack_f1)}\n- Exact lanes / relative shape / chord class: ${pct(s.exact_lane_rate)} / ${pct(s.relative_shape_rate)} / ${pct(s.chord_class_rate)}\n- Recurring-pattern coverage: ${pct(s.recurring_pattern_coverage)}\n\nHighest-error regions\n${regions||'- None reported'}\n\nMy listening and playability review\n- Overall verdict: ${a['human-verdict']}\n- Correct musical part: ${a['human-role']}\n- Density/playability: ${a['human-density']}\n- Repeated patterns: ${a['human-patterns']}\n- Timing: ${a['human-timing']}\n- Hold notes: ${a['human-sustains']}\n- Lane/color behavior: ${a['human-lanes']}\n- Most important area: ${a['human-area']}\n- What works well: ${a['human-good']||'Not provided'}\n- Problem timestamps: ${a['human-times']||'Not provided'}\n- Highest-priority correction: ${a['human-priority']||'Not provided'}\n\nPlease target the highest-priority human-observed problem first. Use the automatic comparison as supporting evidence, not as a command to imitate the official chart. Diagnose and change one general subsystem at a time. Do not special-case this song or use reference notes as generation inputs. Preserve real variations, rests, source IDs, attack timing and lower-difficulty ancestry. Report synthetic tests separately from cached replay, listening review and Xbox playtesting.`
}
REVIEW_FIELDS.forEach(id=>document.getElementById(id).addEventListener('input',persistReview));restoreReview();
document.getElementById('copy-review').onclick=async()=>{persistReview();const text=buildReview(),output=document.getElementById('copilot-output'),status=document.getElementById('copy-status');output.value=text;try{await navigator.clipboard.writeText(text);status.textContent='Copied to clipboard'}catch(e){output.focus();output.select();document.execCommand('copy');status.textContent='Selected and copied; press Ctrl+C if needed'}};
</script>
"""
    return panel.replace("__CONTEXT__", context)
