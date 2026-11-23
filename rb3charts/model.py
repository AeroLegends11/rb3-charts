from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import numpy as np

DIFFICULTIES = {"easy": 60, "medium": 72, "hard": 84, "expert": 96}
TRACKS = {"guitar": "PART GUITAR", "bass": "PART BASS", "drums": "PART DRUMS", "keys": "PART KEYS"}

@dataclass
class Note:
    start: float
    end: float
    lanes: tuple[int, ...]
    confidence: float = 1.0
    source_id: str = ""
    pitch: float = 0
    toms: tuple[int, ...] = ()

@dataclass
class TempoMap:
    """Consecutive quarter-note positions in original audio seconds.

    Piecewise-linear conversion is inverted using the SAME map on export.
    No global-BPM conversion, so tempo drift cannot accumulate timing error.
    """
    beats: list[float]
    numerator: int = 4
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self):
        b = np.asarray(self.beats)
        if len(b) < 2 or not np.isfinite(b).all() or np.any(np.diff(b) <= 0):
            raise ValueError("Beat times must contain at least two strictly increasing finite seconds.")
        if self.numerator not in (3, 4, 5, 6, 7):
            raise ValueError("Supported quarter-note meters: 3/4 through 7/4.")

    def beat_at(self, seconds: float) -> float:
        i = int(np.clip(np.searchsorted(self.beats, seconds, side="right") - 1, 0, len(self.beats)-2))
        return i + (seconds-self.beats[i])/(self.beats[i+1]-self.beats[i])

    def seconds_at(self, beat: float) -> float:
        i = int(np.clip(np.floor(beat), 0, len(self.beats)-2))
        return self.beats[i] + (beat-i)*(self.beats[i+1]-self.beats[i])

    @property
    def bpm(self):
        return float(60 / np.median(np.diff(self.beats)))

@dataclass
class Options:
    source: Path
    output: Path = Path("output")
    title: str = ""
    artist: str = ""
    album: str = ""
    year: str = ""
    genre: str = ""
    instruments: tuple[str, ...] = ("guitar", "bass", "drums")
    chart_profile: str = "auto"
    quality: str = "precision"
    device: str = "auto"
    stems: Path | None = None
    beats_file: Path | None = None
    bpm: float | None = None
    first_beat: float = 0.0
    meter: int = 4
    confidence: float = 0.35
    onyx: Path | None = None
    compile: bool = True
    clean_guitar_stem: bool = True

def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, default=lambda x: asdict(x) if hasattr(x, "__dataclass_fields__") else str(x)), encoding="utf-8")
