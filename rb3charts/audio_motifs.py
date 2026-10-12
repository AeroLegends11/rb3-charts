from pathlib import Path

import numpy as np


def analyze_guitar_repetitions(path, window_seconds=4.0):
    """Find repeated multi-second guitar regions without using chart lanes."""
    import librosa

    audio, sample_rate = librosa.load(Path(path), sr=22050, mono=True)
    hop_length = 2205
    chroma = librosa.feature.chroma_cqt(
        y=audio, sr=sample_rate, hop_length=hop_length
    )
    onset = librosa.onset.onset_strength(
        y=audio, sr=sample_rate, hop_length=hop_length
    )[None, :]
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)
    onset /= np.percentile(onset, 95) + 1e-6
    rms /= np.percentile(rms, 95) + 1e-6
    features = np.nan_to_num(
        np.vstack((chroma, onset, rms)),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    frames = max(1, int(round(window_seconds * sample_rate / hop_length)))
    frame_step = 2
    embeddings = []
    times = []
    for start in range(0, features.shape[1] - frames + 1, frame_step):
        embedding = features[:, start:start + frames].reshape(-1)
        embedding = embedding - np.mean(embedding)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
        embeddings.append(embedding)
        times.append(start * hop_length / sample_rate)
    if len(embeddings) < 2:
        return {
            "method": "four-second chroma, onset and energy self-similarity",
            "window_seconds": window_seconds,
            "regions": [],
        }
    embeddings = np.asarray(embeddings, dtype=np.float32)
    similarities = embeddings @ embeddings.T
    frame_seconds = frame_step * hop_length / sample_rate
    minimum_separation = max(1, int(round(8.0 / frame_seconds)))
    neighbors = min(5, len(embeddings) - 1)
    nearest = []
    for index in range(len(embeddings)):
        similarities[
            index,
            max(0, index - minimum_separation):
            min(len(embeddings), index + minimum_separation + 1),
        ] = -2
        nearest.append(set(np.argpartition(similarities[index], -neighbors)[-neighbors:]))
    anchors = []
    for left, rows in enumerate(nearest):
        for right in rows:
            if (
                right > left
                and left in nearest[right]
                and similarities[left, right] >= .78
            ):
                anchors.append((
                    times[left],
                    times[right],
                    float(similarities[left, right]),
                ))
    groups = []
    for anchor in sorted(anchors, key=lambda row: (row[0], row[1] - row[0])):
        lag = anchor[1] - anchor[0]
        candidates = [
            rows for rows in groups
            if abs((rows[-1][1] - rows[-1][0]) - lag) <= .45
            and 0 <= anchor[0] - rows[-1][0] <= 2.1
        ]
        group = min(
            candidates,
            key=lambda rows: abs((rows[-1][1] - rows[-1][0]) - lag),
            default=None,
        )
        if group is None:
            groups.append([anchor])
        else:
            group.append(anchor)
    regions = []
    for rows in groups:
        left_start = min(row[0] for row in rows)
        left_end = max(row[0] for row in rows) + window_seconds
        right_start = min(row[1] for row in rows)
        right_end = max(row[1] for row in rows) + window_seconds
        duration = min(left_end - left_start, right_end - right_start)
        if duration < window_seconds or len(rows) < 2:
            continue
        regions.append({
            "left_start": round(left_start, 6),
            "left_end": round(left_start + duration, 6),
            "right_start": round(right_start, 6),
            "right_end": round(right_start + duration, 6),
            "duration": round(duration, 6),
            "similarity": round(float(np.mean([row[2] for row in rows])), 6),
            "anchors": len(rows),
        })
    regions.sort(
        key=lambda row: (
            -row["duration"],
            -row["similarity"],
            row["left_start"],
        )
    )
    kept = []
    for row in regions:
        duplicate = any(
            abs(row["left_start"] - old["left_start"]) < 1.0
            and abs(row["right_start"] - old["right_start"]) < 1.0
            for old in kept
        )
        if not duplicate:
            kept.append(row)
        if len(kept) >= 24:
            break
    return {
        "method": "four-second chroma, onset and energy self-similarity",
        "window_seconds": window_seconds,
        "embedding_hop_seconds": frame_seconds,
        "similarity_threshold": .78,
        "regions": kept,
    }
