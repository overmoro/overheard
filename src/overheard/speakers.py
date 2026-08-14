"""Speaker identity that persists across meetings.

The diarizer returns a voice embedding per segment. Storing those against
confirmed names lets the next meeting recognise a returning voice directly,
rather than guessing from the order people happened to speak in.

This is the last inference in the naming path. Attendee-order matching is a
convention: defensible, deterministic, and still a guess. A matched embedding is
evidence.

Library lives at ~/.config/overheard/speakers.json:

    {"Sarah Kelly": {"embedding": [...], "samples": 3, "updated": "2026-08-14"}}
"""

import json
import sys
from datetime import date
from pathlib import Path

from overheard import config as cfg

LIBRARY_PATH = cfg.CONFIG_DIR / "speakers.json"

# Cosine similarity above which two embeddings are treated as the same person.
# Deliberately cautious: a wrong name on a transcript is worse than a missing
# one, since the reader has no way to tell it is wrong.
DEFAULT_THRESHOLD = 0.70


def _cosine(a, b) -> float:
    import numpy as np

    va, vb = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    if va.shape != vb.shape or not va.size:
        return 0.0
    denominator = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(va, vb) / denominator)


class SpeakerLibrary:
    """Known voices, keyed by name."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else LIBRARY_PATH
        self._entries: dict[str, dict] = {}
        self.load()

    # ------------------------------------------------------------------

    def load(self) -> None:
        if not self.path.exists():
            self._entries = {}
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            self._entries = data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError) as e:
            print(f"[overheard] could not read speaker library: {e}", file=sys.stderr)
            self._entries = {}

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w") as f:
                json.dump(self._entries, f, indent=2)
        except OSError as e:
            print(f"[overheard] could not write speaker library: {e}", file=sys.stderr)

    # ------------------------------------------------------------------

    def names(self) -> list[str]:
        return sorted(self._entries)

    def describe(self) -> list[tuple[str, int, str]]:
        """(name, samples, updated) for each known voice, for the UI."""
        return [
            (name, int(entry.get("samples", 1)), str(entry.get("updated", "")))
            for name, entry in sorted(self._entries.items())
        ]

    def forget(self, name: str) -> bool:
        if name in self._entries:
            del self._entries[name]
            self.save()
            return True
        return False

    # ------------------------------------------------------------------

    def match(self, embedding, threshold: float = DEFAULT_THRESHOLD) -> tuple[str | None, float]:
        """Best matching name for an embedding, or (None, score)."""
        best_name, best_score = None, 0.0
        for name, entry in self._entries.items():
            stored = entry.get("embedding")
            if not stored:
                continue
            score = _cosine(embedding, stored)
            if score > best_score:
                best_name, best_score = name, score
        if best_name is not None and best_score >= threshold:
            return best_name, best_score
        return None, best_score

    def match_labels(
        self, label_embeddings: dict[str, list], threshold: float = DEFAULT_THRESHOLD
    ) -> dict[str, str]:
        """Match diarized labels to known names, one name per label.

        Resolves highest-scoring pairs first so two labels can never claim the
        same person: in a single recording, one name belongs to one voice.
        """
        scored = []
        for label, embedding in label_embeddings.items():
            name, score = self.match(embedding, threshold)
            if name is not None:
                scored.append((score, label, name))

        matched: dict[str, str] = {}
        claimed: set[str] = set()
        for _, label, name in sorted(scored, reverse=True):
            if label in matched or name in claimed:
                continue
            matched[label] = name
            claimed.add(name)
        return matched

    def remember(self, name: str, embedding) -> None:
        """Fold an embedding into the stored average for a name."""
        import numpy as np

        vector = np.asarray(embedding, dtype="float64")
        if not vector.size:
            return

        entry = self._entries.get(name)
        if entry and entry.get("embedding"):
            stored = np.asarray(entry["embedding"], dtype="float64")
            if stored.shape == vector.shape:
                samples = int(entry.get("samples", 1))
                # Running mean, so one noisy meeting cannot dominate a voice
                # that has been heard many times.
                vector = (stored * samples + vector) / (samples + 1)
                samples += 1
            else:
                samples = 1
        else:
            samples = 1

        self._entries[name] = {
            "embedding": [float(x) for x in vector],
            "samples": samples,
            "updated": date.today().isoformat(),
        }


def mean_embeddings(turns: list[dict]) -> dict[str, list]:
    """Average each speaker's per-segment embeddings, weighted by duration.

    Longer turns carry more of the voice, so they should weigh more than a
    two-word interjection.
    """
    import numpy as np

    accumulated: dict[str, tuple] = {}
    for turn in turns:
        embedding = turn.get("embedding")
        if not embedding:
            continue
        weight = max(0.0, float(turn.get("end", 0)) - float(turn.get("start", 0)))
        if weight <= 0:
            continue
        vector = np.asarray(embedding, dtype="float64") * weight
        speaker = turn["speaker"]
        if speaker in accumulated:
            total, total_weight = accumulated[speaker]
            accumulated[speaker] = (total + vector, total_weight + weight)
        else:
            accumulated[speaker] = (vector, weight)

    return {
        speaker: (total / total_weight).tolist()
        for speaker, (total, total_weight) in accumulated.items()
        if total_weight > 0
    }
