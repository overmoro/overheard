"""Who the voices belong to, within a meeting and across them.

Two related jobs, kept together because they are two ends of one decision.
``build_speaker_map`` decides what name a diarized label gets, weighing three
signals: a voice matched against the library (evidence), which track carried the
audio (measurement), and the attendee list order (convention). The library below
is where the first of those comes from.

The diarizer returns a voice embedding per segment. Storing those against
confirmed names lets the next meeting recognise a returning voice directly,
rather than guessing from the order people happened to speak in.

Attendee-order matching is a convention: defensible, deterministic, and still a
guess. A matched embedding is evidence.

Library lives at ~/.config/overheard/speakers.json:

    {"Sarah Kelly": {"embedding": [...], "samples": 3, "updated": "2026-08-14"}}
"""

import json
import sys
from datetime import date
from pathlib import Path

from overheard import settings

LIBRARY_PATH = settings.CONFIG_DIR / "speakers.json"

# Cosine similarity above which two embeddings are treated as the same person.
# Deliberately cautious: a wrong name on a transcript is worse than a missing
# one, since the reader has no way to tell it is wrong.
DEFAULT_THRESHOLD = 0.70


def build_speaker_map(
    local_labels: list[str],
    remote_labels: list[str],
    attendees: list[str],
    mic_speaker: str | None = None,
    known: dict[str, str] | None = None,
) -> dict[str, str]:
    """Map diarized labels to real names, using which track each came from.

    Labels arrive already split by source: ``local_labels`` were heard on the
    microphone and ``remote_labels`` through the system audio. That split is
    knowledge rather than inference, so the first mic speaker is the person at
    this machine and gets ``mic_speaker`` outright.

    ``known`` holds labels already identified by voice against the speaker
    library. Those win outright: a matched embedding is evidence, where the rest
    of this function is convention.

    Everyone else is filled from the attendee list in first-speech order, which
    is deterministic and explainable, but still a guess. Additional mic speakers
    occur when several people share the laptop in an in-person meeting.
    """
    known: dict[str, str] = dict(known or {})
    mapping: dict[str, str] = dict(known)
    claimed = {name.strip().casefold() for name in known.values()}
    # Guard against whitespace-only entries: the details panel has free-text
    # attendee fields, and a name of "  " would render as a broken [[ ]] link.
    remaining = [a for a in attendees
                 if a.strip() and a.strip().casefold() not in claimed]

    # Also guard against handing out a name the library already claimed for a
    # different voice, which would show one person as two speakers.
    if (
        local_labels
        and mic_speaker
        and local_labels[0] not in mapping
        and mic_speaker.strip().casefold() not in claimed
    ):
        mapping[local_labels[0]] = mic_speaker
        # Don't hand the local speaker's name out twice if they're also listed
        remaining = [
            a for a in remaining
            if a.strip().casefold() != mic_speaker.strip().casefold()
        ]

    # Remote speakers first: they are the ones the attendee list describes.
    for label in remote_labels + local_labels:
        if label in mapping:
            continue
        if not remaining:
            break
        mapping[label] = remaining.pop(0)

    return mapping


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
        # best_score starts unset rather than at 0.0: cosine similarity runs from
        # -1 to 1, and flooring at zero reported 0.0 for a genuinely negative
        # match, which made the returned score useless for diagnosing a
        # threshold and stopped comparisons working over the full range.
        best_name: str | None = None
        best_score: float | None = None

        for name, entry in self._entries.items():
            stored = entry.get("embedding")
            if not stored:
                continue
            score = _cosine(embedding, stored)
            if best_score is None or score > best_score:
                best_name, best_score = name, score

        if best_score is None:
            return None, 0.0      # nothing stored to compare against
        if best_score >= threshold:
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
