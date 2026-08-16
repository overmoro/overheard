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

import contextlib
import json
import os
import shutil
import sys
import tempfile
from datetime import date
from pathlib import Path

from overheard import settings

LIBRARY_PATH = settings.CONFIG_DIR / "speakers.json"

# Cosine similarity above which two embeddings are treated as the same person.
# Deliberately cautious: a wrong name on a transcript is worse than a missing
# one, since the reader has no way to tell it is wrong.
#
# Derived, not restated. Writing 0.70 here as well would mean a user who raised
# speaker_match_threshold in their config still got 0.70 from any caller that
# relied on this default, which is the two-defaults bug settings.py was built
# to remove.
DEFAULT_THRESHOLD = settings.DEFAULTS["speaker_match_threshold"]


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
    known = dict(known or {})
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

    @property
    def _backup_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".bak")

    def load(self) -> None:
        """Read the library, falling back to the backup if the file is unusable.

        This used to reset to {} on any read failure, print a line to a stderr
        the user never sees, and carry on. Every voice the app had ever learned
        was gone at that point, and the next save wrote the empty dict over the
        file, so a single truncated write was permanent and silent.

        A voice library is accumulated slowly, over many meetings, and cannot be
        reconstructed from anything: the audio it came from is deleted unless
        keep_recordings is on. So a failed read is treated as something to
        recover from rather than something to shrug at.
        """
        for candidate, described in ((self.path, "library"),
                                     (self._backup_path, "backup library")):
            if not candidate.exists():
                continue
            try:
                with open(candidate) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"[overheard] could not read speaker {described}: {e}",
                      file=sys.stderr)
                continue
            if not isinstance(data, dict):
                print(f"[overheard] speaker {described} was not an object",
                      file=sys.stderr)
                continue
            if candidate is not self.path:
                print(f"[overheard] recovered {len(data)} voices from the backup",
                      file=sys.stderr)
            self._entries = data
            return

        self._entries = {}

    def save(self) -> None:
        """Write atomically, keeping the previous good copy as a backup.

        Two separate hazards, and the plain `open(path, "w")` this replaces had
        both. It truncates before it writes, so a crash, a full disk or a kill
        partway through leaves a half-written file that load() cannot parse. And
        it kept no previous copy, so a logically wrong but syntactically valid
        write, which is what a shape mismatch in remember() produced, was
        unrecoverable the moment it landed.

        Writing to a temp file in the same directory and renaming makes the
        replacement atomic: os.replace either happens or does not, and a reader
        sees the old file or the new one, never a partial one. Same directory
        matters, since a rename across filesystems is not atomic.
        """
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                # Copy rather than rename: a rename would leave no library at
                # all in the window before the new one lands.
                shutil.copy2(self.path, self._backup_path)

            fd, tmp_name = tempfile.mkstemp(
                dir=self.path.parent, prefix=self.path.name, suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(self._entries, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_name, self.path)
            except BaseException:
                # Including KeyboardInterrupt and SystemExit: leaving a stray
                # temp file next to the library is worse than the interruption.
                with contextlib.suppress(OSError):
                    os.unlink(tmp_name)
                raise
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
        """Fold an embedding into the stored average for a name.

        Refuses to discard an accumulated profile. On a shape mismatch this used
        to overwrite the entry and reset samples to 1, so a single malformed
        vector replaced a voice heard across a dozen meetings, silently and with
        no way back. That is not hypothetical: it is what a width bug in the
        diarizer guard actually did, and the whole reason this method is now
        careful.

        The rule: a stored profile with more than one sample in it is evidence,
        and one disagreeing vector is not enough to throw it away. The caller
        cannot make this judgement, since it does not know what is on disk.

        A legitimate width change upstream lands here too, and is deliberately
        NOT auto-accepted: it would be indistinguishable from the corruption
        above. It reports itself instead and names the way out, which is why
        forget() exists.
        """
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
            elif int(entry.get("samples", 1)) > 1:
                print(
                    f"[overheard] refusing to replace the voice profile for "
                    f"{name}: stored {stored.shape[0]} values over "
                    f"{entry.get('samples')} meetings, got {vector.shape[0]}. "
                    f"Forget this speaker in Preferences to learn them again.",
                    file=sys.stderr,
                )
                return
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
