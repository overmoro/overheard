"""Formatting a diarized transcript as markdown for the vault.

The output is an Obsidian note: YAML frontmatter the vault can query on, and a
body where every named speaker is a ``[[wikilink]]`` so the transcript joins the
graph rather than sitting outside it.
"""

from datetime import datetime
from pathlib import Path


def fallback_speaker_names(segments: list[dict], named: dict[str, str]) -> dict[str, str]:
    """Name any unmapped speakers 'Speaker 1', 'Speaker 2', ... by first appearance.

    Numbering is assigned across the merged transcript rather than per track, so
    the reader sees one consistent sequence regardless of which side each voice
    arrived on.
    """
    display: dict[str, str] = {}
    for segment in segments:
        label = segment.get("speaker")
        if not label or label in named or label in display:
            continue
        display[label] = f"Speaker {len(display) + 1}"
    return display


def write_markdown(
    result: dict,
    output_path: str,
    meeting_details=None,   # MeetingDetails | None
    speaker_map: dict[str, str] | None = None,
) -> None:
    """Format a diarized result as a markdown transcript."""
    now = datetime.now()
    title = now.strftime("%-d %B %Y, %-I:%M%p").replace("AM", "am").replace("PM", "pm")

    lines = []

    # ---- YAML frontmatter --------------------------------------------------
    if meeting_details is not None:
        created = now.strftime("%Y-%m-%d")
        meeting_title = f"[[{meeting_details.name}]]" if meeting_details.name else "[[Meeting]]"
        attendee_lines = ""
        if meeting_details.attendees:
            formatted = [f'  - "[[{name.strip()}]]"'
                         for name in meeting_details.attendees if name and name.strip()]
            if formatted:
                attendee_lines = "\n" + "\n".join(formatted)
        else:
            attendee_lines = ""

        location_val = meeting_details.location or ""
        source_val = meeting_details.source or "in-person"

        lines.append("---")
        lines.append("type: transcript")
        lines.append(f"created: {created}")
        lines.append(f"source: {source_val}")
        lines.append(f"location: {location_val}")
        lines.append(f'meeting: "{meeting_title}"')
        lines.append(f"attendees:{attendee_lines if attendee_lines else ' []'}")
        lines.append("status: inbox")
        lines.append("topics: []")
        lines.append("---")
        lines.append("")

    # ---- Heading -----------------------------------------------------------
    lines.append(f"# Meeting \u2014 {title}\n")

    # ---- Transcript body ---------------------------------------------------
    speaker_map = speaker_map or {}
    segments = result.get("segments", [])
    fallback_names = fallback_speaker_names(segments, speaker_map)
    current_speaker = None
    current_ts = ""
    current_texts: list[str] = []

    def _flush_speaker():
        if current_speaker and current_texts:
            paragraph = " ".join(current_texts)
            lines.append(f"\n**{current_speaker}** {current_ts}\n{paragraph}")

    for seg in segments:
        raw_speaker = seg.get("speaker") or "SPEAKER_00"
        if raw_speaker in speaker_map:
            display_speaker = f"[[{speaker_map[raw_speaker]}]]"
        else:
            display_speaker = fallback_names.get(raw_speaker, raw_speaker)

        text = seg.get("text", "").strip()
        if not text:
            continue

        if display_speaker != current_speaker:
            _flush_speaker()
            start = seg.get("start", 0)
            current_ts = f"[{int(start)//3600:02d}:{(int(start)%3600)//60:02d}:{int(start)%60:02d}]"
            current_speaker = display_speaker
            current_texts = [text]
        else:
            current_texts.append(text)

    _flush_speaker()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
