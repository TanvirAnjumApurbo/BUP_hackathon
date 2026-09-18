"""Deterministic time-window extraction from operator-note text.

Measured on gpt-4o-mini, hour extraction is the weakest link in the whole
pipeline: it is context-sensitive and gets the same phrase right or wrong
depending on the surrounding sentence. "6 PM until 9 PM" came back as [18, 19]
inside a battery-reserve note and [18, 19, 20] inside a grid-cap note.

So the model does not own this. It owns classification (which directive type,
is the note relevant) and the numeric value. The hours come from here.

Windows are start-inclusive and end-exclusive:  hours = [h for h in range(start, end)]
  "1 PM to 3 PM"            -> [13, 14]
  "between 11 AM and 2 PM"  -> [11, 12, 13]
  "6 PM until 9 PM"         -> [18, 19, 20]

Confidence matters. When both endpoints carry an explicit meridiem or are in
24-hour form there is nothing to guess, and this module overrides the model.
When the text says "from one until three" the meridiem is genuinely ambiguous,
and the model's reading of context is better than any rule we could write, so we
defer to it.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}

_TIME = (
    r"(?:"
    r"\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)"   # 2 PM, 2:30 p.m.
    r"|\d{1,2}:\d{2}"                                # 13:00
    r"|noon|midday|midnight"
    r"|" + "|".join(WORD_NUMBERS) +                  # one .. twelve
    r"|\d{1,2}"                                      # bare 13
    r")"
)

_CONNECTOR = r"(?:\s*(?:-|–|—|to|until|till|through|thru|and)\s*)"

_RANGE = re.compile(
    r"(?:from\s+|between\s+|during\s+)?(" + _TIME + r")" + _CONNECTOR + r"(" + _TIME + r")",
    re.IGNORECASE,
)

_MERIDIEM = re.compile(r"(a\.?m\.?|p\.?m\.?)", re.IGNORECASE)


@dataclass(frozen=True)
class Window:
    hours: List[int]
    start: int
    end: int
    confidence: str  # "high" when nothing had to be guessed, else "low"
    source: str      # the substring we parsed, for logging


def _parse_endpoint(token: str) -> tuple[Optional[int], Optional[str]]:
    """Return (hour_in_12h_or_24h, meridiem or None). Hour is not yet normalised."""
    text = token.strip().lower()

    if text in ("noon", "midday"):
        return 12, "pm"
    if text == "midnight":
        return 12, "am"
    if text in WORD_NUMBERS:
        return WORD_NUMBERS[text], None

    meridiem_match = _MERIDIEM.search(text)
    meridiem = meridiem_match.group(1).replace(".", "").lower() if meridiem_match else None

    digits = re.match(r"(\d{1,2})(?::(\d{2}))?", text)
    if not digits:
        return None, meridiem
    hour = int(digits.group(1))

    # A 24-hour reading like 13:00 or a bare 18 needs no meridiem.
    if meridiem is None and (digits.group(2) is not None or hour > 12):
        return hour, "24h"
    return hour, meridiem


def _to_24h(hour: Optional[int], meridiem: Optional[str]) -> Optional[int]:
    if hour is None:
        return None
    if meridiem == "24h":
        return hour if 0 <= hour <= 23 else None
    if meridiem == "pm":
        return 12 if hour == 12 else (hour + 12 if hour < 12 else hour)
    if meridiem == "am":
        return 0 if hour == 12 else hour
    return hour if 0 <= hour <= 23 else None


def parse_window(text: str) -> Optional[Window]:
    """Window(s) found in the note, or None if the text has no range at all.

    A note can legitimately carry two ranges ("from 10 AM until noon and again
    from 2 PM until 4 PM"), but a second range can equally belong to something
    else ("capped from 6 PM to 9 PM following the 2 PM to 4 PM inspection").
    A regex cannot tell those apart, so when more than one range is present we
    return the union but mark it low confidence, which hands the decision to the
    model. It reads context; we only provide the backstop.
    """
    matches = list(_RANGE.finditer(text or ""))
    if not matches:
        return None
    if len(matches) > 1:
        merged: List[int] = []
        sources = []
        for match in matches:
            window = _single_window(match)
            if window is None:
                continue
            merged.extend(window.hours)
            sources.append(window.source)
        if not merged:
            return None
        return Window(
            hours=sorted(set(merged)),
            start=min(merged),
            end=max(merged) + 1,
            confidence="low",
            source=" + ".join(sources),
        )
    return _single_window(matches[0])


def _single_window(match: "re.Match") -> Optional[Window]:
    left_raw, right_raw = match.group(1), match.group(2)
    left_hour, left_meridiem = _parse_endpoint(left_raw)
    right_hour, right_meridiem = _parse_endpoint(right_raw)
    if left_hour is None or right_hour is None:
        return None

    # "1-3 PM" — a shared trailing meridiem applies to both endpoints. That is
    # unambiguous English, not a guess, so it keeps high confidence.
    if left_meridiem is None and right_meridiem in ("am", "pm"):
        left_meridiem = right_meridiem
    if right_meridiem is None and left_meridiem in ("am", "pm"):
        right_meridiem = left_meridiem

    # Only a window with no meridiem anywhere is genuinely ambiguous — "from one
    # until three" could be either half of the day. There we defer to the model.
    guessed = left_meridiem is None or right_meridiem is None

    start = _to_24h(left_hour, left_meridiem)
    end = _to_24h(right_hour, right_meridiem)
    if start is None or end is None or not (0 <= start <= 23) or not (0 <= end <= 24):
        return None

    if start == end:
        return None

    if end < start:
        # Overnight window. Hours must still be emitted in ascending order.
        hours = sorted(list(range(start, 24)) + list(range(0, end)))
    else:
        hours = list(range(start, min(end, 24)))

    if not hours:
        return None

    return Window(
        hours=hours,
        start=start,
        end=end,
        confidence="low" if guessed else "high",
        source=match.group(0).strip(),
    )
