"""A Japanese sentence as what a text-to-speech model reads: Open JTalk's phonemes, each with
where its mora stands against the pitch accent.

Japanese words differ by where the pitch falls (箸 against 橋), and writing does not show it.
Open JTalk's front end reads the sentence, splits it into accent phrases and estimates each
one's accent; its full-context labels say for every phoneme
  a1  how many morae its mora is from the phrase's accent nucleus (0 on it, negative before);
      a phrase with no fall counts from its start, so a1 is then the mora's place itself
  a2  the mora's place in its accent phrase, from 1
  a3  the same counted from the phrase's end
and for the sentence whether it is a question. That is all `pmvc.tts` is told of the prosody.
The front end's reading and accent can be wrong; the recordings are learned from as labelled.
"""

import re
from dataclasses import dataclass

# what a silence's (sil, pau) accent places are written as
NONE: int = -99
# the furthest places told apart: beyond them a mora is as the one at the limit
NUCLEUS: int = 12
PLACE: int = 16

_PHONE = re.compile(r"-(.+?)\+")
_ACCENT = re.compile(r"/A:(-?\d+)\+(\d+)\+(\d+)/")
_ASKED = re.compile(r"/E:[^!]*!(\d+)_")


@dataclass(frozen=True)
class Phone:
    """A phoneme as Open JTalk names it (`sil` at both ends, `pau`, `cl` for っ, `N` for ん,
    capital vowels for devoiced ones) with its mora's places (NONE for a silence), and whether
    its sentence asks. `sp` is not Open JTalk's: a pause the speaker made where the text has
    none (`pmvc.align` puts it in)."""

    p: str
    a1: int
    a2: int
    a3: int
    ask: bool
    # the language whose front end read it: `pmvc.english` gives the three places other meanings
    lang: str = "ja"

    @property
    def silent(self) -> bool:
        return self.p in ("sil", "pau", "sp")

    @property
    def ends_mora(self) -> bool:
        """Whether a mora can end with it: a vowel, ん or っ. (In English the places are a
        phoneme's own, and any phoneme can be its word's last.)"""
        return self.lang != "ja" or self.p in ("a", "i", "u", "e", "o", "A", "I", "U", "E", "O", "N", "cl")

    def ids(self, phones: list[str]) -> list[int]:
        """The phone as five small numbers, each from 0: its place in `phones` (a model's list
        of phonemes), the three accent places clipped (0 for a silence), the question."""
        if self.a1 == NONE:
            return [phones.index(self.p), 0, 0, 0, int(self.ask)]
        return [phones.index(self.p), max(-NUCLEUS, min(NUCLEUS, self.a1)) + NUCLEUS + 1, min(PLACE, self.a2), min(PLACE, self.a3), int(self.ask)]


# how many values each of `Phone.ids`' numbers after the first can take
SIZES: tuple[int, int, int, int] = (2 * NUCLEUS + 2, PLACE + 1, PLACE + 1, 2)


def border(phones: list[Phone], k: int) -> bool:
    """Whether an accent phrase ends before phoneme `k` and the next starts with it (in English: a word)."""
    return k > 0 and not phones[k].silent and phones[k].a2 == 1 and phones[k - 1].ends_mora and phones[k - 1].a3 == 1


def read(text: str, lang: str = "ja") -> list[Phone]:
    """The sentence's phonemes, from the silence before it to the silence after, as the
    language's front end reads it."""
    if lang == "en":
        from .english import read as read_english

        return read_english(text)
    import pyopenjtalk

    labels: list[str] = pyopenjtalk.extract_fullcontext(text)
    if not labels:
        # nothing in the text that is read aloud (punctuation alone, a picture): the two silences
        return [Phone("sil", NONE, NONE, NONE, False), Phone("sil", NONE, NONE, NONE, False)]
    # the last label, the closing silence, says whether the phrase before it asks
    asked = _ASKED.search(labels[-1])
    ask = asked is not None and asked.group(1) == "1"
    out: list[Phone] = []
    for label in labels:
        m = _PHONE.search(label)
        assert m is not None, label
        accent = _ACCENT.search(label)
        a1, a2, a3 = (int(g) for g in accent.groups()) if accent else (NONE, NONE, NONE)
        out.append(Phone(m.group(1), a1, a2, a3, ask))
    return out
