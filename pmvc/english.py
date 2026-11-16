"""An English sentence as what a text-to-speech model reads: CHIMERA's own front end.

CHIMERA has an English front end of its own (`crates/chimera/src/phonetics/english.rs`: the CMU
dictionary, letter-to-sound rules for a word that is not in it, numbers spelt out, function
words reduced, /t/ flapped and /l/ darkened as Americans do), which its planner turns into
gestures by rule. Here the same front end reads the sentence, and a model (`pmvc.tts`) learns
what to do with the phonemes from recordings instead. The engine is not changed:
`pmvc_engine.English` calls the front end as it is, and the browser could call it too.

A phoneme is a `pmvc.japanese.Phone` whose three places mean, in English:
  a1  the stress of its syllable (0 none, 1 primary, 2 secondary): a vowel's own, a consonant's
      that of the next vowel in its word, or of the last one if none follows
  a2  its place in its word, from 1
  a3  the same counted from the word's end
so that a word begins where a2 is 1 after an a3 of 1 (`pmvc.japanese.border`). Its name is the
front end's IPA symbol. The front end splits a sentence into phrases at its punctuation: a
`pau` stands between two, and the sentence asks if its last phrase does.
"""

import gzip
import json
import re
import unicodedata
from functools import cache
from typing import Any

from .japanese import NONE, Phone
from .paths import ROOT

# abbreviations the dictionary reads as something else (its "dr" is a drive)
SPELT: dict[str, str] = {"mr.": "mister", "mrs.": "missus", "ms.": "miz", "dr.": "doctor", "st.": "saint", "prof.": "professor", "vs.": "versus", "etc.": "et cetera"}


@cache
def front() -> Any:
    import pmvc_engine

    path = ROOT / "third-party" / "chimera-vocal-synth" / "public" / "data" / "cmudict.txt.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return pmvc_engine.English(f.read())


def read(text: str) -> list[Phone]:
    """The sentence's phonemes, from the silence before it to the silence after."""
    # the front end knows the letters of the dictionary: an accent goes (café), a curly quote is a straight one
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)).translate(str.maketrans("’‘“”", "''\"\""))
    text = re.sub(r"\b[A-Za-z]+\.", lambda m: SPELT.get(m.group(0).lower(), m.group(0)), text)
    phrases: list[dict[str, Any]] = [q for q in json.loads(front().phrases(text)) if q["segments"]]
    ask = bool(phrases) and phrases[-1]["intonation"] == "question"
    quiet = Phone("sil", NONE, NONE, NONE, ask, "en")
    out: list[Phone] = [quiet]
    for k, q in enumerate(phrases):
        word: list[dict[str, Any]] = []
        for seg in q["segments"]:
            word.append(seg)
            if not seg.get("wordEnd"):
                continue
            stress = [s.get("stress") or 0 for s in word]
            vowels = [i for i, s in enumerate(word) if s.get("stress") is not None and (s["stress"] > 0 or is_vowel(s["ipa"]))]
            for i, s in enumerate(word):
                after = [v for v in vowels if v >= i]
                of = (after[0] if after else vowels[-1]) if vowels else i
                out.append(Phone(s["ipa"], stress[of], i + 1, len(word) - i, ask, "en"))
            word = []
        if k < len(phrases) - 1:
            out.append(Phone("pau", NONE, NONE, NONE, ask, "en"))
    return [*out, quiet]


def is_vowel(ipa: str) -> bool:
    """Whether the front end's symbol is a vowel (a syllable's centre)."""
    return ipa[0] in "aeiouæɑɒɔəɚɛɜɝɪʊʌ"
