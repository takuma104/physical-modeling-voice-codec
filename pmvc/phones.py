"""A recording's aligned phones as input for CHIMERA's planner: the phones the speaker said,
for as long as the speaker held them, split into breath groups at the speaker's pauses.

The ARPAbet-to-IPA table, the function words and the two American allophones (flapping, dark
/l/) follow CHIMERA's own English front end (`crates/chimera/src/phonetics/english.rs`), so
the planner sees the phones it was written for.
"""

from dataclasses import dataclass

from .tsp import Utterance

ARPA_IPA: dict[str, str] = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ", "EH": "ɛ", "ER": "ɝ",
    "EY": "eɪ", "IH": "ɪ", "IY": "i", "OW": "oʊ", "OY": "ɔɪ", "UH": "ʊ", "UW": "u",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f", "G": "g", "HH": "h", "JH": "dʒ",
    "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}  # fmt: skip
VOWELS: frozenset[str] = frozenset(
    ["ɑ", "æ", "ʌ", "ə", "ɔ", "aʊ", "aɪ", "ɛ", "ɝ", "ɚ", "eɪ", "ɪ", "i", "oʊ", "ɔɪ", "ʊ", "u"]
)
STOPS: frozenset[str] = frozenset(["p", "t", "k", "b", "d", "g", "tʃ", "dʒ"])
FUNCTION_WORDS: frozenset[str] = frozenset(
    "a an the of to in on at by for from with as and or but nor so if is am are was were be been "
    "do does did has have had will would can could should shall may might must it its i you he "
    "she we they me him her us them my your his our their this that than then there not no".split()
)

# A silence between two phones at least this long (s) starts a new breath group. The aligner
# marks silences from 30 ms; shorter ones than this go to the phones on either side.
PAUSE_MIN: float = 0.15
# An intervocalic /t d/ held longer than this (s) was not flapped by the speaker.
FLAP_MAX: float = 0.06


@dataclass
class Seg:
    ipa: str
    start: float
    end: float
    stress: int
    word_end: bool

    @property
    def dur(self) -> float:
        return self.end - self.start


def to_segments(utt: Utterance) -> list[Seg]:
    """The recording's phones as planner segments, on the recording's own times."""
    segs: list[Seg] = []
    word_of: list[int] = []
    for p in utt.phones:
        base = p.label.rstrip("012")
        if base not in ARPA_IPA:
            raise ValueError(f"{utt.id}: no phone for {p.label!r}")
        digit = p.label[len(base) :]
        stress = int(digit) if digit else 0
        ipa = ARPA_IPA[base]
        if base == "AH" and stress == 0:
            ipa = "ə"
        if base == "ER" and stress == 0:
            ipa = "ɚ"
        mid = 0.5 * (p.start + p.end)
        w = next((i for i, x in enumerate(utt.words) if x.start <= mid < x.end), -1)
        word_of.append(w)
        segs.append(Seg(ipa, p.start, p.end, stress if ipa in VOWELS else 0, False))
    for i, s in enumerate(segs):
        s.word_end = i + 1 == len(segs) or word_of[i + 1] != word_of[i]
        # monosyllabic function words carry no accent in running speech
        if word_of[i] >= 0 and utt.words[word_of[i]].label in FUNCTION_WORDS:
            s.stress = 0
    _allophones(segs)
    return segs


def _allophones(segs: list[Seg]) -> None:
    for i in range(1, len(segs) - 1):
        prev, s, nxt = segs[i - 1], segs[i], segs[i + 1]
        flappable = (prev.ipa in VOWELS or prev.ipa == "ɹ") and nxt.ipa in VOWELS and nxt.stress == 0
        # CHIMERA's rule, plus the recording's evidence: a tap is short
        if s.ipa in ("t", "d") and flappable and not prev.word_end and s.dur <= FLAP_MAX:
            s.ipa = "ɾ"
    for i, s in enumerate(segs):
        if s.ipa == "l":
            next_vowel = i + 1 < len(segs) and segs[i + 1].ipa in VOWELS
            if not next_vowel or s.word_end:
                s.ipa = "ɫ"


def to_phrases(segs: list[Seg]) -> list[list[Seg]]:
    """Breath groups: split at pauses, with shorter silences given to the neighbouring phones (a
    stop's closure takes the silence before it; otherwise the two neighbours share it)."""
    phrases: list[list[Seg]] = [[]]
    for i, s in enumerate(segs):
        s = Seg(s.ipa, s.start, s.end, s.stress, s.word_end)
        cur = phrases[-1]
        if cur:
            gap = s.start - cur[-1].end
            if gap >= PAUSE_MIN:
                phrases.append([s])
                continue
            if gap > 1e-6:
                if s.ipa in STOPS:
                    s.start -= gap
                elif cur[-1].ipa in STOPS:
                    cur[-1].end += gap
                else:
                    cur[-1].end += 0.5 * gap
                    s.start = cur[-1].end
        cur.append(s)
    return phrases


# The planner splits these into two vowel targets (two planned segments).
DIPHTHONGS: frozenset[str] = frozenset(["aɪ", "aʊ", "ɔɪ", "eɪ", "oʊ"])


def phrase_specs(phrases: list[list[Seg]], shift: float, durations: bool = True) -> list[dict[str, object]]:
    """The phrases as `pmvc-synth` takes them, `shift` seconds later than their own times.
    Without `durations` the planner's rules time the phones."""
    out: list[dict[str, object]] = []
    for n, ph in enumerate(phrases):
        segs: list[dict[str, object]] = []
        for s in ph:
            seg: dict[str, object] = {"ipa": s.ipa, "stress": s.stress, "wordEnd": s.word_end}
            if durations:
                seg["dur"] = round(s.dur, 4)
            segs.append(seg)
        out.append(
            {
                "t0": round(ph[0].start + shift, 4),
                # only the sentence's end falls; a pause inside it does not
                "intonation": "statement" if n + 1 == len(phrases) else "flat",
                "segments": segs,
            }
        )
    return out


def relayout(phrases: list[list[Seg]], durs: list[list[float]]) -> list[list[Seg]]:
    """The phrases with other phone durations: the first starts where it did, and each pause
    between phrases stays as long as it was."""
    out: list[list[Seg]] = []
    for n, (ph, ds) in enumerate(zip(phrases, durs)):
        t = ph[0].start if n == 0 else out[-1][-1].end + (ph[0].start - phrases[n - 1][-1].end)
        new: list[Seg] = []
        for s, d in zip(ph, ds):
            new.append(Seg(s.ipa, t, t + d, s.stress, s.word_end))
            t += d
        out.append(new)
    return out


def boundary_times(phrases: list[list[Seg]]) -> list[float]:
    """Every phone boundary in order: each phrase's start, then the end of each of its phones."""
    return [t for ph in phrases for t in (ph[0].start, *(s.end for s in ph))]
