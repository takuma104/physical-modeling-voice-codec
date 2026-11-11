"""The TSP speech database (P. Kabal, McGill, 2018): 23 adults reading Harvard sentences at
48 kHz in an anechoic room, with the phone alignment CHIMERA's repository commits
(Montreal Forced Aligner, ARPAbet with stress)."""

import gzip
from dataclasses import dataclass, field
from pathlib import Path

from .paths import CHIMERA, TSP_DIR


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    label: str


@dataclass
class Utterance:
    id: str
    speaker: str
    text: str
    phones: list[Interval] = field(default_factory=list)
    words: list[Interval] = field(default_factory=list)

    @property
    def sex(self) -> str:
        """"F" or "M" (the speaker codes start with it)."""
        return self.speaker[0]

    @property
    def body(self) -> str:
        """The CHIMERA body that reads this speaker's sentences."""
        return "female" if self.sex == "F" else "male"

    @property
    def wav(self) -> Path:
        return TSP_DIR / "48k" / self.speaker / f"{self.id}.wav"


def load_utterances() -> dict[str, Utterance]:
    """Every aligned adult recording, by id (`FA01_01`)."""
    tools = CHIMERA / "tools" / "tsp"
    utts: dict[str, Utterance] = {}
    for line in (tools / "sentences.tsv").read_text().splitlines():
        uid, speaker, text = line.split("\t")
        utts[uid] = Utterance(uid, speaker, text)
    with gzip.open(tools / "alignment.tsv.gz", "rt") as f:
        for line in f:
            uid, tier, start, end, label = line.rstrip("\n").split("\t")
            iv = Interval(float(start), float(end), label)
            (utts[uid].phones if tier == "phone" else utts[uid].words).append(iv)
    return utts
