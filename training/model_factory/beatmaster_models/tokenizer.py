from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

TOKEN_PATTERN = re.compile(r"\[[^\]]+\]|\w+(?:['’-]\w+)*|[^\w\s]", re.UNICODE)
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<sep>"]


class BeatMasterTokenizer:
    """Small transparent tokenizer for captions and multilingual lyrics.

    It preserves Unicode words used in Sranan Tongo, Dutch, Sarnami,
    Romanized Hindi and Caribbean languages without requiring an external model.
    """

    def __init__(self, vocabulary: dict[str, int]) -> None:
        for index, token in enumerate(SPECIAL_TOKENS):
            if vocabulary.get(token) != index:
                raise ValueError(f"Vocabulary must map {token} to {index}")
        self.vocabulary = vocabulary
        self.inverse = {index: token for token, index in vocabulary.items()}

    @staticmethod
    def normalize(text: str) -> str:
        return unicodedata.normalize("NFKC", text).strip()

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        return TOKEN_PATTERN.findall(cls.normalize(text))

    @classmethod
    def train(cls, texts: Iterable[str], vocabulary_size: int = 16000, minimum_frequency: int = 2) -> "BeatMasterTokenizer":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(cls.tokenize(text))
        available = max(0, vocabulary_size - len(SPECIAL_TOKENS))
        ranked = [token for token, count in counter.most_common() if count >= minimum_frequency][:available]
        vocabulary = {token: index for index, token in enumerate(SPECIAL_TOKENS + ranked)}
        return cls(vocabulary)

    @property
    def pad_id(self) -> int:
        return self.vocabulary["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.vocabulary["<unk>"]

    @property
    def bos_id(self) -> int:
        return self.vocabulary["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.vocabulary["<eos>"]

    def encode(self, text: str, *, add_special_tokens: bool = True, maximum_length: int | None = None) -> list[int]:
        values = [self.vocabulary.get(token, self.unk_id) for token in self.tokenize(text)]
        if add_special_tokens:
            values = [self.bos_id, *values, self.eos_id]
        if maximum_length is not None:
            values = values[:maximum_length]
            if add_special_tokens and values and values[-1] != self.eos_id:
                values[-1] = self.eos_id
        return values

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        tokens: list[str] = []
        for token_id in token_ids:
            token = self.inverse.get(int(token_id), "<unk>")
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        text = " ".join(tokens)
        text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
        text = re.sub(r"([([])\s+", r"\1", text)
        return text.strip()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": 1, "vocabulary": self.vocabulary}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BeatMasterTokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls({str(token): int(index) for token, index in payload["vocabulary"].items()})
