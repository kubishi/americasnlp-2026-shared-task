"""Train/val split of the dev set for honest evaluation.

The dev split has 50 examples per language (70 for hch incl. pilot). We
deterministically partition by row id into TRAIN (visible to the
generator agent) and VAL (held out, used only by the evaluator).

Once the test set drops and we're producing a real submission, we'll
re-run the generator with `train_frac=1.0` so it sees ALL dev examples.
The val/train split is purely for our internal measurement.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Set

from americasnlp.data import load_split
from americasnlp.languages import LanguageConfig

DEFAULT_TRAIN_FRAC = 0.6   # 30/20 on dev
DEFAULT_SEED = 17


@dataclass(frozen=True)
class SplitIds:
    train: frozenset[str]
    val: frozenset[str]

    def __contains__(self, row_id: str) -> bool:
        return row_id in self.train or row_id in self.val


def split_dev(
    lang: LanguageConfig,
    data_root: Path,
    *,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    seed: int = DEFAULT_SEED,
    extra_train_splits: tuple[str, ...] = ("pilot",),
) -> SplitIds:
    """Deterministic partition of dev row ids into train + val.

    `extra_train_splits` (default: pilot) are folded entirely into train —
    the agent always sees pilot rows in addition to its dev-train slice
    (only Wixárika has a pilot, but the option is uniform).
    """
    dev_rows = load_split(lang, "dev", data_root)
    dev_ids = sorted(r["id"] for r in dev_rows)
    rng = random.Random(seed)
    shuffled = list(dev_ids)
    rng.shuffle(shuffled)
    n_train = int(round(len(shuffled) * train_frac))
    train: Set[str] = set(shuffled[:n_train])
    val: Set[str] = set(shuffled[n_train:])

    for split in extra_train_splits:
        try:
            extras = load_split(lang, split, data_root)
        except FileNotFoundError:
            continue
        for r in extras:
            train.add(r["id"])

    return SplitIds(train=frozenset(train), val=frozenset(val))
