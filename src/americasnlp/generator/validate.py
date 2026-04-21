"""Acceptance check for a generated yaduha-{iso} package.

The agent calls this after every edit. It reports:
  - whether the package imports cleanly
  - whether `LanguageLoader.load_language(iso)` returns a Language with at
    least one Sentence type
  - whether each Sentence type's get_examples() renders to a non-empty
    target string with no `[english]` placeholder leakage
  - vocab coverage of training-data tokens
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class SentenceTypeReport:
    name: str
    n_examples: int
    placeholder_leaks: int     # examples whose render contains `[english]`
    empty_renders: int
    sample_renders: List[str] = field(default_factory=list)


@dataclass
class PackageValidation:
    iso: str
    importable: bool
    error: Optional[str] = None
    language_name: Optional[str] = None
    sentence_types: List[SentenceTypeReport] = field(default_factory=list)
    vocab_counts: dict = field(default_factory=dict)  # {category: count}
    vocab_coverage: Optional[float] = None            # 0..1 over training tokens

    @property
    def passed(self) -> bool:
        if not self.importable:
            return False
        if not self.sentence_types:
            return False
        for rep in self.sentence_types:
            if rep.empty_renders > 0:
                return False
        return True

    def summary(self) -> str:
        lines = [f"package: yaduha-{self.iso}",
                 f"importable: {self.importable}"]
        if self.error:
            lines.append(f"error: {self.error}")
        if self.language_name:
            lines.append(f"language.name: {self.language_name!r}")
        for st in self.sentence_types:
            lines.append(
                f"  {st.name}: {st.n_examples} examples, "
                f"{st.placeholder_leaks} placeholder leak(s), "
                f"{st.empty_renders} empty render(s)")
            for s in st.sample_renders:
                lines.append(f"    e.g. {s!r}")
        if self.vocab_counts:
            lines.append("vocab: " + ", ".join(
                f"{k}={v}" for k, v in self.vocab_counts.items()))
        if self.vocab_coverage is not None:
            lines.append(f"training-token coverage: {self.vocab_coverage:.1%}")
        lines.append(f"PASS: {self.passed}")
        return "\n".join(lines)


def _reload_package(module_name: str):
    # Drop cached imports so successive validate() calls see the latest edits.
    for key in list(sys.modules):
        if key == module_name or key.startswith(module_name + "."):
            del sys.modules[key]
    return importlib.import_module(module_name)


def validate_package(
    iso: str,
    *,
    training_tokens: Optional[List[str]] = None,
) -> PackageValidation:
    module_name = f"yaduha_{iso}"
    result = PackageValidation(iso=iso, importable=False)

    try:
        mod = _reload_package(module_name)
    except Exception as exc:  # noqa: BLE001
        result.error = f"import {module_name!r} failed: {exc}"
        return result
    result.importable = True

    lang = getattr(mod, "language", None)
    if lang is None:
        result.error = f"{module_name}.language is None"
        return result
    result.language_name = getattr(lang, "name", None)

    for SentType in getattr(lang, "sentence_types", ()):
        examples = SentType.get_examples()
        renders = [str(ex) for _, ex in examples]
        leaks = sum(1 for r in renders if "[" in r and "]" in r)
        empties = sum(1 for r in renders if not r.strip())
        result.sentence_types.append(SentenceTypeReport(
            name=SentType.__name__,
            n_examples=len(examples),
            placeholder_leaks=leaks,
            empty_renders=empties,
            sample_renders=renders[:3],
        ))

    vocab_mod_name = f"{module_name}.vocab"
    try:
        vocab = importlib.import_module(vocab_mod_name)
        for cat in ("NOUNS", "TRANSITIVE_VERBS", "INTRANSITIVE_VERBS"):
            entries = getattr(vocab, cat, None)
            if entries is not None:
                result.vocab_counts[cat] = len(entries)
        if training_tokens:
            target_tokens = set()
            for cat in ("NOUNS", "TRANSITIVE_VERBS", "INTRANSITIVE_VERBS"):
                for ent in getattr(vocab, cat, []):
                    target_tokens.update(ent.target.lower().split())
            hit = sum(1 for t in training_tokens if t.lower() in target_tokens)
            result.vocab_coverage = hit / len(training_tokens)
    except Exception as exc:  # noqa: BLE001
        result.error = (result.error or "") + f"; vocab inspection failed: {exc}"

    return result
