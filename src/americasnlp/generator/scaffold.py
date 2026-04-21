"""Write the empty `yaduha-{iso}/` skeleton the agent then fills in.

Everything here is *deterministic*: same input arguments → same files. The
agent only modifies `__init__.py` and `vocab.py` after this runs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


PYPROJECT_TEMPLATE = '''\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "yaduha-{iso}"
version = "0.1.0"
description = "{display_name} ({iso}) language package for the Yaduha framework"
requires-python = ">=3.10"
dependencies = ["yaduha>=0.3"]

[project.entry-points."yaduha.languages"]
{iso} = "yaduha_{iso}:language"
'''


VOCAB_PLACEHOLDER = '''\
"""{display_name} vocabulary.

GENERATED STARTER FILE — replace with citation-grounded entries before merge.
Entries should have an `english` lemma key and a `target` form attested by a
published reference. See the package docstring for source list.
"""
from yaduha.language import VocabEntry

NOUNS: list[VocabEntry] = []

TRANSITIVE_VERBS: list[VocabEntry] = []

INTRANSITIVE_VERBS: list[VocabEntry] = []
'''


INIT_PLACEHOLDER = '''\
"""{display_name} ({iso}) — Yaduha language package.

GENERATED STARTER FILE. Implement the full grammar here following the
`yaduha-hch` pattern: enums (Number, Person, TenseAspect), Pydantic models
(Noun, Verb, TransitiveVerb, IntransitiveVerb), and at minimum two sentence
types (SubjectVerbSentence, SubjectVerbObjectSentence).

Word order: {word_order_hint}
"""
from yaduha.language import Language, Sentence  # noqa: F401

# TODO: import vocabulary, define enums, define sentence types, then construct
# the module-level `language` object below.
language = None  # type: ignore[assignment]
'''


def scaffold_package(
    iso: str,
    display_name: str,
    *,
    repo_root: Path,
    word_order_hint: str = "verify against published sources",
    overwrite: bool = False,
) -> Path:
    """Write `yaduha-{iso}/` skeleton files. Returns the package root path.

    Skips existing files unless `overwrite=True`. Does not modify the parent
    pyproject.toml or pyrightconfig.json — those are managed by the user.
    """
    pkg_root = repo_root / f"yaduha-{iso}"
    code_dir = pkg_root / f"yaduha_{iso}"
    code_dir.mkdir(parents=True, exist_ok=True)

    files = {
        pkg_root / "pyproject.toml":
            PYPROJECT_TEMPLATE.format(iso=iso, display_name=display_name),
        code_dir / "__init__.py":
            INIT_PLACEHOLDER.format(iso=iso,
                                    display_name=display_name,
                                    word_order_hint=word_order_hint),
        code_dir / "vocab.py":
            VOCAB_PLACEHOLDER.format(display_name=display_name),
    }
    for path, content in files.items():
        if path.exists() and not overwrite:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return pkg_root
