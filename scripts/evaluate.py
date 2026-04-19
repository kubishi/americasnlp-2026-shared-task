"""Evaluate AmericasNLP 2026 translation across all target languages.

Captions images in the target language using three approaches:
  1. structured:          Vision model → structured sentence → translate_structured_sentence
  2. translator-pipeline: Vision model → English caption → structured sentence → translate_structured_sentence
  3. translator-agentic:  Vision model → English caption → agentic translation with vocab/grammar prompt

structured and translator-pipeline require a yaduha language package to be installed for the
selected language. translator-agentic falls back to a generic prompt if no package is available.

Computes ChrF++ scores against reference translations and saves results to CSV.

Usage:
    uv run python scripts/evaluate.py --language wixarika
    uv run python scripts/evaluate.py --language bribri --split dev
    uv run python scripts/evaluate.py --language maya --model gpt-4o --output results/maya.csv
"""

from abc import ABC, abstractmethod
import argparse
import base64
import csv
import importlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, ClassVar, Dict, Generic, List, Optional, Tuple, Type, Union, cast

from dotenv import load_dotenv
from pydantic import BaseModel, create_model
from sacrebleu.metrics.chrf import CHRF

from yaduha.language import Sentence
from yaduha.tool import Tool
from yaduha.tool.english_to_sentences import EnglishToSentencesTool, TSentenceType, SentenceList
from yaduha.agent import Agent, AgentResponse
from yaduha.translator.agentic import AgenticTranslator

load_dotenv()

from yaduha.agent.openai import OpenAIAgent

import openai.resources.chat  # warm up import cache before threading


# ---------------------------------------------------------------------------
# Language configuration
# ---------------------------------------------------------------------------

LANGUAGE_CONFIG: Dict[str, Dict[str, str]] = {
    "wixarika": {
        "yaduha_code": "hch",
        "display_name": "Wixárika",
        "pilot_file": "americasnlp2026/data/pilot/wixarika.jsonl",
        "dev_file":   "americasnlp2026/data/dev/wixarika/wixarika.jsonl",
    },
    "bribri": {
        "yaduha_code": "bzd",
        "display_name": "Bribri",
        "pilot_file": "",
        "dev_file":   "americasnlp2026/data/dev/bribri/bribri.jsonl",
    },
    "guarani": {
        "yaduha_code": "grn",
        "display_name": "Guaraní",
        "pilot_file": "",
        "dev_file":   "americasnlp2026/data/dev/guarani/guarani.jsonl",
    },
    "maya": {
        "yaduha_code": "yua",
        "display_name": "Yucatec Maya",
        "pilot_file": "",
        "dev_file":   "americasnlp2026/data/dev/maya/maya.jsonl",
    },
    "nahuatl": {
        "yaduha_code": "nlv",
        "display_name": "Orizaba Nahuatl",
        "pilot_file": "",
        "dev_file":   "americasnlp2026/data/dev/nahuatl/nahuatl.jsonl",
    },
}


# ---------------------------------------------------------------------------
# translate_structured_sentence
# ---------------------------------------------------------------------------

class SentenceTranslationResult(BaseModel):
    source: Dict[str, Any]
    language_code: str
    sentence_type: str
    target: str


def translate_structured_sentence(
    sentence: Dict[str, Any],
    language_code: str,
) -> SentenceTranslationResult:
    """Translate a structured sentence dict to the target language.

    Loads the language package via LanguageLoader, tries each registered sentence
    type in order, and calls str(sentence) to render the target-language string —
    the same rendering step used internally by PipelineTranslator.

    Args:
        sentence:       Dict matching one of the language's sentence-type schemas
                        (the JSON output of EnglishToSentencesTool or StructuredCaptionTool).
        language_code:  Yaduha language code (e.g. 'hch', 'bzd').

    Returns:
        SentenceTranslationResult with the rendered target string.

    Raises:
        ValueError: If no registered sentence type can parse the dict.
    """
    from yaduha.loader import LanguageLoader

    language = LanguageLoader.load_language(language_code)
    errors: List[str] = []

    for SentenceType in language.sentence_types:
        try:
            parsed = SentenceType.model_validate(sentence)
            return SentenceTranslationResult(
                source=sentence,
                language_code=language_code,
                sentence_type=SentenceType.__name__,
                target=str(parsed),
            )
        except Exception as exc:
            errors.append(f"{SentenceType.__name__}: {exc}")

    raise ValueError(
        f"Could not parse sentence into any type for language '{language_code}'.\n"
        + "\n".join(errors)
    )


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def resolve_image_path(data_path: Path, record: Dict[str, Any]) -> Path:
    """Resolve image path relative to the data file directory, with fallback."""
    data_dir = data_path.parent
    candidate = data_dir / record["filename"]
    if candidate.exists():
        return candidate
    # Fallback: look for the bare filename inside data_dir/images/
    image_name = Path(record["filename"]).name
    fallback = data_dir / "images" / image_name
    return fallback if fallback.exists() else candidate

# def image_data_url(path: Path) -> str:
#     """Return a base64 data URL for the image, normalised to JPEG via Pillow.

#     Using the file extension to infer MIME type is unreliable — some dataset
#     images are WebP or unknown formats stored with a .jpg extension. Pillow
#     opens any supported format and re-encodes to JPEG, so OpenAI always
#     receives a valid image/jpeg payload regardless of the original file.
#     """
#     from PIL import Image
#     import io as _io

#     with Image.open(path) as img:
#         if img.mode not in ("RGB", "L"):
#             img = img.convert("RGB")
#         buf = _io.BytesIO()
#         img.save(buf, format="JPEG")
#         encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
#     return f"data:image/jpeg;base64,{encoded}"


def load_data(data_path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_chrf(hypothesis: str, reference: str) -> float:
    chrf = CHRF(word_order=2)
    return chrf.corpus_score([hypothesis], [[reference]]).score


# ---------------------------------------------------------------------------
# Shared image-captioning tools
# ---------------------------------------------------------------------------

class StructuredCaptionTool(Tool[AgentResponse[SentenceList[TSentenceType]]]):
    """Ask the VLM to describe an image as a structured sentence JSON."""
    agent: Agent
    name: ClassVar[str] = "caption_image"
    description: ClassVar[str] = "Caption images using a Sentence Model."
    SentenceType: Type[TSentenceType] | tuple[Type[Sentence], ...]

    def _run(self, image_path: str) -> AgentResponse[SentenceList[TSentenceType]]:
        base64_image = encode_image(Path(image_path))

        if isinstance(self.SentenceType, tuple):
            sentence_union = (
                self.SentenceType[0] if len(self.SentenceType) == 1
                else Union[tuple(self.SentenceType)]
            )
            TargetSentenceList = create_model(
                "TargetSentenceList",
                sentences=(List[sentence_union], ...),
                __base__=BaseModel,
            )
        else:
            TargetSentenceList = create_model(
                "TargetSentenceList",
                sentences=(List[self.SentenceType], ...),
                __base__=SentenceList[self.SentenceType],
            )

        response = self.agent.get_response(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "what's in this image?"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "auto",
                    }},
                ],
            }],
            response_format=TargetSentenceList,
        )
        return cast(AgentResponse[SentenceList[TSentenceType]], response)


class EnglishCaptionTool(Tool[str]):
    """Ask the VLM to describe an image in free-form English."""
    agent: Agent
    name: ClassVar[str] = "english_caption"
    description: ClassVar[str] = "Caption images in English."

    def _run(self, image_path: str) -> str:
        base64_image = encode_image(Path(image_path))
        response = self.agent.get_response(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the contents of this image in English."},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "auto",
                    }},
                ],
            }]
        )
        return response.content.strip()


# ---------------------------------------------------------------------------
# Captioner implementations
# ---------------------------------------------------------------------------

class Captioner(Tool[str], ABC):
    """Base class for image captioners that produce target-language strings."""
    name: ClassVar[str] = "captioner"
    description: ClassVar[str] = "Caption an image in the target language."

    def _run(self, image_path: str) -> str:
        return self.caption(Path(image_path))

    @abstractmethod
    def caption(self, image_path: Path) -> str:
        pass


class StructuredCaptioner(Captioner, Generic[TSentenceType]):
    """Vision model → structured sentence → translate_structured_sentence → target string.

    The VLM directly produces a structured sentence JSON which is then rendered
    into the target language via translate_structured_sentence (no intermediate
    English caption step).
    """
    agent: Agent
    language_code: str
    SentenceType: Type[TSentenceType] | tuple[Type[Sentence], ...]

    def caption(self, image_path: Path) -> str:
        tool = StructuredCaptionTool[TSentenceType](
            agent=self.agent, SentenceType=self.SentenceType
        )
        response = tool(str(image_path))
        parts = []
        for sentence in response.content.sentences:
            result = translate_structured_sentence(sentence.model_dump(), self.language_code)
            parts.append(result.target)
        return " ".join(parts)


class PipelineCaptioner(Captioner, Generic[TSentenceType]):
    """Vision model → English caption → EnglishToSentencesTool → translate_structured_sentence → target string.

    Replaces PipelineTranslator: English parsing is still done via EnglishToSentencesTool,
    but rendering uses translate_structured_sentence (LanguageLoader-driven, language-agnostic).
    """
    agent: Agent
    language_code: str
    SentenceType: Type[TSentenceType] | tuple[Type[Sentence], ...]

    def caption(self, image_path: Path) -> str:
        english = EnglishCaptionTool(agent=self.agent)(str(image_path))
        parser = EnglishToSentencesTool(agent=self.agent, SentenceType=self.SentenceType)
        sentence_list = parser(english).content
        parts = []
        for sentence in sentence_list.sentences:
            result = translate_structured_sentence(sentence.model_dump(), self.language_code)
            parts.append(result.target)
        return " ".join(parts)


class AgenticCaptioner(Captioner):
    """Vision model → English caption → agentic translation with vocab/grammar prompt.

    Does not require a yaduha language package; falls back to a generic prompt
    if no language-specific prompts module is available.
    """
    agent: Agent
    system_prompt: str

    def caption(self, image_path: Path) -> str:
        english = EnglishCaptionTool(agent=self.agent)(str(image_path))
        translator = AgenticTranslator(
            agent=self.agent, system_prompt=self.system_prompt, tools=None
        )
        return translator.translate(english).target.strip()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate AmericasNLP 2026 image captioning → target language translation"
    )
    parser.add_argument(
        "--language", default="wixarika", choices=list(LANGUAGE_CONFIG.keys()),
        help="Target language to evaluate (default: wixarika)",
    )
    parser.add_argument(
        "--split", default="pilot", choices=["pilot", "dev"],
        help="Data split to evaluate on (default: pilot)",
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini", choices=["gpt-4o", "gpt-4o-mini"],
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path (default: results/{language}_{split}_{model}.csv)",
    )
    args = parser.parse_args()

    lang_cfg = LANGUAGE_CONFIG[args.language]
    yaduha_code: str = lang_cfg["yaduha_code"]
    display_name: str = lang_cfg["display_name"]

    # Resolve data path from --language + --split
    data_file = lang_cfg.get(f"{args.split}_file", "")
    if not data_file:
        print(f"Error: No {args.split} data available for {display_name}.", file=sys.stderr)
        sys.exit(1)
    data_path = Path(data_file)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    output_path = (
        Path(args.output) if args.output
        else Path(f"results/{args.language}_{args.split}_{args.model}.csv")
    )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    records = load_data(data_path)
    print(f"Loaded {len(records)} examples from {data_path} ({display_name})")

    agent = OpenAIAgent(model=args.model, api_key=api_key, temperature=0.0)

    # Try to load the yaduha language package (required for structured + pipeline methods)
    sentence_types: Optional[Tuple[Type[Sentence], ...]] = None
    try:
        from yaduha.loader import LanguageLoader
        language = LanguageLoader.load_language(yaduha_code)
        sentence_types = language.sentence_types
        print(f"Loaded yaduha package '{yaduha_code}' ({len(sentence_types)} sentence types)")
    except Exception as e:
        print(
            f"Warning: No yaduha package for '{yaduha_code}' ({e}). "
            "structured and translator-pipeline methods will be skipped."
        )

    # Build agentic system prompt — language-specific if a prompts module exists, else generic
    try:
        prompts_mod = importlib.import_module(f"yaduha_{yaduha_code}.prompts")
        agentic_prompt: str = prompts_mod.get_prompt(
            include_vocab=True,
            include_examples=list(sentence_types) if sentence_types else [],
        )
    except Exception:
        agentic_prompt = (
            f"You are a translator. Translate the following English sentence into "
            f"{display_name} as accurately as possible. Respond with only the translation."
        )

    # Assemble captioners — structured and pipeline require an installed yaduha package
    captioners: Dict[str, Captioner] = {}
    if sentence_types is not None:
        captioners["structured"] = StructuredCaptioner(
            agent=agent, language_code=yaduha_code, SentenceType=sentence_types,
        )
        captioners["translator-pipeline"] = PipelineCaptioner(
            agent=agent, language_code=yaduha_code, SentenceType=sentence_types,
        )
    captioners["translator-agentic"] = AgenticCaptioner(
        agent=agent, system_prompt=agentic_prompt,
    )

    # Load existing results to resume interrupted runs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["example_id", "method", "translation", "reference", "chrf_score"]
    existing_results: List[Dict[str, Any]] = []
    completed: set = set()
    if output_path.exists():
        with open(output_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and set(fieldnames).issubset(set(reader.fieldnames)):
                for row in reader:
                    row["example_id"] = int(row["example_id"])
                    row["chrf_score"] = float(row["chrf_score"])
                    existing_results.append(row)
                    completed.add((row["example_id"], row["method"]))
                print(f"Loaded {len(existing_results)} existing results from {output_path}")
            else:
                print(f"Warning: {output_path} has unexpected headers — ignoring and starting fresh.")

    all_tasks = [
        (i, record, method, captioner)
        for i, record in enumerate(records)
        for method, captioner in captioners.items()
        if (i, method) not in completed
    ]

    if not all_tasks:
        print("All tasks already completed, nothing to do.")
        results = existing_results
    else:
        print(f"Skipping {len(completed)} completed, running {len(all_tasks)} remaining tasks...")

        results: List[Dict[str, Any]] = list(existing_results)
        write_lock = threading.Lock()

        if not existing_results:
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()

        def process_one(
            i: int, record: Dict[str, Any], method: str, captioner: Captioner
        ) -> Dict[str, Any]:
            image_path = resolve_image_path(data_path, record)
            reference = record["target_caption"]
            try:
                translation = captioner(str(image_path))
                chrf_score = compute_chrf(translation, reference)
                print(f"[{i+1}/{len(records)}] {method}: ChrF++={chrf_score:.2f}")
                return {
                    "example_id": i, "method": method,
                    "translation": translation, "reference": reference,
                    "chrf_score": chrf_score,
                }
            except Exception as e:
                print(f"[{i+1}/{len(records)}] {method}: ERROR - {e}", file=sys.stderr)
                return {
                    "example_id": i, "method": method,
                    "translation": f"[ERROR: {e}]", "reference": reference,
                    "chrf_score": 0.0,
                }

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(process_one, i, rec, method, cap): (i, method, rec)
                for i, rec, method, cap in all_tasks
            }
            for future in as_completed(futures, timeout=300):
                i, method, record = futures[future]
                try:
                    result = future.result(timeout=60)
                except Exception as e:
                    result = {
                        "example_id": i, "method": method,
                        "translation": f"[TIMEOUT: {e}]",
                        "reference": record["target_caption"],
                        "chrf_score": 0.0,
                    }
                with write_lock:
                    results.append(result)
                    with open(output_path, "a", newline="", encoding="utf-8") as f:
                        csv.DictWriter(f, fieldnames=fieldnames).writerow(result)

    # Re-sort and rewrite final CSV
    results.sort(key=lambda r: (r["example_id"], r["method"]))
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_path}")
    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {display_name} | {args.split} | {args.model} | {len(records)} examples")
    print("=" * 60)
    for method in captioners:
        scores = [r["chrf_score"] for r in results if r["method"] == method]
        if scores:
            print(
                f"  {method:<25} mean={sum(scores)/len(scores):.2f}  "
                f"min={min(scores):.2f}  max={max(scores):.2f}"
            )


if __name__ == "__main__":
    main()
