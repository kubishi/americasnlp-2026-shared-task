"""Evaluate Wixárika translation on AmericasNLP 2026 pilot data.

Translates Spanish captions to Wixárika using two approaches:
  1. Pipeline: Structured outputs via yaduha-hch sentence types
  2. Agentic: Free-form translation with vocabulary/grammar prompts

Computes ChrF++ scores against reference translations and saves results to CSV.

Usage:
    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --model gpt-4o --output results/eval_gpt4o.csv
"""

from abc import ABC, abstractmethod
import argparse
import base64
import csv
from http import client
import json
import os
from pydoc import text
import sys
import time
from pathlib import Path
from typing import ClassVar, Generic, List, Type, Union, cast
import openai
from base64 import b64encode

from dotenv import load_dotenv
from pydantic import BaseModel, create_model
from sacrebleu.metrics.chrf import CHRF

from yaduha.language import Sentence
from yaduha.tool import Tool
from yaduha.tool.english_to_sentences import TSentenceType, SentenceList
from yaduha.agent import Agent, AgentResponse
from yaduha.translator import Translator

load_dotenv()

from yaduha.agent.openai import OpenAIAgent
from yaduha.translator.pipeline import PipelineTranslator
from yaduha.translator.agentic import AgenticTranslator
from yaduha_hch import SubjectVerbSentence, SubjectVerbObjectSentence
from yaduha_hch.prompts import get_prompt



def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def load_data(data_path: Path) -> list[dict]:
    """Load JSONL data from the pilot set."""
    records = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_chrf(hypothesis: str, reference: str) -> float:
    """Compute ChrF++ score for a single hypothesis/reference pair."""
    chrf = CHRF(word_order=2)
    score = chrf.corpus_score([hypothesis], [[reference]])
    return score.score

class Captioner(Tool[str], ABC):
    """Base class for translators that translate text to a target language and back to the source language."""
    name: ClassVar[str] = "translator"
    description: ClassVar[str] = "Translate text to the target language and back to the source language."

    def _run(self, image_path: str) -> str:
        """Caption the image at the given path.

        Args:
            image_path (str): The path to the image file.
        Returns:
            str: The caption in the target language.
        """
        return self.caption(Path(image_path))   

    @abstractmethod
    def caption(self, image_path: Path) -> str:
        pass


class StructuredCaptionTool(Tool[AgentResponse[SentenceList[TSentenceType]]]):
    agent: "Agent"
    name: ClassVar[str] = "caption_image"
    description: ClassVar[str] = "Caption images using a Sentence Model"
    SentenceType: Type[TSentenceType] | tuple[Type[Sentence], ...]

    def _run(self, image_path: str) -> AgentResponse[SentenceList[TSentenceType]]:
        base64_image = encode_image(image_path)
    
        if isinstance(self.SentenceType, tuple):
            # Create a discriminated union type for multiple sentence types
            if len(self.SentenceType) == 1:
                sentence_union = self.SentenceType[0]
            else:
                sentence_union = Union[tuple(self.SentenceType)]

            TargetSentenceList = create_model(
                "TargetSentenceList",
                sentences=(List[sentence_union], ...),
                __base__=BaseModel
            )
        else:
            # Single sentence type (backward compatible)
            TargetSentenceList = create_model(
                "TargetSentenceList",
                sentences=(List[self.SentenceType], ...),
                __base__=SentenceList[self.SentenceType]
            )

        response = self.agent.get_response(
            messages=[
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": "what's in this image?" },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "auto"
                            }
                        },
                    ],
                }
            ],
            response_format=TargetSentenceList
        )

        return cast(AgentResponse[SentenceList[TSentenceType]], response)

class EnglishCaptionTool(Tool[str]):
    agent: Agent
    name: ClassVar[str] = "english_caption"
    description: ClassVar[str] = "Caption images in English using a free-form response."

    def _run(self, image_path: str) -> str:
        base64_image = encode_image(image_path)
        response = self.agent.get_response(
            messages=[
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": "Describe the contents of this image in English." },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "auto"
                            }
                        },
                    ],
                }
            ]
        )

        return response.content.strip()

class StructuredCaptioner(Captioner, Generic[TSentenceType]):
    agent: Agent
    SentenceType: Type[TSentenceType] | tuple[Type[Sentence], ...]

    def caption(self, image_path: Path) -> str:
        tool = StructuredCaptionTool[TSentenceType](agent=self.agent, SentenceType=self.SentenceType)
        response = tool(image_path)
        return " ".join(str(sentence) for sentence in response.content.sentences)

class TranslationCaptioner(Captioner):
    agent: Agent
    translator: Translator

    def caption(self, image_path: Path) -> str:
        tool = EnglishCaptionTool(agent=self.agent)
        response = tool(image_path)
        translation = self.translator.translate(response)
        return translation.target.strip()

def main():
    parser = argparse.ArgumentParser(description="Evaluate Wixárika translation")
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        choices=["gpt-4o", "gpt-4o-mini"],
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--output", default="results/evaluation_results.csv",
        help="Output CSV path (default: results/evaluation_results.csv)"
    )
    parser.add_argument(
        "--data", default="americasnlp2026/data/pilot/wixarika.jsonl",
        help="Path to JSONL data file"
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    records = load_data(data_path)
    print(f"Loaded {len(records)} examples from {data_path}")

    # Initialize agent
    agent = OpenAIAgent(model=args.model, api_key=api_key, temperature=0.0)

    # Initialize translators
    pipeline = PipelineTranslator.from_language(
        "hch", agent=agent, back_translation_agent=None
    )

    system_prompt = get_prompt(
        include_vocab=True,
        include_examples=[SubjectVerbSentence, SubjectVerbObjectSentence]
    )
    agentic = AgenticTranslator(
        agent=agent,
        system_prompt=system_prompt,
        tools=None
    )

    captioners = {
        "structured": StructuredCaptioner(agent=agent, SentenceType=(SubjectVerbSentence, SubjectVerbObjectSentence)),
        "translator-pipeline": TranslationCaptioner(agent=agent, translator=pipeline),
        "translator-agentic": TranslationCaptioner(agent=agent, translator=agentic),
    }

    # Load existing results to skip completed entries
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["example_id", "method", "translation", "reference", "chrf_score"]

    existing_results = []
    completed = set()
    if output_path.exists():
        with open(output_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["example_id"] = int(row["example_id"])
                row["chrf_score"] = float(row["chrf_score"])
                existing_results.append(row)
                completed.add((row["example_id"], row["method"]))
        print(f"Loaded {len(existing_results)} existing results from {output_path}")

    # Determine which tasks still need to run
    all_tasks = []
    for i, record in enumerate(records):
        for method, captioner in captioners.items():
            if (i, method) not in completed:
                all_tasks.append((i, record, method, captioner))

    if not all_tasks:
        print("All tasks already completed, nothing to do.")
        results = existing_results
    else:
        print(f"Skipping {len(completed)} completed, running {len(all_tasks)} remaining tasks...")

        # Run translations in parallel
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        results = list(existing_results)
        write_lock = threading.Lock()

        # If CSV doesn't exist yet, write the header
        if not existing_results:
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

        def process_one(i: int, record: dict, method: str, captioner: Captioner) -> dict:
            image_path = Path("americasnlp2026/data/pilot") / record["filename"]
            reference = record["target_caption"]
            try:
                translation = captioner(image_path)
                chrf_score = compute_chrf(translation, reference)
                print(f"[{i+1}/{len(records)}] {method}: ChrF++={chrf_score:.2f}")
                return {
                    "example_id": i,
                    "method": method,
                    "translation": translation,
                    "reference": reference,
                    "chrf_score": chrf_score,
                }
            except Exception as e:
                print(f"[{i+1}/{len(records)}] {method}: ERROR - {e}", file=sys.stderr)
                return {
                    "example_id": i,
                    "method": method,
                    "translation": f"[ERROR: {e}]",
                    "reference": reference,
                    "chrf_score": 0.0,
                }

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {}
            for i, record, method, captioner in all_tasks:
                f = executor.submit(process_one, i, record, method, captioner)
                futures[f] = (i, method, record)

            for future in as_completed(futures, timeout=300):
                i, method, record = futures[future]
                try:
                    result = future.result(timeout=60)
                except Exception as e:
                    print(f"[{i+1}/{len(records)}] {method}: TIMEOUT/ERROR - {e}", file=sys.stderr)
                    result = {
                        "example_id": i,
                        "method": method,
                        "translation": f"[TIMEOUT: {e}]",
                        "reference": record["target_caption"],
                        "chrf_score": 0.0,
                    }
                with write_lock:
                    results.append(result)
                    # Append to CSV immediately
                    with open(output_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writerow(result)

    # Re-sort and rewrite the final CSV so rows are in order
    results.sort(key=lambda r: (r["example_id"], r["method"]))
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_path}")

    # Print summary per method
    print("\n" + "=" * 60)
    print(f"SUMMARY ({args.model}, {len(records)} examples)")
    print("=" * 60)
    for method in captioners:
        scores = [r["chrf_score"] for r in results if r["method"] == method]
        if scores:
            print(f"  {method:<25} mean={sum(scores)/len(scores):.2f}  "
                  f"min={min(scores):.2f}  max={max(scores):.2f}")


if __name__ == "__main__":
    main()
