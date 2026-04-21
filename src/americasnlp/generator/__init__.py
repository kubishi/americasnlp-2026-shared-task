"""Language Package Generator.

Produces a yaduha-{iso}/ package given training data (image→target_caption pairs)
and open-web research about the language. The actual orchestration lives in
`agent.py`; the building blocks (extract, validate, scaffold) are reusable on
their own.
"""
from americasnlp.generator.extract import (
    TrainingExample,
    extract_content_words,
    load_training_examples,
)
from americasnlp.generator.scaffold import scaffold_package
from americasnlp.generator.validate import (
    PackageValidation,
    validate_package,
)

__all__ = [
    "TrainingExample",
    "extract_content_words",
    "load_training_examples",
    "scaffold_package",
    "PackageValidation",
    "validate_package",
]
