"""
src/models/classifier.py

Day 4: Zero-shot failure mode classification using DeBERTa NLI.

Given a sensor anomaly (represented as a short text description or as
a retrieved maintenance log snippet), classify it into one of the known
OSHA turbofan failure mode categories — without any labelled training data.

Why zero-shot:
    CMAPSS doesn't provide failure mode labels per window, only RUL.
    Training a supervised classifier would require manual annotation.
    DeBERTa NLI reformulates classification as textual entailment:
    "This engine is experiencing [label]" — does the evidence entail this?

Pipeline position:
    Runs in parallel with FAISS retrieval (Day 3).
    Both outputs feed the LLM reasoning agent (Day 5).

HuggingFace task: Zero-Shot Classification
Model: cross-encoder/nli-deberta-v3-base
"""

from __future__ import annotations
from dataclasses import dataclass
from transformers import pipeline
import torch


# OSHA-grounded turbofan failure mode taxonomy
# Matches the labels used during synthetic log generation
FAILURE_MODES = [
    "bearing wear",
    "blade tip erosion",
    "seal degradation",
    "compressor stall",
    "turbine blade damage",
    "lubrication failure",
    "vibration-induced fatigue",
    "thermal degradation",
    "foreign object damage",
    "oil contamination",
]

# Hypothesis template — DeBERTa NLI checks:
# "This text entails that the engine is experiencing {label}"
HYPOTHESIS_TEMPLATE = "This engine fault is caused by {}."


@dataclass
class ClassificationResult:
    top_label: str
    top_score: float
    all_scores: dict[str, float]   # full distribution over all failure modes

    def is_confident(self, threshold: float = 0.30) -> bool:
        """
        True if top score exceeds threshold.
        Below 0.30 typically means the anomaly pattern is ambiguous
        or doesn't match any known failure mode well.
        """
        return self.top_score >= threshold

    def top_k(self, k: int = 3) -> list[tuple[str, float]]:
        """Return top-k (label, score) pairs sorted by score."""
        return sorted(self.all_scores.items(),
                      key=lambda x: x[1], reverse=True)[:k]


class FailureModeClassifier:
    """
    Zero-shot failure mode classifier.

    Wraps HuggingFace zero-shot-classification pipeline with
    DeBERTa-v3-base trained on NLI tasks.

    Usage:
        clf = FailureModeClassifier()
        result = clf.classify_from_text("High vibration on HPT bearing, oil temp rising")
        result = clf.classify_from_anomaly_score(score=0.87, top_sensor="sensor_2")
        result = clf.classify_from_retrieved_logs(retrieved_logs)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        device: int = -1,   # -1 = CPU, 0 = first GPU
    ):
        print(f"Loading zero-shot classifier: {model_name}")
        self.pipe = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device if torch.cuda.is_available() else -1,
        )
        self.labels = FAILURE_MODES

    def classify_from_text(self, text: str) -> ClassificationResult:
        """
        Classify a free-text description of an anomaly.

        Args:
            text: any text describing the sensor anomaly or maintenance event
                  e.g. "Unusual vibration pattern in bearing region, cycle 287"

        Returns:
            ClassificationResult with scores over all failure modes
        """
        result = self.pipe(
            text,
            candidate_labels=self.labels,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=False,
        )
        scores = dict(zip(result["labels"], result["scores"]))
        return ClassificationResult(
            top_label=result["labels"][0],
            top_score=result["scores"][0],
            all_scores=scores,
        )

    def classify_from_anomaly_score(
        self,
        anomaly_score: float,
        top_sensors: list[str],
        unit_id: int,
        cycle: int,
    ) -> ClassificationResult:
        """
        Construct a synthetic text description from sensor metadata
        and classify it. Used when no retrieved log is available.

        Args:
            anomaly_score: reconstruction error from autoencoder
            top_sensors: list of sensor names with highest reconstruction error
                         (these are the most anomalous channels)
            unit_id: engine unit
            cycle: current operating cycle
        """
        # Map sensor names to physical interpretations
        SENSOR_SEMANTICS = {
            "sensor_2":  "fan inlet temperature anomaly",
            "sensor_3":  "LPC outlet temperature anomaly",
            "sensor_4":  "HPC outlet temperature anomaly",
            "sensor_7":  "HPC outlet pressure anomaly",
            "sensor_8":  "physical fan speed anomaly",
            "sensor_9":  "core speed anomaly",
            "sensor_11": "HPC outlet static pressure anomaly",
            "sensor_12": "fuel flow ratio anomaly",
            "sensor_13": "corrected fan speed anomaly",
            "sensor_14": "corrected core speed anomaly",
            "sensor_15": "bypass ratio anomaly",
            "sensor_17": "bleed enthalpy anomaly",
            "sensor_20": "HPT coolant bleed anomaly",
            "sensor_21": "LPT coolant bleed anomaly",
        }

        sensor_descriptions = [
            SENSOR_SEMANTICS.get(s, f"{s} anomaly") for s in top_sensors[:3]
        ]
        desc = ", ".join(sensor_descriptions)

        text = (
            f"Engine unit {unit_id} at cycle {cycle} shows elevated anomaly score "
            f"({anomaly_score:.3f}). Most degraded signals: {desc}."
        )
        return self.classify_from_text(text)

    def classify_from_retrieved_logs(
        self,
        retrieved_logs: list[dict],
        max_logs: int = 3,
    ) -> ClassificationResult:
        """
        Classify by concatenating retrieved maintenance log texts.
        Using retrieved evidence produces more grounded classification
        than sensor-description-only prompts.

        Args:
            retrieved_logs: list of dicts from CrossModalRetriever.retrieve()
            max_logs: use top-N retrieved logs (more = richer context, slower)
        """
        if not retrieved_logs:
            raise ValueError("No retrieved logs provided")

        context = " | ".join(
            log["text"] for log in retrieved_logs[:max_logs]
        )
        return self.classify_from_text(context)

    def batch_classify(
        self,
        texts: list[str],
        batch_size: int = 16,
    ) -> list[ClassificationResult]:
        """
        Classify a list of texts efficiently in batches.
        Used during evaluation over all test windows.
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for text in batch:
                results.append(self.classify_from_text(text))
        return results


if __name__ == "__main__":
    # Smoke test
    clf = FailureModeClassifier()

    test_cases = [
        "High oil temperature in bearing assembly, stage 2. Unusual vibration pattern detected.",
        "Compressor inlet pressure drop at high altitude. Possible ice ingestion.",
        "HPT blade cooling channel blockage suspected. Exhaust temperature above limits.",
    ]

    for text in test_cases:
        result = clf.classify_from_text(text)
        print(f"\nText: {text[:60]}...")
        print(f"  Top label:  {result.top_label} ({result.top_score:.3f})")
        print(f"  Confident:  {result.is_confident()}")
        print(f"  Top 3:      {result.top_k(3)}")
