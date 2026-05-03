#!/usr/bin/env python3
"""
metrics.py — MURU-BENCH Evaluation Metrics

Implements the six core metrics for evaluating model performance:
  1. Accuracy@exact  — answer within ground truth CI
  2. ECE             — Expected Calibration Error
  3. Category Breakdown — per-category accuracy
  4. Difficulty Scaling — accuracy by difficulty level
  5. Overconfidence Rate — confidence > accuracy frequency
  6. Reasoning Chain Quality — framework match score

Usage:
    from evaluation.metrics import MURUMetrics
    metrics = MURUMetrics(problems, predictions)
    results = metrics.compute_all()
"""

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Prediction:
    """A model's prediction for a single MURU-BENCH problem."""
    problem_id: str
    predicted_answer: float  # model's point estimate
    predicted_confidence: float  # model's stated confidence (0-1)
    predicted_interval: tuple[float, float] | None = None  # model's CI if given
    predicted_framework: str | None = None  # model's framework choice
    raw_response: str = ""  # full model response text


@dataclass
class ProblemResult:
    """Evaluation result for a single problem."""
    problem_id: str
    category: str
    difficulty: int
    correct: bool  # answer falls within ground truth CI
    model_confidence: float
    ground_truth_pe: float
    model_pe: float
    absolute_error: float
    framework_match: bool
    overconfident: bool  # confidence > accuracy indicator


class MURUMetrics:
    """Compute all MURU-BENCH evaluation metrics."""

    def __init__(self, problems: list[dict], predictions: list[Prediction]):
        self.problems = {p["id"]: p for p in problems}
        self.predictions = {p.problem_id: p for p in predictions}
        self.results: list[ProblemResult] = []
        self._evaluate()

    def _evaluate(self):
        """Evaluate each prediction against ground truth."""
        for pid, pred in self.predictions.items():
            if pid not in self.problems:
                continue

            problem = self.problems[pid]
            gt = problem["ground_truth"]
            ci = gt["confidence_interval"]

            # Accuracy@exact: is the prediction within the ground truth CI?
            correct = ci[0] <= pred.predicted_answer <= ci[1]

            # Framework match
            framework_match = (
                pred.predicted_framework == problem["required_framework"]
                if pred.predicted_framework
                else False
            )

            # Absolute error from point estimate
            absolute_error = abs(pred.predicted_answer - gt["point_estimate"])

            # Overconfidence: model is confident but wrong
            overconfident = pred.predicted_confidence > 0.7 and not correct

            self.results.append(ProblemResult(
                problem_id=pid,
                category=problem["category"],
                difficulty=problem["difficulty"],
                correct=correct,
                model_confidence=pred.predicted_confidence,
                ground_truth_pe=gt["point_estimate"],
                model_pe=pred.predicted_answer,
                absolute_error=absolute_error,
                framework_match=framework_match,
                overconfident=overconfident,
            ))

    # ─────────────────────────────────────────────────────────────
    # Metric 1: Accuracy@exact
    # ─────────────────────────────────────────────────────────────

    def accuracy(self) -> float:
        """Overall accuracy: fraction of predictions within ground truth CI."""
        if not self.results:
            return 0.0
        return sum(r.correct for r in self.results) / len(self.results)

    # ─────────────────────────────────────────────────────────────
    # Metric 2: Expected Calibration Error (ECE)
    # ─────────────────────────────────────────────────────────────

    def ece(self, n_bins: int = 10) -> float:
        """
        Expected Calibration Error using uniform binning.

        ECE = Σ (|B_m| / N) * |acc(B_m) - conf(B_m)|

        where B_m is the set of predictions in bin m.
        """
        if not self.results:
            return 0.0

        bins = defaultdict(list)
        for r in self.results:
            bin_idx = min(int(r.model_confidence * n_bins), n_bins - 1)
            bins[bin_idx].append(r)

        ece_val = 0.0
        n = len(self.results)
        for bin_idx, bin_results in bins.items():
            bin_acc = sum(r.correct for r in bin_results) / len(bin_results)
            bin_conf = sum(r.model_confidence for r in bin_results) / len(bin_results)
            ece_val += (len(bin_results) / n) * abs(bin_acc - bin_conf)

        return ece_val

    def calibration_curve(self, n_bins: int = 10) -> list[dict]:
        """Return calibration data for plotting reliability diagrams."""
        if not self.results:
            return []

        bins = defaultdict(list)
        for r in self.results:
            bin_idx = min(int(r.model_confidence * n_bins), n_bins - 1)
            bins[bin_idx].append(r)

        curve = []
        for bin_idx in range(n_bins):
            if bin_idx not in bins:
                continue
            bin_results = bins[bin_idx]
            curve.append({
                "bin_lower": bin_idx / n_bins,
                "bin_upper": (bin_idx + 1) / n_bins,
                "mean_confidence": sum(r.model_confidence for r in bin_results) / len(bin_results),
                "accuracy": sum(r.correct for r in bin_results) / len(bin_results),
                "count": len(bin_results),
            })
        return curve

    # ─────────────────────────────────────────────────────────────
    # Metric 3: Category Breakdown
    # ─────────────────────────────────────────────────────────────

    def accuracy_by_category(self) -> dict[str, dict]:
        """Accuracy and count per category."""
        groups = defaultdict(list)
        for r in self.results:
            groups[r.category].append(r)

        return {
            cat: {
                "accuracy": sum(r.correct for r in results) / len(results),
                "count": len(results),
                "mean_error": sum(r.absolute_error for r in results) / len(results),
            }
            for cat, results in sorted(groups.items())
        }

    # ─────────────────────────────────────────────────────────────
    # Metric 4: Difficulty Scaling
    # ─────────────────────────────────────────────────────────────

    def accuracy_by_difficulty(self) -> dict[int, dict]:
        """Accuracy per difficulty level."""
        groups = defaultdict(list)
        for r in self.results:
            groups[r.difficulty].append(r)

        return {
            diff: {
                "accuracy": sum(r.correct for r in results) / len(results),
                "count": len(results),
                "mean_error": sum(r.absolute_error for r in results) / len(results),
            }
            for diff, results in sorted(groups.items())
        }

    # ─────────────────────────────────────────────────────────────
    # Metric 5: Overconfidence Rate
    # ─────────────────────────────────────────────────────────────

    def overconfidence_rate(self) -> float:
        """Fraction of predictions where the model is high-confidence but wrong."""
        if not self.results:
            return 0.0
        return sum(r.overconfident for r in self.results) / len(self.results)

    # ─────────────────────────────────────────────────────────────
    # Metric 6: Reasoning Chain Quality
    # ─────────────────────────────────────────────────────────────

    def framework_match_rate(self) -> float:
        """Fraction of predictions where the model chose the correct reasoning framework."""
        matched = [r for r in self.results if r.framework_match is not None]
        if not matched:
            return 0.0
        return sum(r.framework_match for r in matched) / len(matched)

    # ─────────────────────────────────────────────────────────────
    # All metrics
    # ─────────────────────────────────────────────────────────────

    def compute_all(self) -> dict:
        """Compute and return all metrics as a dictionary."""
        return {
            "accuracy": round(self.accuracy(), 4),
            "ece": round(self.ece(), 4),
            "overconfidence_rate": round(self.overconfidence_rate(), 4),
            "framework_match_rate": round(self.framework_match_rate(), 4),
            "accuracy_by_category": self.accuracy_by_category(),
            "accuracy_by_difficulty": self.accuracy_by_difficulty(),
            "calibration_curve": self.calibration_curve(),
            "total_evaluated": len(self.results),
        }

    def summary(self) -> str:
        """Human-readable summary of all metrics."""
        results = self.compute_all()
        lines = [
            f"\n{'═' * 60}",
            f"  MURU-BENCH Evaluation Results",
            f"{'═' * 60}",
            f"",
            f"  Total problems evaluated: {results['total_evaluated']}",
            f"  Accuracy@exact:           {results['accuracy']:.1%}",
            f"  ECE:                      {results['ece']:.4f}",
            f"  Overconfidence rate:       {results['overconfidence_rate']:.1%}",
            f"  Framework match rate:      {results['framework_match_rate']:.1%}",
            f"",
            f"  {'─' * 50}",
            f"  Accuracy by Category",
            f"  {'─' * 50}",
        ]
        for cat, data in results["accuracy_by_category"].items():
            lines.append(f"    {cat:<35} {data['accuracy']:.1%}  (n={data['count']})")

        lines.extend([
            f"",
            f"  {'─' * 50}",
            f"  Accuracy by Difficulty",
            f"  {'─' * 50}",
        ])
        for diff, data in results["accuracy_by_difficulty"].items():
            stars = "★" * diff + "☆" * (5 - diff)
            lines.append(f"    Level {diff} {stars}  {data['accuracy']:.1%}  (n={data['count']})")

        lines.append("")
        return "\n".join(lines)
