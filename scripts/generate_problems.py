#!/usr/bin/env python3
"""
generate_problems.py — MURU-BENCH Parametric Problem Generator

Generates validated problem variations from parameterized templates.
Each template defines a problem structure with randomizable parameters,
producing unique problems with different numerical values while
maintaining mathematical correctness.

Usage:
    python scripts/generate_problems.py --category bayesian_updating --n 50
    python scripts/generate_problems.py --all --n 100
    python scripts/generate_problems.py --template medical_test --n 20
    python scripts/generate_problems.py --list-templates
    python scripts/generate_problems.py --dry-run --n 5

Templates produce problems with:
  - Randomized numerical parameters within valid ranges
  - Automatically computed ground truth (point estimates + CIs)
  - Contextual variation (different domains/scenarios)
  - Difficulty calibrated by parameter complexity
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "train"
sys.path.insert(0, str(PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────

def get_next_id() -> int:
    """Find the next available MURU ID number."""
    existing = set()
    for subdir in ["train", "validation", "test"]:
        for f in (PROJECT_ROOT / "data" / subdir).rglob("MURU-*.json"):
            try:
                num = int(f.stem.split("-")[1])
                existing.add(num)
            except (IndexError, ValueError):
                continue
    return max(existing, default=0) + 1


def round_sig(x: float, sig: int = 3) -> float:
    """Round to significant figures."""
    if x == 0:
        return 0
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)


def bayes(prior: float, sensitivity: float, specificity: float) -> float:
    """Compute P(H|+) using Bayes' theorem."""
    fp_rate = 1 - specificity
    p_pos = sensitivity * prior + fp_rate * (1 - prior)
    if p_pos == 0:
        return 0
    return sensitivity * prior / p_pos


def bayes_negative(prior: float, sensitivity: float, specificity: float) -> float:
    """Compute P(H|-) using Bayes' theorem for negative evidence."""
    fn_rate = 1 - sensitivity
    p_neg = fn_rate * prior + specificity * (1 - prior)
    if p_neg == 0:
        return 0
    return fn_rate * prior / p_neg


# ──────────────────────────────────────────────────────────────
# Template Registry
# ──────────────────────────────────────────────────────────────

@dataclass
class ProblemTemplate:
    """A parameterized template for generating problem variations."""
    name: str
    category: str
    difficulty_range: tuple[int, int]
    description: str
    generate: Callable[[int, int], dict]  # (problem_id, difficulty) -> problem dict


TEMPLATES: dict[str, ProblemTemplate] = {}


def register_template(template: ProblemTemplate):
    TEMPLATES[template.name] = template


# ──────────────────────────────────────────────────────────────
# Template: Medical Diagnostic Test (Bayesian Updating)
# ──────────────────────────────────────────────────────────────

MEDICAL_CONTEXTS = [
    {"disease": "a rare genetic condition", "test": "a blood screening test", "domain": "genetics"},
    {"disease": "early-stage lung cancer", "test": "a CT scan screening", "domain": "oncology"},
    {"disease": "Type 2 diabetes", "test": "an HbA1c test", "domain": "endocrinology"},
    {"disease": "celiac disease", "test": "a tTG-IgA antibody test", "domain": "gastroenterology"},
    {"disease": "tuberculosis", "test": "a skin tuberculin test", "domain": "infectious disease"},
    {"disease": "HIV", "test": "a rapid antibody test", "domain": "virology"},
    {"disease": "coronary artery disease", "test": "a stress ECG test", "domain": "cardiology"},
    {"disease": "hepatitis C", "test": "an anti-HCV screening test", "domain": "hepatology"},
    {"disease": "prostate cancer", "test": "a PSA blood test", "domain": "urology"},
    {"disease": "iron deficiency anemia", "test": "a serum ferritin test", "domain": "hematology"},
    {"disease": "hypothyroidism", "test": "a TSH blood test", "domain": "endocrinology"},
    {"disease": "rheumatoid arthritis", "test": "an anti-CCP antibody test", "domain": "rheumatology"},
    {"disease": "deep vein thrombosis", "test": "a D-dimer blood test", "domain": "vascular medicine"},
    {"disease": "meningitis", "test": "a lumbar puncture CSF analysis", "domain": "neurology"},
    {"disease": "Lyme disease", "test": "an ELISA screening test", "domain": "infectious disease"},
]


def generate_medical_test(problem_id: int, difficulty: int) -> dict:
    ctx = random.choice(MEDICAL_CONTEXTS)

    if difficulty <= 2:
        sensitivity = round(random.uniform(0.85, 0.99), 2)
        specificity = round(random.uniform(0.80, 0.98), 2)
        prev_low = round(random.uniform(0.005, 0.03), 3)
        prev_high = round(prev_low * random.uniform(1.5, 3.0), 3)
        prev_mid = round((prev_low + prev_high) / 2, 3)
    else:
        sensitivity = round(random.uniform(0.75, 0.95), 2)
        specificity = round(random.uniform(0.70, 0.95), 2)
        prev_low = round(random.uniform(0.01, 0.10), 3)
        prev_high = round(prev_low * random.uniform(1.5, 3.0), 3)
        prev_mid = round((prev_low + prev_high) / 2, 3)

    # Compute ground truth
    ppv_low = round_sig(bayes(prev_low, sensitivity, specificity), 3)
    ppv_mid = round_sig(bayes(prev_mid, sensitivity, specificity), 3)
    ppv_high = round_sig(bayes(prev_high, sensitivity, specificity), 3)

    ci = sorted([ppv_low, ppv_high])
    ci_lower = round_sig(ci[0], 3)
    ci_upper = round_sig(ci[1], 3)

    # Ensure valid CI
    if ci_lower >= ci_upper:
        ci_upper = round(ci_lower + 0.01, 3)

    prev_pct_low = f"{prev_low*100:.1f}%"
    prev_pct_high = f"{prev_high*100:.1f}%"

    stem = (
        f"A hospital uses {ctx['test']} to screen for {ctx['disease']}. "
        f"The test has a sensitivity of {sensitivity*100:.0f}% and a specificity of {specificity*100:.0f}%. "
        f"The prevalence of {ctx['disease']} in the patient population is estimated to be between "
        f"{prev_pct_low} and {prev_pct_high}, with uncertainty due to regional variation and "
        f"demographic differences in the hospital's catchment area. "
        f"A patient receives a positive test result. What is the probability that "
        f"the patient actually has {ctx['disease']}? Provide your answer as a probability "
        f"with a confidence interval reflecting the uncertainty in prevalence."
    )

    answer = (
        f"P({ctx['disease']} | positive test) ranges from {ci_lower} to {ci_upper} "
        f"as prevalence varies from {prev_pct_low} to {prev_pct_high}. "
        f"Point estimate at {prev_mid*100:.1f}% prevalence: {ppv_mid}."
    )

    fp_rate = round(1 - specificity, 2)
    steps = [
        f"Apply Bayes' theorem: P(D|+) = P(+|D)×P(D) / [P(+|D)×P(D) + P(+|¬D)×P(¬D)], where P(+|¬D) = 1-specificity = {fp_rate}.",
        f"At prevalence {prev_mid*100:.1f}%: P(D|+) = {sensitivity}×{prev_mid} / ({sensitivity}×{prev_mid} + {fp_rate}×{round(1-prev_mid,3)}) ≈ {ppv_mid}.",
        f"At prevalence {prev_pct_low}: P(D|+) = {sensitivity}×{prev_low} / ({sensitivity}×{prev_low} + {fp_rate}×{round(1-prev_low,3)}) ≈ {ppv_low}.",
        f"At prevalence {prev_pct_high}: P(D|+) = {sensitivity}×{prev_high} / ({sensitivity}×{prev_high} + {fp_rate}×{round(1-prev_high,3)}) ≈ {ppv_high}.",
        f"The confidence interval [{ci_lower}, {ci_upper}] reflects the range of posterior probabilities over plausible prevalence values.",
    ]

    failure_modes = [
        f"Confusing sensitivity ({sensitivity*100:.0f}%) with positive predictive value",
        "Providing only a point estimate without acknowledging the prevalence uncertainty",
        "Ignoring the base rate (prevalence) and using the test sensitivity as the answer",
    ]

    if difficulty >= 3:
        failure_modes.append("Not recognizing the strong base rate effect when prevalence is low")

    return {
        "id": f"MURU-{problem_id:04d}",
        "category": "bayesian_updating",
        "difficulty": difficulty,
        "stem": stem,
        "uncertainty_type": "parameter_uncertainty",
        "required_framework": "bayesian_inference",
        "ground_truth": {
            "answer": answer,
            "point_estimate": ppv_mid,
            "confidence_interval": [ci_lower, ci_upper],
            "ci_level": 0.90,
        },
        "solution_steps": steps,
        "common_failure_modes": failure_modes,
        "metadata": {
            "author": "generator_medical_test",
            "reviewed": False,
            "source_inspiration": "original",
        },
    }


register_template(ProblemTemplate(
    name="medical_test",
    category="bayesian_updating",
    difficulty_range=(1, 3),
    description="Medical diagnostic test with uncertain prevalence (Bayes' theorem)",
    generate=generate_medical_test,
))


# ──────────────────────────────────────────────────────────────
# Template: Quality Control / Manufacturing (Bayesian Updating)
# ──────────────────────────────────────────────────────────────

QC_CONTEXTS = [
    {"product": "microchips", "source_a": "Fabrication Line A", "source_b": "Fabrication Line B", "domain": "semiconductor"},
    {"product": "pharmaceutical tablets", "source_a": "Batch Process Alpha", "source_b": "Batch Process Beta", "domain": "pharma"},
    {"product": "automotive parts", "source_a": "Robot Arm Station 1", "source_b": "Robot Arm Station 2", "domain": "automotive"},
    {"product": "solar panels", "source_a": "Automated Line East", "source_b": "Automated Line West", "domain": "energy"},
    {"product": "food packages", "source_a": "Packaging Line Morning Shift", "source_b": "Packaging Line Night Shift", "domain": "food"},
    {"product": "aircraft rivets", "source_a": "Forge A", "source_b": "Forge B", "domain": "aerospace"},
    {"product": "PCB boards", "source_a": "SMT Line 1", "source_b": "SMT Line 2", "domain": "electronics"},
    {"product": "glass bottles", "source_a": "Furnace Alpha", "source_b": "Furnace Beta", "domain": "packaging"},
    {"product": "textile rolls", "source_a": "Loom Section East", "source_b": "Loom Section West", "domain": "textile"},
    {"product": "steel beams", "source_a": "Mill A", "source_b": "Mill B", "domain": "construction"},
]


def generate_quality_control(problem_id: int, difficulty: int) -> dict:
    ctx = random.choice(QC_CONTEXTS)

    # Source A: known defect rate, source B: uncertain
    prop_a = round(random.uniform(0.50, 0.80), 2)
    prop_b = round(1 - prop_a, 2)
    defect_a = round(random.uniform(0.01, 0.05), 3)
    defect_b_low = round(random.uniform(0.03, 0.08), 3)
    defect_b_high = round(defect_b_low + random.uniform(0.02, 0.06), 3)
    defect_b_mid = round((defect_b_low + defect_b_high) / 2, 3)

    def posterior_b(defect_b):
        p_def = defect_a * prop_a + defect_b * prop_b
        if p_def == 0:
            return 0
        return defect_b * prop_b / p_def

    post_low = round_sig(posterior_b(defect_b_low), 3)
    post_mid = round_sig(posterior_b(defect_b_mid), 3)
    post_high = round_sig(posterior_b(defect_b_high), 3)

    ci = sorted([post_low, post_high])
    if ci[0] >= ci[1]:
        ci[1] = round(ci[0] + 0.01, 3)

    stem = (
        f"A factory produces {ctx['product']} using two sources. {ctx['source_a']} handles "
        f"{prop_a*100:.0f}% of production with a known defect rate of {defect_a*100:.1f}%. "
        f"{ctx['source_b']} handles {prop_b*100:.0f}% of production, but its defect rate "
        f"is uncertain — recent quality audits suggest it lies between {defect_b_low*100:.1f}% "
        f"and {defect_b_high*100:.1f}%, depending on raw material batch quality. "
        f"A randomly selected {ctx['product'].rstrip('s')} is found to be defective. "
        f"What is the probability it came from {ctx['source_b']}?"
    )

    answer = (
        f"P({ctx['source_b']} | defective) ranges from {ci[0]} to {ci[1]} as "
        f"{ctx['source_b']}'s defect rate varies from {defect_b_low*100:.1f}% to {defect_b_high*100:.1f}%. "
        f"Point estimate at {defect_b_mid*100:.1f}%: {post_mid}."
    )

    steps = [
        f"Apply Bayes' theorem: P(B|def) = P(def|B)×P(B) / [P(def|A)×P(A) + P(def|B)×P(B)].",
        f"At defect rate {defect_b_mid*100:.1f}%: P(B|def) = {defect_b_mid}×{prop_b} / ({defect_a}×{prop_a} + {defect_b_mid}×{prop_b}) ≈ {post_mid}.",
        f"At defect rate {defect_b_low*100:.1f}%: P(B|def) ≈ {post_low}.",
        f"At defect rate {defect_b_high*100:.1f}%: P(B|def) ≈ {post_high}.",
        f"The interval [{ci[0]}, {ci[1]}] captures the range of posterior probabilities.",
    ]

    return {
        "id": f"MURU-{problem_id:04d}",
        "category": "bayesian_updating",
        "difficulty": difficulty,
        "stem": stem,
        "uncertainty_type": "parameter_uncertainty",
        "required_framework": "bayesian_inference",
        "ground_truth": {
            "answer": answer,
            "point_estimate": post_mid,
            "confidence_interval": [ci[0], ci[1]],
            "ci_level": 0.90,
        },
        "solution_steps": steps,
        "common_failure_modes": [
            f"Using only the midpoint defect rate and providing a single answer",
            f"Ignoring the production proportions ({prop_a*100:.0f}/{prop_b*100:.0f} split)",
            "Comparing defect rates directly without applying Bayes' theorem",
        ],
        "metadata": {
            "author": "generator_quality_control",
            "reviewed": False,
            "source_inspiration": "original",
        },
    }


register_template(ProblemTemplate(
    name="quality_control",
    category="bayesian_updating",
    difficulty_range=(1, 3),
    description="Quality control with uncertain defect rate from one source (Bayes' theorem)",
    generate=generate_quality_control,
))


# ──────────────────────────────────────────────────────────────
# Template: Sequential Pipeline (Conditional Probability Chains)
# ──────────────────────────────────────────────────────────────

PIPELINE_CONTEXTS = [
    {"domain": "hiring", "stages": ["resume screening", "phone interview", "onsite interview", "offer acceptance"],
     "entity": "candidate", "outcome": "hired"},
    {"domain": "manufacturing", "stages": ["raw material inspection", "machining", "assembly", "final QA"],
     "entity": "product", "outcome": "shipped"},
    {"domain": "loan processing", "stages": ["application review", "credit check", "underwriting", "approval"],
     "entity": "loan application", "outcome": "approved"},
    {"domain": "drug development", "stages": ["preclinical testing", "Phase I trial", "Phase II trial", "FDA review"],
     "entity": "drug candidate", "outcome": "approved for market"},
    {"domain": "space mission", "stages": ["design review", "testing", "launch", "orbital insertion"],
     "entity": "mission", "outcome": "successful"},
    {"domain": "software release", "stages": ["code review", "unit testing", "integration testing", "deployment"],
     "entity": "release", "outcome": "deployed without incidents"},
    {"domain": "grant application", "stages": ["departmental review", "external peer review", "panel discussion", "funding decision"],
     "entity": "proposal", "outcome": "funded"},
    {"domain": "immigration", "stages": ["document verification", "background check", "interview", "visa issuance"],
     "entity": "application", "outcome": "approved"},
]


def generate_sequential_pipeline(problem_id: int, difficulty: int) -> dict:
    ctx = random.choice(PIPELINE_CONTEXTS)

    if difficulty <= 2:
        n_stages = 3
    elif difficulty <= 3:
        n_stages = 4
    else:
        n_stages = min(len(ctx["stages"]), 4)

    stages = ctx["stages"][:n_stages]

    # Generate pass rates — one stage has uncertainty
    uncertain_stage = random.randint(0, n_stages - 1)
    rates = []
    for i in range(n_stages):
        if i == uncertain_stage:
            low = round(random.uniform(0.55, 0.80), 2)
            high = round(low + random.uniform(0.08, 0.20), 2)
            high = min(high, 0.99)
            mid = round((low + high) / 2, 2)
            rates.append({"low": low, "mid": mid, "high": high, "uncertain": True})
        else:
            rate = round(random.uniform(0.70, 0.98), 2)
            rates.append({"low": rate, "mid": rate, "high": rate, "uncertain": False})

    # Compute overall probabilities
    def chain_prob(scenario):
        p = 1.0
        for r in rates:
            p *= r[scenario]
        return round_sig(p, 3)

    p_low = chain_prob("low")
    p_mid = chain_prob("mid")
    p_high = chain_prob("high")

    ci = sorted([p_low, p_high])
    if ci[0] >= ci[1]:
        ci[1] = round(ci[0] + 0.01, 3)

    # Build stem
    rate_descriptions = []
    for i, (stage, rate) in enumerate(zip(stages, rates)):
        if i == 0:
            prefix = f"A {ctx['entity']} enters the {ctx['domain']} pipeline. The probability of passing {stage}"
        else:
            prefix = f"If the previous stage is passed, the probability of passing {stage}"

        if rate["uncertain"]:
            rate_descriptions.append(
                f"{prefix} is between {rate['low']*100:.0f}% and {rate['high']*100:.0f}% "
                f"(uncertain due to evaluator variability)."
            )
        else:
            rate_descriptions.append(f"{prefix} is {rate['mid']*100:.0f}%.")

    stem = " ".join(rate_descriptions) + (
        f" What is the overall probability that the {ctx['entity']} is {ctx['outcome']}?"
    )

    uncertain_name = stages[uncertain_stage]
    answer = (
        f"P({ctx['outcome']}) ranges from {ci[0]} to {ci[1]} as the {uncertain_name} pass rate "
        f"varies from {rates[uncertain_stage]['low']*100:.0f}% to {rates[uncertain_stage]['high']*100:.0f}%. "
        f"Point estimate: {p_mid}."
    )

    steps = [
        f"The {ctx['entity']} must pass all {n_stages} stages sequentially: P(success) = {'×'.join(f'P(stage {i+1})' for i in range(n_stages))}.",
    ]
    for scenario, label in [("mid", "mid-range"), ("low", "lower bound"), ("high", "upper bound")]:
        val = chain_prob(scenario)
        calc = " × ".join(f"{r[scenario]}" for r in rates)
        steps.append(f"At {label}: P(success) = {calc} = {val}.")

    steps.append(
        f"The uncertainty in the {uncertain_name} stage creates a "
        f"{abs(ci[1]-ci[0])*100:.1f} percentage point range in the overall success probability."
    )

    return {
        "id": f"MURU-{problem_id:04d}",
        "category": "conditional_probability_chains",
        "difficulty": difficulty,
        "stem": stem,
        "uncertainty_type": "parameter_uncertainty",
        "required_framework": "bayesian_inference",
        "ground_truth": {
            "answer": answer,
            "point_estimate": p_mid,
            "confidence_interval": [ci[0], ci[1]],
            "ci_level": 0.90,
        },
        "solution_steps": steps,
        "common_failure_modes": [
            "Adding probabilities instead of multiplying for sequential stages",
            f"Using a single value for the {uncertain_name} rate without acknowledging uncertainty",
            "Not recognizing that uncertainty at one stage propagates through all subsequent stages",
        ],
        "metadata": {
            "author": "generator_pipeline",
            "reviewed": False,
            "source_inspiration": "original",
        },
    }


register_template(ProblemTemplate(
    name="sequential_pipeline",
    category="conditional_probability_chains",
    difficulty_range=(1, 4),
    description="Multi-stage sequential pipeline with one uncertain stage",
    generate=generate_sequential_pipeline,
))


# ──────────────────────────────────────────────────────────────
# Template: Search / Detection (Bayesian Negative Evidence)
# ──────────────────────────────────────────────────────────────

SEARCH_CONTEXTS = [
    {"target": "a missing person", "area_a": "the forest zone", "area_b": "the mountain zone", "domain": "search and rescue"},
    {"target": "a submarine", "area_a": "Sector Alpha", "area_b": "Sector Bravo", "domain": "naval search"},
    {"target": "a lost drone", "area_a": "the urban area", "area_b": "the farmland area", "domain": "asset recovery"},
    {"target": "a buried artifact", "area_a": "Dig Site North", "area_b": "Dig Site South", "domain": "archaeology"},
    {"target": "a gas leak", "area_a": "Building Wing A", "area_b": "Building Wing B", "domain": "safety inspection"},
    {"target": "a network intrusion", "area_a": "the web server cluster", "area_b": "the database cluster", "domain": "cybersecurity"},
    {"target": "a tumor", "area_a": "the left lobe", "area_b": "the right lobe", "domain": "radiology"},
    {"target": "an oil deposit", "area_a": "Block X", "area_b": "Block Y", "domain": "petroleum exploration"},
]


def generate_search_detection(problem_id: int, difficulty: int) -> dict:
    ctx = random.choice(SEARCH_CONTEXTS)

    prior_a = round(random.uniform(0.30, 0.70), 2)
    prior_b = round(1 - prior_a, 2)
    det_low = round(random.uniform(0.50, 0.70), 2)
    det_high = round(det_low + random.uniform(0.10, 0.25), 2)
    det_high = min(det_high, 0.95)
    det_mid = round((det_low + det_high) / 2, 2)

    def posterior_b(det):
        """P(target in B | not found in A)"""
        p_not_found = (1 - det) * prior_a + 1.0 * prior_b
        if p_not_found == 0:
            return 0
        return prior_b / p_not_found

    post_low = round_sig(posterior_b(det_low), 3)
    post_mid = round_sig(posterior_b(det_mid), 3)
    post_high = round_sig(posterior_b(det_high), 3)

    ci = sorted([post_low, post_high])
    if ci[0] >= ci[1]:
        ci[1] = round(ci[0] + 0.01, 3)

    stem = (
        f"A {ctx['domain']} team is looking for {ctx['target']}. Based on initial analysis, "
        f"they estimate a {prior_a*100:.0f}% probability that {ctx['target']} is in {ctx['area_a']} "
        f"and a {prior_b*100:.0f}% probability it is in {ctx['area_b']}. They search {ctx['area_a']} "
        f"first and do not find {ctx['target']}. However, their search effectiveness is uncertain "
        f"— conditions vary, and they estimate their detection probability in {ctx['area_a']} is "
        f"between {det_low*100:.0f}% and {det_high*100:.0f}%. Given they did not find {ctx['target']} "
        f"in {ctx['area_a']}, what is the updated probability that {ctx['target']} is in {ctx['area_b']}?"
    )

    answer = (
        f"P({ctx['area_b']} | not found in {ctx['area_a']}) ranges from {ci[0]} to {ci[1]} "
        f"as detection probability varies from {det_low*100:.0f}% to {det_high*100:.0f}%. "
        f"Point estimate at {det_mid*100:.0f}% detection: {post_mid}."
    )

    steps = [
        f"Apply Bayes' theorem for negative evidence: P(B|¬found_A) = P(¬found_A|B)×P(B) / P(¬found_A).",
        f"If target is in B: P(¬found in A | in B) = 1 (cannot find what isn't there).",
        f"If target is in A: P(¬found in A | in A) = 1 - detection_probability.",
        f"P(¬found_A) = (1-det)×{prior_a} + 1×{prior_b}.",
        f"At detection {det_mid*100:.0f}%: P(B|¬found) = {prior_b} / ({round(1-det_mid,2)}×{prior_a} + {prior_b}) ≈ {post_mid}.",
        f"At detection {det_low*100:.0f}%: P(B|¬found) ≈ {post_low}.",
        f"At detection {det_high*100:.0f}%: P(B|¬found) ≈ {post_high}.",
    ]

    return {
        "id": f"MURU-{problem_id:04d}",
        "category": "bayesian_updating",
        "difficulty": difficulty,
        "stem": stem,
        "uncertainty_type": "parameter_uncertainty",
        "required_framework": "bayesian_inference",
        "ground_truth": {
            "answer": answer,
            "point_estimate": post_mid,
            "confidence_interval": [ci[0], ci[1]],
            "ci_level": 0.90,
        },
        "solution_steps": steps,
        "common_failure_modes": [
            f"Assuming not finding in {ctx['area_a']} means it is definitely in {ctx['area_b']}",
            "Ignoring the uncertain detection probability and using a single value",
            "Not applying Bayes' theorem correctly for negative evidence (absence of detection)",
        ],
        "metadata": {
            "author": "generator_search",
            "reviewed": False,
            "source_inspiration": "original",
        },
    }


register_template(ProblemTemplate(
    name="search_detection",
    category="bayesian_updating",
    difficulty_range=(1, 2),
    description="Search and detection with uncertain coverage (negative evidence updating)",
    generate=generate_search_detection,
))


# ──────────────────────────────────────────────────────────────
# Template: Decision with Uncertain Payoffs (Decision Theory)
# ──────────────────────────────────────────────────────────────

DECISION_CONTEXTS = [
    {"scenario": "crop choice", "option_a": "planting wheat", "option_b": "planting corn",
     "factor": "rainfall", "domain": "agriculture"},
    {"scenario": "product launch", "option_a": "the premium version", "option_b": "the budget version",
     "factor": "market demand", "domain": "business"},
    {"scenario": "treatment choice", "option_a": "the established treatment", "option_b": "the experimental treatment",
     "factor": "patient response rate", "domain": "medicine"},
    {"scenario": "route selection", "option_a": "the highway route", "option_b": "the scenic route",
     "factor": "traffic conditions", "domain": "logistics"},
    {"scenario": "energy source", "option_a": "solar panels", "option_b": "wind turbines",
     "factor": "weather patterns", "domain": "energy"},
    {"scenario": "server configuration", "option_a": "horizontal scaling", "option_b": "vertical scaling",
     "factor": "traffic load patterns", "domain": "cloud computing"},
]


def generate_decision_payoff(problem_id: int, difficulty: int) -> dict:
    ctx = random.choice(DECISION_CONTEXTS)

    # Option A: safe, predictable
    payoff_a = round(random.uniform(50, 200), 0)

    # Option B: risky, depends on uncertain factor
    p_good_low = round(random.uniform(0.30, 0.45), 2)
    p_good_high = round(p_good_low + random.uniform(0.15, 0.25), 2)
    p_good_mid = round((p_good_low + p_good_high) / 2, 2)

    payoff_b_good = round(random.uniform(payoff_a * 1.5, payoff_a * 3.0), 0)
    payoff_b_bad = round(random.uniform(payoff_a * 0.1, payoff_a * 0.5), 0)

    # Expected values
    def ev_b(p_good):
        return round(p_good * payoff_b_good + (1 - p_good) * payoff_b_bad, 1)

    ev_b_low = ev_b(p_good_low)
    ev_b_mid = ev_b(p_good_mid)
    ev_b_high = ev_b(p_good_high)

    ci = sorted([ev_b_low, ev_b_high])

    stem = (
        f"A decision-maker must choose between two options. "
        f"Option A ({ctx['option_a']}) yields a guaranteed return of ${payoff_a:.0f}K. "
        f"Option B ({ctx['option_b']}) depends on {ctx['factor']}: with probability "
        f"{p_good_low*100:.0f}-{p_good_high*100:.0f}% (uncertain), it yields ${payoff_b_good:.0f}K; "
        f"otherwise it yields ${payoff_b_bad:.0f}K. "
        f"Compute the expected value of each option and determine which is better. "
        f"How does the uncertainty in {ctx['factor']} affect the recommendation?"
    )

    # Determine breakeven
    # payoff_a = p × payoff_b_good + (1-p) × payoff_b_bad
    p_breakeven = (payoff_a - payoff_b_bad) / (payoff_b_good - payoff_b_bad) if payoff_b_good != payoff_b_bad else 0.5
    p_breakeven = round(p_breakeven, 3)

    if ev_b_mid > payoff_a:
        recommendation = f"Option B is preferred at most assumptions (EV_B > EV_A when P(good) > {p_breakeven*100:.1f}%)."
    else:
        recommendation = f"Option A is preferred at most assumptions (EV_A > EV_B when P(good) < {p_breakeven*100:.1f}%)."

    answer = (
        f"EV(A) = ${payoff_a:.0f}K (fixed). EV(B) ranges from ${ci[0]:.1f}K to ${ci[1]:.1f}K. "
        f"Point estimate: ${ev_b_mid:.1f}K. {recommendation}"
    )

    steps = [
        f"EV(A) = ${payoff_a:.0f}K (guaranteed).",
        f"EV(B) = P(good)×${payoff_b_good:.0f}K + (1-P(good))×${payoff_b_bad:.0f}K.",
        f"At P(good)={p_good_mid}: EV(B) = {p_good_mid}×{payoff_b_good:.0f} + {round(1-p_good_mid,2)}×{payoff_b_bad:.0f} = ${ev_b_mid:.1f}K.",
        f"At P(good)={p_good_low}: EV(B) = ${ev_b_low:.1f}K.",
        f"At P(good)={p_good_high}: EV(B) = ${ev_b_high:.1f}K.",
        f"Breakeven probability: P(good) = ({payoff_a:.0f}-{payoff_b_bad:.0f}) / ({payoff_b_good:.0f}-{payoff_b_bad:.0f}) = {p_breakeven}.",
        f"The decision switches at P(good) = {p_breakeven*100:.1f}%, which {'falls within' if p_good_low <= p_breakeven <= p_good_high else 'falls outside'} the uncertainty range.",
    ]

    return {
        "id": f"MURU-{problem_id:04d}",
        "category": "decision_under_uncertainty",
        "difficulty": difficulty,
        "stem": stem,
        "uncertainty_type": "parameter_uncertainty",
        "required_framework": "decision_theory",
        "ground_truth": {
            "answer": answer,
            "point_estimate": ev_b_mid,
            "confidence_interval": [ci[0], ci[1]],
            "ci_level": 0.90,
        },
        "solution_steps": steps,
        "common_failure_modes": [
            "Picking a single probability and giving a definitive recommendation without sensitivity analysis",
            "Not computing the breakeven probability where the optimal decision switches",
            "Ignoring the range of outcomes and focusing only on expected values (risk-neutral assumption)",
        ],
        "metadata": {
            "author": "generator_decision",
            "reviewed": False,
            "source_inspiration": "original",
        },
    }


register_template(ProblemTemplate(
    name="decision_payoff",
    category="decision_under_uncertainty",
    difficulty_range=(1, 3),
    description="Two-option decision with uncertain outcome probability",
    generate=generate_decision_payoff,
))


# ──────────────────────────────────────────────────────────────
# Template: Sample Mean with Uncertainty (Distribution Estimation)
# ──────────────────────────────────────────────────────────────

SAMPLE_MEAN_CONTEXTS = [
    {"measurement": "response time (ms)", "entity": "website users", "domain": "web performance", "unit": "ms"},
    {"measurement": "weight (grams)", "entity": "cereal boxes", "domain": "food manufacturing", "unit": "g"},
    {"measurement": "battery life (hours)", "entity": "smartphone models", "domain": "electronics testing", "unit": "hours"},
    {"measurement": "commute time (minutes)", "entity": "city workers", "domain": "urban planning", "unit": "min"},
    {"measurement": "yield per hectare (tonnes)", "entity": "farms", "domain": "agriculture", "unit": "tonnes"},
    {"measurement": "diastolic blood pressure (mmHg)", "entity": "patients", "domain": "clinical study", "unit": "mmHg"},
    {"measurement": "download speed (Mbps)", "entity": "subscribers", "domain": "ISP analysis", "unit": "Mbps"},
    {"measurement": "assembly time (seconds)", "entity": "factory workers", "domain": "industrial engineering", "unit": "sec"},
    {"measurement": "fuel consumption (L/100km)", "entity": "vehicles", "domain": "automotive testing", "unit": "L/100km"},
    {"measurement": "wait time (minutes)", "entity": "emergency room visitors", "domain": "healthcare operations", "unit": "min"},
]


def generate_sample_mean(problem_id: int, difficulty: int) -> dict:
    ctx = random.choice(SAMPLE_MEAN_CONTEXTS)

    # Generate sample statistics
    if difficulty <= 2:
        n = random.choice([20, 25, 30, 40, 50])
        mean = round(random.uniform(20, 500), 1)
        std = round(mean * random.uniform(0.10, 0.30), 1)
        n_outliers = 0
    else:
        n = random.choice([15, 20, 25, 30])
        mean = round(random.uniform(30, 400), 1)
        std = round(mean * random.uniform(0.15, 0.40), 1)
        n_outliers = random.randint(2, max(3, n // 8))

    # Compute CI using t-distribution approximation
    # t critical values (approximate for common df)
    t_crit = {10: 2.228, 14: 2.145, 15: 2.131, 19: 2.093, 20: 2.086,
              24: 2.064, 25: 2.060, 29: 2.045, 30: 2.042, 39: 2.023,
              40: 2.021, 49: 2.010, 50: 2.009}
    df = n - 1
    t = t_crit.get(df, 2.0)  # fallback

    se = round(std / math.sqrt(n), 2)
    margin = round(t * se, 2)
    ci_lower = round(mean - margin, 1)
    ci_upper = round(mean + margin, 1)

    if n_outliers > 0:
        # Trimmed mean scenario
        trim_mean = round(mean * random.uniform(0.92, 0.98), 1)
        trim_std = round(std * random.uniform(0.60, 0.80), 1)
        n_trim = n - n_outliers
        df_trim = n_trim - 1
        t_trim = t_crit.get(df_trim, 2.0)
        se_trim = round(trim_std / math.sqrt(n_trim), 2)
        margin_trim = round(t_trim * se_trim, 2)
        ci_lower_trim = round(trim_mean - margin_trim, 1)
        ci_upper_trim = round(trim_mean + margin_trim, 1)

        # Combined interval
        final_ci_lower = min(ci_lower, ci_lower_trim)
        final_ci_upper = max(ci_upper, ci_upper_trim)
        point_est = round((mean + trim_mean) / 2, 1)
    else:
        final_ci_lower = ci_lower
        final_ci_upper = ci_upper
        point_est = mean
        trim_mean = None

    stem_base = (
        f"A {ctx['domain']} study measures the {ctx['measurement']} for a sample of "
        f"{n} {ctx['entity']}. The sample mean is {mean} {ctx['unit']} with a sample "
        f"standard deviation of {std} {ctx['unit']}."
    )

    if n_outliers > 0:
        stem = stem_base + (
            f" However, {n_outliers} of the {n} observations appear to be outliers "
            f"(values more than 2.5 standard deviations from the mean). If these outliers "
            f"are removed, the remaining {n_trim} observations have a mean of {trim_mean} "
            f"{ctx['unit']} and standard deviation of {trim_std} {ctx['unit']}. "
            f"Estimate the true population mean and provide a 95% confidence interval "
            f"that accounts for the uncertainty about whether outliers should be included."
        )
    else:
        stem = stem_base + (
            f" Assuming the data are approximately normally distributed, estimate the "
            f"true population mean and provide a 95% confidence interval."
        )

    if n_outliers > 0:
        answer = (
            f"With all data: mean = {mean}, 95% CI = [{ci_lower}, {ci_upper}]. "
            f"Without outliers: mean = {trim_mean}, 95% CI = [{ci_lower_trim}, {ci_upper_trim}]. "
            f"Combined interval accounting for analytic uncertainty: [{final_ci_lower}, {final_ci_upper}]. "
            f"Point estimate: {point_est} {ctx['unit']}."
        )
        steps = [
            f"With full sample (n={n}): SE = {std}/√{n} = {se}. Margin = t({df},0.025) × {se} = {t} × {se} ≈ {margin}.",
            f"Full sample 95% CI: {mean} ± {margin} = [{ci_lower}, {ci_upper}].",
            f"With trimmed sample (n={n_trim}): SE = {trim_std}/√{n_trim} = {se_trim}. Margin = {t_trim} × {se_trim} ≈ {margin_trim}.",
            f"Trimmed 95% CI: {trim_mean} ± {margin_trim} = [{ci_lower_trim}, {ci_upper_trim}].",
            f"The outlier treatment choice creates structural uncertainty. Combined interval: [{final_ci_lower}, {final_ci_upper}].",
        ]
        failure_modes = [
            "Using only the full sample without considering outlier impact",
            "Removing outliers without reporting both analyses",
            "Not recognizing that the choice of analysis method is itself a source of uncertainty",
        ]
    else:
        answer = (
            f"Population mean ≈ {mean} {ctx['unit']}. "
            f"95% CI = [{ci_lower}, {ci_upper}]. Point estimate: {point_est} {ctx['unit']}."
        )
        steps = [
            f"Standard error: SE = s/√n = {std}/√{n} = {se}.",
            f"Degrees of freedom: df = {n} - 1 = {df}.",
            f"t-critical value: t({df}, 0.025) ≈ {t}.",
            f"Margin of error: {t} × {se} = {margin}.",
            f"95% CI: {mean} ± {margin} = [{ci_lower}, {ci_upper}].",
        ]
        failure_modes = [
            "Using z-critical value instead of t-critical for small samples",
            f"Computing SE as {std}/n instead of {std}/√n",
            "Not reporting the confidence interval alongside the point estimate",
        ]

    return {
        "id": f"MURU-{problem_id:04d}",
        "category": "distribution_estimation",
        "difficulty": difficulty,
        "stem": stem,
        "uncertainty_type": "data_uncertainty",
        "required_framework": "frequentist_inference",
        "ground_truth": {
            "answer": answer,
            "point_estimate": point_est,
            "confidence_interval": [final_ci_lower, final_ci_upper],
            "ci_level": 0.95,
        },
        "solution_steps": steps,
        "common_failure_modes": failure_modes,
        "metadata": {
            "author": "generator_sample_mean",
            "reviewed": False,
            "source_inspiration": "original",
        },
    }


register_template(ProblemTemplate(
    name="sample_mean",
    category="distribution_estimation",
    difficulty_range=(1, 3),
    description="Sample mean estimation with t-distribution CI, optional outlier treatment",
    generate=generate_sample_mean,
))


# ──────────────────────────────────────────────────────────────
# Template: Measurement with Systematic Bias (Distribution Est.)
# ──────────────────────────────────────────────────────────────

BIAS_CONTEXTS = [
    {"instrument": "a bathroom scale", "what": "body weight", "unit": "kg", "domain": "health monitoring"},
    {"instrument": "a speedometer", "what": "vehicle speed", "unit": "km/h", "domain": "traffic enforcement"},
    {"instrument": "a thermometer", "what": "room temperature", "unit": "°C", "domain": "climate control"},
    {"instrument": "an air quality sensor", "what": "PM2.5 concentration", "unit": "µg/m³", "domain": "environmental monitoring"},
    {"instrument": "a flow meter", "what": "water flow rate", "unit": "L/min", "domain": "water management"},
    {"instrument": "a fuel gauge", "what": "fuel level", "unit": "%", "domain": "fleet management"},
    {"instrument": "a noise meter", "what": "noise level", "unit": "dB", "domain": "occupational safety"},
    {"instrument": "a pressure gauge", "what": "tire pressure", "unit": "psi", "domain": "vehicle maintenance"},
]


def generate_measurement_bias(problem_id: int, difficulty: int) -> dict:
    ctx = random.choice(BIAS_CONTEXTS)

    n = random.choice([10, 15, 20, 25, 30])
    raw_mean = round(random.uniform(20, 200), 1)
    raw_std = round(raw_mean * random.uniform(0.05, 0.15), 1)

    # Systematic bias: instrument reads X% too high or too low
    bias_low_pct = round(random.uniform(1, 5), 1)
    bias_high_pct = round(bias_low_pct + random.uniform(2, 6), 1)
    bias_mid_pct = round((bias_low_pct + bias_high_pct) / 2, 1)

    # Randomly choose if bias is positive (overreading) or negative (underreading)
    if random.random() < 0.5:
        bias_direction = "overestimates"
        corrected_low = round(raw_mean / (1 + bias_high_pct / 100), 1)
        corrected_mid = round(raw_mean / (1 + bias_mid_pct / 100), 1)
        corrected_high = round(raw_mean / (1 + bias_low_pct / 100), 1)
    else:
        bias_direction = "underestimates"
        corrected_low = round(raw_mean / (1 - bias_high_pct / 100), 1)
        corrected_mid = round(raw_mean / (1 - bias_mid_pct / 100), 1)
        corrected_high = round(raw_mean / (1 - bias_low_pct / 100), 1)

    ci = sorted([corrected_low, corrected_high])

    # Also add sampling variability
    se_raw = round(raw_std / math.sqrt(n), 2)
    ci_sampling_half = round(2.0 * se_raw, 1)

    final_ci_lower = round(ci[0] - ci_sampling_half, 1)
    final_ci_upper = round(ci[1] + ci_sampling_half, 1)

    stem = (
        f"An engineer takes {n} measurements of {ctx['what']} using {ctx['instrument']}. "
        f"The measurements have a mean of {raw_mean} {ctx['unit']} and a standard deviation "
        f"of {raw_std} {ctx['unit']}. However, calibration tests reveal that {ctx['instrument']} "
        f"systematically {bias_direction} the true value by between {bias_low_pct}% and "
        f"{bias_high_pct}%. Estimate the true {ctx['what']} and provide a confidence interval "
        f"that accounts for both the sampling variability and the uncertain systematic bias."
    )

    answer = (
        f"Raw mean: {raw_mean} {ctx['unit']}. After correcting for the {bias_mid_pct}% bias: "
        f"≈ {corrected_mid} {ctx['unit']}. Full interval accounting for bias uncertainty "
        f"({bias_low_pct}%-{bias_high_pct}%) and sampling variability: "
        f"[{final_ci_lower}, {final_ci_upper}]. Point estimate: {corrected_mid} {ctx['unit']}."
    )

    steps = [
        f"Raw sample mean: {raw_mean} {ctx['unit']} (n={n}, s={raw_std}).",
        f"Sampling SE: {raw_std}/√{n} = {se_raw}. Sampling margin ≈ ±{ci_sampling_half}.",
        f"Bias correction: instrument {bias_direction} by {bias_low_pct}%-{bias_high_pct}%.",
        f"At {bias_mid_pct}% correction: true value ≈ {corrected_mid} {ctx['unit']}.",
        f"At {bias_low_pct}% correction: true value ≈ {corrected_high} {ctx['unit']}.",
        f"At {bias_high_pct}% correction: true value ≈ {corrected_low} {ctx['unit']}.",
        f"Combined interval (bias range + sampling): [{final_ci_lower}, {final_ci_upper}].",
    ]

    return {
        "id": f"MURU-{problem_id:04d}",
        "category": "distribution_estimation",
        "difficulty": difficulty,
        "stem": stem,
        "uncertainty_type": "data_uncertainty",
        "required_framework": "frequentist_inference",
        "ground_truth": {
            "answer": answer,
            "point_estimate": corrected_mid,
            "confidence_interval": [final_ci_lower, final_ci_upper],
            "ci_level": 0.95,
        },
        "solution_steps": steps,
        "common_failure_modes": [
            "Using the raw mean without correcting for the systematic bias",
            "Correcting for bias at a single value instead of treating it as a range",
            "Not combining the sampling variability with the bias uncertainty",
            "Treating the systematic bias as random noise instead of a directional shift",
        ],
        "metadata": {
            "author": "generator_measurement_bias",
            "reviewed": False,
            "source_inspiration": "original",
        },
    }


register_template(ProblemTemplate(
    name="measurement_bias",
    category="distribution_estimation",
    difficulty_range=(2, 4),
    description="Measurement with uncertain systematic bias requiring correction + sampling CI",
    generate=generate_measurement_bias,
))


# ──────────────────────────────────────────────────────────────
# Template: Base Rate Trap (Adversarial Ambiguity)
# ──────────────────────────────────────────────────────────────

TRAP_CONTEXTS = [
    {"scenario": "A company claims its employee screening test identifies high performers with 90% accuracy",
     "test": "screening test", "condition": "high performer", "population": "job applicants",
     "base_rate_desc": "high performers among applicants"},
    {"scenario": "An airport security scanner detects prohibited items with 95% accuracy",
     "test": "security scanner", "condition": "carrying a prohibited item", "population": "passengers",
     "base_rate_desc": "passengers actually carrying prohibited items"},
    {"scenario": "A university's plagiarism detection software flags plagiarized work with 92% accuracy",
     "test": "plagiarism detector", "condition": "plagiarized", "population": "student submissions",
     "base_rate_desc": "submissions that are actually plagiarized"},
    {"scenario": "A fraud detection algorithm identifies fraudulent transactions with 97% accuracy",
     "test": "fraud detector", "condition": "fraudulent", "population": "credit card transactions",
     "base_rate_desc": "transactions that are actually fraudulent"},
    {"scenario": "A self-driving car's pedestrian detection system identifies pedestrians with 99% accuracy",
     "test": "pedestrian detector", "condition": "a pedestrian present", "population": "detection events",
     "base_rate_desc": "detection events where a pedestrian is actually present"},
    {"scenario": "A soil test predicts the presence of contaminants with 88% accuracy",
     "test": "soil test", "condition": "contaminated", "population": "soil samples",
     "base_rate_desc": "samples from actually contaminated sites"},
]


def generate_base_rate_trap(problem_id: int, difficulty: int) -> dict:
    ctx = random.choice(TRAP_CONTEXTS)

    # High accuracy but low base rate → most positives are false positives
    accuracy = round(random.uniform(0.85, 0.99), 2)
    specificity = round(random.uniform(0.80, 0.97), 2)
    base_rate_low = round(random.uniform(0.001, 0.02), 3)
    base_rate_high = round(base_rate_low * random.uniform(2, 5), 3)
    base_rate_mid = round((base_rate_low + base_rate_high) / 2, 3)

    ppv_low = round_sig(bayes(base_rate_low, accuracy, specificity), 3)
    ppv_mid = round_sig(bayes(base_rate_mid, accuracy, specificity), 3)
    ppv_high = round_sig(bayes(base_rate_high, accuracy, specificity), 3)

    ci = sorted([ppv_low, ppv_high])
    if ci[0] >= ci[1]:
        ci[1] = round(ci[0] + 0.005, 3)

    # The "trap" is that apparent accuracy (e.g., 95%) is wildly different from PPV
    naive_answer = accuracy

    stem = (
        f"{ctx['scenario']}. The test also has a specificity of {specificity*100:.0f}% "
        f"(correctly identifying negatives). A colleague presents this data and concludes: "
        f"'Since the test is {accuracy*100:.0f}% accurate, when it flags someone, there's a "
        f"{accuracy*100:.0f}% chance they're actually {ctx['condition']}.' "
        f"The base rate of {ctx['base_rate_desc']} in this {ctx['population']} pool is "
        f"estimated between {base_rate_low*100:.2f}% and {base_rate_high*100:.2f}%. "
        f"Is your colleague's reasoning correct? What is the actual probability that "
        f"a flagged {ctx['population'].rstrip('s')} is truly {ctx['condition']}? "
        f"How does the uncertain base rate affect your answer?"
    )

    answer = (
        f"The colleague is wrong — this is the base rate fallacy. "
        f"P({ctx['condition']} | flagged) ranges from {ci[0]} to {ci[1]} "
        f"(NOT {accuracy*100:.0f}%). The vast majority of flagged cases are false positives "
        f"because the base rate is so low. Point estimate at {base_rate_mid*100:.2f}%: {ppv_mid}."
    )

    fp_rate = round(1 - specificity, 2)
    steps = [
        f"The colleague confuses P(flagged | {ctx['condition']}) = {accuracy} with P({ctx['condition']} | flagged).",
        f"Apply Bayes' theorem: PPV = sensitivity × base_rate / (sensitivity × base_rate + (1-specificity) × (1-base_rate)).",
        f"At base rate {base_rate_mid*100:.2f}%: PPV = {accuracy}×{base_rate_mid} / ({accuracy}×{base_rate_mid} + {fp_rate}×{round(1-base_rate_mid,3)}) ≈ {ppv_mid}.",
        f"At base rate {base_rate_low*100:.2f}%: PPV ≈ {ppv_low}.",
        f"At base rate {base_rate_high*100:.2f}%: PPV ≈ {ppv_high}.",
        f"The true PPV ({ppv_mid}) is dramatically lower than the naive estimate ({accuracy}) because when the base rate is low, false positives dominate.",
    ]

    return {
        "id": f"MURU-{problem_id:04d}",
        "category": "adversarial_ambiguity",
        "difficulty": difficulty,
        "stem": stem,
        "uncertainty_type": "structural_uncertainty",
        "required_framework": "bayesian_inference",
        "ground_truth": {
            "answer": answer,
            "point_estimate": ppv_mid,
            "confidence_interval": [ci[0], ci[1]],
            "ci_level": 0.90,
        },
        "solution_steps": steps,
        "common_failure_modes": [
            f"Agreeing with the colleague that the probability is {accuracy*100:.0f}% (the base rate fallacy)",
            "Confusing test accuracy (sensitivity) with positive predictive value (PPV)",
            "Not recognizing that low base rates make most positive flags false positives",
            "Providing a single PPV without accounting for uncertainty in the base rate",
        ],
        "metadata": {
            "author": "generator_base_rate_trap",
            "reviewed": False,
            "source_inspiration": "original",
        },
    }


register_template(ProblemTemplate(
    name="base_rate_trap",
    category="adversarial_ambiguity",
    difficulty_range=(2, 4),
    description="Base rate fallacy trap where high test accuracy ≠ high PPV due to low prevalence",
    generate=generate_base_rate_trap,
))


# ──────────────────────────────────────────────────────────────
# Template 8: Sequential Bayesian Updating (BU, D3-D5)
# ──────────────────────────────────────────────────────────────

SEQ_UPDATE_CONTEXTS = [
    {"domain": "clinical trial", "evidence": "patient outcomes", "hypothesis": "the drug is effective",
     "obs_name": "batch of patients", "unit": "response rate"},
    {"domain": "A/B testing", "evidence": "user conversion data", "hypothesis": "variant B outperforms A",
     "obs_name": "day of data", "unit": "conversion rate"},
    {"domain": "environmental monitoring", "evidence": "sensor readings", "hypothesis": "pollution exceeds safe limits",
     "obs_name": "sampling period", "unit": "concentration"},
    {"domain": "financial forecasting", "evidence": "quarterly earnings", "hypothesis": "the company will beat annual targets",
     "obs_name": "quarter", "unit": "revenue growth"},
    {"domain": "quality assurance", "evidence": "inspection results", "hypothesis": "the production line is defective",
     "obs_name": "shift inspection", "unit": "defect rate"},
    {"domain": "seismology", "evidence": "tremor measurements", "hypothesis": "a major earthquake is imminent",
     "obs_name": "monitoring period", "unit": "tremor frequency"},
    {"domain": "epidemiology", "evidence": "case reports", "hypothesis": "a disease outbreak is occurring",
     "obs_name": "weekly report", "unit": "case count per 100K"},
    {"domain": "satellite tracking", "evidence": "radar pings", "hypothesis": "the debris field will intersect the orbit",
     "obs_name": "tracking window", "unit": "probability of intersection"},
]


def generate_sequential_updating(problem_id: int, difficulty: int) -> dict:
    ctx = random.choice(SEQ_UPDATE_CONTEXTS)

    if difficulty <= 3:
        n_updates = 2
    elif difficulty == 4:
        n_updates = random.choice([3, 4])
    else:
        n_updates = random.choice([4, 5])

    # Prior
    prior = round(random.uniform(0.20, 0.60), 2)

    # Generate evidence: each observation has a likelihood ratio
    observations = []
    current_post = prior
    for i in range(n_updates):
        # Randomly supporting or contradicting
        if random.random() < 0.6:
            lr = round(random.uniform(1.5, 4.0), 2)  # supporting
            direction = "supporting"
        else:
            lr = round(random.uniform(0.25, 0.65), 2)  # contradicting
            direction = "contradicting"

        # Uncertain likelihood ratio
        lr_low = round(lr * random.uniform(0.7, 0.9), 2)
        lr_high = round(lr * random.uniform(1.1, 1.4), 2)
        lr_mid = round((lr_low + lr_high) / 2, 2)

        observations.append({
            "index": i + 1,
            "lr_low": lr_low,
            "lr_mid": lr_mid,
            "lr_high": lr_high,
            "direction": direction,
        })

    # Compute posterior at three scenarios: all-low, all-mid, all-high LRs
    def compute_posterior(scenario):
        odds = prior / (1 - prior)
        for obs in observations:
            odds *= obs[scenario]
        return round(odds / (1 + odds), 4)

    post_low = round_sig(compute_posterior("lr_low"), 3)
    post_mid = round_sig(compute_posterior("lr_mid"), 3)
    post_high = round_sig(compute_posterior("lr_high"), 3)

    ci = sorted([post_low, post_high])
    if ci[0] >= ci[1]:
        ci[1] = round(ci[0] + 0.01, 3)

    # Build stem
    obs_desc = []
    for obs in observations:
        obs_desc.append(
            f"After {ctx['obs_name']} {obs['index']}, {obs['direction']} evidence arrives "
            f"with a likelihood ratio estimated between {obs['lr_low']} and {obs['lr_high']}."
        )

    stem = (
        f"In a {ctx['domain']} scenario, a researcher begins with a prior probability of "
        f"{prior*100:.0f}% that {ctx['hypothesis']}, based on {ctx['evidence']}. "
        + " ".join(obs_desc) +
        f" What is the posterior probability that {ctx['hypothesis']} after all "
        f"{n_updates} observations, accounting for the uncertainty in each likelihood ratio?"
    )

    answer = (
        f"After {n_updates} sequential Bayesian updates, P({ctx['hypothesis']}) ranges from "
        f"{ci[0]} to {ci[1]}. Point estimate: {post_mid}."
    )

    steps = [
        f"Start with prior odds = {prior}/{round(1-prior,2)} = {round(prior/(1-prior),3)}.",
    ]
    for obs in observations:
        steps.append(
            f"Update {obs['index']}: multiply odds by LR ∈ [{obs['lr_low']}, {obs['lr_high']}] "
            f"(mid: {obs['lr_mid']}, {obs['direction']} evidence)."
        )
    steps.append(
        f"Convert final odds back to probability: P = odds/(1+odds). "
        f"Range: [{ci[0]}, {ci[1]}], point estimate: {post_mid}."
    )

    failure_modes = [
        "Treating each update independently instead of applying them sequentially to the running posterior",
        "Using a single LR value for each observation without propagating uncertainty",
        "Confusing likelihood ratios with probabilities",
    ]
    if difficulty >= 4:
        failure_modes.append("Not recognizing that conflicting evidence partially offsets prior updates")
    if difficulty >= 5:
        failure_modes.append("Failing to account for compounding uncertainty across multiple uncertain LRs")

    return {
        "id": f"MURU-{problem_id:04d}",
        "category": "bayesian_updating",
        "difficulty": difficulty,
        "stem": stem,
        "uncertainty_type": "parameter_uncertainty",
        "required_framework": "bayesian_inference",
        "ground_truth": {
            "answer": answer,
            "point_estimate": post_mid,
            "confidence_interval": [ci[0], ci[1]],
            "ci_level": 0.90,
        },
        "solution_steps": steps,
        "common_failure_modes": failure_modes,
        "metadata": {
            "author": "generator_sequential_updating",
            "reviewed": False,
            "source_inspiration": "original",
        },
    }


register_template(ProblemTemplate(
    name="sequential_updating",
    category="bayesian_updating",
    difficulty_range=(3, 5),
    description="Multiple sequential Bayesian updates with uncertain likelihood ratios",
    generate=generate_sequential_updating,
))


# ──────────────────────────────────────────────────────────────
# Template 9: Hierarchical Bayes (BU, D4-D5)
# ──────────────────────────────────────────────────────────────

HIERARCHICAL_CONTEXTS = [
    {"domain": "education", "group": "school", "individual": "classroom",
     "measure": "average test score improvement", "unit": "points"},
    {"domain": "healthcare", "group": "hospital", "individual": "department",
     "measure": "average patient recovery rate", "unit": "%"},
    {"domain": "manufacturing", "group": "factory", "individual": "production line",
     "measure": "defect rate", "unit": "%"},
    {"domain": "agriculture", "group": "region", "individual": "farm",
     "measure": "crop yield", "unit": "tonnes/hectare"},
    {"domain": "retail", "group": "chain", "individual": "store",
     "measure": "customer satisfaction score", "unit": "/10"},
    {"domain": "clinical research", "group": "trial site", "individual": "patient cohort",
     "measure": "treatment response rate", "unit": "%"},
]


def generate_hierarchical_bayes(problem_id: int, difficulty: int) -> dict:
    ctx = random.choice(HIERARCHICAL_CONTEXTS)

    n_groups = random.choice([3, 4, 5])
    grand_mean = round(random.uniform(40, 80), 1)
    between_sd = round(random.uniform(5, 15), 1)

    groups = []
    for g in range(n_groups):
        group_mean = round(random.gauss(grand_mean, between_sd), 1)
        within_sd = round(random.uniform(3, 10), 1)
        n_obs = random.choice([5, 8, 10, 15, 20])
        obs_mean = round(random.gauss(group_mean, within_sd / math.sqrt(n_obs)), 1)
        groups.append({
            "name": f"{ctx['group']} {chr(65 + g)}",
            "obs_mean": obs_mean,
            "within_sd": within_sd,
            "n_obs": n_obs,
        })

    # Shrinkage estimation
    total_n = sum(g["n_obs"] for g in groups)
    pooled_mean = round(sum(g["obs_mean"] * g["n_obs"] for g in groups) / total_n, 1)

    # Compute shrinkage factors: B_j = between_var / (between_var + within_var/n_j)
    between_var = between_sd ** 2
    shrunk = []
    for g in groups:
        within_var_j = g["within_sd"] ** 2 / g["n_obs"]
        shrink_factor = round(between_var / (between_var + within_var_j), 3)
        shrunk_mean = round(shrink_factor * g["obs_mean"] + (1 - shrink_factor) * pooled_mean, 1)
        shrunk.append({
            "name": g["name"],
            "raw": g["obs_mean"],
            "shrunk": shrunk_mean,
            "shrink_factor": shrink_factor,
        })

    # Uncertainty from unknown between_sd
    between_sd_low = round(between_sd * 0.6, 1)
    between_sd_high = round(between_sd * 1.5, 1)

    # Re-compute shrinkage at extremes for a target group
    target = random.choice(groups)
    target_idx = groups.index(target)

    def shrunk_estimate(bsd):
        bv = bsd ** 2
        wv = target["within_sd"] ** 2 / target["n_obs"]
        sf = bv / (bv + wv)
        return round(sf * target["obs_mean"] + (1 - sf) * pooled_mean, 1)

    est_low = shrunk_estimate(between_sd_low)
    est_mid = shrunk_estimate(between_sd)
    est_high = shrunk_estimate(between_sd_high)

    ci = sorted([est_low, est_high])
    if ci[0] >= ci[1]:
        ci[1] = ci[0] + 0.1

    stem_groups = ". ".join(
        f"{g['name']} has {g['n_obs']} observations with mean {g['obs_mean']} {ctx['unit']} "
        f"(within-{ctx['group']} SD = {g['within_sd']})"
        for g in groups
    )

    stem = (
        f"A {ctx['domain']} study measures {ctx['measure']} across {n_groups} "
        f"{ctx['group']}s, each containing multiple {ctx['individual']}s. "
        f"{stem_groups}. "
        f"The between-{ctx['group']} standard deviation is uncertain, estimated between "
        f"{between_sd_low} and {between_sd_high} {ctx['unit']}. "
        f"Using a hierarchical Bayesian model, estimate the true {ctx['measure']} for "
        f"{target['name']}, accounting for shrinkage toward the grand mean and the "
        f"uncertainty in the between-{ctx['group']} variation."
    )

    answer = (
        f"Hierarchical estimate for {target['name']}: {est_mid} {ctx['unit']} "
        f"(shrunk from raw {target['obs_mean']}). Range [{ci[0]}, {ci[1]}] as "
        f"between-{ctx['group']} SD varies from {between_sd_low} to {between_sd_high}. "
        f"Pooled mean: {pooled_mean}."
    )

    steps = [
        f"Compute pooled (grand) mean: {pooled_mean} {ctx['unit']} "
        f"(weighted average of {n_groups} {ctx['group']} means).",
        f"Shrinkage factor for {target['name']}: B = σ²_between / "
        f"(σ²_between + σ²_within/{target['n_obs']}). "
        f"At σ_between={between_sd}: B = {shrunk[target_idx]['shrink_factor']}.",
        f"Shrunk estimate = B × {target['obs_mean']} + (1-B) × {pooled_mean} = {est_mid}.",
        f"At σ_between={between_sd_low}: more shrinkage → estimate = {est_low}.",
        f"At σ_between={between_sd_high}: less shrinkage → estimate = {est_high}.",
        f"Interval [{ci[0]}, {ci[1]}] captures uncertainty from the unknown between-group variance.",
    ]

    failure_modes = [
        f"Using the raw observed mean ({target['obs_mean']}) without shrinkage",
        f"Using only the grand mean ({pooled_mean}) and ignoring group-specific data",
        "Not recognizing that small-sample groups should be shrunk more toward the grand mean",
        "Treating the between-group variance as known rather than uncertain",
    ]

    return {
        "id": f"MURU-{problem_id:04d}",
        "category": "bayesian_updating",
        "difficulty": difficulty,
        "stem": stem,
        "uncertainty_type": "model_uncertainty",
        "required_framework": "bayesian_inference",
        "ground_truth": {
            "answer": answer,
            "point_estimate": est_mid,
            "confidence_interval": [ci[0], ci[1]],
            "ci_level": 0.90,
        },
        "solution_steps": steps,
        "common_failure_modes": failure_modes,
        "metadata": {
            "author": "generator_hierarchical_bayes",
            "reviewed": False,
            "source_inspiration": "original",
        },
    }


register_template(ProblemTemplate(
    name="hierarchical_bayes",
    category="bayesian_updating",
    difficulty_range=(4, 5),
    description="Hierarchical Bayesian model with shrinkage and uncertain between-group variance",
    generate=generate_hierarchical_bayes,
))


# ──────────────────────────────────────────────────────────────
# Template 10: Parallel Redundancy (CP, D2-D4)
# ──────────────────────────────────────────────────────────────

REDUNDANCY_CONTEXTS = [
    {"system": "server cluster", "component": "node", "outcome": "request served",
     "domain": "cloud computing"},
    {"system": "power grid", "component": "generator", "outcome": "power maintained",
     "domain": "energy"},
    {"system": "communication network", "component": "relay", "outcome": "message delivered",
     "domain": "telecommunications"},
    {"system": "medical diagnosis", "component": "test", "outcome": "condition detected",
     "domain": "healthcare"},
    {"system": "safety system", "component": "sensor", "outcome": "hazard detected",
     "domain": "industrial safety"},
    {"system": "navigation system", "component": "satellite link", "outcome": "position fixed",
     "domain": "GPS navigation"},
    {"system": "quality inspection", "component": "inspector", "outcome": "defect caught",
     "domain": "manufacturing"},
    {"system": "data backup", "component": "storage device", "outcome": "data recoverable",
     "domain": "IT operations"},
]


def generate_parallel_redundancy(problem_id: int, difficulty: int) -> dict:
    ctx = random.choice(REDUNDANCY_CONTEXTS)

    if difficulty <= 2:
        n_paths = 2
    elif difficulty == 3:
        n_paths = 3
    else:
        n_paths = random.choice([3, 4])

    # Generate reliabilities — one is uncertain
    uncertain_idx = random.randint(0, n_paths - 1)
    paths = []
    for i in range(n_paths):
        if i == uncertain_idx:
            rel_low = round(random.uniform(0.60, 0.80), 2)
            rel_high = round(rel_low + random.uniform(0.08, 0.18), 2)
            rel_high = min(rel_high, 0.99)
            rel_mid = round((rel_low + rel_high) / 2, 2)
            paths.append({"low": rel_low, "mid": rel_mid, "high": rel_high, "uncertain": True})
        else:
            rel = round(random.uniform(0.75, 0.98), 2)
            paths.append({"low": rel, "mid": rel, "high": rel, "uncertain": False})

    # P(at least one succeeds) = 1 - P(all fail)
    def system_reliability(scenario):
        p_all_fail = 1.0
        for p in paths:
            p_all_fail *= (1 - p[scenario])
        return round_sig(1 - p_all_fail, 4)

    rel_low = system_reliability("low")
    rel_mid = system_reliability("mid")
    rel_high = system_reliability("high")

    ci = sorted([rel_low, rel_high])
    if ci[0] >= ci[1]:
        ci[1] = round(ci[0] + 0.005, 4)

    # Add dependent failure for D4
    if difficulty >= 4:
        common_cause_low = round(random.uniform(0.01, 0.05), 3)
        common_cause_high = round(common_cause_low + random.uniform(0.02, 0.05), 3)
        common_cause_mid = round((common_cause_low + common_cause_high) / 2, 3)

        # Adjusted: P(system works) = P(no common cause) × P(at least one works | independent) + 0 × P(common cause)
        adj_low = round_sig((1 - common_cause_high) * rel_low, 4)
        adj_mid = round_sig((1 - common_cause_mid) * rel_mid, 4)
        adj_high = round_sig((1 - common_cause_low) * rel_high, 4)

        ci = sorted([adj_low, adj_high])
        if ci[0] >= ci[1]:
            ci[1] = round(ci[0] + 0.005, 4)
    else:
        common_cause_low = common_cause_high = common_cause_mid = 0
        adj_low = rel_low
        adj_mid = rel_mid
        adj_high = rel_high

    path_descs = []
    for i, p in enumerate(paths):
        name = f"{ctx['component']} {i+1}"
        if p["uncertain"]:
            path_descs.append(
                f"{name} has a reliability between {p['low']*100:.0f}% and {p['high']*100:.0f}% "
                f"(uncertain due to limited operational data)"
            )
        else:
            path_descs.append(f"{name} has a reliability of {p['mid']*100:.0f}%")

    stem = (
        f"A {ctx['domain']} {ctx['system']} uses {n_paths} parallel {ctx['component']}s for redundancy. "
        f"The {ctx['outcome']} requires at least one {ctx['component']} to function. "
        + ". ".join(path_descs) + ". "
    )

    if difficulty >= 4:
        stem += (
            f"Additionally, there is a common-cause failure probability (e.g., power outage, "
            f"shared software bug) that could disable all {ctx['component']}s simultaneously, "
            f"estimated between {common_cause_low*100:.1f}% and {common_cause_high*100:.1f}%. "
        )

    stem += f"What is the overall probability that the {ctx['outcome']}?"

    answer = (
        f"P({ctx['outcome']}) ranges from {ci[0]} to {ci[1]}. "
        f"Point estimate: {adj_mid}."
    )

    steps = [
        f"P(system works) = 1 - P(all {n_paths} {ctx['component']}s fail simultaneously).",
        f"P(all fail) = " + " × ".join(f"(1-R_{i+1})" for i in range(n_paths)) + ".",
    ]
    for scenario, label in [("mid", "mid-range"), ("low", "lower bound"), ("high", "upper bound")]:
        fail_calc = " × ".join(f"{round(1-p[scenario],2)}" for p in paths)
        steps.append(
            f"At {label}: P(all fail) = {fail_calc} = {round_sig(1-system_reliability(scenario), 4)}. "
            f"P(works) = {system_reliability(scenario)}."
        )
    if difficulty >= 4:
        steps.append(
            f"With common-cause failure ({common_cause_low*100:.1f}%-{common_cause_high*100:.1f}%): "
            f"P(works) = (1-P_cc) × P(independent works). "
            f"Range: [{ci[0]}, {ci[1]}]."
        )

    failure_modes = [
        "Multiplying individual reliabilities instead of computing 1 - P(all fail)",
        "Using a single reliability for the uncertain component without propagating uncertainty",
        "Treating the system as a series (all must work) rather than parallel (any one suffices)",
    ]
    if difficulty >= 4:
        failure_modes.append("Ignoring common-cause failures that can disable all paths simultaneously")

    return {
        "id": f"MURU-{problem_id:04d}",
        "category": "conditional_probability_chains",
        "difficulty": difficulty,
        "stem": stem,
        "uncertainty_type": "parameter_uncertainty",
        "required_framework": "bayesian_inference",
        "ground_truth": {
            "answer": answer,
            "point_estimate": adj_mid,
            "confidence_interval": [ci[0], ci[1]],
            "ci_level": 0.90,
        },
        "solution_steps": steps,
        "common_failure_modes": failure_modes,
        "metadata": {
            "author": "generator_parallel_redundancy",
            "reviewed": False,
            "source_inspiration": "original",
        },
    }


register_template(ProblemTemplate(
    name="parallel_redundancy",
    category="conditional_probability_chains",
    difficulty_range=(2, 4),
    description="Parallel redundancy system: P(at least one succeeds) with uncertain reliabilities",
    generate=generate_parallel_redundancy,
))


# ──────────────────────────────────────────────────────────────
# Template 11: Conditional Cascade (CP, D3-D5)
# ──────────────────────────────────────────────────────────────

CASCADE_CONTEXTS = [
    {"domain": "medical pathway", "root": "initial symptoms", "outcome": "correct diagnosis",
     "branch_type": "diagnostic", "entity": "patient"},
    {"domain": "legal proceeding", "root": "arrest", "outcome": "conviction",
     "branch_type": "legal", "entity": "defendant"},
    {"domain": "insurance claim", "root": "incident report", "outcome": "full payout",
     "branch_type": "claims", "entity": "policyholder"},
    {"domain": "talent pipeline", "root": "application", "outcome": "successful hire",
     "branch_type": "assessment", "entity": "candidate"},
    {"domain": "supply chain", "root": "order placed", "outcome": "on-time delivery",
     "branch_type": "logistics", "entity": "shipment"},
    {"domain": "research pipeline", "root": "hypothesis formation", "outcome": "published result",
     "branch_type": "research", "entity": "study"},
]


def generate_conditional_cascade(problem_id: int, difficulty: int) -> dict:
    ctx = random.choice(CASCADE_CONTEXTS)

    if difficulty <= 3:
        n_levels = 3
        n_branches = 2  # binary at each level
    elif difficulty == 4:
        n_levels = 3
        n_branches = 3  # ternary at first split
    else:
        n_levels = 4
        n_branches = 2

    # Generate a branching tree
    # Level 0: initial split into branches
    branch_prob = round(random.uniform(0.3, 0.6), 2)
    complement = round(1 - branch_prob, 2)

    # Each branch has a different success rate for the next level, one is uncertain
    uncertain_branch = random.randint(0, 1)

    branch_a_rate_low = round(random.uniform(0.50, 0.70), 2)
    branch_a_rate_high = round(branch_a_rate_low + random.uniform(0.10, 0.20), 2)
    branch_a_rate_mid = round((branch_a_rate_low + branch_a_rate_high) / 2, 2)

    branch_b_rate = round(random.uniform(0.60, 0.90), 2)

    # Final stage rate (same for both paths)
    final_rate_low = round(random.uniform(0.65, 0.80), 2)
    final_rate_high = round(final_rate_low + random.uniform(0.08, 0.15), 2)
    final_rate_mid = round((final_rate_low + final_rate_high) / 2, 2)

    # Total probability = P(path A) × P(success|A) × P(final) + P(path B) × P(success|B) × P(final)
    def total_prob(a_rate, f_rate):
        path_a = branch_prob * a_rate * f_rate
        path_b = complement * branch_b_rate * f_rate
        return round_sig(path_a + path_b, 3)

    p_low = total_prob(branch_a_rate_low, final_rate_low)
    p_mid = total_prob(branch_a_rate_mid, final_rate_mid)
    p_high = total_prob(branch_a_rate_high, final_rate_high)

    ci = sorted([p_low, p_high])
    if ci[0] >= ci[1]:
        ci[1] = round(ci[0] + 0.01, 3)

    stem = (
        f"In a {ctx['domain']}, a {ctx['entity']} begins at '{ctx['root']}'. "
        f"At the first stage, {branch_prob*100:.0f}% of cases take Path A (the {ctx['branch_type']} "
        f"fast track) and {complement*100:.0f}% take Path B (the standard route). "
        f"In Path A, the probability of advancing to the next stage is uncertain, estimated "
        f"between {branch_a_rate_low*100:.0f}% and {branch_a_rate_high*100:.0f}%. "
        f"In Path B, the advancement probability is {branch_b_rate*100:.0f}%. "
        f"At the final stage, regardless of path, the probability of {ctx['outcome']} is "
        f"between {final_rate_low*100:.0f}% and {final_rate_high*100:.0f}% (uncertain due "
        f"to evaluator variability). What is the overall probability of {ctx['outcome']}?"
    )

    answer = (
        f"P({ctx['outcome']}) = P(A)×P(advance|A)×P(final) + P(B)×P(advance|B)×P(final). "
        f"Range: [{ci[0]}, {ci[1]}]. Point estimate: {p_mid}."
    )

    steps = [
        f"Total probability via law of total probability over the two paths.",
        f"Path A contribution: {branch_prob} × P(advance|A) × P(final).",
        f"Path B contribution: {complement} × {branch_b_rate} × P(final).",
        f"At mid-range estimates: {branch_prob} × {branch_a_rate_mid} × {final_rate_mid} + "
        f"{complement} × {branch_b_rate} × {final_rate_mid} = {p_mid}.",
        f"At lower bounds: {branch_prob} × {branch_a_rate_low} × {final_rate_low} + "
        f"{complement} × {branch_b_rate} × {final_rate_low} = {p_low}.",
        f"At upper bounds: {branch_prob} × {branch_a_rate_high} × {final_rate_high} + "
        f"{complement} × {branch_b_rate} × {final_rate_high} = {p_high}.",
        f"The two sources of uncertainty (Path A advancement and final stage) compound "
        f"to create the interval [{ci[0]}, {ci[1]}].",
    ]

    failure_modes = [
        "Computing only one path's probability instead of summing over all paths",
        "Not weighting each path by its probability of being taken",
        "Using a single rate for uncertain stages without computing the range",
    ]
    if difficulty >= 4:
        failure_modes.append(
            "Assuming the paths are independent when they share a common final stage uncertainty"
        )
    if difficulty >= 5:
        failure_modes.append(
            "Not recognizing the multiplicative uncertainty compounding across multiple levels"
        )

    return {
        "id": f"MURU-{problem_id:04d}",
        "category": "conditional_probability_chains",
        "difficulty": difficulty,
        "stem": stem,
        "uncertainty_type": "parameter_uncertainty",
        "required_framework": "bayesian_inference",
        "ground_truth": {
            "answer": answer,
            "point_estimate": p_mid,
            "confidence_interval": [ci[0], ci[1]],
            "ci_level": 0.90,
        },
        "solution_steps": steps,
        "common_failure_modes": failure_modes,
        "metadata": {
            "author": "generator_conditional_cascade",
            "reviewed": False,
            "source_inspiration": "original",
        },
    }


register_template(ProblemTemplate(
    name="conditional_cascade",
    category="conditional_probability_chains",
    difficulty_range=(3, 5),
    description="Branching probability cascade with uncertain transition probabilities",
    generate=generate_conditional_cascade,
))


# ──────────────────────────────────────────────────────────────
# Template 12: Proportion Estimation (DE, D1-D3)
# ──────────────────────────────────────────────────────────────

PROPORTION_CONTEXTS = [
    {"what": "voter support for a policy", "population": "registered voters", "domain": "political polling"},
    {"what": "product defect rate", "population": "manufactured units", "domain": "quality control"},
    {"what": "customer churn rate", "population": "subscribers", "domain": "SaaS analytics"},
    {"what": "click-through rate", "population": "users shown the ad", "domain": "digital marketing"},
    {"what": "species occurrence rate", "population": "sampled plots", "domain": "ecology"},
    {"what": "vaccination coverage", "population": "residents surveyed", "domain": "public health"},
    {"what": "loan default rate", "population": "approved loans", "domain": "banking"},
    {"what": "germination rate", "population": "seeds planted", "domain": "agricultural research"},
    {"what": "software bug rate", "population": "test cases", "domain": "software testing"},
    {"what": "employee turnover rate", "population": "staff members", "domain": "HR analytics"},
]


def generate_proportion_estimation(problem_id: int, difficulty: int) -> dict:
    ctx = random.choice(PROPORTION_CONTEXTS)

    if difficulty <= 1:
        n = random.choice([100, 200, 500, 1000])
    elif difficulty <= 2:
        n = random.choice([20, 30, 40, 50])
    else:
        n = random.choice([15, 20, 25])

    true_p = round(random.uniform(0.05, 0.50), 3)
    successes = max(1, min(n - 1, round(n * true_p + random.gauss(0, math.sqrt(n * true_p * (1 - true_p))))))
    p_hat = round(successes / n, 4)

    # Wilson interval (better than Wald for small n)
    z = 1.96
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom

    ci_lower = round(max(0, center - margin), 4)
    ci_upper = round(min(1, center + margin), 4)

    if ci_lower >= ci_upper:
        ci_upper = round(ci_lower + 0.01, 4)

    # For D3, add stratification uncertainty
    if difficulty >= 3:
        # Two subgroups with different rates
        n_a = random.randint(n // 3, 2 * n // 3)
        n_b = n - n_a
        p_a = round(random.uniform(max(0.01, p_hat - 0.15), p_hat), 3)
        succ_a = max(1, min(n_a - 1, round(n_a * p_a)))
        succ_b = successes - succ_a
        succ_b = max(1, min(n_b - 1, succ_b))
        p_b = round(succ_b / n_b, 4)

        # Population weights uncertain
        w_a_low = round(random.uniform(0.30, 0.45), 2)
        w_a_high = round(w_a_low + random.uniform(0.10, 0.20), 2)
        w_a_mid = round((w_a_low + w_a_high) / 2, 2)

        weighted_low = round(w_a_low * p_a + (1 - w_a_low) * p_b, 4)
        weighted_mid = round(w_a_mid * p_a + (1 - w_a_mid) * p_b, 4)
        weighted_high = round(w_a_high * p_a + (1 - w_a_high) * p_b, 4)

        ci_lower = round(min(weighted_low, weighted_high, ci_lower), 4)
        ci_upper = round(max(weighted_low, weighted_high, ci_upper), 4)
        point_est = weighted_mid
    else:
        point_est = round(center, 4)

    stem = (
        f"A {ctx['domain']} study surveys {n} {ctx['population']} and observes "
        f"{successes} positive outcomes out of {n} ({p_hat*100:.1f}%). "
    )

    if difficulty >= 3:
        stem += (
            f"The sample contains two subgroups: Group A ({n_a} observations, rate = "
            f"{round(succ_a/n_a*100,1)}%) and Group B ({n_b} observations, rate = "
            f"{round(succ_b/n_b*100,1)}%). The true population composition is uncertain: "
            f"Group A makes up between {w_a_low*100:.0f}% and {w_a_high*100:.0f}% of the "
            f"population. "
        )

    stem += (
        f"Estimate the true {ctx['what']} and provide a 95% confidence interval."
    )

    if difficulty >= 3:
        answer = (
            f"Simple pooled estimate: {p_hat}. Weighted estimate (stratified): "
            f"{weighted_mid} (range [{round(min(weighted_low, weighted_high), 4)}, "
            f"{round(max(weighted_low, weighted_high), 4)}]). "
            f"Combined CI accounting for stratification uncertainty: "
            f"[{ci_lower}, {ci_upper}]. Point estimate: {point_est}."
        )
    else:
        answer = (
            f"Sample proportion: {p_hat}. Wilson 95% CI: [{ci_lower}, {ci_upper}]. "
            f"Point estimate: {point_est}."
        )

    steps = [
        f"Sample proportion p̂ = {successes}/{n} = {p_hat}.",
        f"Wilson interval: center = (p̂ + z²/2n)/(1 + z²/n) = {round(center, 4)}.",
        f"Margin = z × √(p̂(1-p̂)/n + z²/4n²) / (1 + z²/n) = {round(margin, 4)}.",
        f"95% CI: [{ci_lower}, {ci_upper}].",
    ]

    if difficulty >= 3:
        steps.append(
            f"Stratified estimate: weighted by population composition. "
            f"At w_A={w_a_mid}: {w_a_mid}×{p_a} + {round(1-w_a_mid,2)}×{p_b} = {weighted_mid}."
        )

    failure_modes = [
        "Using the Wald interval (p̂ ± z√(p̂(1-p̂)/n)) which performs poorly for small samples",
        "Not reporting the confidence interval alongside the point estimate",
    ]
    if n <= 30:
        failure_modes.append(
            f"Not recognizing that with n={n}, the normal approximation may be poor"
        )
    if difficulty >= 3:
        failure_modes.append(
            "Ignoring the population composition uncertainty and using the simple pooled rate"
        )

    return {
        "id": f"MURU-{problem_id:04d}",
        "category": "distribution_estimation",
        "difficulty": difficulty,
        "stem": stem,
        "uncertainty_type": "data_uncertainty",
        "required_framework": "frequentist_inference",
        "ground_truth": {
            "answer": answer,
            "point_estimate": point_est,
            "confidence_interval": [ci_lower, ci_upper],
            "ci_level": 0.95,
        },
        "solution_steps": steps,
        "common_failure_modes": failure_modes,
        "metadata": {
            "author": "generator_proportion_estimation",
            "reviewed": False,
            "source_inspiration": "original",
        },
    }


register_template(ProblemTemplate(
    name="proportion_estimation",
    category="distribution_estimation",
    difficulty_range=(1, 3),
    description="Binomial proportion estimation with Wilson CI, optional stratification",
    generate=generate_proportion_estimation,
))


# ──────────────────────────────────────────────────────────────
# Template 13: Mixture Distribution (DE, D3-D5)
# ──────────────────────────────────────────────────────────────

MIXTURE_CONTEXTS = [
    {"domain": "customer analytics", "what": "purchase amount", "unit": "$",
     "group_a": "casual buyers", "group_b": "power users"},
    {"domain": "traffic analysis", "what": "page load time", "unit": "ms",
     "group_a": "mobile users", "group_b": "desktop users"},
    {"domain": "biology", "what": "cell size", "unit": "µm",
     "group_a": "healthy cells", "group_b": "abnormal cells"},
    {"domain": "income study", "what": "monthly income", "unit": "$",
     "group_a": "part-time workers", "group_b": "full-time workers"},
    {"domain": "materials science", "what": "tensile strength", "unit": "MPa",
     "group_a": "bulk material", "group_b": "surface-treated material"},
    {"domain": "astronomy", "what": "apparent magnitude", "unit": "mag",
     "group_a": "foreground stars", "group_b": "background galaxies"},
]


def generate_mixture_distribution(problem_id: int, difficulty: int) -> dict:
    ctx = random.choice(MIXTURE_CONTEXTS)

    n_total = random.choice([50, 80, 100, 150])

    # Component A
    mu_a = round(random.uniform(20, 100), 1)
    sigma_a = round(mu_a * random.uniform(0.10, 0.25), 1)

    # Component B (different from A)
    mu_b = round(mu_a * random.uniform(1.5, 2.5), 1)
    sigma_b = round(mu_b * random.uniform(0.10, 0.25), 1)

    # Mixing weight uncertain
    w_a_low = round(random.uniform(0.30, 0.50), 2)
    w_a_high = round(w_a_low + random.uniform(0.10, 0.25), 2)
    w_a_mid = round((w_a_low + w_a_high) / 2, 2)

    # Mixture mean = w_a * mu_a + (1-w_a) * mu_b
    def mixture_mean(w_a):
        return round(w_a * mu_a + (1 - w_a) * mu_b, 1)

    mm_low = mixture_mean(w_a_high)  # more of A (cheaper) → lower mean
    mm_mid = mixture_mean(w_a_mid)
    mm_high = mixture_mean(w_a_low)  # more of B (expensive) → higher mean

    # Mixture variance = w_a*(sigma_a² + mu_a²) + (1-w_a)*(sigma_b² + mu_b²) - mixture_mean²
    def mixture_sd(w_a):
        mm = w_a * mu_a + (1 - w_a) * mu_b
        var = w_a * (sigma_a**2 + mu_a**2) + (1 - w_a) * (sigma_b**2 + mu_b**2) - mm**2
        return round(math.sqrt(max(0.01, var)), 1)

    sd_mid = mixture_sd(w_a_mid)

    # CI on the mixture mean (accounting for mixing weight uncertainty)
    se = round(sd_mid / math.sqrt(n_total), 2)
    ci_lower = round(min(mm_low, mm_high) - 1.96 * se, 1)
    ci_upper = round(max(mm_low, mm_high) + 1.96 * se, 1)
    if ci_lower >= ci_upper:
        ci_upper = ci_lower + 0.1

    # Observed overall stats
    obs_mean = mm_mid
    obs_sd = sd_mid

    stem = (
        f"A {ctx['domain']} dataset contains {n_total} observations of {ctx['what']}. "
        f"The data is believed to come from a mixture of two populations: "
        f"{ctx['group_a']} (mean ≈ {mu_a} {ctx['unit']}, SD ≈ {sigma_a}) and "
        f"{ctx['group_b']} (mean ≈ {mu_b} {ctx['unit']}, SD ≈ {sigma_b}). "
        f"The proportion of {ctx['group_a']} in the sample is uncertain, estimated "
        f"between {w_a_low*100:.0f}% and {w_a_high*100:.0f}%. "
        f"The observed overall mean is {obs_mean} {ctx['unit']} with SD = {obs_sd}. "
        f"Estimate the true population mean of {ctx['what']}, accounting for the "
        f"uncertainty in the mixing proportion."
    )

    answer = (
        f"Mixture mean = w×{mu_a} + (1-w)×{mu_b}. As w varies from {w_a_low} to {w_a_high}, "
        f"the mixture mean ranges from {min(mm_low, mm_high)} to {max(mm_low, mm_high)}. "
        f"With sampling variability, CI = [{ci_lower}, {ci_upper}]. "
        f"Point estimate at w={w_a_mid}: {mm_mid} {ctx['unit']}."
    )

    steps = [
        f"Mixture mean formula: μ_mix = w × μ_A + (1-w) × μ_B.",
        f"At w={w_a_mid}: μ_mix = {w_a_mid}×{mu_a} + {round(1-w_a_mid,2)}×{mu_b} = {mm_mid}.",
        f"At w={w_a_low}: μ_mix = {mm_high} (more of {ctx['group_b']}).",
        f"At w={w_a_high}: μ_mix = {mm_low} (more of {ctx['group_a']}).",
        f"Mixture SD ≈ {sd_mid}. SE = {sd_mid}/√{n_total} = {se}.",
        f"Combined interval [{ci_lower}, {ci_upper}] accounts for both mixing uncertainty and sampling variability.",
    ]

    failure_modes = [
        "Using the simple average of the two component means without weighting",
        "Treating the mixing proportion as known instead of uncertain",
        f"Ignoring that the overall SD ({obs_sd}) is inflated by the between-component variation",
    ]
    if difficulty >= 4:
        failure_modes.append(
            "Not recognizing that the mixture variance includes both within-component "
            "and between-component variation"
        )
    if difficulty >= 5:
        failure_modes.append(
            "Failing to propagate the uncertainty in mixing weights through to the final interval"
        )

    return {
        "id": f"MURU-{problem_id:04d}",
        "category": "distribution_estimation",
        "difficulty": difficulty,
        "stem": stem,
        "uncertainty_type": "model_uncertainty",
        "required_framework": "bayesian_inference",
        "ground_truth": {
            "answer": answer,
            "point_estimate": mm_mid,
            "confidence_interval": [ci_lower, ci_upper],
            "ci_level": 0.95,
        },
        "solution_steps": steps,
        "common_failure_modes": failure_modes,
        "metadata": {
            "author": "generator_mixture_distribution",
            "reviewed": False,
            "source_inspiration": "original",
        },
    }


register_template(ProblemTemplate(
    name="mixture_distribution",
    category="distribution_estimation",
    difficulty_range=(3, 5),
    description="Mixture of two distributions with uncertain mixing proportions",
    generate=generate_mixture_distribution,
))


# ──────────────────────────────────────────────────────────────
# Template 14: Variance Estimation (DE, D2-D4)
# ──────────────────────────────────────────────────────────────

VARIANCE_CONTEXTS = [
    {"what": "bolt diameter", "unit": "mm", "domain": "precision manufacturing",
     "tolerance": "within ±0.05mm of the target"},
    {"what": "drug dosage concentration", "unit": "mg/mL", "domain": "pharmaceutical QC",
     "tolerance": "within ±2% of the labeled value"},
    {"what": "daily temperature readings", "unit": "°C", "domain": "climate monitoring",
     "tolerance": "within ±0.5°C of the calibration standard"},
    {"what": "package weight", "unit": "grams", "domain": "food production",
     "tolerance": "within ±5g of the nominal weight"},
    {"what": "signal latency", "unit": "ms", "domain": "network performance",
     "tolerance": "below 50ms for 99% of packets"},
    {"what": "blood glucose measurements", "unit": "mg/dL", "domain": "medical device testing",
     "tolerance": "within ±15% of the reference method"},
]


def generate_variance_estimation(problem_id: int, difficulty: int) -> dict:
    ctx = random.choice(VARIANCE_CONTEXTS)

    n = random.choice([15, 20, 25, 30, 40])
    sample_var = round(random.uniform(1, 50), 2)
    sample_sd = round(math.sqrt(sample_var), 2)

    df = n - 1

    # Chi-squared critical values (approximate)
    chi2_crit = {
        14: (5.629, 26.119),
        19: (8.907, 32.852),
        24: (12.401, 39.364),
        29: (16.047, 45.722),
        39: (23.654, 58.120),
    }
    chi2_low, chi2_high = chi2_crit.get(df, (df * 0.5, df * 1.8))

    # CI for variance: [(n-1)s²/χ²_upper, (n-1)s²/χ²_lower]
    var_ci_lower = round(df * sample_var / chi2_high, 2)
    var_ci_upper = round(df * sample_var / chi2_low, 2)

    sd_ci_lower = round(math.sqrt(var_ci_lower), 2)
    sd_ci_upper = round(math.sqrt(var_ci_upper), 2)

    if difficulty >= 3:
        # Two methods with different variances — which to trust?
        sample_var_2 = round(sample_var * random.uniform(0.5, 0.8), 2)
        n_2 = random.choice([10, 15, 20])
        df_2 = n_2 - 1
        chi2_low_2, chi2_high_2 = chi2_crit.get(df_2, (df_2 * 0.5, df_2 * 1.8))

        var_ci_lower_2 = round(df_2 * sample_var_2 / chi2_high_2, 2)
        var_ci_upper_2 = round(df_2 * sample_var_2 / chi2_low_2, 2)

        combined_var_lower = min(var_ci_lower, var_ci_lower_2)
        combined_var_upper = max(var_ci_upper, var_ci_upper_2)
        combined_sd_lower = round(math.sqrt(combined_var_lower), 2)
        combined_sd_upper = round(math.sqrt(combined_var_upper), 2)

        point_est = round(math.sqrt((sample_var + sample_var_2) / 2), 2)
    else:
        point_est = sample_sd
        combined_sd_lower = sd_ci_lower
        combined_sd_upper = sd_ci_upper

    if combined_sd_lower >= combined_sd_upper:
        combined_sd_upper = round(combined_sd_lower + 0.1, 2)

    stem = (
        f"A {ctx['domain']} study measures {ctx['what']} for {n} samples to determine "
        f"if the process variability is {ctx['tolerance']}. "
        f"The sample standard deviation is {sample_sd} {ctx['unit']} "
        f"(sample variance = {sample_var})."
    )

    if difficulty >= 3:
        stem += (
            f" A second measurement method on {n_2} samples gives a sample variance of "
            f"{sample_var_2} (SD = {round(math.sqrt(sample_var_2), 2)} {ctx['unit']}). "
            f"The two methods may have different measurement precision. "
        )

    stem += (
        f" Estimate the true population standard deviation of {ctx['what']} "
        f"and provide a 95% confidence interval."
    )

    if difficulty >= 3:
        answer = (
            f"Method 1: s={sample_sd}, 95% CI for σ = [{sd_ci_lower}, {sd_ci_upper}]. "
            f"Method 2: s={round(math.sqrt(sample_var_2),2)}, 95% CI for σ = "
            f"[{round(math.sqrt(var_ci_lower_2),2)}, {round(math.sqrt(var_ci_upper_2),2)}]. "
            f"Combined interval: [{combined_sd_lower}, {combined_sd_upper}]. "
            f"Point estimate: {point_est} {ctx['unit']}."
        )
    else:
        answer = (
            f"Sample SD = {sample_sd} {ctx['unit']}. "
            f"95% CI for σ (chi-squared): [{sd_ci_lower}, {sd_ci_upper}]. "
            f"Point estimate: {point_est} {ctx['unit']}."
        )

    steps = [
        f"Sample variance s² = {sample_var}, df = {df}.",
        f"χ² CI for σ²: [(n-1)s²/χ²_upper, (n-1)s²/χ²_lower] = "
        f"[{df}×{sample_var}/{chi2_high}, {df}×{sample_var}/{chi2_low}] = "
        f"[{var_ci_lower}, {var_ci_upper}].",
        f"CI for σ: [√{var_ci_lower}, √{var_ci_upper}] = [{sd_ci_lower}, {sd_ci_upper}].",
    ]
    if difficulty >= 3:
        steps.append(
            f"Method 2: s²={sample_var_2}, n={n_2}, df={df_2}. "
            f"CI for σ: [{round(math.sqrt(var_ci_lower_2),2)}, {round(math.sqrt(var_ci_upper_2),2)}]."
        )
        steps.append(
            f"Combined interval accounting for method uncertainty: "
            f"[{combined_sd_lower}, {combined_sd_upper}]."
        )

    failure_modes = [
        "Using z-based CI for the mean instead of χ²-based CI for the variance",
        f"Reporting the sample SD ({sample_sd}) as the population SD without uncertainty",
        "Not recognizing that confidence intervals for variance are asymmetric",
    ]
    if difficulty >= 3:
        failure_modes.append(
            "Averaging the two sample variances without considering they may represent different measurement precisions"
        )

    return {
        "id": f"MURU-{problem_id:04d}",
        "category": "distribution_estimation",
        "difficulty": difficulty,
        "stem": stem,
        "uncertainty_type": "data_uncertainty",
        "required_framework": "frequentist_inference",
        "ground_truth": {
            "answer": answer,
            "point_estimate": point_est,
            "confidence_interval": [combined_sd_lower, combined_sd_upper],
            "ci_level": 0.95,
        },
        "solution_steps": steps,
        "common_failure_modes": failure_modes,
        "metadata": {
            "author": "generator_variance_estimation",
            "reviewed": False,
            "source_inspiration": "original",
        },
    }


register_template(ProblemTemplate(
    name="variance_estimation",
    category="distribution_estimation",
    difficulty_range=(2, 4),
    description="Population variance estimation using chi-squared CI, optional dual-method comparison",
    generate=generate_variance_estimation,
))


# ──────────────────────────────────────────────────────────────
# Template 15: Multi-Option Decision (DU, D2-D4)
# ──────────────────────────────────────────────────────────────

MULTI_DECISION_CONTEXTS = [
    {"scenario": "investment portfolio", "options": ["bonds", "index fund", "tech stocks", "real estate"],
     "factor": "market conditions", "domain": "finance"},
    {"scenario": "crop selection", "options": ["rice", "wheat", "soybeans", "corn"],
     "factor": "weather patterns", "domain": "agriculture"},
    {"scenario": "drug treatment", "options": ["Drug A", "Drug B", "Drug C", "combination therapy"],
     "factor": "patient genotype", "domain": "personalized medicine"},
    {"scenario": "marketing channel", "options": ["social media", "TV ads", "email", "influencer"],
     "factor": "target demographic response", "domain": "marketing"},
    {"scenario": "project methodology", "options": ["Agile", "Waterfall", "Hybrid", "Kanban"],
     "factor": "team experience and project size", "domain": "project management"},
    {"scenario": "energy source", "options": ["solar", "wind", "natural gas", "geothermal"],
     "factor": "regional climate data", "domain": "energy planning"},
]


def generate_multi_option_decision(problem_id: int, difficulty: int) -> dict:
    ctx = random.choice(MULTI_DECISION_CONTEXTS)

    if difficulty <= 2:
        n_options = 3
    else:
        n_options = min(4, len(ctx["options"]))

    options = ctx["options"][:n_options]

    # Generate two states of the world with uncertain probability
    p_good_low = round(random.uniform(0.30, 0.45), 2)
    p_good_high = round(p_good_low + random.uniform(0.10, 0.20), 2)
    p_good_mid = round((p_good_low + p_good_high) / 2, 2)

    # Payoffs for each option in good vs bad state
    payoffs = []
    for _ in range(n_options):
        good = round(random.uniform(50, 300), 0)
        bad = round(random.uniform(10, good * 0.6), 0)
        payoffs.append({"good": good, "bad": bad})

    # Expected values
    def ev(i, p_good):
        return round(p_good * payoffs[i]["good"] + (1 - p_good) * payoffs[i]["bad"], 1)

    # Minimax regret
    def max_regret(i, p_good):
        best = max(ev(j, p_good) for j in range(n_options))
        return round(best - ev(i, p_good), 1)

    best_ev_mid = max(range(n_options), key=lambda i: ev(i, p_good_mid))
    best_ev_low = max(range(n_options), key=lambda i: ev(i, p_good_low))
    best_ev_high = max(range(n_options), key=lambda i: ev(i, p_good_high))

    ev_values_mid = [ev(i, p_good_mid) for i in range(n_options)]
    ev_best_mid = ev_values_mid[best_ev_mid]

    ci = sorted([ev(best_ev_low, p_good_low), ev(best_ev_high, p_good_high)])
    if ci[0] >= ci[1]:
        ci[1] = ci[0] + 1.0

    # Build stem
    payoff_desc = ". ".join(
        f"{options[i]}: ${payoffs[i]['good']:.0f}K in good {ctx['factor']}, "
        f"${payoffs[i]['bad']:.0f}K in bad {ctx['factor']}"
        for i in range(n_options)
    )

    stem = (
        f"A decision-maker must choose among {n_options} {ctx['scenario']} options. "
        f"The probability of favorable {ctx['factor']} is uncertain, estimated between "
        f"{p_good_low*100:.0f}% and {p_good_high*100:.0f}%. "
        f"The payoffs (in $K) for each option are: {payoff_desc}. "
        f"Which option maximizes expected value? Does the optimal choice change across "
        f"the range of probability estimates? What is the expected value of the best option?"
    )

    # Check if decision switches
    if best_ev_low == best_ev_high:
        switch_note = f"{options[best_ev_mid]} is optimal across the entire uncertainty range."
    else:
        switch_note = (
            f"The optimal choice switches: {options[best_ev_low]} at P(good)={p_good_low}, "
            f"{options[best_ev_high]} at P(good)={p_good_high}."
        )

    answer = (
        f"At P(good)={p_good_mid}: best option is {options[best_ev_mid]} "
        f"with EV=${ev_best_mid:.1f}K. {switch_note} "
        f"EV range of best option: [{ci[0]}, {ci[1]}]."
    )

    steps = [f"Compute EV for each option: EV = P(good)×payoff_good + P(bad)×payoff_bad."]
    for i in range(n_options):
        steps.append(
            f"{options[i]}: EV = {p_good_mid}×{payoffs[i]['good']:.0f} + "
            f"{round(1-p_good_mid,2)}×{payoffs[i]['bad']:.0f} = ${ev(i, p_good_mid):.1f}K."
        )
    steps.append(f"Best at mid-range: {options[best_ev_mid]} (${ev_best_mid:.1f}K).")
    steps.append(f"Sensitivity: check endpoints P(good)={p_good_low} and {p_good_high}.")
    steps.append(f"{switch_note}")

    failure_modes = [
        "Evaluating options only at the midpoint probability without sensitivity analysis",
        "Not identifying the breakeven probability where the optimal choice switches",
        "Ignoring downside risk and focusing only on expected value",
    ]
    if difficulty >= 4:
        failure_modes.append(
            "Not performing minimax regret analysis to find the most robust choice"
        )

    return {
        "id": f"MURU-{problem_id:04d}",
        "category": "decision_under_uncertainty",
        "difficulty": difficulty,
        "stem": stem,
        "uncertainty_type": "parameter_uncertainty",
        "required_framework": "decision_theory",
        "ground_truth": {
            "answer": answer,
            "point_estimate": ev_best_mid,
            "confidence_interval": [ci[0], ci[1]],
            "ci_level": 0.90,
        },
        "solution_steps": steps,
        "common_failure_modes": failure_modes,
        "metadata": {
            "author": "generator_multi_option_decision",
            "reviewed": False,
            "source_inspiration": "original",
        },
    }


register_template(ProblemTemplate(
    name="multi_option_decision",
    category="decision_under_uncertainty",
    difficulty_range=(2, 4),
    description="Multi-option decision with uncertain state probabilities and sensitivity analysis",
    generate=generate_multi_option_decision,
))


# ──────────────────────────────────────────────────────────────
# Template 16: Sequential Decision / Value of Info (DU, D3-D5)
# ──────────────────────────────────────────────────────────────

SEQ_DECISION_CONTEXTS = [
    {"domain": "R&D pipeline", "stage": "prototype", "gate": "go/no-go decision",
     "invest": "development cost", "payoff": "market revenue", "info": "pilot study"},
    {"domain": "oil exploration", "stage": "seismic survey", "gate": "drill decision",
     "invest": "drilling cost", "payoff": "oil revenue", "info": "seismic data"},
    {"domain": "clinical trials", "stage": "Phase I", "gate": "Phase II decision",
     "invest": "trial cost", "payoff": "drug sales", "info": "Phase I results"},
    {"domain": "startup funding", "stage": "seed round", "gate": "Series A decision",
     "invest": "capital", "payoff": "exit value", "info": "product-market fit data"},
    {"domain": "mine development", "stage": "geological survey", "gate": "mine construction",
     "invest": "construction cost", "payoff": "mineral value", "info": "core sample analysis"},
    {"domain": "movie production", "stage": "script development", "gate": "greenlighting",
     "invest": "production budget", "payoff": "box office revenue", "info": "test screening"},
]


def generate_sequential_decision(problem_id: int, difficulty: int) -> dict:
    ctx = random.choice(SEQ_DECISION_CONTEXTS)

    # Prior probability of success
    p_success_low = round(random.uniform(0.25, 0.40), 2)
    p_success_high = round(p_success_low + random.uniform(0.15, 0.25), 2)
    p_success_mid = round((p_success_low + p_success_high) / 2, 2)

    # Investment and payoffs
    invest_cost = round(random.uniform(100, 500), 0)
    payoff_success = round(invest_cost * random.uniform(3, 8), 0)
    payoff_failure = 0

    # Info cost and accuracy
    info_cost = round(invest_cost * random.uniform(0.05, 0.15), 0)
    info_accuracy_low = round(random.uniform(0.65, 0.75), 2)
    info_accuracy_high = round(info_accuracy_low + random.uniform(0.10, 0.20), 2)
    info_accuracy_mid = round((info_accuracy_low + info_accuracy_high) / 2, 2)

    # EV without info: max(0, p_success × payoff - invest_cost)
    def ev_no_info(p):
        ev_invest = p * payoff_success - invest_cost
        return round(max(0, ev_invest), 1)

    # EV with info (simplified: info is a binary signal)
    def ev_with_info(p, acc):
        # P(positive signal) = p*acc + (1-p)*(1-acc)
        p_pos = p * acc + (1 - p) * (1 - acc)
        # P(success | positive signal)
        p_succ_given_pos = p * acc / p_pos if p_pos > 0 else 0
        # EV if we get positive signal and invest
        ev_invest_pos = p_succ_given_pos * payoff_success - invest_cost
        # EV with info = P(pos) × max(0, ev_invest_pos) + P(neg) × 0 - info_cost
        ev = p_pos * max(0, ev_invest_pos) - info_cost
        return round(ev, 1)

    ev_no_info_mid = ev_no_info(p_success_mid)
    ev_with_info_mid = ev_with_info(p_success_mid, info_accuracy_mid)
    voi_mid = round(ev_with_info_mid - ev_no_info_mid, 1)

    ev_no_info_low = ev_no_info(p_success_low)
    ev_with_info_low = ev_with_info(p_success_low, info_accuracy_low)
    ev_no_info_high = ev_no_info(p_success_high)
    ev_with_info_high = ev_with_info(p_success_high, info_accuracy_high)

    # CI on VOI — VOI is non-monotonic, include mid in the range
    voi_low = round(ev_with_info_low - ev_no_info_low, 1)
    voi_high = round(ev_with_info_high - ev_no_info_high, 1)

    ci_vals = sorted([min(voi_low, voi_high, voi_mid), max(voi_low, voi_high, voi_mid)])
    if ci_vals[0] >= ci_vals[1]:
        ci_vals[1] = ci_vals[0] + 1.0

    stem = (
        f"In {ctx['domain']}, a decision-maker is at the {ctx['stage']} stage and must decide "
        f"whether to proceed to {ctx['gate']}. The investment cost is ${invest_cost:.0f}K. "
        f"If the project succeeds (estimated probability {p_success_low*100:.0f}%-{p_success_high*100:.0f}%), "
        f"it yields ${payoff_success:.0f}K; if it fails, the investment is lost. "
        f"Before deciding, the decision-maker can purchase a {ctx['info']} for ${info_cost:.0f}K. "
        f"This {ctx['info']} correctly predicts success/failure with accuracy "
        f"{info_accuracy_low*100:.0f}%-{info_accuracy_high*100:.0f}%. "
        f"Should the decision-maker buy the information? What is the value of this "
        f"information, and how does it depend on the uncertain success probability and "
        f"information accuracy?"
    )

    answer = (
        f"EV without info at mid-range: ${ev_no_info_mid:.1f}K. "
        f"EV with info at mid-range: ${ev_with_info_mid:.1f}K. "
        f"Value of information: ${voi_mid:.1f}K. "
        f"VOI range: [${ci_vals[0]:.1f}K, ${ci_vals[1]:.1f}K]. "
        f"{'Information is worth purchasing.' if voi_mid > 0 else 'Information may not be worth the cost.'}"
    )

    steps = [
        f"EV without info: max(0, P(success)×${payoff_success:.0f}K - ${invest_cost:.0f}K).",
        f"At P={p_success_mid}: EV_no_info = max(0, {p_success_mid}×{payoff_success:.0f} - {invest_cost:.0f}) = ${ev_no_info_mid:.1f}K.",
        f"With info: P(positive signal) = P(success)×accuracy + P(failure)×(1-accuracy).",
        f"P(success|positive) = P(success)×accuracy / P(positive signal).",
        f"EV_with_info = P(pos)×max(0, P(success|pos)×{payoff_success:.0f} - {invest_cost:.0f}) - {info_cost:.0f}.",
        f"At mid-range: EV_with_info = ${ev_with_info_mid:.1f}K.",
        f"VOI = EV_with_info - EV_no_info = ${voi_mid:.1f}K.",
        f"Range across uncertain parameters: [${ci_vals[0]:.1f}K, ${ci_vals[1]:.1f}K].",
    ]

    failure_modes = [
        "Not computing the conditional probability P(success|positive signal) correctly",
        "Forgetting to subtract the information cost from the EV with info",
        "Assuming perfect information and overestimating VoI",
        "Not recognizing that VoI depends on both the prior probability and information accuracy",
    ]
    if difficulty >= 5:
        failure_modes.append(
            "Not recognizing the option value: negative EV without info doesn't mean info is worthless"
        )

    return {
        "id": f"MURU-{problem_id:04d}",
        "category": "decision_under_uncertainty",
        "difficulty": difficulty,
        "stem": stem,
        "uncertainty_type": "epistemic_uncertainty",
        "required_framework": "decision_theory",
        "ground_truth": {
            "answer": answer,
            "point_estimate": voi_mid,
            "confidence_interval": [ci_vals[0], ci_vals[1]],
            "ci_level": 0.90,
        },
        "solution_steps": steps,
        "common_failure_modes": failure_modes,
        "metadata": {
            "author": "generator_sequential_decision",
            "reviewed": False,
            "source_inspiration": "original",
        },
    }


register_template(ProblemTemplate(
    name="sequential_decision",
    category="decision_under_uncertainty",
    difficulty_range=(3, 5),
    description="Sequential decision with value of information under dual uncertainty",
    generate=generate_sequential_decision,
))


# ──────────────────────────────────────────────────────────────
# Template 17: Insurance / Expected Loss Pricing (DU, D3-D5)
# ──────────────────────────────────────────────────────────────

INSURANCE_CONTEXTS = [
    {"peril": "flood", "asset": "residential property", "region": "a coastal city",
     "domain": "flood insurance"},
    {"peril": "cyber attack", "asset": "corporate data", "region": "a financial institution",
     "domain": "cyber insurance"},
    {"peril": "wildfire", "asset": "agricultural land", "region": "a fire-prone region",
     "domain": "agricultural insurance"},
    {"peril": "earthquake", "asset": "commercial buildings", "region": "a seismically active zone",
     "domain": "earthquake insurance"},
    {"peril": "product liability claim", "asset": "consumer products", "region": "the US market",
     "domain": "product liability"},
    {"peril": "supply chain disruption", "asset": "manufacturing operations", "region": "a global supply chain",
     "domain": "business interruption"},
]


def generate_insurance_pricing(problem_id: int, difficulty: int) -> dict:
    ctx = random.choice(INSURANCE_CONTEXTS)

    # Frequency: Poisson rate (uncertain)
    freq_low = round(random.uniform(0.5, 2.0), 2)
    freq_high = round(freq_low + random.uniform(1.0, 3.0), 2)
    freq_mid = round((freq_low + freq_high) / 2, 2)

    # Severity: Lognormal parameters (uncertain mean)
    sev_mean_low = round(random.uniform(50, 200), 0)
    sev_mean_high = round(sev_mean_low * random.uniform(1.5, 2.5), 0)
    sev_mean_mid = round((sev_mean_low + sev_mean_high) / 2, 0)

    # Expected loss = frequency × severity
    def expected_loss(freq, sev):
        return round(freq * sev, 1)

    el_low = expected_loss(freq_low, sev_mean_low)
    el_mid = expected_loss(freq_mid, sev_mean_mid)
    el_high = expected_loss(freq_high, sev_mean_high)

    ci = sorted([el_low, el_high])
    if ci[0] >= ci[1]:
        ci[1] = ci[0] + 10.0

    # Risk loading (for premium)
    risk_load = round(random.uniform(0.15, 0.35), 2)
    premium_mid = round(el_mid * (1 + risk_load), 1)

    stem = (
        f"An insurer is pricing a {ctx['domain']} policy for {ctx['asset']} in {ctx['region']}. "
        f"Historical data suggests the annual frequency of {ctx['peril']} events follows a "
        f"Poisson distribution with rate λ estimated between {freq_low} and {freq_high} events/year. "
        f"Each event causes a loss estimated between ${sev_mean_low:.0f}K and ${sev_mean_high:.0f}K "
        f"(with the average loss being uncertain due to limited historical claims data). "
        f"The insurer uses a risk loading factor of {risk_load*100:.0f}% above the expected loss. "
        f"What is the fair premium, and what range of premiums is justified given the "
        f"uncertainty in both frequency and severity?"
    )

    premium_low = round(el_low * (1 + risk_load), 1)
    premium_high = round(el_high * (1 + risk_load), 1)
    premium_ci = sorted([premium_low, premium_high])

    answer = (
        f"Expected loss = λ × mean severity. Range: ${ci[0]:.1f}K to ${ci[1]:.1f}K. "
        f"Point estimate: ${el_mid:.1f}K. "
        f"With {risk_load*100:.0f}% loading: premium range "
        f"[${premium_ci[0]:.1f}K, ${premium_ci[1]:.1f}K]. "
        f"Point estimate premium: ${premium_mid:.1f}K."
    )

    steps = [
        f"Expected loss = E[N] × E[X] where N ~ Poisson(λ), X = loss per event.",
        f"At λ={freq_mid}, severity=${sev_mean_mid:.0f}K: EL = {freq_mid}×{sev_mean_mid:.0f} = ${el_mid:.1f}K.",
        f"At λ={freq_low}, severity=${sev_mean_low:.0f}K: EL = ${el_low:.1f}K (lower bound).",
        f"At λ={freq_high}, severity=${sev_mean_high:.0f}K: EL = ${el_high:.1f}K (upper bound).",
        f"Premium = EL × (1 + risk load) = EL × {1+risk_load:.2f}.",
        f"Premium range: [${premium_ci[0]:.1f}K, ${premium_ci[1]:.1f}K]. Mid: ${premium_mid:.1f}K.",
    ]

    if difficulty >= 4:
        steps.append(
            f"Note: this assumes independence between frequency and severity. "
            f"In practice, high-frequency years may also have higher severity (correlation), "
            f"which would widen the range further."
        )
    if difficulty >= 5:
        steps.append(
            f"For tail risk: the 99th percentile loss (Value at Risk) would be substantially "
            f"higher than the expected loss, requiring additional capital reserves."
        )

    failure_modes = [
        "Using only the midpoint estimates without computing the full range",
        "Adding frequency and severity instead of multiplying",
        "Not including the risk loading in the premium calculation",
        "Treating the expected loss as the premium without risk margin",
    ]
    if difficulty >= 4:
        failure_modes.append(
            "Assuming frequency and severity are independent when they may be correlated"
        )
    if difficulty >= 5:
        failure_modes.append(
            "Ignoring tail risk (VaR/TVaR) and pricing only based on expected loss"
        )

    return {
        "id": f"MURU-{problem_id:04d}",
        "category": "decision_under_uncertainty",
        "difficulty": difficulty,
        "stem": stem,
        "uncertainty_type": "parameter_uncertainty",
        "required_framework": "decision_theory",
        "ground_truth": {
            "answer": answer,
            "point_estimate": premium_mid,
            "confidence_interval": [premium_ci[0], premium_ci[1]],
            "ci_level": 0.90,
        },
        "solution_steps": steps,
        "common_failure_modes": failure_modes,
        "metadata": {
            "author": "generator_insurance_pricing",
            "reviewed": False,
            "source_inspiration": "original",
        },
    }


register_template(ProblemTemplate(
    name="insurance_pricing",
    category="decision_under_uncertainty",
    difficulty_range=(3, 5),
    description="Insurance premium pricing with uncertain Poisson frequency × uncertain severity",
    generate=generate_insurance_pricing,
))


# ──────────────────────────────────────────────────────────────
# Template 18: Simpson's Paradox (AA, D3-D5)
# ──────────────────────────────────────────────────────────────

SIMPSONS_CONTEXTS = [
    {"domain": "medical treatment", "treatment": "Drug X vs. Drug Y",
     "outcome": "recovery rate", "confounder": "disease severity",
     "group_a": "mild cases", "group_b": "severe cases"},
    {"domain": "university admissions", "treatment": "Department A vs. Department B",
     "outcome": "acceptance rate", "confounder": "applicant pool competitiveness",
     "group_a": "less competitive applicants", "group_b": "highly competitive applicants"},
    {"domain": "employee training", "treatment": "Training Program X vs. Y",
     "outcome": "performance improvement", "confounder": "initial skill level",
     "group_a": "junior employees", "group_b": "senior employees"},
    {"domain": "advertising", "treatment": "Platform A vs. Platform B",
     "outcome": "conversion rate", "confounder": "product category",
     "group_a": "low-consideration products", "group_b": "high-consideration products"},
    {"domain": "legal outcomes", "treatment": "Judge A vs. Judge B",
     "outcome": "acquittal rate", "confounder": "case severity",
     "group_a": "misdemeanors", "group_b": "felonies"},
    {"domain": "school performance", "treatment": "School A vs. School B",
     "outcome": "test score improvement", "confounder": "socioeconomic background",
     "group_a": "affluent area students", "group_b": "lower-income area students"},
]


def generate_simpsons_paradox(problem_id: int, difficulty: int) -> dict:
    ctx = random.choice(SIMPSONS_CONTEXTS)

    # Design rates so Simpson's paradox occurs:
    # Treatment X is better in BOTH subgroups, but worse overall due to composition

    # Group A (larger population)
    n_a_x = random.choice([80, 100, 120, 150])
    n_a_y = random.choice([20, 30, 40])
    rate_a_x = round(random.uniform(0.75, 0.90), 2)  # X is better in group A
    rate_a_y = round(rate_a_x - random.uniform(0.05, 0.10), 2)  # Y is worse in group A

    # Group B (smaller for X, larger for Y)
    n_b_x = random.choice([20, 30, 40])
    n_b_y = random.choice([80, 100, 120])
    rate_b_x = round(random.uniform(0.40, 0.60), 2)  # X is better in group B
    rate_b_y = round(rate_b_x - random.uniform(0.05, 0.10), 2)  # Y is worse in group B

    # Aggregate rates (paradox: Y appears better overall)
    succ_x = round(rate_a_x * n_a_x + rate_b_x * n_b_x)
    total_x = n_a_x + n_b_x
    overall_x = round(succ_x / total_x, 3)

    succ_y = round(rate_a_y * n_a_y + rate_b_y * n_b_y)
    total_y = n_a_y + n_b_y
    overall_y = round(succ_y / total_y, 3)

    # Ensure paradox exists (overall Y > overall X even though X > Y in both subgroups)
    # If not, swap group sizes to force it
    if overall_x >= overall_y:
        n_a_x, n_a_y = n_a_y, n_a_x
        n_b_x, n_b_y = n_b_y, n_b_x
        succ_x = round(rate_a_x * n_a_x + rate_b_x * n_b_x)
        total_x = n_a_x + n_b_x
        overall_x = round(succ_x / total_x, 3)
        succ_y = round(rate_a_y * n_a_y + rate_b_y * n_b_y)
        total_y = n_a_y + n_b_y
        overall_y = round(succ_y / total_y, 3)

    # Uncertain group sizes in the population
    w_a_low = round(random.uniform(0.30, 0.45), 2)
    w_a_high = round(w_a_low + random.uniform(0.10, 0.20), 2)
    w_a_mid = round((w_a_low + w_a_high) / 2, 2)

    # Adjusted rate for Treatment X with population weighting
    def adj_rate_x(w_a):
        return round(w_a * rate_a_x + (1 - w_a) * rate_b_x, 3)

    def adj_rate_y(w_a):
        return round(w_a * rate_a_y + (1 - w_a) * rate_b_y, 3)

    diff_low = round(adj_rate_x(w_a_low) - adj_rate_y(w_a_low), 3)
    diff_mid = round(adj_rate_x(w_a_mid) - adj_rate_y(w_a_mid), 3)
    diff_high = round(adj_rate_x(w_a_high) - adj_rate_y(w_a_high), 3)

    ci = sorted([diff_low, diff_high])
    if ci[0] >= ci[1]:
        ci[1] = round(ci[0] + 0.005, 3)

    stem = (
        f"In a {ctx['domain']} study comparing {ctx['treatment']}: "
        f"Overall, Treatment Y has a higher {ctx['outcome']} ({overall_y*100:.1f}% vs "
        f"{overall_x*100:.1f}%). However, when stratified by {ctx['confounder']}: "
        f"In {ctx['group_a']} (n_X={n_a_x}, n_Y={n_a_y}): Treatment X achieves "
        f"{rate_a_x*100:.0f}% vs Treatment Y's {rate_a_y*100:.0f}%. "
        f"In {ctx['group_b']} (n_X={n_b_x}, n_Y={n_b_y}): Treatment X achieves "
        f"{rate_b_x*100:.0f}% vs Treatment Y's {rate_b_y*100:.0f}%. "
        f"The true population proportion of {ctx['group_a']} is uncertain, estimated "
        f"between {w_a_low*100:.0f}% and {w_a_high*100:.0f}%. "
        f"Which treatment is actually better, and by how much?"
    )

    answer = (
        f"This is Simpson's Paradox. Treatment X is better in BOTH subgroups. "
        f"The aggregate reversal occurs because X is disproportionately applied to "
        f"{ctx['group_b']} (harder cases). Adjusted difference (X - Y): "
        f"[{ci[0]}, {ci[1]}], point estimate: {diff_mid}. "
        f"Treatment X is genuinely better by {diff_mid*100:.1f} percentage points."
    )

    steps = [
        f"Aggregate rates: X={overall_x*100:.1f}%, Y={overall_y*100:.1f}% (Y appears better).",
        f"Stratified: In {ctx['group_a']}: X={rate_a_x*100:.0f}% > Y={rate_a_y*100:.0f}%. "
        f"In {ctx['group_b']}: X={rate_b_x*100:.0f}% > Y={rate_b_y*100:.0f}%.",
        f"X is better in BOTH subgroups — this is Simpson's Paradox.",
        f"The paradox arises because {ctx['treatment'].split(' vs.')[0].strip()} is applied more to "
        f"{ctx['group_b']} (lower baseline rate), skewing the aggregate.",
        f"Population-adjusted rate for X: w_A×{rate_a_x} + (1-w_A)×{rate_b_x}.",
        f"Population-adjusted rate for Y: w_A×{rate_a_y} + (1-w_A)×{rate_b_y}.",
        f"Difference X-Y at w_A={w_a_mid}: {diff_mid}. Range: [{ci[0]}, {ci[1]}].",
    ]

    failure_modes = [
        "Using the aggregate rates and concluding Treatment Y is better",
        "Not recognizing Simpson's Paradox despite seeing reversed trends in subgroups",
        "Ignoring the confounding variable (unequal group composition between treatments)",
        "Computing adjusted rates with a single population weight without uncertainty",
    ]
    if difficulty >= 5:
        failure_modes.append(
            "Not considering that additional unobserved confounders could further modify the comparison"
        )

    return {
        "id": f"MURU-{problem_id:04d}",
        "category": "adversarial_ambiguity",
        "difficulty": difficulty,
        "stem": stem,
        "uncertainty_type": "structural_uncertainty",
        "required_framework": "bayesian_inference",
        "ground_truth": {
            "answer": answer,
            "point_estimate": diff_mid,
            "confidence_interval": [ci[0], ci[1]],
            "ci_level": 0.90,
        },
        "solution_steps": steps,
        "common_failure_modes": failure_modes,
        "metadata": {
            "author": "generator_simpsons_paradox",
            "reviewed": False,
            "source_inspiration": "original",
        },
    }


register_template(ProblemTemplate(
    name="simpsons_paradox",
    category="adversarial_ambiguity",
    difficulty_range=(3, 5),
    description="Simpson's Paradox with uncertain population composition",
    generate=generate_simpsons_paradox,
))


# ──────────────────────────────────────────────────────────────
# Template 19: Reference Class Problem (AA, D3-D5)
# ──────────────────────────────────────────────────────────────

REFERENCE_CLASS_CONTEXTS = [
    {"scenario": "project duration estimation", "subject": "a software project",
     "class_a": "all IT projects in the firm", "class_b": "similar-sized agile projects",
     "class_c": "projects by this same team", "domain": "project management"},
    {"scenario": "recidivism prediction", "subject": "a parolee",
     "class_a": "all parolees nationally", "class_b": "parolees with similar offense type",
     "class_c": "parolees from this specific facility", "domain": "criminal justice"},
    {"scenario": "startup valuation", "subject": "a tech startup",
     "class_a": "all venture-backed startups", "class_b": "SaaS startups at Series B",
     "class_c": "startups in this specific niche", "domain": "venture capital"},
    {"scenario": "patient prognosis", "subject": "a cancer patient",
     "class_a": "all cancer patients", "class_b": "patients with this cancer type and stage",
     "class_c": "patients at this hospital with this oncologist", "domain": "oncology"},
    {"scenario": "construction cost estimation", "subject": "a bridge project",
     "class_a": "all infrastructure projects nationally", "class_b": "similar-scale bridge projects",
     "class_c": "projects by this contractor in this region", "domain": "civil engineering"},
    {"scenario": "insurance claim prediction", "subject": "a homeowner",
     "class_a": "all homeowners nationally", "class_b": "homeowners in this ZIP code",
     "class_c": "homeowners with this specific home profile", "domain": "home insurance"},
]


def generate_reference_class_problem(problem_id: int, difficulty: int) -> dict:
    ctx = random.choice(REFERENCE_CLASS_CONTEXTS)

    # Three different reference classes give different base rates
    rate_a = round(random.uniform(0.15, 0.35), 2)
    rate_b = round(random.uniform(0.25, 0.50), 2)
    rate_c = round(random.uniform(0.35, 0.65), 2)

    # Ensure they're meaningfully different
    while abs(rate_a - rate_b) < 0.08 or abs(rate_b - rate_c) < 0.08:
        rate_b = round(random.uniform(0.25, 0.50), 2)
        rate_c = round(random.uniform(0.35, 0.65), 2)

    # Sample sizes (narrower class → smaller n → wider CI)
    n_a = random.choice([500, 1000, 2000])
    n_b = random.choice([50, 100, 200])
    n_c = random.choice([15, 20, 30])

    # CIs for each reference class
    def wilson_ci(p, n):
        z = 1.96
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        margin = z * math.sqrt((p * (1 - p) / n + z**2 / (4 * n**2))) / denom
        return (round(max(0, center - margin), 3), round(min(1, center + margin), 3))

    ci_a = wilson_ci(rate_a, n_a)
    ci_b = wilson_ci(rate_b, n_b)
    ci_c = wilson_ci(rate_c, n_c)

    # The "answer" spans all reference classes
    overall_lower = min(ci_a[0], ci_b[0], ci_c[0])
    overall_upper = max(ci_a[1], ci_b[1], ci_c[1])

    # Point estimate: weighted average (more weight to narrower/more specific class)
    if difficulty >= 4:
        # At higher difficulty, the weighting itself is uncertain
        w_a = round(random.uniform(0.15, 0.30), 2)
        w_b = round(random.uniform(0.30, 0.40), 2)
        w_c = round(1 - w_a - w_b, 2)
    else:
        w_a, w_b, w_c = 0.20, 0.35, 0.45

    point_est = round(w_a * rate_a + w_b * rate_b + w_c * rate_c, 3)

    if overall_lower >= overall_upper:
        overall_upper = round(overall_lower + 0.01, 3)

    stem = (
        f"For {ctx['scenario']}, the base rate for {ctx['subject']} depends on which "
        f"reference class is used: "
        f"Reference Class A ({ctx['class_a']}, n={n_a}): base rate = {rate_a*100:.0f}%. "
        f"Reference Class B ({ctx['class_b']}, n={n_b}): base rate = {rate_b*100:.0f}%. "
        f"Reference Class C ({ctx['class_c']}, n={n_c}): base rate = {rate_c*100:.0f}%. "
        f"Each reference class has a different level of specificity and sample size. "
        f"What is the appropriate base rate to use, and how should the choice of "
        f"reference class affect the uncertainty in the prediction?"
    )

    answer = (
        f"The reference class problem: different classes give rates of {rate_a*100:.0f}%, "
        f"{rate_b*100:.0f}%, {rate_c*100:.0f}%. No single class is objectively 'correct'. "
        f"Combined interval spanning all classes: [{overall_lower}, {overall_upper}]. "
        f"Weighted point estimate (specificity-weighted): {point_est}."
    )

    steps = [
        f"Class A ({ctx['class_a']}): rate={rate_a}, n={n_a}, CI={list(ci_a)}. "
        f"Large sample but least specific.",
        f"Class B ({ctx['class_b']}): rate={rate_b}, n={n_b}, CI={list(ci_b)}. "
        f"Moderate specificity and sample size.",
        f"Class C ({ctx['class_c']}): rate={rate_c}, n={n_c}, CI={list(ci_c)}. "
        f"Most specific but smallest sample (widest CI).",
        f"The reference class problem has no definitive solution — the 'correct' base rate "
        f"depends on which class membership is considered most relevant.",
        f"Weighted estimate (weights reflecting specificity): "
        f"{w_a}×{rate_a} + {w_b}×{rate_b} + {w_c}×{rate_c} = {point_est}.",
        f"Full uncertainty range spanning all reference classes: [{overall_lower}, {overall_upper}].",
    ]

    failure_modes = [
        "Choosing one reference class without acknowledging the others",
        "Using only the most specific class (Class C) despite its small sample size",
        "Using only the broadest class (Class A) despite its low relevance",
        "Not recognizing that the choice of reference class is itself a source of fundamental uncertainty",
    ]
    if difficulty >= 5:
        failure_modes.append(
            "Not considering that the reference classes may overlap, complicating the combination"
        )

    return {
        "id": f"MURU-{problem_id:04d}",
        "category": "adversarial_ambiguity",
        "difficulty": difficulty,
        "stem": stem,
        "uncertainty_type": "structural_uncertainty",
        "required_framework": "bayesian_inference",
        "ground_truth": {
            "answer": answer,
            "point_estimate": point_est,
            "confidence_interval": [overall_lower, overall_upper],
            "ci_level": 0.90,
        },
        "solution_steps": steps,
        "common_failure_modes": failure_modes,
        "metadata": {
            "author": "generator_reference_class",
            "reviewed": False,
            "source_inspiration": "original",
        },
    }


register_template(ProblemTemplate(
    name="reference_class_problem",
    category="adversarial_ambiguity",
    difficulty_range=(3, 5),
    description="Reference class problem: multiple valid base rates from different reference populations",
    generate=generate_reference_class_problem,
))


# ──────────────────────────────────────────────────────────────
# Template 20: Anchoring Manipulation (AA, D2-D4)
# ──────────────────────────────────────────────────────────────

ANCHORING_CONTEXTS = [
    {"domain": "real estate", "what": "property value", "unit": "$K",
     "anchor_source": "a listing price set by a motivated seller",
     "data_source": "recent comparable sales in the neighborhood"},
    {"domain": "salary negotiation", "what": "fair compensation", "unit": "$K/year",
     "anchor_source": "an initial offer from the employer",
     "data_source": "industry salary surveys and cost-of-living data"},
    {"domain": "project estimation", "what": "project completion time", "unit": "months",
     "anchor_source": "a sales team's promise to the client",
     "data_source": "historical data on similar projects"},
    {"domain": "risk assessment", "what": "annual probability of failure", "unit": "%",
     "anchor_source": "an expert's quick estimate during a meeting",
     "data_source": "actuarial records and failure databases"},
    {"domain": "market sizing", "what": "total addressable market", "unit": "$M",
     "anchor_source": "a venture capitalist's optimistic projection",
     "data_source": "bottom-up analysis from actual customer data"},
    {"domain": "environmental", "what": "contamination level", "unit": "ppm",
     "anchor_source": "an initial field screening measurement",
     "data_source": "laboratory analysis of soil samples"},
]


def generate_anchoring_manipulation(problem_id: int, difficulty: int) -> dict:
    ctx = random.choice(ANCHORING_CONTEXTS)

    # The anchor (potentially misleading)
    true_value = round(random.uniform(50, 500), 0)
    anchor_bias = random.choice([-1, 1])  # over or under
    anchor = round(true_value * (1 + anchor_bias * random.uniform(0.30, 0.60)), 0)

    # Data-based estimates
    data_mean = round(true_value * random.uniform(0.90, 1.10), 0)
    data_sd = round(true_value * random.uniform(0.10, 0.25), 0)
    n_data = random.choice([10, 15, 20, 25])

    se = round(data_sd / math.sqrt(n_data), 1)
    data_ci_lower = round(data_mean - 1.96 * se, 1)
    data_ci_upper = round(data_mean + 1.96 * se, 1)

    # Anchored estimate (biased toward anchor)
    anchor_weight_low = round(random.uniform(0.10, 0.25), 2)
    anchor_weight_high = round(anchor_weight_low + random.uniform(0.10, 0.25), 2)
    anchor_weight_mid = round((anchor_weight_low + anchor_weight_high) / 2, 2)

    def anchored_est(w):
        return round(w * anchor + (1 - w) * data_mean, 1)

    est_low = anchored_est(anchor_weight_low)
    est_mid = anchored_est(anchor_weight_mid)
    est_high = anchored_est(anchor_weight_high)

    ci_lower = min(est_low, est_high, data_ci_lower)
    ci_upper = max(est_low, est_high, data_ci_upper)

    if ci_lower >= ci_upper:
        ci_upper = ci_lower + 1.0

    bias_word = "overestimates" if anchor > data_mean else "underestimates"

    stem = (
        f"In {ctx['domain']}, you are asked to estimate the {ctx['what']}. "
        f"You are first told that {ctx['anchor_source']} puts the value at "
        f"{anchor} {ctx['unit']}. You then collect your own data: {ctx['data_source']} "
        f"based on {n_data} observations give a mean of {data_mean} {ctx['unit']} "
        f"with SD = {data_sd} {ctx['unit']}. "
        f"How should you combine the anchor with your data? How much, if at all, "
        f"should the anchor influence your estimate? Provide a point estimate and "
        f"confidence interval."
    )

    answer = (
        f"The anchor ({anchor} {ctx['unit']}) likely {bias_word} the true value. "
        f"Data-based estimate: {data_mean} {ctx['unit']}, CI = [{data_ci_lower}, {data_ci_upper}]. "
        f"If the anchor receives 0-{anchor_weight_high*100:.0f}% weight, the estimate "
        f"ranges from {min(est_low, est_high)} to {max(est_low, est_high)}. "
        f"Point estimate: {data_mean} {ctx['unit']} (data-only, unanchored)."
    )

    steps = [
        f"Anchor: {anchor} {ctx['unit']} (from {ctx['anchor_source']} — potentially biased).",
        f"Data: mean = {data_mean}, SD = {data_sd}, n = {n_data}. SE = {se}.",
        f"Data-based 95% CI: [{data_ci_lower}, {data_ci_upper}].",
        f"The anchor {bias_word} relative to the data by "
        f"{abs(anchor - data_mean):.0f} {ctx['unit']} ({abs(anchor - data_mean)/data_mean*100:.0f}%).",
        f"If we assign the anchor weight w ∈ [0, {anchor_weight_high}]: "
        f"estimate = w×{anchor} + (1-w)×{data_mean}.",
        f"At w={anchor_weight_mid}: estimate = {est_mid}.",
        f"Best practice: the data-based estimate ({data_mean}) should dominate since "
        f"it has an objective basis. The anchor should receive minimal weight.",
    ]

    failure_modes = [
        f"Being anchored by the initial value ({anchor}) and insufficiently adjusting toward the data",
        "Giving equal weight to the anchor and data despite the anchor's questionable provenance",
        f"Not recognizing that the anchor ({anchor}) and data ({data_mean}) are substantially different, "
        f"suggesting bias in the anchor",
    ]
    if difficulty >= 3:
        failure_modes.append(
            "Not computing the data-based confidence interval and relying on intuitive adjustment from the anchor"
        )
    if difficulty >= 4:
        failure_modes.append(
            "Failing to consider the source credibility: data from systematic measurement "
            "should outweigh a single expert estimate or listing price"
        )

    return {
        "id": f"MURU-{problem_id:04d}",
        "category": "adversarial_ambiguity",
        "difficulty": difficulty,
        "stem": stem,
        "uncertainty_type": "epistemic_uncertainty",
        "required_framework": "bayesian_inference",
        "ground_truth": {
            "answer": answer,
            "point_estimate": data_mean,
            "confidence_interval": [ci_lower, ci_upper],
            "ci_level": 0.95,
        },
        "solution_steps": steps,
        "common_failure_modes": failure_modes,
        "metadata": {
            "author": "generator_anchoring_manipulation",
            "reviewed": False,
            "source_inspiration": "original",
        },
    }


register_template(ProblemTemplate(
    name="anchoring_manipulation",
    category="adversarial_ambiguity",
    difficulty_range=(2, 4),
    description="Anchoring bias manipulation with data-based correction",
    generate=generate_anchoring_manipulation,
))


# ──────────────────────────────────────────────────────────────
# Main: Generate and Save
# ──────────────────────────────────────────────────────────────

def save_problem(problem: dict, dry_run: bool = False) -> bool:
    """Save a problem to disk and validate it."""
    filepath = DATA_DIR / f"{problem['id']}.json"

    if filepath.exists():
        return False  # skip duplicates

    if dry_run:
        print(f"  [DRY RUN] Would create {problem['id']} ({problem['category']}, D{problem['difficulty']})")
        return True

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(problem, f, indent=2)
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate MURU-BENCH problems from templates.")
    parser.add_argument("--n", type=int, default=10, help="Number of problems to generate.")
    parser.add_argument("--category", "-c", type=str, help="Filter templates by category.")
    parser.add_argument("--template", "-t", type=str, help="Use a specific template.")
    parser.add_argument("--all", action="store_true", help="Generate from all templates equally.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving.")
    parser.add_argument("--list-templates", action="store_true", help="List available templates.")
    parser.add_argument("--validate", action="store_true", help="Run validation after generation.")
    args = parser.parse_args()

    if args.list_templates:
        print(f"\n  Available Templates ({len(TEMPLATES)}):\n")
        for name, tmpl in sorted(TEMPLATES.items()):
            print(f"    {name:<25} {tmpl.category:<35} D{tmpl.difficulty_range[0]}-{tmpl.difficulty_range[1]}  {tmpl.description}")
        return

    if args.seed is not None:
        random.seed(args.seed)

    # Select templates
    if args.template:
        if args.template not in TEMPLATES:
            print(f"Unknown template: {args.template}. Use --list-templates.")
            sys.exit(1)
        templates = [TEMPLATES[args.template]]
    elif args.category:
        templates = [t for t in TEMPLATES.values() if t.category == args.category]
        if not templates:
            print(f"No templates for category: {args.category}")
            sys.exit(1)
    else:
        templates = list(TEMPLATES.values())

    print(f"\n{'═' * 60}")
    print(f"  MURU-BENCH Problem Generator")
    print(f"  Templates: {', '.join(t.name for t in templates)}")
    print(f"  Target: {args.n} problems")
    print(f"{'═' * 60}\n")

    next_id = get_next_id()
    generated = 0
    failures = 0

    for i in range(args.n):
        template = templates[i % len(templates)]
        d_low, d_high = template.difficulty_range
        difficulty = random.randint(d_low, d_high)
        pid = next_id + i

        try:
            problem = template.generate(pid, difficulty)
            if save_problem(problem, dry_run=args.dry_run):
                generated += 1
                if not args.dry_run:
                    print(f"  ✓ {problem['id']}  {template.name:<20} D{difficulty}  {problem['category']}")
            else:
                failures += 1
        except Exception as e:
            failures += 1
            print(f"  ✗ MURU-{pid:04d}  Error: {e}")

    print(f"\n{'─' * 60}")
    print(f"  Generated: {generated}  |  Skipped/Failed: {failures}")

    if args.validate and not args.dry_run and generated > 0:
        print(f"\n  Running validation...")
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "validate.py")],
            capture_output=True, text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)


if __name__ == "__main__":
    main()
