#!/usr/bin/env python3
"""
run_eval.py — MURU-BENCH Model Evaluator

Runs language models against MURU-BENCH problems via their APIs
and scores them using the metrics module.

Usage:
    python evaluation/run_eval.py --model gpt-4o --subset data/test/
    python evaluation/run_eval.py --model llama-3.1-70b --n 50 --save
    python evaluation/run_eval.py --model gemini-1.5-flash --subset data/test/ --save

API keys (set as environment variables):
    OPENAI_API_KEY       — for GPT models
    ANTHROPIC_API_KEY    — for Claude models
    GOOGLE_API_KEY       — for Gemini models (free at aistudio.google.com)
    GROQ_API_KEY         — for Llama/Mixtral via Groq (free at console.groq.com)
    TOGETHER_API_KEY     — for Llama/Mistral via Together AI

Supported models:
    OpenAI:     gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
    Anthropic:  claude-3.5-sonnet, claude-3-opus, claude-3-haiku
    Google:     gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash
    Groq (FREE): llama-3.1-70b, llama-3.1-8b, mixtral-8x7b, gemma2-9b
    Together:   meta-llama/Llama-3-70b, mistralai/Mixtral-8x7B
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import MURUMetrics, Prediction


# ──────────────────────────────────────────────────────────────────────
# Prompt template
# ──────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a mathematical reasoning expert. You will be given a problem involving mathematical uncertainty. You must:

1. Identify the correct mathematical framework (bayesian_inference, frequentist_inference, decision_theory, information_theory, or monte_carlo).
2. Show your reasoning step-by-step.
3. Provide your final answer as a single number (point estimate).
4. Provide a confidence interval [lower, upper] for your answer.
5. State your confidence in your answer as a probability between 0 and 1.

Format your response EXACTLY as follows at the end:

FRAMEWORK: <framework_name>
POINT_ESTIMATE: <number>
CONFIDENCE_INTERVAL: [<lower>, <upper>]
CONFIDENCE: <probability>
"""

USER_PROMPT_TEMPLATE = """Problem:
{stem}

Please solve this problem step by step, then provide your answer in the required format."""


# ──────────────────────────────────────────────────────────────────────
# API Clients
# ──────────────────────────────────────────────────────────────────────

class ModelClient:
    """Base class for model API clients."""

    def query(self, prompt: str, system: str = "") -> str:
        raise NotImplementedError


class OpenAIClient(ModelClient):
    def __init__(self, model: str):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai>=1.12.0")
        self.client = OpenAI()
        self.model = model

    def query(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
        )
        return response.choices[0].message.content or ""


class AnthropicClient(ModelClient):
    def __init__(self, model: str):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic>=0.18.0")
        self.client = anthropic.Anthropic()
        self.model = model

    def query(self, prompt: str, system: str = "") -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text if response.content else ""


class GoogleClient(ModelClient):
    def __init__(self, model: str):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Install google-generativeai: pip install google-generativeai>=0.4.0")
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model)

    def query(self, prompt: str, system: str = "") -> str:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        response = self.model.generate_content(full_prompt)
        return response.text or ""


class GroqClient(ModelClient):
    """Groq API client — FREE tier for Llama 3, Mixtral, Gemma.

    Get your free API key at: https://console.groq.com
    Supports: llama-3.1-70b-versatile, llama-3.1-8b-instant,
              mixtral-8x7b-32768, gemma2-9b-it
    """
    # Map short names to current Groq model IDs (verified 2026-05).
    MODEL_MAP = {
        "llama-3.3-70b": "llama-3.3-70b-versatile",
        "llama-3.1-70b": "llama-3.3-70b-versatile",   # alias to current 70B
        "llama-3.1-8b": "llama-3.1-8b-instant",
        "llama-3-70b": "llama-3.3-70b-versatile",
        "llama-3-8b": "llama-3.1-8b-instant",
        "llama-4-scout": "meta-llama/llama-4-scout-17b-16e-instruct",
        "gpt-oss-120b": "openai/gpt-oss-120b",
        "gpt-oss-20b": "openai/gpt-oss-20b",
        "qwen3-32b": "qwen/qwen3-32b",
    }

    def __init__(self, model: str):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Get a FREE key at https://console.groq.com"
            )
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai>=1.12.0")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = self.MODEL_MAP.get(model.lower(), model)

    def query(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
        )
        return response.choices[0].message.content or ""


class TogetherClient(ModelClient):
    """Together AI client — cheap Llama/Mistral models.

    Get your API key at: https://together.ai ($5 free credits on signup)
    """

    def __init__(self, model: str):
        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError(
                "TOGETHER_API_KEY not set. Get $5 free credits at https://together.ai"
            )
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai>=1.12.0")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.together.xyz/v1",
        )
        self.model = model

    def query(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
        )
        return response.choices[0].message.content or ""


class OpenRouterClient(ModelClient):
    """OpenRouter — unified API for many open and frontier models.

    Get your API key at: https://openrouter.ai (free tier on many models;
    pay-as-you-go for the rest). OpenAI-compatible endpoint.
    Use the "or:" prefix or the full org/model[:free] slug to route here.
    """

    def __init__(self, model: str):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set. Get a key at https://openrouter.ai"
            )
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai>=1.12.0")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        # Strip the "or:" routing prefix if present.
        self.model = model[3:] if model.startswith("or:") else model

    def query(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
        )
        return response.choices[0].message.content or ""


def get_client(model_name: str) -> ModelClient:
    """Create the appropriate API client for the given model name."""
    model_lower = model_name.lower()

    # Explicit OpenRouter prefix takes priority.
    if model_name.startswith("or:"):
        return OpenRouterClient(model_name)
    # Groq-hosted open models (must be checked before OpenAI/etc. since
    # gpt-oss-* contains "gpt" and would otherwise route to the OpenAI client).
    if any(x in model_lower for x in ["gpt-oss", "qwen", "llama-4", "llama-3"]):
        return GroqClient(model_name)
    # OpenAI
    elif any(x in model_lower for x in ["gpt", "o1", "o3"]):
        return OpenAIClient(model_name)
    # Anthropic
    elif "claude" in model_lower:
        return AnthropicClient(model_name)
    # Gemini (Google AI Studio — free)
    elif "gemini" in model_lower:
        return GoogleClient(model_name)
    # Groq (free Llama/Mixtral/Gemma)
    elif any(x in model_lower for x in ["llama", "mixtral", "gemma"]):
        if os.environ.get("TOGETHER_API_KEY") and not os.environ.get("GROQ_API_KEY"):
            return TogetherClient(model_name)
        return GroqClient(model_name)
    # Together AI (explicit)
    elif "/" in model_name:  # Together uses org/model format
        return TogetherClient(model_name)
    else:
        print(f"ERROR: Unknown model provider for '{model_name}'.")
        print("Supported models:")
        print("  OpenAI:    gpt-4o, gpt-4o-mini, gpt-3.5-turbo")
        print("  Anthropic: claude-3.5-sonnet, claude-3-opus")
        print("  Google:    gemini-1.5-pro, gemini-1.5-flash (FREE)")
        print("  Groq:      llama-3.1-70b, mixtral-8x7b, gemma2-9b (FREE)")
        print("  Together:  meta-llama/Llama-3-70b (org/model format)")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────
# Response parser
# ──────────────────────────────────────────────────────────────────────

def parse_response(response: str) -> dict:
    """Extract structured answer from model response."""
    result = {
        "framework": None,
        "point_estimate": None,
        "confidence_interval": None,
        "confidence": 0.5,  # default
    }

    # Extract FRAMEWORK
    fw_match = re.search(r"FRAMEWORK:\s*(\w+)", response, re.IGNORECASE)
    if fw_match:
        result["framework"] = fw_match.group(1).lower()

    # Extract POINT_ESTIMATE
    pe_match = re.search(r"POINT_ESTIMATE:\s*([-+]?\d*\.?\d+)", response, re.IGNORECASE)
    if pe_match:
        result["point_estimate"] = float(pe_match.group(1))

    # Extract CONFIDENCE_INTERVAL
    ci_match = re.search(
        r"CONFIDENCE_INTERVAL:\s*\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]",
        response, re.IGNORECASE
    )
    if ci_match:
        result["confidence_interval"] = (float(ci_match.group(1)), float(ci_match.group(2)))

    # Extract CONFIDENCE
    conf_match = re.search(r"CONFIDENCE:\s*([-+]?\d*\.?\d+)", response, re.IGNORECASE)
    if conf_match:
        result["confidence"] = float(conf_match.group(1))

    return result


# ──────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ──────────────────────────────────────────────────────────────────────

def load_problems(subset_dir: str) -> list[dict]:
    """Load problems from a directory."""
    problems = []
    path = Path(subset_dir)
    for filepath in sorted(path.rglob("MURU-*.json")):
        try:
            with open(filepath) as f:
                problems.append(json.load(f))
        except (json.JSONDecodeError, IOError):
            continue
    return problems


def run_evaluation(
    model_name: str,
    problems: list[dict],
    max_n: int | None = None,
    delay: float = 0.5,
    seed: int = 42,
) -> tuple[list[Prediction], list[dict]]:
    """Run model on all problems and collect predictions.

    Problems are always evaluated in a deterministic, seed-shuffled order
    rather than sorted-ID order. This matters when a run is truncated early
    (e.g. a hosted-API daily-token cap): the completed prefix is then a
    difficulty-representative random subset instead of the easy-skewed
    front of the ID-sorted test set. The shuffle is seeded, so the order
    is reproducible and ``--n`` yields the same subset across runs.
    """
    client = get_client(model_name)

    import random
    problems = list(problems)
    random.Random(seed).shuffle(problems)
    if max_n and max_n < len(problems):
        problems = problems[:max_n]

    predictions = []
    raw_results = []

    for i, problem in enumerate(problems):
        print(f"  [{i + 1}/{len(problems)}] {problem['id']} ... ", end="", flush=True)

        prompt = USER_PROMPT_TEMPLATE.format(stem=problem["stem"])

        # Retry with exponential backoff for rate limits
        max_retries = 5
        response = None
        tpd_exhausted = False
        for attempt in range(max_retries + 1):
            try:
                response = client.query(prompt, system=SYSTEM_PROMPT)
                break  # Success
            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()
                # Detect daily-token-budget exhaustion: retrying within the
                # same calendar day is futile. Exit the loop and let the
                # outer code save partial results.
                is_tpd = "tokens per day" in err_str.lower() or "TPD" in err_str
                if is_tpd:
                    print(f"✗ (TPD exhausted; saving partial results)")
                    raw_results.append({
                        "problem_id": problem["id"],
                        "response": "",
                        "parsed": {},
                        "success": False,
                        "error": err_str,
                    })
                    tpd_exhausted = True
                    break
                if is_rate_limit and attempt < max_retries:
                    wait = 8 * (2 ** attempt)  # 8, 16, 32, 64, 128 seconds
                    print(f"⏳ rate limited, waiting {wait}s (retry {attempt+1}/{max_retries})... ", end="", flush=True)
                    time.sleep(wait)
                    continue
                else:
                    print(f"✗ (error: {e})")
                    raw_results.append({
                        "problem_id": problem["id"],
                        "response": "",
                        "parsed": {},
                        "success": False,
                        "error": err_str,
                    })
                    response = None
                    break

        if tpd_exhausted:
            print(f"  Stopping early: model has exhausted its daily token budget. "
                  f"Returning {len(predictions)} predictions on {i+1} attempted problems.")
            break

        if response is not None:
            parsed = parse_response(response)

            if parsed["point_estimate"] is not None:
                pred = Prediction(
                    problem_id=problem["id"],
                    predicted_answer=parsed["point_estimate"],
                    predicted_confidence=parsed["confidence"],
                    predicted_interval=parsed["confidence_interval"],
                    predicted_framework=parsed["framework"],
                    raw_response=response,
                )
                predictions.append(pred)
                print(f"✓ (est={parsed['point_estimate']:.3f}, conf={parsed['confidence']:.2f})")
            else:
                print(f"✗ (could not parse response)")

            raw_results.append({
                "problem_id": problem["id"],
                "response": response,
                "parsed": parsed,
                "success": parsed["point_estimate"] is not None,
            })

        time.sleep(delay)

    return predictions, raw_results


def main():
    parser = argparse.ArgumentParser(description="Run MURU-BENCH evaluation.")
    parser.add_argument("--model", "-m", required=True, help="Model name (e.g., gpt-4o, claude-3.5-sonnet)")
    parser.add_argument("--subset", "-s", default=str(PROJECT_ROOT / "data" / "test"), help="Problem directory.")
    parser.add_argument("--n", type=int, help="Max number of problems to evaluate.")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls (seconds).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the deterministic evaluation-order shuffle (keeps quota-truncated runs difficulty-representative).")
    parser.add_argument("--save", action="store_true", help="Save results to evaluation/baselines/.")
    args = parser.parse_args()

    print(f"\n{'═' * 60}")
    print(f"  MURU-BENCH Evaluation")
    print(f"  Model: {args.model}")
    print(f"  Subset: {args.subset}")
    print(f"{'═' * 60}\n")

    problems = load_problems(args.subset)
    if not problems:
        print(f"No problems found in {args.subset}")
        sys.exit(1)

    print(f"  Loaded {len(problems)} problems\n")

    predictions, raw_results = run_evaluation(
        args.model, problems, max_n=args.n, delay=args.delay, seed=args.seed
    )

    if not predictions:
        print("\n  No valid predictions obtained. Cannot compute metrics.")
        sys.exit(1)

    # Compute metrics
    metrics = MURUMetrics(problems, predictions)
    print(metrics.summary())

    # Save results
    if args.save:
        baselines_dir = PROJECT_ROOT / "evaluation" / "baselines"
        baselines_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_slug = args.model.replace("/", "_").replace(".", "_")

        results_data = {
            "model": args.model,
            "timestamp": timestamp,
            "seed": args.seed,
            "n_problems": len(problems),
            "n_predictions": len(predictions),
            "metrics": metrics.compute_all(),
            "raw_results": raw_results,
        }

        outpath = baselines_dir / f"{model_slug}_{timestamp}.json"
        with open(outpath, "w") as f:
            json.dump(results_data, f, indent=2, default=str)

        print(f"\n  Results saved to: {outpath.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
