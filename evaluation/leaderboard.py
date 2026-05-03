#!/usr/bin/env python3
"""
leaderboard.py — MURU-BENCH Leaderboard Generator

Aggregates all evaluation results (baselines + real models) and generates
a formatted leaderboard table for README.md and paper updates.

Usage:
    python evaluation/leaderboard.py                 # Print leaderboard
    python evaluation/leaderboard.py --update-readme # Auto-update README.md
    python evaluation/leaderboard.py --latex          # Output LaTeX table
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINES_DIR = PROJECT_ROOT / "evaluation" / "baselines"


def load_all_results() -> list[dict]:
    """Load all evaluation result files."""
    results = []

    # Load individual result files
    for f in sorted(BASELINES_DIR.glob("*.json")):
        if f.name in ("summary.json", "bootstrap_summary.json"):
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
            if "metrics" in data and "model" in data:
                results.append({
                    "model": data["model"],
                    "accuracy": data["metrics"].get("accuracy", 0),
                    "ece": data["metrics"].get("ece", 1.0),
                    "overconfidence": data["metrics"].get("overconfidence_rate", 0),
                    "framework_match": data["metrics"].get("framework_match_rate", 0),
                    "n_problems": data.get("n_predictions", data.get("n_problems", 0)),
                    "timestamp": data.get("timestamp", ""),
                    "source": "api" if "raw_results" in data else "simulator",
                    "d5_accuracy": _get_d5(data),
                })
        except (json.JSONDecodeError, IOError, KeyError):
            continue

    # Also try summary.json for simulated baselines
    summary_path = BASELINES_DIR / "summary.json"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            existing_models = {r["model"] for r in results}
            for name, data in summary["models"].items():
                if name not in existing_models:
                    m = data["metrics"]
                    diff = m.get("accuracy_by_difficulty", {})
                    d5 = diff.get("5", diff.get(5, {}))
                    results.append({
                        "model": name,
                        "accuracy": m["accuracy"],
                        "ece": m["ece"],
                        "overconfidence": m["overconfidence_rate"],
                        "framework_match": m["framework_match_rate"],
                        "n_problems": m["total_evaluated"],
                        "timestamp": "",
                        "source": "simulator",
                        "d5_accuracy": d5.get("accuracy", None),
                    })
        except (json.JSONDecodeError, IOError, KeyError):
            pass

    # Sort by accuracy descending
    results.sort(key=lambda x: -x["accuracy"])
    return results


def _get_d5(data: dict) -> float | None:
    """Extract D5 accuracy from result data."""
    try:
        diff = data["metrics"]["accuracy_by_difficulty"]
        d5 = diff.get("5", diff.get(5, {}))
        return d5.get("accuracy", None)
    except (KeyError, TypeError):
        return None


def format_model_name(name: str) -> str:
    """Format model name for display."""
    name = name.replace("_", " ").replace("-", " ")
    # Capitalize known model names
    replacements = {
        "gpt 4o": "GPT-4o",
        "gpt 4o mini": "GPT-4o-mini",
        "gpt 3.5 turbo": "GPT-3.5-Turbo",
        "claude 3.5 sonnet": "Claude 3.5 Sonnet",
        "claude 3 opus": "Claude 3 Opus",
        "gemini 1.5 pro": "Gemini 1.5 Pro",
        "gemini 1.5 flash": "Gemini 1.5 Flash",
        "gemini 2.0 flash": "Gemini 2.0 Flash",
        "llama 3.1 70b": "Llama 3.1 70B",
        "llama 3.1 8b": "Llama 3.1 8B",
        "mixtral 8x7b": "Mixtral 8x7B",
        "random baseline": "Random Baseline",
        "heuristic baseline": "Heuristic Baseline",
        "competent model": "Competent (sim.)",
        "strong model": "Strong (sim.)",
        "expert model": "Expert (sim.)",
    }
    for k, v in replacements.items():
        if k in name.lower():
            return v
    return name.title()


def generate_markdown_table(results: list[dict]) -> str:
    """Generate markdown leaderboard table."""
    lines = [
        "## 🏆 Leaderboard",
        "",
        "| Rank | Model | Acc@CI ↑ | ECE ↓ | OvConf ↓ | FwMatch ↑ | D5 Acc | Source |",
        "|------|-------|---------|-------|----------|-----------|--------|--------|",
    ]

    for i, r in enumerate(results, 1):
        name = format_model_name(r["model"])
        d5 = f"{r['d5_accuracy']:.0%}" if r["d5_accuracy"] is not None else "—"
        source = "🤖 API" if r["source"] == "api" else "🔧 Sim."
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}"

        lines.append(
            f"| {medal} | **{name}** | {r['accuracy']:.1%} | "
            f"{r['ece']:.3f} | {r['overconfidence']:.1%} | "
            f"{r['framework_match']:.1%} | {d5} | {source} |"
        )

    lines.extend([
        "",
        f"*Last updated: {datetime.now().strftime('%Y-%m-%d')}*",
        "",
        "**Legend**: Acc@CI = Accuracy (point estimate within ground-truth CI), "
        "ECE = Expected Calibration Error, OvConf = Overconfidence Rate, "
        "FwMatch = Framework Match Rate, D5 = Difficulty 5 accuracy. "
        "🤖 = real model via API, 🔧 = simulated baseline.",
    ])

    return "\n".join(lines)


def generate_latex_table(results: list[dict]) -> str:
    """Generate LaTeX leaderboard table."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{MURU-BENCH Leaderboard. Models ranked by Accuracy@CI on the test set ($n=301$).}",
        r"\label{tab:leaderboard}",
        r"\begin{tabular}{rlcccc}",
        r"\toprule",
        r"\textbf{\#} & \textbf{Model} & \textbf{Acc@CI}$\uparrow$ & "
        r"\textbf{ECE}$\downarrow$ & \textbf{OvConf}$\downarrow$ & \textbf{D5} \\",
        r"\midrule",
    ]

    for i, r in enumerate(results, 1):
        name = format_model_name(r["model"])
        d5 = f"{r['d5_accuracy']:.0%}" if r["d5_accuracy"] is not None else "---"
        lines.append(
            f"  {i} & {name} & {r['accuracy']:.1%} & {r['ece']:.3f} & "
            f"{r['overconfidence']:.1%} & {d5} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def update_readme(table: str):
    """Update README.md with the leaderboard."""
    readme_path = PROJECT_ROOT / "README.md"
    content = readme_path.read_text()

    # Find and replace leaderboard section
    start_marker = "## 🏆 Leaderboard"
    end_markers = ["## ", "---"]

    if start_marker in content:
        start = content.index(start_marker)
        # Find the next section after leaderboard
        rest = content[start + len(start_marker):]
        end = len(content)
        for marker in end_markers:
            idx = rest.find(f"\n{marker}")
            if idx > 0 and start + len(start_marker) + idx < end:
                end = start + len(start_marker) + idx

        content = content[:start] + table + "\n\n" + content[end:]
    else:
        # Insert before baseline results
        insert_before = "## 📈 Baseline Results"
        if insert_before in content:
            idx = content.index(insert_before)
            content = content[:idx] + table + "\n\n" + content[idx:]
        else:
            content += "\n\n" + table

    readme_path.write_text(content)
    print(f"  ✓ Updated {readme_path.relative_to(PROJECT_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="MURU-BENCH Leaderboard")
    parser.add_argument("--update-readme", action="store_true", help="Update README.md")
    parser.add_argument("--latex", action="store_true", help="Output LaTeX table")
    args = parser.parse_args()

    results = load_all_results()

    if not results:
        print("No evaluation results found in evaluation/baselines/")
        print("Run 'python evaluation/run_baselines.py --save' first.")
        sys.exit(1)

    print(f"\n{'═' * 60}")
    print(f"  MURU-BENCH Leaderboard ({len(results)} entries)")
    print(f"{'═' * 60}\n")

    table = generate_markdown_table(results)
    print(table)

    if args.latex:
        print("\n" + "─" * 60)
        print(generate_latex_table(results))

    if args.update_readme:
        update_readme(table)


if __name__ == "__main__":
    main()
