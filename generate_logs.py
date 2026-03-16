"""
src/ingestion/generate_logs.py

Generates synthetic maintenance logs grounded in turbofan domain terminology.
Uses GPT-4o to produce varied, realistic maintenance records that will form
the text corpus for cross-modal retrieval.

Research note: Synthetic log generation is a legitimate and reproducible
approach when real maintenance data is proprietary. The generator is seeded
with real turbofan component taxonomy and OSHA-aligned maintenance vocabulary
to ensure domain fidelity.
"""

import json
import random
import argparse
from pathlib import Path
from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from env

# Turbofan component taxonomy (from GE/P&W maintenance manuals, public domain)
COMPONENTS = [
    "high-pressure turbine (HPT) disc",
    "low-pressure turbine (LPT) blade",
    "high-pressure compressor (HPC) rotor",
    "fan blade tip",
    "combustor liner",
    "turbine blade tip clearance",
    "bearing assembly (stage 2)",
    "oil seal",
    "fuel nozzle",
    "inlet guide vane",
    "turbine shroud",
    "rotor shaft",
]

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

ACTIONS = [
    "replaced",
    "inspected and cleared",
    "measured tip clearance — within spec",
    "borescope inspection — no anomaly",
    "scheduled for replacement within 200 cycles",
    "emergency shutdown triggered",
    "oil sample sent for spectrographic analysis",
    "vibration dampening adjusted",
]

SYSTEM_PROMPT = """You are a senior aircraft engine maintenance technician with
20 years of experience writing maintenance logs for turbofan engines.

Write realistic, concise maintenance log entries. Each log should:
- Reference specific engine unit numbers and cycle counts
- Name specific components using technical terminology
- Describe the observed anomaly and the action taken
- Vary in tone (some urgent, some routine)
- Be 2-4 sentences long
- Never use generic language like "checked the engine" — be specific

Output ONLY the log entry text. No preamble, no metadata."""


def generate_log(
    unit_id: int,
    cycle: int,
    failure_mode: str,
    component: str,
    is_critical: bool,
) -> str:
    """Generate a single maintenance log entry via GPT-4o."""

    urgency = "critical — engine grounded" if is_critical else "routine inspection"

    user_prompt = f"""Write a maintenance log entry for:
- Engine Unit: {unit_id}
- Operating Cycle: {cycle}
- Observed Issue: {failure_mode} in {component}
- Urgency: {urgency}
- Action taken: {random.choice(ACTIONS)}

Be specific. Use real turbofan maintenance language."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=200,
        temperature=0.85,   # high temp for variety across logs
    )

    return response.choices[0].message.content.strip()


def generate_corpus(
    n_logs: int = 1200,
    out_path: str = "data/synthetic_logs/maintenance_corpus.json",
    seed: int = 42,
) -> list[dict]:
    """
    Generate full maintenance log corpus.

    Each entry is a dict with:
        - log_id: unique identifier
        - unit_id: engine unit (maps loosely to CMAPSS unit_ids)
        - cycle: operating cycle at time of log
        - failure_mode: ground-truth failure category
        - component: affected component
        - is_critical: whether this was a near-failure event
        - text: the generated log text

    The (failure_mode, is_critical) fields serve as ground truth for
    evaluating retrieval quality — a retrieved log is "correct" if its
    failure_mode matches the query sensor's true failure mode.
    """
    random.seed(seed)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    corpus = []
    print(f"Generating {n_logs} maintenance logs...")

    for i in range(n_logs):
        unit_id      = random.randint(1, 100)
        cycle        = random.randint(50, 350)
        failure_mode = random.choice(FAILURE_MODES)
        component    = random.choice(COMPONENTS)
        is_critical  = random.random() < 0.25   # 25% critical events

        try:
            text = generate_log(unit_id, cycle, failure_mode,
                                component, is_critical)
        except Exception as e:
            print(f"  [warn] Log {i} generation failed: {e}. Using template.")
            text = (
                f"Unit {unit_id}, Cycle {cycle}: Routine inspection of "
                f"{component}. Observed signs of {failure_mode}. "
                f"{'Grounded for immediate repair.' if is_critical else 'Logged for next maintenance window.'}"
            )

        entry = {
            "log_id":       i,
            "unit_id":      unit_id,
            "cycle":        cycle,
            "failure_mode": failure_mode,
            "component":    component,
            "is_critical":  is_critical,
            "text":         text,
        }
        corpus.append(entry)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_logs} logs generated")

            # Save incrementally — don't lose progress on API failures
            with open(out_path, "w") as f:
                json.dump(corpus, f, indent=2)

    with open(out_path, "w") as f:
        json.dump(corpus, f, indent=2)

    print(f"Corpus saved to {out_path}")
    print(f"Critical events: {sum(e['is_critical'] for e in corpus)}/{n_logs}")
    return corpus


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_logs",   default=1200, type=int)
    parser.add_argument("--out_path", default="data/synthetic_logs/maintenance_corpus.json")
    parser.add_argument("--seed",     default=42,   type=int)
    args = parser.parse_args()

    generate_corpus(args.n_logs, args.out_path, args.seed)
