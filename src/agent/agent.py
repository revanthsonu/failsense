"""
src/agent/agent.py

Day 5: LLM reasoning agent — the output layer of FailSense.

Takes a live sensor anomaly and synthesizes a structured diagnosis
by combining three inputs:
  1. Cross-modal retrieval results (from FAISS, Day 3)
  2. Zero-shot failure mode classification (DeBERTa NLI, Day 4)
  3. Historical anomaly memory (sliding window of last 24h per engine)

Three tools available to the agent:
  - get_sensor_history: retrieves recent anomaly scores for a unit
  - query_maintenance_db: full-text search over maintenance corpus
  - estimate_rul: heuristic RUL estimate from anomaly score trajectory

Output: structured Pydantic object — failure mode, confidence,
estimated RUL, mechanistic explanation, recommended action, citations.

Model: GPT-4o via OpenAI API (falls back to llama-3-70b via Groq)
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from collections import deque

import numpy as np
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.append(str(Path(__file__).parent.parent))
from retrieval.build_index import CrossModalRetriever
from models.classifier import FailureModeClassifier, ClassificationResult


# ─── Pydantic output schema ────────────────────────────────────────────────

class FailureDiagnosis(BaseModel):
    """
    Structured output from the LLM reasoning agent.
    Every field is required — the agent must commit to a value.
    Uncertainty is expressed through confidence scores, not omission.
    """
    failure_mode: str = Field(
        description="Primary failure mode (from OSHA turbofan taxonomy)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Agent confidence in failure_mode (0–1)"
    )
    estimated_rul_cycles: int = Field(
        ge=0,
        description="Estimated remaining useful life in operating cycles"
    )
    urgency: str = Field(
        description="One of: CRITICAL / HIGH / MEDIUM / LOW"
    )
    explanation: str = Field(
        description="2–3 sentence mechanistic explanation of the failure"
    )
    recommended_action: str = Field(
        description="Specific maintenance action with timeline"
    )
    evidence_log_ids: list[int] = Field(
        default_factory=list,
        description="Log IDs from maintenance corpus that informed this diagnosis"
    )
    sensor_contributors: list[str] = Field(
        default_factory=list,
        description="Sensor channels with highest anomaly contribution"
    )


# ─── Sliding memory per engine ─────────────────────────────────────────────

@dataclass
class EngineMemory:
    """
    Sliding window memory of anomaly events per engine unit.
    Stores the last N anomaly scores + diagnoses so the agent
    can reason about trends, not just point-in-time anomalies.
    """
    unit_id: int
    max_len: int = 48   # ~24 hours at 30-min sampling
    anomaly_scores: deque = field(default_factory=deque)
    diagnoses: deque = field(default_factory=deque)

    def add(self, score: float, cycle: int, diagnosis: Optional[FailureDiagnosis] = None):
        if len(self.anomaly_scores) >= self.max_len:
            self.anomaly_scores.popleft()
            self.diagnoses.popleft()
        self.anomaly_scores.append({"score": score, "cycle": cycle})
        self.diagnoses.append(diagnosis)

    def trend(self) -> str:
        """Describes whether anomaly score is rising, stable, or falling."""
        if len(self.anomaly_scores) < 3:
            return "insufficient history"
        scores = [e["score"] for e in self.anomaly_scores]
        recent = np.mean(scores[-5:])
        earlier = np.mean(scores[:-5]) if len(scores) > 5 else scores[0]
        delta = recent - earlier
        if delta > 0.05:
            return f"rising ({delta:+.3f} over last {len(scores)} readings)"
        elif delta < -0.05:
            return f"improving ({delta:+.3f})"
        return "stable"

    def summary(self) -> str:
        if not self.anomaly_scores:
            return "No history available."
        scores = [e["score"] for e in self.anomaly_scores]
        cycles = [e["cycle"] for e in self.anomaly_scores]
        return (
            f"Unit {self.unit_id}: {len(scores)} readings over cycles "
            f"{min(cycles)}–{max(cycles)}. "
            f"Mean anomaly: {np.mean(scores):.3f}, max: {max(scores):.3f}. "
            f"Trend: {self.trend()}."
        )


# ─── Tool definitions ──────────────────────────────────────────────────────

# Module-level memory store (persists across agent calls in one session)
_engine_memory_store: dict[int, EngineMemory] = {}
_retriever: Optional[CrossModalRetriever] = None
_maintenance_corpus: list[dict] = []


def init_agent_context(
    retriever: CrossModalRetriever,
    maintenance_corpus: list[dict],
):
    """Call once before running the agent to inject dependencies."""
    global _retriever, _maintenance_corpus
    _retriever           = retriever
    _maintenance_corpus  = maintenance_corpus


@tool
def get_sensor_history(unit_id: int) -> str:
    """
    Retrieve the anomaly score history for a specific engine unit.
    Returns trend analysis and summary statistics.
    Use this to understand if degradation is accelerating or stable.
    """
    if unit_id not in _engine_memory_store:
        return f"No history recorded for unit {unit_id} in current session."
    return _engine_memory_store[unit_id].summary()


@tool
def query_maintenance_db(query: str) -> str:
    """
    Search the maintenance log corpus for records relevant to a symptom or failure mode.
    Use natural language queries like 'bearing wear high temperature' or 'compressor stall'.
    Returns the 3 most relevant maintenance records with their outcomes.
    """
    if not _maintenance_corpus:
        return "Maintenance corpus not loaded."

    # Simple keyword match — FAISS retrieval is used for the main pipeline;
    # this tool gives the agent free-form search capability
    query_lower = query.lower()
    matches = [
        log for log in _maintenance_corpus
        if any(word in log["text"].lower() for word in query_lower.split())
    ]

    if not matches:
        return "No matching maintenance records found."

    # Return top 3 by relevance (word overlap count)
    def relevance(log):
        return sum(1 for w in query_lower.split() if w in log["text"].lower())

    top3 = sorted(matches, key=relevance, reverse=True)[:3]
    result = []
    for log in top3:
        critical = "CRITICAL" if log["is_critical"] else "routine"
        result.append(
            f"[Log {log['log_id']} — {log['failure_mode']} — {critical}] "
            f"Unit {log['unit_id']}, Cycle {log['cycle']}: {log['text']}"
        )
    return "\n\n".join(result)


@tool
def estimate_rul(unit_id: int, current_anomaly_score: float) -> str:
    """
    Estimate remaining useful life in operating cycles based on current
    anomaly score and historical trend for the engine unit.
    Returns a range estimate with confidence level.
    """
    # Heuristic RUL model (before full regression is trained in eval/):
    # Anomaly score > 0.8: imminent failure (< 20 cycles)
    # Anomaly score 0.5–0.8: moderate degradation (20–80 cycles)
    # Anomaly score 0.2–0.5: early degradation (80–200 cycles)
    # Anomaly score < 0.2: healthy (> 200 cycles)

    memory = _engine_memory_store.get(unit_id)
    trend  = memory.trend() if memory else "unknown"

    if current_anomaly_score > 0.80:
        rul_range  = "< 20 cycles"
        confidence = "HIGH"
    elif current_anomaly_score > 0.50:
        rul_range  = "20–80 cycles"
        confidence = "MEDIUM"
    elif current_anomaly_score > 0.20:
        rul_range  = "80–200 cycles"
        confidence = "MEDIUM"
    else:
        rul_range  = "> 200 cycles"
        confidence = "LOW (healthy range)"

    return (
        f"Unit {unit_id} — RUL estimate: {rul_range} "
        f"(confidence: {confidence}, anomaly score: {current_anomaly_score:.3f}, "
        f"trend: {trend})"
    )


# ─── Agent ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are FailSense, an expert industrial predictive maintenance AI.

You diagnose turbofan engine anomalies by reasoning over:
- Live sensor anomaly embeddings (already processed into retrieval results)
- Historical maintenance records from the corpus
- Per-engine anomaly trend history

You have access to three tools:
1. get_sensor_history — check degradation trend for an engine unit
2. query_maintenance_db — search maintenance records by symptom
3. estimate_rul — get heuristic remaining useful life estimate

Always:
- Call get_sensor_history and estimate_rul before concluding
- Base your failure_mode on the retrieved evidence and zero-shot classification
- Be specific about the mechanical cause, not just "anomaly detected"
- Express urgency as: CRITICAL (< 20 cycles), HIGH (20–80), MEDIUM (80–200), LOW (> 200)
- Cite specific log IDs that informed your diagnosis

Your final answer MUST be valid JSON matching the FailureDiagnosis schema.
Do not include markdown code fences — return raw JSON only."""


class FailSenseAgent:
    """
    LLM reasoning agent for FailSense.

    Wraps LangChain tool-calling agent with:
    - GPT-4o as backbone (Groq/llama-3 fallback)
    - 3 domain tools
    - Per-engine sliding memory
    - Structured Pydantic output
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.1,   # low temp for consistent structured output
    ):
        # Try OpenAI first, fall back to Groq
        if os.getenv("OPENAI_API_KEY"):
            self.llm = ChatOpenAI(model=model, temperature=temperature)
        elif os.getenv("GROQ_API_KEY"):
            from langchain_groq import ChatGroq
            self.llm = ChatGroq(model="llama-3-70b-8192", temperature=temperature)
        else:
            raise EnvironmentError(
                "Set OPENAI_API_KEY or GROQ_API_KEY in .env"
            )

        self.tools = [get_sensor_history, query_maintenance_db, estimate_rul]

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True,
        )

    def diagnose(
        self,
        unit_id: int,
        cycle: int,
        anomaly_score: float,
        retrieved_logs: list[dict],
        classification: ClassificationResult,
        top_sensors: list[str],
    ) -> FailureDiagnosis:
        """
        Run the agent to produce a structured diagnosis.

        Args:
            unit_id: engine unit number
            cycle: current operating cycle
            anomaly_score: reconstruction error from autoencoder
            retrieved_logs: top-k results from CrossModalRetriever
            classification: zero-shot failure mode classification
            top_sensors: sensor names with highest reconstruction error

        Returns:
            FailureDiagnosis — structured, validated output
        """
        # Update engine memory
        if unit_id not in _engine_memory_store:
            _engine_memory_store[unit_id] = EngineMemory(unit_id)
        _engine_memory_store[unit_id].add(anomaly_score, cycle)

        # Build context summary for the agent
        retrieved_summary = "\n".join([
            f"  Log {log['log_id']} (similarity {log.get('similarity', 0):.3f}): "
            f"{log['failure_mode']} — {log['text'][:120]}..."
            for log in retrieved_logs[:5]
        ])

        user_message = f"""
Engine unit {unit_id} at cycle {cycle}:

ANOMALY SCORE: {anomaly_score:.4f}
TOP DEGRADED SENSORS: {', '.join(top_sensors[:3])}

ZERO-SHOT CLASSIFICATION:
  Top label: {classification.top_label} (score: {classification.top_score:.3f})
  Top 3: {classification.top_k(3)}

CROSS-MODAL RETRIEVAL (top 5 similar past events):
{retrieved_summary}

Please diagnose this anomaly. Use your tools to check sensor history
and estimate RUL. Return a JSON object matching the FailureDiagnosis schema.
""".strip()

        # Run agent
        response = self.executor.invoke({"input": user_message})
        raw_output = response["output"]

        # Parse structured output
        try:
            # Strip any accidental markdown fences
            clean = raw_output.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data  = json.loads(clean)
            diagnosis = FailureDiagnosis(**data)
        except Exception as e:
            # Fallback: construct from classification if JSON parse fails
            print(f"[WARN] JSON parse failed ({e}), using classification fallback")
            diagnosis = FailureDiagnosis(
                failure_mode       = classification.top_label,
                confidence         = classification.top_score,
                estimated_rul_cycles = 100,
                urgency            = "MEDIUM",
                explanation        = f"Anomaly score {anomaly_score:.3f} with {classification.top_label} pattern. Fallback diagnosis.",
                recommended_action = "Inspect and schedule maintenance within next 50 cycles.",
                evidence_log_ids   = [l["log_id"] for l in retrieved_logs[:3]],
                sensor_contributors= top_sensors[:3],
            )

        # Store diagnosis in memory
        _engine_memory_store[unit_id].diagnoses.append(diagnosis)
        return diagnosis


if __name__ == "__main__":
    # Smoke test with mock data (no real models needed)
    print("FailureDiagnosis schema:")
    print(json.dumps(FailureDiagnosis.model_json_schema(), indent=2))

    print("\nEngineMemory test:")
    mem = EngineMemory(unit_id=42)
    for i, score in enumerate([0.1, 0.15, 0.2, 0.35, 0.5, 0.65, 0.8]):
        mem.add(score, cycle=100 + i * 10)
    print(mem.summary())
