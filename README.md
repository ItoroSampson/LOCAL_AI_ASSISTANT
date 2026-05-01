# Local LLM Benchmarking & Evaluation Framework
### *A Production-Grade MLOps Approach to Edge AI Model Selection*

> **TL;DR:** Built a Python-based evaluation framework that benchmarks Small Language Models (SLMs) on 16GB hardware using hard performance metrics and an LLM-as-a-Judge architecture — replacing subjective assessment with reproducible, data-driven model selection.

---

## The Problem

Leaderboard benchmarks are measured on A100s. Your production environment is not an A100.

When deploying LLMs at the edge — on constrained hardware, without GPU acceleration — you need a framework that answers one question: **which model actually works on *this* machine, for *this* task?** This project builds that framework from scratch.

---

## What This Framework Measures

| Metric | What It Tells You |
|--------|-------------------|
| **TPS** (Tokens Per Second) | Raw inference throughput — how fast the model generates |
| **TTFT** (Time to First Token) | Perceived latency — how long before the user sees anything |
| **RAM Delta** | Memory footprint — headroom left for the OS and other processes |
| **Logic Accuracy** | Reasoning quality — validated against ground-truth logic benchmarks |

---

## Results

Tested on a **16GB Windows laptop** (CPU-only inference via Ollama):

| Model | Avg. TPS | Avg. TTFT | RAM Usage | Logic Accuracy | Verdict |
|:------|:--------:|:---------:|:---------:|:--------------:|:-------:|
| **Llama 3.2 (1B)** | **10.3** | **3.2s** | **< 1GB** | ❌ Low (Hallucinates) | Real-time chat only |
| **Phi-4-Mini (3.8B)** | 5.1 | 5.4s | ~2.4GB | ✅ High (Precise) | **Edge Champion** |
| **Mistral (7B)** | 3.1 | 7.2s | ~4GB+ | ✅ High (Verbose) | Judge / Baseline |

**Key finding:** Llama 3.2 is 2x faster than Phi-4-Mini but fails consistently on multi-step logical reasoning. For MLOps pipelines and structured extraction, Phi-4-Mini is the clear winner. Speed without accuracy is not production-ready.

---

## Architecture

### Evaluation Strategy: Full-Circle Cross-Evaluation

To minimise judge bias — a known failure mode in single-judge eval systems — the framework uses two versioned evaluation pipelines:

```
┌─────────────────────────────────────────────────────────┐
│                    judge.py (v1)                        │
│                                                         │
│  Llama 3.2 ──┐                                          │
│               ├──► Mistral 7B (Judge) ──► Score + Log  │
│  Phi-4-Mini ──┘                                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   judge_v2.py (v2)                      │
│                                                         │
│  Mistral 7B ──► Phi-4-Mini (CoT Judge)                 │
│                     │                                   │
│                     ├─ 1. Solve problem internally      │
│                     ├─ 2. Compare to student answer     │
│                     └─ 3. Output structured JSON score  │
└─────────────────────────────────────────────────────────┘
```

**Why v2 matters:** v1 revealed that Mistral occasionally scored confident-but-wrong Llama responses as correct ("confidence bias"). v2 forces the judge to use Chain of Thought — solving the problem before grading — which eliminated this failure mode entirely.

### Data Pipeline

```
Ollama (local inference)
    │
    ▼
Python automation (batch inference + timing)
    │
    ▼
Structured JSON (schema-validated responses + metadata)
    │
    ▼
CSV export (historical analysis + model comparison)
    │
    ▼
Prometheus ──► Grafana (real-time CPU / RAM / GPU dashboards)
```

---

## Engineering Decisions Worth Noting

**Model Garbage Collection:** Mistral 7B caused 25s cold-start TTFTs by consuming RAM that wasn't released between runs. A Python GC routine was implemented to explicitly unload model weights between inferences — reducing memory contention on constrained hardware. Production edge AI systems need resource lifecycle management, not just inference code.

**Reason-then-Rate Prompting:** The v2 judge uses a CoT prompt that forces the model to produce its own step-by-step solution before evaluating the student's answer. This is a pattern used in production eval pipelines to improve judge reliability.

**Structured Output Enforcement:** All model responses are parsed and validated against a JSON schema before being written to CSV. Silent schema failures are caught and logged — not silently corrupted.

---

## Project Structure

```
LOCAL_AI_ASSISTANT/
│
├── judge.py               # v1 — Mistral as direct judge
├── judge_v2.py            # v2 — Phi-4-Mini with CoT reasoning (enhanced)
├── evaluation_result.csv  # Structured dataset of all benchmark runs
├── grafana/
│   └── dashboard.json     # Pre-configured Grafana dashboard template
└── README.md
```

---

## Setup

### Prerequisites
- [Ollama](https://ollama.com/) installed and running locally
- Python 3.10+
- 16GB RAM recommended

### 1. Clone the repo
```bash
git clone https://github.com/ItoroSampson/LOCAL_AI_ASSISTANT
cd LOCAL_AI_ASSISTANT
```

### 2. Install dependencies
```bash
pip install ollama pandas
```

### 3. Pull the models
```bash
ollama pull llama3.2:1b
ollama pull phi4-mini:latest
ollama pull mistral:latest
```

### 4. Run the benchmark
```bash
# v1: Mistral as judge
python judge.py

# v2: Phi-4-Mini with Chain of Thought (recommended)
python judge_v2.py
```

Results are automatically exported to `evaluation_result.csv`.

---

## MLOps Practices Applied

- **Systematic measurement** — hard metrics replace subjective quality assessment
- **Versioned evaluation pipeline** — v1 and v2 are discrete, comparable evaluation strategies
- **Explainability** — all judge outputs include reasoning chains before scores
- **Observability** — Prometheus + Grafana for real-time resource monitoring
- **Structured output** — schema-validated JSON throughout the pipeline

---

## What's Next

- [ ] Extend with RAG integration to test retrieval-augmented reasoning across models
- [ ] Add Langfuse for persistent trace logging and eval scoring
- [ ] Test quantised variants (GGUF Q4/Q8) to measure accuracy-compression trade-offs

---

## Author

**Itoro Sampson** — Cloud AI/Gen AI Engineer  
Building LLM-powered systems, serverless AI pipelines on AWS, and production-grade MLOps infrastructure.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/itoro-sampson-10477b245/)
[![X](https://img.shields.io/badge/X-Follow-000000?style=flat&logo=x)](https://x.com/itoro_samp?s=11)
[![GitHub](https://img.shields.io/badge/GitHub-ItoroSampson-181717?style=flat&logo=github)](https://github.com/ItoroSampson)
