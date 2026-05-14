# NL-HEMS-DR Bench (VS/VS Code Project)

A research prototype + benchmark for **“Beyond Dashboards”** smart home energy management:
- Users give **fuzzy natural-language commands** (e.g., “Guests tonight—keep warm but don’t run up the bill”)
- A **parser** (stub baseline or LLM) turns language into **structured intent**
- A **fuzzy mapping layer** converts intent into numeric preferences
- A **constraint optimizer (OR-Tools CP-SAT)** generates a **DR-aware schedule**
- A **benchmark runner + Streamlit dashboard** evaluates outcomes and visualizes results

This project is meant to help you **demo** and **measure** how conversational interaction affects:
- comfort vs cost tradeoffs
- demand response (DR) compliance
- clarification behavior (ask vs act)

---

## 1) What is “the model” here?

This is NOT “one neural network you train.” It’s a **hybrid pipeline**:

1. **Natural Language Parser**
   - `stub`: keyword/rule baseline
   - `llm`: OpenAI-compatible endpoint (Vector Engine) returns JSON intent

2. **Fuzzy Preference Mapper**
   - Converts “warm / don’t spend much / tonight” into:
     - comfort targets
     - weights (comfort vs cost vs DR)
     - time windows

3. **Optimization**
   - CP-SAT selects HVAC on/off decisions to minimize:
     - energy cost penalty
     - DR penalty
     - comfort violations
     - switching penalty

4. **Benchmark + Visualization**
   - Runs many samples and exports CSV/JSON + charts (Streamlit + paper PNG figures)

---

## 2) Folder structure


nl-hems-dr-bench-v2/
├─ app.py
├─ requirements.txt
├─ .streamlit/
│ └─ secrets.toml
├─ data/
│ ├─ samples/
│ │ └─ benchmark_samples.jsonl
│ └─ outputs/
├─ src/
│ ├─ core.py
│ ├─ clarification.py
│ ├─ llm_parser.py
│ ├─ benchmark_runner.py
│ ├─ comparison_runner.py
│ └─ paper_figures.py
└─ config/
└─ (optional: dashboard params JSON)

---

## 3) Setup in Visual Studio Code (recommended)

### 3.1 Open the project
- Open **VS Code**
- **File → Open Folder** → `~/Projects/nl-hems-dr-bench-v2`

### 3.2 Create + activate venv (macOS)
In VS Code Terminal:

```bash
cd ~/Projects/nl-hems-dr-bench-v2
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip

**3.3 Install dependencies**

python3 -m pip install ortools pydantic pandas numpy scikit-learn streamlit altair matplotlib


**3.4 Select interpreter in VS Code**

Cmd+Shift+P → Python: Select Interpreter

Choose: .../nl-hems-dr-bench-v2/.venv/bin/python


**4) Running the benchmark (CLI)**
*4.1 Ensure dataset exists*

data/samples/benchmark_samples.jsonl must contain one JSON object per line.

Example line:
{"sample_id":"S1","command":"Guests tonight, keep it warm but don’t run up the bill.","gold":{"expect_guest_event":true,"clarification_needed":true,"min_temp_c":21.0}}

