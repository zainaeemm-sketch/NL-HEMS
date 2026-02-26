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
3.3 Install dependencies
python3 -m pip install ortools pydantic pandas numpy scikit-learn streamlit altair matplotlib

(Optional) Save dependencies:

python3 -m pip freeze > requirements.txt
3.4 Select interpreter in VS Code

Cmd+Shift+P → Python: Select Interpreter

Choose: .../nl-hems-dr-bench-v2/.venv/bin/python

4) Running the benchmark (CLI)
4.1 Ensure dataset exists

data/samples/benchmark_samples.jsonl must contain one JSON object per line.

Example line:

{"sample_id":"S1","command":"Guests tonight, keep it warm but don’t run up the bill.","gold":{"expect_guest_event":true,"clarification_needed":true,"min_temp_c":21.0}}
4.2 Run one benchmark
source .venv/bin/activate
python3 src/benchmark_runner.py

Outputs:

data/outputs/benchmark_results.csv

data/outputs/benchmark_results.json

4.3 Run comparisons (stub vs llm × soft vs hard comfort)
python3 src/comparison_runner.py

Outputs:

data/outputs/comparison_results.csv

data/outputs/comparison_summary.csv

5) Streamlit dashboard (professional UI + graphs)
5.1 Start Streamlit

Always run from project root:

cd ~/Projects/nl-hems-dr-bench-v2
source .venv/bin/activate
streamlit run app.py

Dashboard features:

KPI cards

Per-sample graphs (comfort violations, DR compliance)

Tradeoff scatter plot (comfort vs DR)

Editable input sliders + price table

Dataset editor + JSONL validation

Back navigation button

6) Editing inputs (so results are not all the same)

Your results can look “similar” if the environment is constant.
You can change inputs in three ways:

A) Streamlit global overrides (recommended)

Use sidebar controls:

horizon hours

initial temp

DR hours selection

price table edits

heat-loss and heat-gain dynamics

weight multipliers for comfort/cost/DR

B) Per-sample context inside JSONL

Add a context object per sample:

{
  "sample_id":"S9",
  "command":"No one is home tonight, save energy.",
  "gold":{"min_temp_c":19.0},
  "context":{
    "initial_temp_c":22.0,
    "dr_event_hours":[20,21],
    "loss_c_per_hour":0.6,
    "heat_c_per_hour":0.7,
    "price":{"16":2,"17":2,"18":5,"19":9,"20":9,"21":9,"22":3,"23":2}
  }
}
C) Change the natural-language commands

Different intent → different fuzzy mapping → different schedules.

7) LLM setup (Vector Engine OpenAI-compatible)
IMPORTANT SECURITY NOTE

Never paste API keys into code or chat. If a key was exposed, rotate/revoke it immediately.

7.1 Recommended: .streamlit/secrets.toml

Create:
nl-hems-dr-bench-v2/.streamlit/secrets.toml

Example:

OPENAI_BASE_URL = "https://api.vectorengine.ai/v1"
OPENAI_API_KEY = "YOUR_KEY_HERE"
LLM_MODEL = "gpt-5-mini-2025-08-07"
LLM_BACKEND = "auto"

Then in Streamlit the sidebar should show:

API key set = True

7.2 CLI env vars (alternative)
export OPENAI_BASE_URL=https://api.vectorengine.ai/v1
export OPENAI_API_KEY='YOUR_KEY_HERE'
export LLM_MODEL='gpt-5-mini-2025-08-07'
export LLM_BACKEND=auto
7.3 Switch stub vs LLM

In src/benchmark_runner.py set:

PARSER_MODE = "llm"   # or "stub"

If LLM fails, the system should fallback to stub and record:

parser_used, parser_fallback, parser_error

8) Generate paper figures as PNG (IEEE-ready)
8.1 Run figure generator
source .venv/bin/activate
python3 src/paper_figures.py --outdir paper_figures

It reads latest comparison_results*.csv / comparison_summary*.csv (or provide explicit paths).

Outputs saved in:

paper_figures/fig_*.png

paper_figures/figures_manifest.json

9) Troubleshooting
“python: command not found” (macOS)

Use python3 everywhere:

python3 --version
venv not active (no (.venv) in prompt)
source .venv/bin/activate
Streamlit shows API Key set = False

Ensure you ran Streamlit from project root:

cd ~/Projects/nl-hems-dr-bench-v2
streamlit run app.py

Confirm secrets path:

ls -la .streamlit/secrets.toml
JSONDecodeError in dataset

Your JSONL file must be one JSON object per line.
Use the Dataset Editor page in Streamlit to validate.

Same schedule every time

Change:

prices/DR hours/dynamics via Streamlit overrides

or add per-sample context

10) Using Visual Studio (full IDE) instead of VS Code

This project is Python-first. Visual Studio can run Python, but VS Code is simpler.
If you must use Visual Studio:

ensure it uses the same .venv interpreter

run scripts from the project root

keep python3 usage on macOS

11) What to tell professors (short pitch)

“This system turns fuzzy natural-language household goals into DR-aware schedules by using an LLM to extract structured intent, mapping language into fuzzy preferences, and then using OR-Tools constraint optimization to compute safe schedules. We benchmark outcomes like comfort violations and DR compliance, and show how interaction design (clarification vs direct action) changes performance.”

License

Add your preferred license here (MIT/Apache-2.0/etc).