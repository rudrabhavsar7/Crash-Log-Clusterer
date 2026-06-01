# Crash-Log Clusterer

An intelligent pipeline for grouping and diagnosing Android crash logs using sentence embeddings, HDBSCAN clustering, and LLMs for root cause labelling.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set up your environment variables:
   ```bash
   cp .env.example .env
   ```
   Add your `GROQ_API_KEY` to `.env`.

3. Ensure `config.yaml` is populated with the desired hyperparameters.

## Run (single command)

Run the full pipeline (generate data, embed, cluster, label, and evaluate) with a single command:
```bash
python run.py
```

You can also skip specific stages:
```bash
python run.py --skip-data --skip-embed
```

Then, start the interactive Streamlit dashboard:
```bash
streamlit run src/dashboard.py
```

## Eval

The evaluation compares the generated HDBSCAN and KMeans clusters against hand-labelled ground truth.
You can run only the evaluation step via:
```bash
python run.py --eval-only
```
Or by running pytest on the quality gates:
```bash
pytest eval/test_gates.py
```
This ensures the pipeline achieves ≥80% recall and ≥70% purity.

## Architecture

1. **Preprocessing & Embedding (`src/embed_cluster.py`)**: Strips noise from raw stack traces and maps them into dense vectors using `sentence-transformers` (`all-MiniLM-L6-v2`).
2. **Clustering (`src/embed_cluster.py`)**: Uses HDBSCAN to group traces of variable density while effectively isolating unclusterable noise points.
3. **LLM Labelling (`src/llm_labeller.py`)**: Submits the 3 most central traces from each valid cluster to an LLM (Groq `llama-3.1-8b-instant`) to synthesize a human-readable root cause summary.
4. **Evaluation (`eval/run_eval.py`)**: Calculates cluster overlap metrics against `eval/ground_truth.json` to monitor clustering quality over time.
5. **Dashboard (`src/dashboard.py`)**: A Streamlit frontend for interacting with clusters, reading synthesized labels, and visualizing metrics.

## What I Would Do With More Time

- Real Bugsnag webhook integration
- Online/incremental clustering for new incoming traces
- Better embedding model (e.g. CodeBERT for code-aware embeddings)
