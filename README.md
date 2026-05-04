# RAG Heuristics Discovery

Modular Python framework for retrieval-augmented heuristic discovery in combinatorial optimization.

## Status: Discovery Pipeline Run Successfully
The pipeline has been verified with **gpt-5-mini-2025-08-07**.

### Initial Run Results (10 iterations)
- **Baseline MDDC-like:** 4110.33
- **Best Discovered Heuristic:** 4115.59 (Feasible)
- **Data Ingested:** 7,411 normalized chunks

## Local Setup

1. **Clone and Install:**
   ```bash
   git clone https://github.com/asefshahriar/RAG_huristics_discovery.git
   cd RAG_huristics_discovery
   pip install -e .
   ```

2. **Configure Environment:**
   Create a `.env` file based on `.env.example`:
   ```env
   OPENAI_API_KEY=your_key_here
   RHD_MODEL_NAME=gpt-5-mini-2025-08-07
   ```

3. **Run Discovery:**
   ```bash
   # Ingest raw data
   python -m rag_heuristics.cli ingest
   
   # Build RAG index
   python -m rag_heuristics.cli build-index
   
   # Run discovery iterations
   python -m rag_heuristics.cli train --iterations 10
   ```

## Colab-first Quickstart (No local setup)

1. Open a new Colab notebook.
2. Clone your repo and install:
   - `!git clone <your-repo-url>`
   - `%cd RAG_huristics_discovery`
   - `!pip install -r requirements-colab.txt`
   - `!pip install -e .`
3. Add secrets in Colab:
   - Set `OPENAI_API_KEY` in Colab Secrets (optional; system has fallback generation if not set).
4. Prepare corpus and run:
   - `!python -m rag_heuristics.cli ingest --source-dir data/raw`
   - `!python -m rag_heuristics.cli build-index`
   - `!python -m rag_heuristics.cli train --problem-type single_machine_total_tardiness --iterations 60 --seed-strategy mdd`
   - `!python -m rag_heuristics.cli compare-tracks --problem-type single_machine_total_tardiness --iterations 60`
   - `!python -m rag_heuristics.cli evaluate --problem-type single_machine_total_tardiness --iterations 40`

## Outputs
- `reports/discovered_algorithms.md`: Paper-style report of discovered heuristics.
- `reports/training_log.jsonl`: Detailed iteration-by-iteration logs.
- `reports/benchmark_summary.json`: formal evaluation results.

## Notes
- The project includes an SMTT plugin and is structured to add future problem plugins.
- Retrieval supports a fallback text mode when vector dependencies are unavailable.
- Uses `gpt-5-mini-2025-08-07` by default with local embeddings.
