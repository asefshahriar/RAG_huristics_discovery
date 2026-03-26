# RAG Heuristics Discovery

Modular Python framework for retrieval-augmented heuristic discovery in combinatorial optimization.

## Colab-first Quickstart (No local setup)

1. Open a new Colab notebook.
2. Clone your repo and install:
   - `!git clone <your-repo-url>`
   - `%cd RAG_huristics_discovery`
   - `!pip install -r requirements-colab.txt`
3. Add secrets in Colab:
   - Set `OPENAI_API_KEY` in Colab Secrets (optional; system has fallback generation if not set).
4. Prepare corpus and run:
   - `!python -m rag_heuristics.cli ingest --source-dir data/raw`
   - `!python -m rag_heuristics.cli build-index`
   - `!python -m rag_heuristics.cli train --problem-type single_machine_total_tardiness --iterations 60`
   - `!python -m rag_heuristics.cli evaluate --problem-type single_machine_total_tardiness --iterations 40`
5. Outputs:
   - `reports/benchmark_summary.json`
   - `reports/benchmark_report.md`
   - `reports/training_log.jsonl`

## Notes
- The project includes an SMTT plugin and is structured to add future problem plugins.
- Retrieval supports a fallback text mode when vector dependencies are unavailable.
