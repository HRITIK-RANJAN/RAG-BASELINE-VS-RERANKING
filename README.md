# RAG: Baseline vs. Re-Ranking

CL3410 Language Models and Agents, Module 7 (RAG), assignment A1.

Two retrieval pipelines answer questions about one complex PDF, the **Netrunner Comprehensive Rules v26.03** (a 99-page card-game rulebook full of exceptions, negations and cross-references). Both pipelines use the **same frozen chunked corpus**:

| | Pipeline 1 (baseline) | Pipeline 2 (re-ranking) |
|---|---|---|
| Retrieve | bi-encoder `all-MiniLM-L6-v2` + FAISS inner product, top 10 | the **same** top 10 |
| Re-rank | none | cross-encoder `ms-marco-MiniLM-L-6-v2` scores all 10 (query, chunk) pairs |
| LLM | top 3 → Answer 1 | top 3 → Answer 2 (identical prompt) |

Everything is in one notebook: **[`rag_baseline_vs_rerank.ipynb`](rag_baseline_vs_rerank.ipynb)**, executed with all outputs saved.

## Results (23 final queries, from the notebook)

| | Pipeline 1 | Pipeline 2 |
|---|---|---|
| Hit@3 (essential gold chunk in the top 3) | 0.652 | 0.696 |
| MRR@10 | 0.637 | 0.687 |
| Correct answers / refusals | 15 / 7 | 16 / 6 |
| Median retrieval time | 4.2 ms | 39.2 ms |
| Median end-to-end time (incl. Gemini) | 1.3 s | 1.8 s |

- Re-ranking **fixed 4** queries (gold chunk moved from rank 4–7 to rank 1), **hurt 3**, left 2 failing and 12 unchanged; 2 queries were out of reach because the gold chunk was not in the top 10 (recall ceiling).
- Fixes were found for the keyword-trap and buried categories; none of the exception or negation candidates was fixed (Section 11.1).
- The cross-encoder adds about **35 ms** per query (about 2% of end-to-end time). Scoring all chunks would cost about **2.7 ms per chunk**, about 46 minutes per query for a 1-million-chunk corpus.

See Sections 8–11 of the notebook for the rank-shift table, bump charts, evidence packs, latency table and scaling plot. The report is in [`report/`](report/).

## How to run

Python 3.12, macOS or Linux.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then paste a free Gemini API key into .env
```

Open `rag_baseline_vs_rerank.ipynb` with the `.venv` kernel and choose **Restart kernel and Run All**. The notebook:
- downloads the PDF and checks its SHA-256 (the PDF is not committed);
- rebuilds the chunks and checks that they are identical to the frozen `artifacts/chunks.jsonl`;
- reads LLM answers from `artifacts/llm_cache.jsonl` and LLM timings from `artifacts/llm_latency.csv`, so a re-run makes no Gemini request and needs no Gemini key. To deliberately create a missing answer or timing, set `CONFIG["allow_live_llm_calls"] = True`; a Gemini key is then required.

Model weights download into `.cache/huggingface` inside the project.

## Models and settings

| Part | Choice |
|---|---|
| Bi-encoder | `sentence-transformers/all-MiniLM-L6-v2` (384-d, normalised; FAISS `IndexFlatIP`) |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | `gemini-3.1-flash-lite` (Google AI Studio free tier), temperature 0, thinking level `MINIMAL`, max 400 output tokens |
| Chunks | 602 structure-aware chunks, at most 220 tokens (bi-encoder tokenizer), header with the section path |
| k | 10 retrieved, 3 sent to the LLM |

Library versions are pinned in `requirements.txt`.

## Repository layout

```
rag_baseline_vs_rerank.ipynb   the notebook (the deliverable)
queries/queries.yaml           48 candidate queries, gold chunks, expected answers (23 final)
queries/answer_labels.yaml     labels of Answer 1 and Answer 2, stage diagnosis per failure query
artifacts/chunks.jsonl         the frozen corpus, plus corpus_manifest.json (SHA-256)
artifacts/embeddings.npy       bi-encoder vectors of all chunks
artifacts/results/<qid>.json   per query: 10 candidates with both ranks and scores, both prompts, both answers
artifacts/*.csv                candidate mining, metrics, rank shifts, latency (raw runs), LLM latency
artifacts/figures/             bump charts, score scatter, latency bars, cross-encoder scaling
report/                        the report (written by the student)
```


