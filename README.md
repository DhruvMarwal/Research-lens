# Research Gap Analyzer

A cross-paper RAG assistant built with LangChain (Lab 9, Activity 1: Academic Research Assistant).

It reads a small set of research PDFs and answers questions **only from what is written in them**, with a page number attached to every claim so the answers can be checked. It can:

- answer a question about **one paper**,
- **synthesise across all papers** ("what limitations do they share?"), or
- return a **structured list of research gaps**, each with the papers that mention it and page-cited evidence. This is the main deliverable.

**Domain:** depression detection with fuzzy, neuro-fuzzy and ML methods (5 open-access papers, 270 chunks).
**Team and roles:** *(fill in names and contributions)*

Everything runs in Google Colab. Embeddings and search are local (no key needed). The LLM is Groq `openai/gpt-oss-120b` (used for the reported results), with Gemini as a fallback.

---

## Files in this submission

| File | What it is |
|---|---|
| `GAI_ResearchGapAnalyzer_clean.ipynb` | The working notebook (Steps 0 to 12) |
| `README.md` | This file |
| `Architecture_Phase1.png` | Architecture diagram (must sit next to this README or the image below shows as broken) |
| `eval/results_v1.json` | Evaluation answers from the **old** pipeline (Gemini, section-gated retrieval) |
| `eval/results_v2_groq.json` | Evaluation answers from the **fixed** pipeline (Groq, whole-paper hybrid retrieval) |
| `eval/structured_gaps_sample.json` | Sample structured output from Step 9 |

---

## Architecture

```
Research Papers (5 PDFs)
      │
      ▼
PDF Text Extraction  (PyMuPDF, two-column aware, [[PAGE n]] markers)
      │
      ▼
Section-aware Chunking  (~1000 chars, 150 overlap, never crosses a section)
      │
      ▼
Metadata Tagging  (paper, section, page)
      │
      ▼
Sentence Embeddings  (all-MiniLM-L6-v2, local)
      │
      ▼
FAISS Vector Store
      │
      ▼
Hybrid Retrieval, per paper  (FAISS + BM25, fused with Reciprocal Rank Fusion)
      │
      ▼
Per-Paper Evidence
      │
      ▼
LLM  (Groq gpt-oss-120b, Gemini fallback)
      │
      ├──────────────┐
      ▼              ▼
Single-Paper    Cross-Paper Synthesis (map → reduce)
Analysis              │
                      ▼
            Structured Research Gap Output (Pydantic)
```

![Pipeline architecture](Architecture_Phase1.png)

### LangChain components used

| Component | Where |
|---|---|
| `RecursiveCharacterTextSplitter`, `Document` | Step 2 (chunking) |
| `HuggingFaceEmbeddings`, `FAISS` vector store | Step 3 (index) |
| `BaseRetriever` (custom `HybridPerPaperRetriever`) | Step 4 (retrieval) |
| `ChatPromptTemplate`, LCEL chains (`prompt \| llm \| parser`), `StrOutputParser` | Steps 7 and 8 |
| `PydanticOutputParser` | Step 9 (structured output) |
| `RunnableWithMessageHistory`, `MessagesPlaceholder` | Step 10 (memory) |

---

## Sample test cases (real outputs from the notebook)

### Test 1: structured gap extraction (`gap_limitations`)

**Question:** *What limitations are repeatedly mentioned across the papers?*

The pipeline retrieves the top chunks from each paper, summarises each paper's evidence (map), then returns a typed `ResearchGapList` (Step 9). It found 8 gaps. Three of them:

```
GAP: Both studies use small, demographically narrow samples, limiting statistical power and generalizability.
  Papers: ['Adegboye2021', 'Saha2024']
  - [Adegboye2021 p7] The system was evaluated on a relatively small test set (40 cases shown in the confusion matrix).
  - [Saha2024 p16] The sample is limited in size and demographic breadth; a more varied sample representing many populations would improve generalizability.

GAP: EEG data collection is restricted to frontal-lobe electrodes, introducing spatial bias and omitting information from other brain regions.
  Papers: ['Khan2024']
  - [Khan2024 p24] The study selected only frontal-lobe electrodes to keep the sensor count low, which introduces a certain degree of bias toward the frontal lobe.

GAP: Data were collected via email or social media from voluntary participants, creating a self-selection bias that may limit broader applicability.
  Papers: ['Saha2024']
  - [Saha2024 p7] Data were collected via email or social media from voluntary participants, implying a self-selection bias ...
```

Each gap lists the papers that mention it, and every piece of evidence carries a paper label and page. The full output is in `eval/structured_gaps_sample.json`.

**What to note:** only one gap was shared by two papers. The strict merge rule (two papers go under one gap only if they share the same *cause*, not just the same downstream effect) kept "small sample" and "frontal-lobe only" separate, as intended. The "40 cases" evidence for Adegboye is the model's reading of a confusion matrix, not a limitation the authors wrote, so it is weaker evidence than the Saha quote next to it.

### Test 2: a hard question, before and after the retrieval fix (`failure_pca_count`)

**Question:** *How many papers use PCA?* The answer sits in the Methods sections, which the old section-gated search never looked at.

| | Answer |
|---|---|
| **v1** (old, section-gated) | "None of the papers mention using PCA ... Therefore, 0 papers out of the provided summaries use PCA." |
| **v2** (whole-paper hybrid retrieval) | "2 papers use PCA." Chattopadhyay2017: applied PCA to extract hidden features and reduce dimensionality (p4), with eigenvalues and seven principal components (p5 to p6). Saha2024 is also listed (p4). Zulfiker2021 is correctly excluded: PCA appears only as a benchmark (p2). |

The fix works: v1 said 0, v2 found Chattopadhyay's real use of PCA with page citations. **But v2 is not fully right.** The Saha2024 hit is quoted as "as a dimensionality reduction method, *they* also employ Principal Component Analysis ...", which reads like a description of *other* work in the paper's related-work text, not Saha's own method. So the "2" is probably an overcount. This is the related-work leakage problem described in the limitations below.

---

## 1. What it does, step by step

| Step | What happens |
|---|---|
| 0. Setup | Installs packages, sets run switches (`RUN_EVAL`, `PROVIDER_ORDER`), defines paths. |
| 1. Parse PDFs | Extracts text page by page, handles two-column layouts, strips headers, footers and watermarks, keeps `[[PAGE n]]` markers. |
| 2. Chunk + tag | ~1000-character chunks that never cross a section boundary. Each chunk is tagged with paper, section and page. |
| 3. Embed + index | Local `all-MiniLM-L6-v2` embeddings and a FAISS index (270 chunks). |
| 4. Per-paper hybrid retrieval | For each paper separately, ranks every chunk by meaning (FAISS) and by keywords (BM25), then fuses the two rankings with Reciprocal Rank Fusion. Sections are only a citation label, never a filter. |
| 5. Connect the LLM | Tries Groq first (`gpt-oss-120b`, then smaller fallbacks), then Gemini, using the first model that answers a test call. |
| 6. Helpers | `safe_invoke` (caching, retry with backoff, stops on a daily quota error) and `check_grounding` (page-number check). |
| 7. Single-paper Q&A | Retrieve, build a page-tagged context, answer with page citations. |
| 8. Cross-paper synthesis | Map: summarise each paper's evidence (5 calls). Reduce: merge into one answer (1 call). |
| 9. Structured output | `analyze_gap()` returns a Pydantic `ResearchGapList`. The main deliverable. |
| 10. Memory | `RunnableWithMessageHistory` so a follow-up like "Of those, which is easiest to fix?" resolves "those". |
| 11. Evaluation | Six fixed questions, a v1 vs v2 scorecard, and a quote-level grounding check. |
| 12. Backup | Zips the whole workspace so a fresh Colab session can resume. |
| Appendix A | Optional stricter-prompt experiment (v3). Not part of the reported results. |

**The key design decision:** early versions only searched "Discussion" or "Conclusion" sections for limitations. That is a trap, because a limitation can sit in Methods (a caveat) or in a results table. The fixed version searches the whole paper by meaning and keeps "section" only as a label.

---

## 2. Setup and how to run

**Colab secrets** (key icon in the sidebar, enable notebook access):

| Secret | Needed for |
|---|---|
| `Groq_API` | Primary LLM (used for the reported results) |
| `Gemini_API` | Optional fallback, used if Groq is unavailable or out of quota |

At least one must be set.

**Fresh start:** run Step 0 to Step 12 in order. Step 1 asks you to upload the 5 PDFs (or put them in `papers/` first).

**Resume from a backup:** run Step 0, then the "Option B" restore cell, skip Steps 1 to 2, and continue from Step 3.

> **Do not use "Run all" repeatedly.** The evaluation is switched **off** by default (`RUN_EVAL = False` in Step 0) so it cannot burn the free-tier quota. Saved outputs in `eval/` are the test evidence. To re-run it, set `RUN_EVAL = True`; it is resumable and skips questions already saved (about 36 LLM calls).

---

## 3. Try it yourself

**One paper** (Step 7):

```python
paper = "Saha2024 (Fuzzy logic depression level)"
question = "What limitations or shortcomings does this paper mention?"

hits = per_paper_retrieve(question, k=3, papers=[paper])[paper]
context = "\n\n".join(f"(p{h.metadata['page']}) {h.page_content}" for h in hits)
answer = safe_invoke(single_paper_chain, {"paper": paper, "context": context, "question": question})
print(answer)
check_grounding(answer, hits)
```

**With memory** (Step 10):

```python
ask_with_memory(paper, "What limitations does this paper mention?")
ask_with_memory(paper, "Of those, which seems easiest for future researchers to actually fix?")
```

**Across all papers, free text** (Step 8, 6 LLM calls):

```python
final, hits = run_topic("gap_limitations", show_retrieval=True)
```

**Across all papers, structured** (Step 9, 6 LLM calls):

```python
t = TOPICS["gap_limitations"]
result, hits = analyze_gap(t["question"], k=t["k"], search=t["search"], per_paper_q=t["per_paper"])
result.model_dump()     # plain dict, ready for a report or slide
```

Paper labels are in `PAPER_LABELS`; the six evaluation questions are defined once in `TOPICS` (Step 4.3).

---

## 4. Evaluation

Six fixed questions run through the cross-paper pipeline, saved after every question. Two runs are kept as evidence:

| Run | File | Pipeline |
|---|---|---|
| v1 | `eval/results_v1.json` | Old: Gemini, section-gated retrieval |
| v2 | `eval/results_v2_groq.json` | Fixed: Groq `gpt-oss-120b`, whole-paper hybrid retrieval |

| Key | Question | Why it is included |
|---|---|---|
| `gap_limitations` | What limitations are repeatedly mentioned across the papers? | Core synthesis |
| `gap_comparisons` | Which approaches or methods are compared across the papers? | Core synthesis |
| `gap_unresolved` | What problems remain unresolved across the papers? | Core synthesis |
| `gap_future_work` | What future work do the authors propose across the papers? | Core synthesis |
| `conflict_accuracy` | Which paper reports the highest accuracy? | **Hard:** the answer is a number inside a results table |
| `failure_pca_count` | How many papers use PCA? | **Hard:** the answer sits in Methods, which the old search never looked at |

### Scorecard (Step 11.2, surface metrics, no API calls)

| Question | v1 "no evidence" phrases | v1 cited pages | v2 "no evidence" phrases | v2 cited pages |
|---|---|---|---|---|
| `gap_limitations` | 2 | 0 | 0 | 5 |
| `gap_comparisons` | 1 | 1 | 0 | 9 |
| `gap_unresolved` | 1 | 1 | 0 | 6 |
| `gap_future_work` | 2 | 3 | 0 | 4 |
| `conflict_accuracy` | 1 | 0 | 0 | 1 |
| `failure_pca_count` | 6 | 2 | 0 | 5 |

These numbers show **whether the system answered**, not whether the answers are correct. "Cited pages" is the count of distinct page numbers in the answer.

### Result

With the fix, every question returned a page-cited answer instead of "no relevant evidence", and the PCA question found Chattopadhyay's use of PCA (v1 reported 0 papers).

**The two hard questions are only partly solved:**
- **PCA:** the count of 2 is probably an overcount (see Test 2 above).
- **Highest accuracy:** v2 surfaces the 96.9 figure but not the 92.03 figure the scorecard checks for, and some accuracy figures come from tables describing *other* studies rather than the paper's own results.

Across the limitation and future-work questions, some answers credit a paper with claims from its related-work section. See Section 5.

---

## 5. Known limitations

- **Related work leaks in.** Retrieval searches the whole paper, so related-work text that describes *other* studies can be retrieved and then attributed to the paper itself. This caused the probable Saha2024 PCA overcount and some accuracy figures taken from other studies' tables.
- **Grounding checks are partial.** `check_grounding` only confirms that cited page numbers exist somewhere in the retrieved set. The stronger `verify_quotes` (Step 11.4) checks that each quoted span appears in the *same paper's* retrieved chunks, but a quote from that paper's related-work section still passes, because it really is in the paper.
- **Chunks can start mid-sentence.** Fixed-size chunking occasionally cuts a sentence in half, which produced one garbled claim.
- **Memory helps the answer, not the search.** In a follow-up like "which of those is easiest to fix?", the pronouns are resolved for writing the answer, but retrieval still searches with the literal follow-up text. A "condense question" step before retrieval would fix this; it is not built.
- **Section labels are tuned to these papers.** The heading vocabulary assumes IMRaD-structured academic papers. It is only a citation label, never a filter, but it would not recognise headings in, say, a legal contract.
- **Free LLM tiers are volatile.** Model availability and quotas change. The notebook falls back across a candidate list and then to the other provider, but if everything is exhausted for the day, retries cannot help; wait for the quota reset.
- **Gap merging is a judgment call by the model.** If it over- or under-merges on a given question, that is a finding about instruction-following at this level of nuance, not a bug.

**Proposed improvements:** exclude or down-weight `related_work` chunks for questions about a paper's *own* methods and limitations; split on sentence boundaries; add the condense-question step for follow-ups; add a quote-to-section check so a quote from related work is flagged.

---

## 6. Saving your work

Step 12 zips everything needed to resume (papers, extracted text, `chunks.json`, FAISS index, all eval results) and downloads it. Upload the zip in a fresh session and use Option B in Step 0.
---

## 👤 Authors

- GitHub: [Dhruv Marwal](https://github.com/DhruvMarwal) , [Priyanshu Jha](https://github.com/Priyanshu0423) , [Shivang Jain](https://github.com/Xopse)
- LinkedIn: [Dhruv Marwal](https://linkedin.com/in/dhruvmarwal) , [Priyanshu Jha](https://linkedin.com/in/priyanshujha-) , [Shivang Jain](https://linkedin.com/in/shivang-jain-69602132a)
---
