# Research Gap Analyzer

A Retrieval-Augmented Generation (RAG) system that analyzes multiple research papers and identifies recurring limitations, unresolved problems, comparisons, and potential research gaps.

The system is designed to help researchers move beyond reading papers individually by retrieving evidence from multiple papers and synthesizing it into structured, page-cited research gaps.

## Problem

Finding research gaps across multiple papers is time-consuming because relevant information can appear in different sections of different papers.

A simple LLM-based approach can also produce unsupported conclusions or miss relevant evidence if retrieval is too restrictive.

This project uses RAG to retrieve relevant evidence from each paper and generate grounded cross-paper analysis.

## What the System Does

Given a collection of research papers, the system can answer questions such as:

- What limitations are repeatedly mentioned across the papers?
- Which approaches or methods are compared?
- What problems remain unresolved?
- What future work do the authors propose?
- Which papers use a particular technique such as PCA?
- What research gaps can be identified across multiple papers?

The final research gaps include the related papers and supporting evidence with page references.

## Dataset

The current system uses five research papers related to machine learning, fuzzy logic, neuro-fuzzy systems, EEG features, and depression analysis.

Each paper is processed independently and assigned a consistent label for retrieval and citation.

## Architecture

```text
Research Papers
      │
      ▼
PDF Text Extraction
      │
      ▼
Section-aware Chunking
      │
      ▼
Metadata Tagging
      │
      ▼
Sentence Embeddings
      │
      ▼
FAISS Vector Store
      │
      ▼
Hybrid Retrieval
(FAISS + BM25)
      │
      ▼
Per-Paper Evidence
      │
      ▼
Gemini LLM
      │
      ├───────────────┐
      ▼               ▼
Single-Paper       Cross-Paper
Analysis           Synthesis
                      │
                      ▼
              Structured Research
                  Gap Output
