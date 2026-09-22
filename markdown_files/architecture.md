# System Architecture Map: Construction Risk Assessment Pipeline

**Purpose:** This document provides a high-level, structural blueprint of the system. Consult this map before modifying any directory, data flow, or core script to ensure downstream dependencies are not broken.

---

## 1. High-Level System Diagram

```text
=======================================================================================
                         OFFLINE PIPELINES (Backend Processing)
=======================================================================================

[Phase 1: ML Pipeline]                       [Phase 2: RAG Pipeline] (Upcoming)
       │                                            │
 ┌─────▼─────┐                                ┌─────▼──────────┐
 │ data/raw/ │ (OSHA/Kaggle CSVs)             │ data/knowledge_base/ │ (ISO/OSHA PDFs)
 └─────┬─────┘                                └─────┬──────────┘
       │                                            │
(phase1_ml_engine.py)                         (phase2_vector_db.py)
       │ TF-IDF & Random Forest                     │ PyMuPDF, LangChain & Embeddings
       │                                            │
 ┌─────▼───────────┐                          ┌─────▼──────────┐
 │ data/processed/ │ (.pkl, .csv)             │ data/chroma_db/│ (Local Vector DB)
 └─────┬───────────┘                          └─────┬──────────┘
       │                                            │
=======│============================================│==================================
       │            ONLINE PIPELINE (Frontend UI)   │
=======│============================================│==================================
       │                                            │
       └────────────────────┐  ┌────────────────────┘
                            ▼  ▼
                       (dashboard.py)
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
      [Live ML Predictor]         [RAG Contract Audit]
    (Categorizes user text)     (Cross-references BOQ vs PDFs)
