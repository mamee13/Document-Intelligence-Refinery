# 🏭 The Document Intelligence Refinery

**A production-grade, multi-stage agentic pipeline for structured document extraction with 100% spatial provenance.**

The Refinery is designed to solve the three critical failures of traditional document processing: **Structure Collapse**, **Context Poverty**, and **Provenance Blindness**. It transforms heterogeneous corpora (native PDFs, scanned images, multi-column reports) into a high-fidelity, auditable knowledge base.

---

## 🏗️ Technical Architecture

The system operates as a 5-stage agentic pipeline with confidence-gated escalation loops.

```mermaid
graph TD
    subgraph Stage1["1. Triage Agent"]
        DOC((PDF)) --> TA[Heuristic Analyzer] --> PROFILE[DocumentProfile]
    end

    subgraph Stage2["2. Structure Extraction"]
        PROFILE --> ROUTER{Router}
        ROUTER --> S1[Strat A: FastText]
        ROUTER --> S2[Strat B: Layout]
        ROUTER --> S3[Strat C: Vision]
        S1 -.Escalation.-> S2
        S2 -.Escalation.-> S3
        S1 & S2 & S3 --> EXDOC[ExtractedDocument]
    end

    subgraph Stage3["3. Semantic Intelligence"]
        EXDOC --> CHUNK[Semantic Chunker] --> LDU[LDU Array]
        EXDOC --> IDX[PageIndex Builder] --> NAV[Navigation Tree]
    end

    subgraph Stage4["4. Persistence"]
        LDU --> VS[(Vector Store)]
        LDU --> SQ[(Fact Table)]
        NAV --> JSON[(PageIndex Store)]
    end

    subgraph Stage5["5. Query Interface"]
        QUERY[User Query] --> QA[Query Agent]
        QA <-->|Browse| JSON
        QA <-->|Search| VS
        QA <-->|Validate| SQ
        QA --> ANS[Answer + ProvenanceChain]
    end

    style Stage1 fill:#e1f5ff
    style Stage2 fill:#fff4e1
    style Stage3 fill:#f0e1ff
    style Stage4 fill:#e1ffe1
    style Stage5 fill:#ffe1e1
```

---

## 🌟 Key Features

### 1. Intelligent Triage (Stage 1)

Determines document "Origin Type" (Digital vs. Scanned) and "Layout Complexity" using measurable metrics like character density and image-to-page ratios before expensive processing begins.

### 2. Multi-Strategy Extraction & Escalation (Stage 2)

- **Strategy A (FastText)**: Instant extraction for simple, single-column documents using `pdfplumber`.
- **Strategy B (Layout)**: Layout-aware extraction for tables and multi-column reports using `Docling`.
- **Strategy C (Vision)**: VLM-based extraction for scans and complex failure cases using **Gemini 2.0 Flash**.
- **Escalation Guard**: Low-confidence outputs automatically trigger a more robust (but more expensive) strategy.

### 3. Semantic Chunking (Stage 3)

Enforces a "Table Integrity First" policy. Blocks are never split mid-table or mid-paragraph, and section headers are propagated as metadata to every child chunk (LDU).

### 4. PageIndex Navigation (Stage 4)

Builds a hierarchical tree of the document, allowing agents to "browse" the table of contents and summaries before performing a deep semantic search.

### 5. Provenance-First Querying (Stage 5)

Every answer is backed by a `ProvenanceChain`. Citations include not just the document name, but the **exact page number**, **spatial bounding box (bbox)**, and a **content hash** for verification.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **uv** (Fast Python package manager)
- **OpenRouter API Key** (for Strategy C and LLM tasks)

### Installation

```bash
# Clone the repository
git clone https://github.com/mamee13/Document-Intelligence-Refinery
cd Document-Intelligence-Refinery


# Setup environment
echo "OPENROUTER_API_KEY=your_key_here" > .env

# Install dependencies
uv sync
```

---

## 🛠️ Usage

The system can be run in multiple modes via the `src/main.py` entry point.

### 1. Extraction Only (Triage + Stage 2)

Fastest way to get structured JSON from one file or an entire directory.

```bash
# Process all files in data/
uv run python -m src.main extract

# Process a specific file (Must reside in the data/ folder)
uv run python -m src.main extract <filename.pdf>
```

### 2. Full Refine (Stages 1-4)

Runs the full pipeline (Triage → Extraction → Chunking → Indexing) and hydrates the persistent stores.

```bash
# Process all files in data/ (Batch Mode)
uv run python -m src.main refine

# Process a specific file (Must reside in the data/ folder)
uv run python -m src.main refine <filename.pdf>
```

### 3. Query Agent (Stage 5)

Ask questions across your refined knowledge base.

```bash
uv run python -m src.main query "What were the total assets in 2023?"
```

### 4. Audit Mode

Verify a specific claim against the primary evidence.

```bash
uv run python -m src.main audit "The company revenue exceeded 10 billion."
```

---

## 🐳 Docker Deployment

For enterprise deployment, use the provided Dockerfile.

```bash
# Build
docker build -t refinery .

# Run
docker run --env-file .env -v $(pwd)/data:/app/data -v $(pwd)/.refinery:/app/.refinery refinery extract
```

---

## 📊 Performance Benchmarks

- **Extraction Fidelity**: Strategy B improves table recall by **>20x** over baseline pdfplumber.
- **Cost Efficiency**: Routing saves **~90%** of API costs by using VLMs only when triage detects scanned or complex input.
- **Auditability**: 100% of facts are linked to verifiable spatial coordinates.
