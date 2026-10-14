# Evaluating Methods of Reliable Question Answering over Governance Documents

Research Project — Curtin University

Author: Lucas Lim  
Supervisor: Professor Tom Gedeon  
Co-Supervisor: Rakib Hasan

## 1. Project Overview

This repository contains the source code, experimental configurations, and supporting materials developed for the research thesis *Evaluating Methods of Reliable Question Answering over Governance Documents*.

The project investigates methods for improving the reliability of Retrieval-Augmented Generation (RAG) when answering questions using governance documents from the IEEE Systems, Man, and Cybernetics Society (SMCS).

Five architectures were implemented and evaluated:

1. **Baseline RAG:** A conventional retrieval-augmented generation pipeline using document retrieval and language-model-based answer generation.
2. **RAG + RISE-Inspired Refinement:** Introduces selective iterative refinement of responses based on evaluation feedback.
3. **RAG + RLVR-Inspired Verification:** Incorporates evidence-based claim verification and targeted correction during inference. This is an RLVR-inspired approach, not a reinforcement-learning training implementation.
4. **RAG + Multi-Model Verification (MMV):** Uses three language models to independently verify generated answers and combines their assessments through majority voting.
5. **RAG + Explainable AI (XAI):** Generates post-hoc explanations, claim-level assessments, and supporting documentary evidence without modifying the generated answers.

The architectures were evaluated using 110 benchmark questions across 16 categories, covering different retrieval, reasoning, and interpretation requirements.

## 2. Repository Structure

The repository is organised into the following directories:

| Folder             | Description                                                                                                         |
|--------------------|---------------------------------------------------------------------------------------------------------------------|
| `BaseRag/`         | Baseline RAG implementation and associated files.                                                                   |
| `RAG+RISE/`        | RISE-inspired refinement implementation and associated files.                                                       |
| `RAG+RLVR/`        | RLVR-inspired verification implementation and associated files.                                                     |
| `RAG+MMV/`         | Multi-model verification implementation and associated files.                                                       |
| `RAG+XAI/`         | Explainable AI implementation and associated files.                                                                 |
| `Necessary Files/` | Supporting files required to execute the architectures, including the benchmark questions and governance documents. |
| `README.md`        | Project documentation and execution instructions.                                                                   |

Each architecture is stored in its own directory, containing the corresponding Python implementation and associated files.

The actual filenames and folder paths should match those used in the final implementation.

## 3. System Requirements

The project uses Python and locally deployed language models through Ollama.

**Required software:**

- Python 3
- Ollama
- Python packages specified in `requirements.txt`

**Hardware:**

- A computer capable of running the required language models locally.
- Sufficient memory and storage for model inference, embeddings, and the document database.

The experiments were conducted using a local computing environment. Exact hardware specifications and software versions should be recorded to support reproducibility.

## 4. Language Models

The following Ollama models were used:

|       Model        | Purpose                                               |
|--------------------|-------------------------------------------------------|
|      `llama3`      | Answer generation and language-model-based processing |
| `nomic-embed-text` | Document and question embeddings                      |
|     `qwen3:8b`     | Additional MMV verifier                               |
|    `mistral:7b`    | Additional MMV verifier                               |

Install Ollama from https://ollama.com.

Download the models using:

```bash
ollama pull llama3
ollama pull nomic-embed-text
ollama pull qwen3:8b
ollama pull mistral:7b
```

Ensure the Ollama service is running before executing the Python scripts.

## 5. Installation

Clone or download the repository and open a terminal in the project directory.

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Alternatively, on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The implementations use libraries including ChromaDB, Ollama, pypdf, ROUGE Score, SacreBLEU, and BERTScore.

## 6. Governance Document Corpus

The retrieval corpus consists of four IEEE-related governance documents:

1. Draft Operational Manual V5.
2. IEEE Constitution and Bylaws.
3. SMCS Constitution and Bylaws, August 2023.
4. SMCS Technical Manual, August 2023.

Place the required PDF files in the `docs/` directory using the filenames expected by the scripts.

The documents are processed and divided into overlapping text chunks before being embedded and stored in ChromaDB.

If the original documents are not included in the repository, users must obtain them from authorised sources.

## 7. RAG Configuration

The baseline retrieval configuration used in the experiments is summarised below.

| Parameter              | Value              |
|------------------------|--------------------|
| Embedding model        | `nomic-embed-text` |
| Generation model       | `llama3`           |
| Vector database        | ChromaDB           |
| Chunk size             | 1,000              |
| Chunk overlap          | 200                |
| Retrieved passages     | Top 12             |
| Evaluation questions   | 110                |
| Correctness score      | 0–3                |
| Correct classification | Score of 3         |

Additional architecture-specific settings, including refinement limits, verification procedures, and prompt configurations, are defined in the corresponding implementation scripts.

## 8. Running the Experiments

Each RAG architecture can be executed independently.

To run an implementation:

1. Download or clone the repository.
2. Install Python, Ollama, and the required dependencies.
3. Download the required Ollama models.
4. Open the `Necessary Files/` directory.
5. Copy the required supporting files and folders into the directory of the architecture you wish to execute (e.g., `BaseRag/` or `RAG+RISE/`).
6. Open a terminal in the selected architecture's directory.
7. Run the corresponding Python script using its actual filename.

For example, if the baseline implementation is named `RAG.py`:

```bash
python "RAG.py"
```

Important: The supporting files from `Necessary Files/` must be placed inside the selected architecture directory before execution. Otherwise, the script may be unable to locate the required governance documents or benchmark questions.

Ensure that the Ollama service is running before executing the scripts.

## 9. Evaluation Framework

The benchmark contains 110 questions distributed across 16 categories, including direct retrieval, cross-document reasoning, semantic retrieval, hallucination traps, ambiguity, contradictions, multi-hop reasoning, temporal interpretation, negation, and conditional questions.

The architectures were assessed using:

**Question-answering performance**
- Correctness accuracy.
- Evaluation rubric scores.
- Category-level accuracy.

**Textual and semantic similarity**
- ROUGE-1, ROUGE-2, and ROUGE-L F1.
- BLEU.
- BERTScore F1.

**Computational performance**
- Total response latency.
- Retrieval and generation durations.
- Language-model calls.
- Embedding calls.

Architecture-specific evaluation also examined refinement outcomes, verification judgments, and explanation behaviour.

## 10. Experimental Results

The final benchmark accuracies were:

| Architecture             | Accuracy |
|--------------------------|----------|
| Baseline RAG             |  94.55%  |
| RISE-Inspired            |  99.09%  |
| RLVR-Inspired            |  97.27%  |
| Multi-Model Verification |  96.36%  |
| Explainable AI           |  95.45%  |

These values represent correctness under the implemented benchmark evaluation framework.

The RISE and RLVR configurations can modify generated answers. In contrast, MMV and XAI perform post-generation assessment without changing the original response. Therefore, accuracy differences for MMV and XAI should not be interpreted as improvements directly caused by their verification or explanation mechanisms.

Detailed experimental outputs and summaries are provided in the `results/` directory, where available.

## 11. Reproducibility and Limitations

The experimental findings are specific to the selected governance documents, benchmark questions, model configurations, and computing environment.

Although the implementations were designed to support consistent evaluation, locally executed language models may exhibit variations across runs or hardware environments.

Furthermore, automated evaluation and verification mechanisms can produce incorrect or inconsistent judgments. Reported benchmark accuracy should therefore not be interpreted as a guarantee of factual correctness.

The RISE-inspired implementation also uses reference-based benchmark evaluation to guide refinement decisions. Such reference answers would not ordinarily be available when answering new questions in a deployed system.

## 12. Academic Attribution

This repository contains the source code and supporting materials developed as part of the research thesis *Evaluating Methods of Reliable Question Answering over Governance Documents* at Curtin University.

The repository is provided for research transparency and reproducibility. Anyone referencing or building upon this work is expected to provide appropriate academic attribution and comply with applicable academic integrity requirements.

Suggested citation::

Lim, L. (2026). *Evaluating Methods of Reliable Question Answering over Governance Documents* [Undergraduate research thesis]. Curtin University.

Repository: [Insert GitHub repository URL]

## 13. License

No open-source license has currently been assigned to this repository. All rights are reserved unless otherwise stated.

Permission to view the repository does not imply permission to reuse, modify, or redistribute its contents beyond applicable legal rights or permissions.

The inclusion of third-party libraries and governance documents remains subject to their respective licenses and usage conditions.
