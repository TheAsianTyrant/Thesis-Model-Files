import os
import re
import json
import shutil
import time
import statistics
from collections import defaultdict

from pypdf import PdfReader
import chromadb
import ollama

from rouge_score import rouge_scorer
import sacrebleu
from bert_score import score as bertscore


# Naming the files, models and settings used for the benchmark

DOCUMENT_FOLDER = "docs"
QUESTIONS_FILE = "Questions.txt"

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "rag_collection"

EMBEDDING_MODEL = "nomic-embed-text"
GENERATION_MODEL = "llama3"
EVALUATION_MODEL = "llama3"

OUTPUT_FILE = "evaluation_results.json"
SUMMARY_JSON_FILE = "evaluation_summary.json"
SUMMARY_TEXT_FILE = "evaluation_summary.txt"

# Retrieval settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 12


# Functions for tracking Ollama durations and the number of model calls

def _ns_to_seconds(value):
    return (value or 0) / 1_000_000_000

def _ollama_durations(response):
    return {
        "ollama_total_duration_seconds": _ns_to_seconds(response.get("total_duration", 0)),
        "ollama_load_duration_seconds": _ns_to_seconds(response.get("load_duration", 0)),
        "ollama_prompt_eval_duration_seconds": _ns_to_seconds(response.get("prompt_eval_duration", 0)),
        "ollama_eval_duration_seconds": _ns_to_seconds(response.get("eval_duration", 0)),
    }

def _empty_cost():
    return {
        "llm_calls": 0,
        "embedding_calls": 0,
        "ollama_total_duration_seconds": 0.0,
        "ollama_load_duration_seconds": 0.0,
        "ollama_prompt_eval_duration_seconds": 0.0,
        "ollama_eval_duration_seconds": 0.0,
    }

def _add_response_cost(cost, response, call_type):
    if call_type == "llm":
        cost["llm_calls"] += 1
    elif call_type == "embedding":
        cost["embedding_calls"] += 1
    for key, value in _ollama_durations(response).items():
        cost[key] += value

def _merge_cost(target, source):
    for key in target:
        target[key] += source.get(key, 0)



# Loading the governance PDF documents and extracting text from each page

print("\n============================================")
print("LOADING DOCUMENTS")
print("============================================")

documents = []

for filename in sorted(os.listdir(DOCUMENT_FOLDER)):

    if not filename.lower().endswith(".pdf"):
        continue

    path = os.path.join(DOCUMENT_FOLDER, filename)

    print(f"Loading: {filename}")

    try:
        reader = PdfReader(path)

        pages = []

        for page_number, page in enumerate(reader.pages, start=1):

            try:
                extracted = page.extract_text()
            except Exception as e:
                print(
                    f"Warning: Could not extract page "
                    f"{page_number} from {filename}: {e}"
                )
                extracted = ""

            if extracted:
                pages.append({
                    "page": page_number,
                    "text": extracted
                })

        documents.append({
            "filename": filename,
            "pages": pages
        })

    except Exception as e:

        print(
            f"ERROR loading {filename}: {e}"
        )


print(f"\nDocuments loaded: {len(documents)}")


# Splitting the documents into overlapping chunks for retrieval

print("\n============================================")
print("CHUNKING DOCUMENTS")
print("============================================")


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


all_chunks = []

chunk_id = 0

for doc in documents:

    for page in doc["pages"]:

        page_chunks = chunk_text(page["text"])

        for chunk in page_chunks:

            all_chunks.append({
                "id": str(chunk_id),
                "text": chunk,
                "source": doc["filename"],
                "page": page["page"]
            })

            chunk_id += 1


print(f"Total chunks created: {len(all_chunks)}")


# Rebuilding ChromaDB so that previous embeddings do not affect the benchmark

print("\n============================================")
print("REBUILDING VECTOR DATABASE")
print("============================================")


# Removing old embeddings to keep each benchmark run independent

if os.path.exists(CHROMA_PATH):

    print("Deleting old ChromaDB...")

    shutil.rmtree(CHROMA_PATH)


client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.create_collection(
    name=COLLECTION_NAME
)

print("New vector database created.")


# Creating and storing embeddings for each document chunk

print("\n============================================")
print("CREATING EMBEDDINGS")
print("============================================")


for i, chunk in enumerate(all_chunks):

    if i % 25 == 0:

        print(
            f"Embedding chunk {i + 1}/{len(all_chunks)}"
        )

    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=chunk["text"]
    )

    embedding = response["embedding"]

    collection.add(
        ids=[chunk["id"]],
        embeddings=[embedding],
        documents=[chunk["text"]],
        metadatas=[{
            "source": chunk["source"],
            "page": chunk["page"]
        }]
    )


print(
    f"Embeddings created: {collection.count()}"
)


# Reading the benchmark questions and their reference answers

print("\n============================================")
print("LOADING QUESTIONS")
print("============================================")


def parse_questions(filename):

    with open(
        filename,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()


    records = text.split("---QUESTION---")

    questions = []


    for record in records:

        record = record.strip()

        if not record:
            continue


        id_match = re.search(
            r"ID:\s*(Q\d+)",
            record
        )


        category_match = re.search(
            r"CATEGORY:\s*(.+)",
            record
        )


        question_match = re.search(
            r"QUESTION:\s*(.*?)(?=\nANSWER:)",
            record,
            re.DOTALL
        )


        answer_match = re.search(
            r"ANSWER:\s*(.*)",
            record,
            re.DOTALL
        )


        if not (
            id_match
            and category_match
            and question_match
            and answer_match
        ):

            print(
                "Warning: Could not parse question record:"
            )

            print(record[:200])

            continue


        questions.append({

            "id":
                id_match.group(1).strip(),

            "category":
                category_match.group(1).strip(),

            "question":
                question_match.group(1).strip(),

            "reference_answer":
                answer_match.group(1).strip()
        })


    return questions


questions = parse_questions(
    QUESTIONS_FILE
)


print(
    f"Questions loaded: {len(questions)}"
)


# Generating the initial RAG answer using the retrieved governance documents

def generate_answer(question):

    cost = _empty_cost()
    pipeline_start = time.perf_counter()
    retrieval_start = time.perf_counter()

    # Converting the question into an embedding for similarity search

    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=question
    )

    _add_response_cost(cost, response, "embedding")
    question_embedding = response["embedding"]


    # Retrieving the most relevant document chunks from ChromaDB

    results = collection.query(

        query_embeddings=[
            question_embedding
        ],

        n_results=RETRIEVAL_K
    )


    retrieved_docs = results["documents"][0]

    retrieved_metadata = results["metadatas"][0]
    retrieval_latency = time.perf_counter() - retrieval_start


    # Including document names and page numbers with the retrieved text

    context_parts = []


    for i, doc in enumerate(retrieved_docs):

        metadata = retrieved_metadata[i]

        context_parts.append(

            f"""
SOURCE:
{metadata["source"]}

PAGE:
{metadata["page"]}

CONTENT:
{doc}
"""
        )


    context = "\n".join(context_parts)


    # Instructing the model to answer using only the retrieved governance documents

    prompt = f"""
You are an accurate question-answering system for
IEEE Systems, Man, and Cybernetics Society governance documents.

Answer the question using ONLY the information contained
in the provided context.

IMPORTANT RULES:

1. Do not invent information.

2. Do not use outside knowledge.

3. If the context contains the answer, give the answer directly.

4. If multiple retrieved passages contain relevant information,
combine them when necessary.

5. If the context does not contain enough information,
say:

"The provided governance documents do not contain enough
information to answer this question."

6. Do not confuse information about IEEE itself with
information about the IEEE Systems, Man, and Cybernetics Society.

7. Do not assume that information about Technical Committees
applies to Society membership, officers, or the Society as a whole.

8. Give a concise answer focused specifically on the question.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


    generation_start = time.perf_counter()
    response = ollama.chat(

        model=GENERATION_MODEL,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    _add_response_cost(cost, response, "llm")
    generation_latency = time.perf_counter() - generation_start
    answer = response["message"]["content"].strip()
    cost.update({
        "retrieval_latency_seconds": retrieval_latency,
        "generation_latency_seconds": generation_latency,
        "baseline_pipeline_latency_seconds": time.perf_counter() - pipeline_start,
    })

    return answer, retrieved_metadata, cost


# Evaluating the generated answer against the reference answer

def evaluate_answer(
    question,
    reference_answer,
    generated_answer,
    cost=None
):


    evaluation_prompt = f"""
You are evaluating a RAG system answering questions
about IEEE Systems, Man, and Cybernetics Society
governance documents.

Your job is to determine whether the GENERATED ANSWER
correctly answers the QUESTION when compared with the
REFERENCE ANSWER.

IMPORTANT:

Evaluate meaning, not wording.

Do NOT penalize an answer simply because it contains
additional information if that additional information is
factually correct and does not contradict the reference.

Do NOT require the generated answer to contain every
word or sentence of the reference answer.

A concise answer can receive a score of 3 if it contains
the essential information needed to answer the question.

However, incorrect additional information IS a problem.

Use these definitions:

CORRECTNESS:
1 = The central claims are factually correct.
0 = The answer contains a factual error about the question.

COMPLETENESS:
1 = The answer contains the important information necessary
    to answer the question.
0 = Important information required to answer the question
    is missing.

FAITHFULNESS:
1 = The answer is supported by the provided governance
    information and does not introduce unsupported claims.
0 = The answer contains hallucinated, unsupported, or
    contradictory information.

SCORE:

3 = Fully correct, complete, and faithful.

2 = Mostly correct, but has a minor omission,
    minor unnecessary detail, or minor issue.

1 = Partially correct, but has a significant error
    or important omission.

0 = Incorrect, contradictory, unrelated, or fails
    to answer the question.

SPECIAL CASE FOR HALLUCINATION TRAP QUESTIONS:

If the reference answer states that the documents
do not contain the requested information, a generated
answer should receive a high score if it correctly
recognises that the information cannot be determined
from the provided documents.

SPECIAL CASE FOR AMBIGUOUS QUESTIONS:

If the reference answer says clarification is required,
the generated answer should not invent a specific
interpretation. It should recognise the ambiguity.

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

GENERATED ANSWER:
{generated_answer}

Return ONLY this format:

CORRECTNESS: <0 or 1>
COMPLETENESS: <0 or 1>
FAITHFULNESS: <0 or 1>
SCORE: <0, 1, 2, or 3>
REASON: <one or two sentences>
"""


    response = ollama.chat(

        model=EVALUATION_MODEL,

        messages=[
            {
                "role": "user",
                "content": evaluation_prompt
            }
        ]
    )


    if cost is not None:
        _add_response_cost(cost, response, "llm")

    evaluation_text = response[
        "message"
    ]["content"].strip()


    # Extracting the scores and reasoning from the evaluator response

    correctness_match = re.search(
        r"CORRECTNESS:\s*([01])",
        evaluation_text,
        re.IGNORECASE
    )


    completeness_match = re.search(
        r"COMPLETENESS:\s*([01])",
        evaluation_text,
        re.IGNORECASE
    )


    faithfulness_match = re.search(
        r"FAITHFULNESS:\s*([01])",
        evaluation_text,
        re.IGNORECASE
    )


    score_match = re.search(
        r"SCORE:\s*([0-3])",
        evaluation_text,
        re.IGNORECASE
    )


    reason_match = re.search(
        r"REASON:\s*(.*)",
        evaluation_text,
        re.IGNORECASE | re.DOTALL
    )


    return {

        "correctness":
            int(correctness_match.group(1))
            if correctness_match
            else None,

        "completeness":
            int(completeness_match.group(1))
            if completeness_match
            else None,

        "faithfulness":
            int(faithfulness_match.group(1))
            if faithfulness_match
            else None,

        "score":
            int(score_match.group(1))
            if score_match
            else None,

        "reason":
            reason_match.group(1).strip()
            if reason_match
            else evaluation_text
    }

# Calculating ROUGE, BLEU and BERTScore to compare generated and reference answers

def calculate_standard_metrics(reference_answer, generated_answer):

    # Measuring word and phrase overlap using ROUGE

    rouge = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )

    rouge_scores = rouge.score(
        reference_answer,
        generated_answer
    )

    rouge1_f1 = rouge_scores["rouge1"].fmeasure
    rouge2_f1 = rouge_scores["rouge2"].fmeasure
    rougeL_f1 = rouge_scores["rougeL"].fmeasure


    # Calculating BLEU similarity against the reference answer

    bleu = sacrebleu.sentence_bleu(
        generated_answer,
        [reference_answer]
    )

    bleu_score = bleu.score / 100.0


    # Measuring semantic similarity using BERTScore

    P, R, F1 = bertscore(
        [generated_answer],
        [reference_answer],
        lang="en",
        verbose=False
    )

    bertscore_precision = P.item()
    bertscore_recall = R.item()
    bertscore_f1 = F1.item()


    return {

        "rouge1_f1":
            rouge1_f1,

        "rouge2_f1":
            rouge2_f1,

        "rougeL_f1":
            rougeL_f1,

        "bleu":
            bleu_score,

        "bertscore_precision":
            bertscore_precision,

        "bertscore_recall":
            bertscore_recall,

        "bertscore_f1":
            bertscore_f1
    }


# Applying RISE-inspired refinement only when the initial answer scores below 3
# The initial RAG answer is generated and evaluated before refinement begins.
# RISE only sees the question, current answer and retrieved evidence, not the reference answer.
# The evaluator decides whether a revised answer improves on the previous score.
# Invalid evaluation scores cannot trigger or approve a revision.

MAX_RISE_ATTEMPTS = 2


# Calling the generation model for the RISE reasoning and revision stages
def chat_for_rise(prompt, cost=None):
    response = ollama.chat(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    if cost is not None:
        _add_response_cost(cost, response, "llm")
    text = response["message"]["content"].strip()
    if not text:
        raise ValueError("RISE model returned an empty response")
    return text


# Retrieving additional passages for the original question or its subquestions
def retrieve_rise_context(query, limit=RETRIEVAL_K, cost=None):
    emb_response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=query)
    if cost is not None:
        _add_response_cost(cost, emb_response, "embedding")
    embedding = emb_response["embedding"]
    response = collection.query(
        query_embeddings=[embedding],
        n_results=min(limit, collection.count()),
        include=["documents", "metadatas"],
    )
    passages = []
    for text, metadata in zip(response["documents"][0], response["metadatas"][0]):
        passages.append({
            "source": metadata["source"],
            "page": metadata["page"],
            "text": text,
        })
    return passages


# Formatting retrieved passages with their source and page information
def format_rise_context(passages):
    return "\n\n".join(
        f'[SOURCE: {p["source"]}, PAGE: {p["page"]}]\n{p["text"]}'
        for p in passages
    )


# Removing duplicate passages and limiting the amount of evidence
def unique_passages(passages, limit=20):
    unique = []
    seen = set()
    for passage in passages:
        key = (passage["source"], passage["page"], passage["text"])
        if key not in seen:
            seen.add(key)
            unique.append(passage)
        if len(unique) >= limit:
            break
    return unique


# Identifying up to two additional evidence questions when needed
def decompose_question(question, initial_answer, context, cost=None):
    prompt = f"""You are planning evidence retrieval for a governance QA task.
Use only the question, initial answer and supplied document excerpts.
Never use a reference answer or assume that a critique is correct.
Identify at most TWO DISTINCT missing evidence requirements. For a simple
single-fact question, output NONE. For multi-hop questions, identify
separate evidence needs, e.g. the authority in one document and the
operational procedure in another. Do not restate the question three ways.
Do not conflate IEEE with SMCS, Technical Councils with Technical Committees,
or different officer roles.
Output exactly one of:
NONE
or
SUBQUESTION: <standalone evidence question>
SUBQUESTION: <standalone evidence question>

ORIGINAL QUESTION: {question}
INITIAL ANSWER: {initial_answer}
DOCUMENT EXCERPTS:
{context}
"""
    raw = chat_for_rise(prompt, cost)
    questions = []
    for line in raw.splitlines():
        match = re.match(r"^\s*SUBQUESTION:\s*(.+?)\s*$", line, re.I)
        if match:
            candidate = match.group(1)
            if candidate.casefold() != question.casefold() and candidate.casefold() not in [q.casefold() for q in questions]:
                questions.append(candidate)
    return questions[:2], raw


# Checking how the retrieved evidence supports the current answer
def document_reasoning(question, initial_answer, context, cost=None):
    return chat_for_rise(f"""You are checking a governance QA answer against DOCUMENTS ONLY.
Explain which cited excerpts directly answer the ORIGINAL QUESTION.
For multi-hop questions, identify each distinct required component and
explain their relationship. Distinguish IEEE vs SMCS, office vs succession,
and Technical Committee vs Technical Council. Identify unsupported claims,
missing essential details and conflicts; if evidence is insufficient say so.
Do not introduce unrelated provisions. Do not use or infer a reference answer.
Do not rewrite the answer yet. Cite [SOURCE: filename, PAGE: n] for claims.

ORIGINAL QUESTION: {question}
INITIAL ANSWER: {initial_answer}
DOCUMENT EXCERPTS:
{context}
""", cost)


# Identifying unsupported claims and possible corrections from the documents
def document_critique(question, initial_answer, reasoning, context, cost=None):
    return chat_for_rise(f"""Audit the INITIAL ANSWER using the ORIGINAL QUESTION and
DOCUMENT EXCERPTS ONLY. The reasoning is a hypothesis, NOT authoritative.
For each proposed correction, give the exact initial claim, the supported
correction, and the relevant [SOURCE: filename, PAGE: n]. Do not demand
irrelevant details. Do not treat rules about different organisations or
roles as contradictions. If no justified correction exists, say NO CHANGE.
Do not use any reference answer, numerical grade or evaluator feedback.

ORIGINAL QUESTION: {question}
INITIAL ANSWER: {initial_answer}
REASONING TO CHECK: {reasoning}
DOCUMENT EXCERPTS:
{context}
""", cost)


# Revising the answer using only corrections supported by retrieved evidence
def revise_from_documents(question, initial_answer, reasoning, critique, context, cost=None):
    return chat_for_rise(f"""You are editing an answer to the ORIGINAL QUESTION.
Preserve supported parts of INITIAL ANSWER. Make ONLY corrections justified
by the DOCUMENT EXCERPTS. Treat the critique as fallible: ignore any claim
that lacks support or concerns a different organisation, office or committee.
For multi-hop questions, explicitly combine the necessary evidence rather
than replacing the answer with one isolated excerpt. Never switch to an
unrelated provision. If no justified change is needed, reproduce the initial
answer exactly. Return ONLY the concise final answer, without preambles,
self-references, critique text or reasoning. Do not use a reference answer.

ORIGINAL QUESTION: {question}
INITIAL ANSWER: {initial_answer}
REASONING: {reasoning}
CRITIQUE: {critique}
DOCUMENT EXCERPTS:
{context}
FINAL ANSWER:""", cost)


# Handling evaluator errors without stopping the benchmark
def safe_evaluate(question, reference_answer, answer, cost=None):
    try:
        evaluation = evaluate_answer(question, reference_answer, answer, cost)
        if evaluation.get("score") not in (0, 1, 2, 3):
            raise ValueError("Evaluator returned no valid score")
        return evaluation
    except Exception as exc:
        return {
            "correctness": None, "completeness": None,
            "faithfulness": None, "score": None,
            "reason": f"Evaluation error: {exc}",
        }


# Handling similarity metric errors without stopping the benchmark
def safe_metrics(reference_answer, answer):
    try:
        return calculate_standard_metrics(reference_answer, answer)
    except Exception as exc:
        print(f"Metric calculation failed: {exc}", flush=True)
        return {
            "rouge1_f1": None, "rouge2_f1": None, "rougeL_f1": None,
            "bleu": None, "bertscore_precision": None,
            "bertscore_recall": None, "bertscore_f1": None,
        }


# Saving results through a temporary file to avoid incomplete JSON output
def save_results_atomically(data):
    temp_file = OUTPUT_FILE + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    os.replace(temp_file, OUTPUT_FILE)


print("\n============================================")
print("STARTING BENCHMARK (FRESH RUN; NO RESUME)")
print("============================================")
print(f"Loaded {len(questions)} questions; 0 completed; {len(questions)} remaining.", flush=True)
if not questions:
    raise RuntimeError("No questions parsed; refusing to overwrite result files.")
if collection.count() == 0:
    raise RuntimeError("No document chunks indexed; refusing to run benchmark.")
results = []
# Starting with an empty results file rather than resuming a previous benchmark
save_results_atomically(results)

for i, item in enumerate(questions):
    print(f"Processing {i + 1}/{len(questions)} ({item['id']})", flush=True)
    initial_answer = ""
    sources = []
    generation_error = None
    computational_cost = _empty_cost()
    question_start = time.perf_counter()
    rise_latency = 0.0
    try:
        initial_answer, sources, baseline_cost = generate_answer(item["question"])
        _merge_cost(computational_cost, baseline_cost)
        retrieval_latency = baseline_cost["retrieval_latency_seconds"]
        generation_latency = baseline_cost["generation_latency_seconds"]
        baseline_pipeline_latency = baseline_cost["baseline_pipeline_latency_seconds"]
    except Exception as exc:
        generation_error = str(exc)
        print(f"ERROR generating {item['id']}: {exc}", flush=True)

    if generation_error:
        initial_evaluation = {
            "correctness": None, "completeness": None,
            "faithfulness": None, "score": None,
            "reason": f"Generation error: {generation_error}",
        }
    else:
        initial_evaluation = safe_evaluate(
            item["question"], item["reference_answer"], initial_answer, computational_cost
        )

    initial_metrics = safe_metrics(item["reference_answer"], initial_answer) if initial_answer else {
        "rouge1_f1": None, "rouge2_f1": None, "rougeL_f1": None,
        "bleu": None, "bertscore_precision": None,
        "bertscore_recall": None, "bertscore_f1": None,
    }
    final_answer = initial_answer
    final_evaluation = initial_evaluation
    final_metrics = initial_metrics
    attempts = []
    rise_status = "skipped_initial_correct" if initial_evaluation["score"] == 3 else "skipped_evaluation_or_generation_error"
    accepted_attempt = None

    if initial_evaluation["score"] in (0, 1, 2):
        rise_start = time.perf_counter()
        rise_status = "attempted_no_improvement"
        best_answer = initial_answer
        best_evaluation = initial_evaluation
        # Using only the question, current answer and retrieved passages for refinement
        for attempt_number in range(1, MAX_RISE_ATTEMPTS + 1):
            attempt = {"attempt": attempt_number, "accepted": False}
            try:
                primary = retrieve_rise_context(item["question"], cost=computational_cost)
                primary_context = format_rise_context(primary)
                subquestions, decomposition_raw = decompose_question(
                    item["question"], best_answer, primary_context, computational_cost
                )
                extras = []
                for subquestion in subquestions:
                    extras.extend(retrieve_rise_context(subquestion, limit=4, cost=computational_cost))
                passages = unique_passages(primary + extras)
                context = format_rise_context(passages)
                reasoning = document_reasoning(item["question"], best_answer, context, computational_cost)
                critique = document_critique(item["question"], best_answer, reasoning, context, computational_cost)
                candidate = revise_from_documents(
                    item["question"], best_answer, reasoning, critique, context, computational_cost
                )
                # Reusing the evaluation when the revised answer has not changed
                candidate_evaluation = (
                    best_evaluation if candidate == best_answer else
                    safe_evaluate(item["question"], item["reference_answer"], candidate, computational_cost)
                )
                attempt.update({
                    "input_answer": best_answer,
                    "subquestions": subquestions,
                    "decomposition_raw": decomposition_raw,
                    "reasoning": reasoning,
                    "critique": critique,
                    "candidate_answer": candidate,
                    "candidate_evaluation": candidate_evaluation,
                    "retrieved_sources": [
                        {"source": p["source"], "page": p["page"]} for p in passages
                    ],
                })
                old_score = best_evaluation["score"]
                new_score = candidate_evaluation["score"]
                if new_score is not None and new_score > old_score:
                    best_answer = candidate
                    best_evaluation = candidate_evaluation
                    accepted_attempt = attempt_number
                    attempt["accepted"] = True
                    rise_status = "improved"
                    print(f"  RISE attempt {attempt_number}: {old_score} -> {new_score} ACCEPTED", flush=True)
                else:
                    print(f"  RISE attempt {attempt_number}: {old_score} -> {new_score} REJECTED", flush=True)
            except Exception as exc:
                attempt["error"] = str(exc)
                print(f"  RISE attempt {attempt_number} failed: {exc}", flush=True)
            attempts.append(attempt)
            if best_evaluation["score"] == 3:
                break
        rise_latency = time.perf_counter() - rise_start
        final_answer = best_answer
        final_evaluation = best_evaluation
        if final_answer != initial_answer:
            final_metrics = safe_metrics(item["reference_answer"], final_answer)

    computational_cost.update({
        "retrieval_latency_seconds": locals().get("retrieval_latency", 0.0),
        "generation_latency_seconds": locals().get("generation_latency", 0.0),
        "baseline_pipeline_latency_seconds": locals().get("baseline_pipeline_latency", 0.0),
        "rise_latency_seconds": rise_latency,
        "end_to_end_latency_seconds": time.perf_counter() - question_start,
    })

    result = {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "reference_answer": item["reference_answer"],
        "generated_answer": final_answer,
        "retrieved_sources": [
            {"source": source["source"], "page": source["page"]}
            for source in sources
        ],
        "evaluation": final_evaluation,
        "standard_metrics": final_metrics,
        "computational_cost": computational_cost,
        "initial_answer": initial_answer,
        "initial_evaluation": initial_evaluation,
        "initial_standard_metrics": initial_metrics,
        "rise": {
            "status": rise_status,
            "attempts": attempts,
            "accepted_attempt": accepted_attempt,
            "changed": final_answer != initial_answer,
            "final_answer": final_answer,
        },
        "status": "complete" if final_evaluation["score"] is not None else "error",
    }
    results.append(result)
    save_results_atomically(results)

print("\n============================================")
print("BENCHMARK COMPLETE")
print("============================================")


# Calculating the overall benchmark accuracy

total_questions = len(results)


# Counting valid evaluations and fully correct answers

valid_results = [

    result

    for result in results

    if result["evaluation"]["score"] is not None
]


valid_total = len(valid_results)


correct_answers = sum(

    1

    for result in valid_results

    if result["evaluation"]["score"] == 3
)


overall_accuracy = (

    correct_answers / valid_total * 100

    if valid_total > 0

    else 0
)


# Collecting accuracy and evaluation metrics for each question category

category_stats = defaultdict(
    lambda: {

        "total": 0,

        "correct": 0,

        "incorrect": 0,

        "accuracy": 0,

        "average_score": 0,

        "average_correctness": 0,

        "average_completeness": 0,

        "average_faithfulness": 0,

        "average_rouge1_f1": 0,

        "average_rouge2_f1": 0,

        "average_rougeL_f1": 0,

        "average_bleu": 0,

        "average_bertscore_precision": 0,

        "average_bertscore_recall": 0,

        "average_bertscore_f1": 0
    }
)


for result in results:

    category = result["category"]

    evaluation = result["evaluation"]


    category_stats[category]["total"] += 1


    # Counting answers with a full evaluation score as correct

    if evaluation["score"] == 3:

        category_stats[category]["correct"] += 1

    else:

        category_stats[category]["incorrect"] += 1


    # Adding the evaluator scores for each category

    if evaluation["score"] is not None:

        category_stats[category][
            "average_score"
        ] += evaluation["score"]


    if evaluation["correctness"] is not None:

        category_stats[category][
            "average_correctness"
        ] += evaluation["correctness"]


    if evaluation["completeness"] is not None:

        category_stats[category][
            "average_completeness"
        ] += evaluation["completeness"]


    if evaluation["faithfulness"] is not None:

        category_stats[category][
            "average_faithfulness"
        ] += evaluation["faithfulness"]

    # Adding text similarity scores for each category

    standard_metrics = result["standard_metrics"]

    if standard_metrics["rouge1_f1"] is not None:

        category_stats[category][
            "average_rouge1_f1"
        ] += standard_metrics["rouge1_f1"]


    if standard_metrics["rouge2_f1"] is not None:

        category_stats[category][
            "average_rouge2_f1"
        ] += standard_metrics["rouge2_f1"]


    if standard_metrics["rougeL_f1"] is not None:

        category_stats[category][
            "average_rougeL_f1"
        ] += standard_metrics["rougeL_f1"]


    if standard_metrics["bleu"] is not None:

        category_stats[category][
            "average_bleu"
        ] += standard_metrics["bleu"]


    if standard_metrics["bertscore_precision"] is not None:

        category_stats[category][
            "average_bertscore_precision"
        ] += standard_metrics["bertscore_precision"]


    if standard_metrics["bertscore_recall"] is not None:

        category_stats[category][
            "average_bertscore_recall"
        ] += standard_metrics["bertscore_recall"]


    if standard_metrics["bertscore_f1"] is not None:

        category_stats[category][
            "average_bertscore_f1"
        ] += standard_metrics["bertscore_f1"]


# Dividing the accumulated scores by the number of questions in each category

for category, stats in category_stats.items():

    total = stats["total"]


    stats["accuracy"] = (

        stats["correct"]
        / total
        * 100

        if total > 0

        else 0
    )


    stats["average_score"] = (

        stats["average_score"]
        / total

        if total > 0

        else 0
    )


    stats["average_correctness"] = (

        stats["average_correctness"]
        / total

        if total > 0

        else 0
    )


    stats["average_completeness"] = (

        stats["average_completeness"]
        / total

        if total > 0

        else 0
    )


    stats["average_faithfulness"] = (

        stats["average_faithfulness"]
        / total

        if total > 0

        else 0
    )

    stats["average_rouge1_f1"] = (
        stats["average_rouge1_f1"] / total
        if total > 0
        else 0
    )

    stats["average_rouge2_f1"] = (
        stats["average_rouge2_f1"] / total
        if total > 0
        else 0
    )

    stats["average_rougeL_f1"] = (
        stats["average_rougeL_f1"] / total
        if total > 0
        else 0
    )

    stats["average_bleu"] = (
        stats["average_bleu"] / total
        if total > 0
        else 0
    )

    stats["average_bertscore_precision"] = (
        stats["average_bertscore_precision"] / total
        if total > 0
        else 0
    )

    stats["average_bertscore_recall"] = (
        stats["average_bertscore_recall"] / total
        if total > 0
        else 0
    )

    stats["average_bertscore_f1"] = (
        stats["average_bertscore_f1"] / total
        if total > 0
        else 0
    )


# Summarising latency, model calls and Ollama processing durations

def _avg(values):
    return sum(values) / len(values) if values else 0.0

latencies = [r["computational_cost"]["end_to_end_latency_seconds"] for r in results]
computational_summary = {
    "average_end_to_end_latency_seconds": _avg(latencies),
    "median_end_to_end_latency_seconds": statistics.median(latencies) if latencies else 0.0,
    "std_end_to_end_latency_seconds": statistics.pstdev(latencies) if len(latencies) > 1 else 0.0,
    "average_retrieval_latency_seconds": _avg([r["computational_cost"]["retrieval_latency_seconds"] for r in results]),
    "average_generation_latency_seconds": _avg([r["computational_cost"]["generation_latency_seconds"] for r in results]),
    "average_rise_latency_seconds": _avg([r["computational_cost"]["rise_latency_seconds"] for r in results]),
    "average_llm_calls": _avg([r["computational_cost"]["llm_calls"] for r in results]),
    "total_llm_calls": sum(r["computational_cost"]["llm_calls"] for r in results),
    "average_embedding_calls": _avg([r["computational_cost"]["embedding_calls"] for r in results]),
    "total_embedding_calls": sum(r["computational_cost"]["embedding_calls"] for r in results),
    "average_ollama_total_duration_seconds": _avg([r["computational_cost"]["ollama_total_duration_seconds"] for r in results]),
    "average_ollama_load_duration_seconds": _avg([r["computational_cost"]["ollama_load_duration_seconds"] for r in results]),
    "average_ollama_prompt_eval_duration_seconds": _avg([r["computational_cost"]["ollama_prompt_eval_duration_seconds"] for r in results]),
    "average_ollama_eval_duration_seconds": _avg([r["computational_cost"]["ollama_eval_duration_seconds"] for r in results]),
}


# Organising the overall and category results into a JSON summary

summary = {

    "overall": {

        "total_questions":
            total_questions,

        "evaluated_questions":
            valid_total,

        "correct":
            correct_answers,

        "incorrect":
            valid_total - correct_answers,

        "accuracy":
            round(
                overall_accuracy,
                2
            )
    },

    "computational_cost": {key: round(value, 6) if isinstance(value, float) else value for key, value in computational_summary.items()},

    "categories": {

        category: {

            "total_questions":
                stats["total"],

            "correct":
                stats["correct"],

            "incorrect":
                stats["incorrect"],

            "accuracy":
                round(
                    stats["accuracy"],
                    2
                ),

            "average_score":
                round(
                    stats["average_score"],
                    2
                ),

            "average_correctness":
                round(
                    stats["average_correctness"],
                    2
                ),

            "average_completeness":
                round(
                    stats["average_completeness"],
                    2
                ),

            "average_faithfulness":
                round(
                    stats["average_faithfulness"],
                    2
                ),

            "average_rouge1_f1":
                round(
                    stats["average_rouge1_f1"],
                    4
                ),

            "average_rouge2_f1":
                round(
                    stats["average_rouge2_f1"],
                    4
                ),

            "average_rougeL_f1":
                round(
                    stats["average_rougeL_f1"],
                    4
                ),

            "average_bleu":
                round(
                    stats["average_bleu"],
                    4
                ),

            "average_bertscore_precision":
                round(
                    stats["average_bertscore_precision"],
                    4
                ),

            "average_bertscore_recall":
                round(
                    stats["average_bertscore_recall"],
                    4
                ),

            "average_bertscore_f1":
                round(
                    stats["average_bertscore_f1"],
                    4
                )
        }

        for category, stats
        in category_stats.items()
    }
}


# Saving the summary statistics as a JSON file

with open(
    SUMMARY_JSON_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4,
        ensure_ascii=False
    )


# Preparing a readable text summary of the benchmark results

summary_lines = []


summary_lines.append(
    "============================================"
)

summary_lines.append(
    "RAG BENCHMARK SUMMARY"
)

summary_lines.append(
    "============================================"
)

summary_lines.append("")


summary_lines.append(
    f"Overall accuracy: "
    f"{overall_accuracy:.2f}%"
)


summary_lines.append(
    f"Questions evaluated: "
    f"{valid_total}/{total_questions}"
)


summary_lines.append("")

summary_lines.append(
    "Accuracy by category:"
)

summary_lines.append("")


for category, stats in category_stats.items():

    summary_lines.append(

        f"{category}: "
        f"{stats['accuracy']:.2f}% "
        f"({stats['correct']}/{stats['total']})"
    )


summary_lines.append("")

summary_lines.append(
    "============================================"
)

summary_lines.append("COMPUTATIONAL COST")
summary_lines.append("============================================")
summary_lines.append("")
summary_lines.append(f"Average end-to-end latency: {computational_summary['average_end_to_end_latency_seconds']:.4f} s")
summary_lines.append(f"Median end-to-end latency: {computational_summary['median_end_to_end_latency_seconds']:.4f} s")
summary_lines.append(f"Latency standard deviation: {computational_summary['std_end_to_end_latency_seconds']:.4f} s")
summary_lines.append(f"Average retrieval latency: {computational_summary['average_retrieval_latency_seconds']:.4f} s")
summary_lines.append(f"Average generation latency: {computational_summary['average_generation_latency_seconds']:.4f} s")
summary_lines.append(f"Average RISE latency: {computational_summary['average_rise_latency_seconds']:.4f} s")
summary_lines.append(f"Average LLM calls/question: {computational_summary['average_llm_calls']:.2f}")
summary_lines.append(f"Total LLM calls: {computational_summary['total_llm_calls']}")
summary_lines.append(f"Average embedding calls/question: {computational_summary['average_embedding_calls']:.2f}")
summary_lines.append(f"Total embedding calls: {computational_summary['total_embedding_calls']}")
summary_lines.append(f"Average Ollama total duration: {computational_summary['average_ollama_total_duration_seconds']:.4f} s")
summary_lines.append(f"Average Ollama load duration: {computational_summary['average_ollama_load_duration_seconds']:.4f} s")
summary_lines.append(f"Average Ollama prompt-eval duration: {computational_summary['average_ollama_prompt_eval_duration_seconds']:.4f} s")
summary_lines.append(f"Average Ollama eval duration: {computational_summary['average_ollama_eval_duration_seconds']:.4f} s")
summary_lines.append("")
summary_lines.append("============================================")
summary_lines.append(
    "ADDITIONAL METRICS"
)

summary_lines.append(
    "============================================"
)

summary_lines.append("")


for category, stats in category_stats.items():

    summary_lines.append(

        f"{category}:"
    )

    summary_lines.append(

        f"  Accuracy: "
        f"{stats['accuracy']:.2f}%"
    )

    summary_lines.append(

        f"  Average Score: "
        f"{stats['average_score']:.2f}/3"
    )

    summary_lines.append(

        f"  Correctness: "
        f"{stats['average_correctness']:.2f}"
    )

    summary_lines.append(

        f"  Completeness: "
        f"{stats['average_completeness']:.2f}"
    )

    summary_lines.append(

        f"  Faithfulness: "
        f"{stats['average_faithfulness']:.2f}"
    )

    summary_lines.append(
        f"  ROUGE-1 F1: "
        f"{stats['average_rouge1_f1']:.4f}"
    )

    summary_lines.append(
        f"  ROUGE-2 F1: "
        f"{stats['average_rouge2_f1']:.4f}"
    )

    summary_lines.append(
        f"  ROUGE-L F1: "
        f"{stats['average_rougeL_f1']:.4f}"
    )

    summary_lines.append(
        f"  BLEU: "
        f"{stats['average_bleu']:.4f}"
    )

    summary_lines.append(
        f"  BERTScore F1: "
        f"{stats['average_bertscore_f1']:.4f}"
    )

    summary_lines.append("")


with open(
    SUMMARY_TEXT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(summary_lines)
    )


# Displaying the final benchmark results and output filenames

print("\n============================================")
print("BENCHMARK SUMMARY")
print("============================================")

print(
    f"Overall accuracy: "
    f"{overall_accuracy:.2f}%"
)

print("\nAccuracy by category:")

for category, stats in category_stats.items():

    print(

        f"{category}: "
        f"{stats['accuracy']:.2f}% "
        f"({stats['correct']}/{stats['total']})"
    )


print("\nFiles created:")

print(
    f"- {OUTPUT_FILE}"
)

print(
    f"- {SUMMARY_JSON_FILE}"
)

print(
    f"- {SUMMARY_TEXT_FILE}"
)

print("\nDone.")