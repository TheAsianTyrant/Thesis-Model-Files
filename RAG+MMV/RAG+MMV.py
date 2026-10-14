# Importing libraries necessary for code functions
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

# Similarity and semantic evaluation metric libraries
from rouge_score import rouge_scorer
import sacrebleu
from bert_score import score as bertscore



# Naming the document corpus, benchmark questions, models, and output files
DOCUMENT_FOLDER = "docs"
QUESTIONS_FILE = "Questions.txt"

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "rag_collection"

EMBEDDING_MODEL = "nomic-embed-text"
GENERATION_MODEL = "llama3"
EVALUATION_MODEL = "llama3"

# Three independent models used to verify generated answers against retrieved evidence
VERIFIER_MODELS = [
    ("llama3", "llama3"),
    ("qwen3", "qwen3:8b"),
    ("mistral", "mistral:7b"),
]
VERIFIER_TEMPERATURE = 0.0

OUTPUT_FILE = "evaluation_results.json"
SUMMARY_JSON_FILE = "evaluation_summary.json"
SUMMARY_TEXT_FILE = "evaluation_summary.txt"

# Settings for document chunking and retrieval
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 12



# Getting and loading PDF documents from the governance corpus
print("\n============================================")
print("LOADING DOCUMENTS")
print("============================================")

documents = []

for filename in sorted(os.listdir(DOCUMENT_FOLDER)):

    # Skip files that are not PDFs
    if not filename.lower().endswith(".pdf"):
        continue

    path = os.path.join(DOCUMENT_FOLDER, filename)

    print(f"Loading: {filename}")

    try:
        reader = PdfReader(path)

        pages = []

        for page_number, page in enumerate(reader.pages, start=1):

            try:
                # Extract text from each page while handling pages that cannot be read
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



# Splitting document pages into overlapping chunks for retrieval
print("\n============================================")
print("CHUNKING DOCUMENTS")
print("============================================")


# Create fixed-length text chunks with overlap to preserve context
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

# Store each chunk with its document name and page number
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



# Rebuilding the vector database to avoid using embeddings from previous runs
print("\n============================================")
print("REBUILDING VECTOR DATABASE")
print("============================================")



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



print("\n============================================")
print("CREATING EMBEDDINGS")
print("============================================")


# Generate an embedding for every chunk and store it with its source metadata
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



# Loading the benchmark questions and reference answers
print("\n============================================")
print("LOADING QUESTIONS")
print("============================================")


# Parse question IDs, categories, questions, and answers from the benchmark file
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



# Helpers for converting and combining Ollama timing information
def _duration_seconds(response, field):
    """Convert Ollama nanosecond duration fields to seconds."""
    value = response.get(field, 0) if hasattr(response, "get") else 0
    try:
        return float(value or 0) / 1_000_000_000.0
    except (TypeError, ValueError):
        return 0.0


def _ollama_durations(response):
    return {
        "ollama_total_duration_seconds": _duration_seconds(response, "total_duration"),
        "ollama_load_duration_seconds": _duration_seconds(response, "load_duration"),
        "ollama_prompt_eval_duration_seconds": _duration_seconds(response, "prompt_eval_duration"),
        "ollama_eval_duration_seconds": _duration_seconds(response, "eval_duration"),
    }


def _add_duration_metrics(target, source):
    for key in (
        "ollama_total_duration_seconds",
        "ollama_load_duration_seconds",
        "ollama_prompt_eval_duration_seconds",
        "ollama_eval_duration_seconds",
    ):
        target[key] = target.get(key, 0.0) + source.get(key, 0.0)



# Generate an answer using retrieved governance document passages
def generate_answer(question):

    pipeline_start = time.perf_counter()
    embedding_calls = 0
    llm_calls = 0


    # Convert the question into an embedding for similarity search
    retrieval_start = time.perf_counter()
    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=question
    )
    embedding_calls += 1

    question_embedding = response["embedding"]



    # Retrieve the most relevant document chunks
    results = collection.query(

        query_embeddings=[
            question_embedding
        ],

        n_results=RETRIEVAL_K
    )


    retrieved_docs = results["documents"][0]

    retrieved_metadata = results["metadatas"][0]
    retrieval_latency = time.perf_counter() - retrieval_start



    # Combine retrieved text with document and page references
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



    # Instruct the model to answer using only the retrieved evidence
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


    # Generate the initial RAG answer and record its computational cost
    generation_start = time.perf_counter()
    response = ollama.chat(

        model=GENERATION_MODEL,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={"temperature": 0.0}
    )


    llm_calls += 1
    generation_latency = time.perf_counter() - generation_start
    answer = response["message"]["content"].strip()

    generation_cost = {
        "retrieval_latency_seconds": retrieval_latency,
        "generation_latency_seconds": generation_latency,
        "rag_latency_seconds": time.perf_counter() - pipeline_start,
        "llm_calls": llm_calls,
        "embedding_calls": embedding_calls,
        **_ollama_durations(response),
    }

    return answer, retrieved_metadata, retrieved_docs, generation_cost



# Verify whether an answer is supported, contradicted, or not found in the retrieved evidence
def verify_with_model(model_name, question, generated_answer, retrieved_docs, retrieved_metadata):
    """Verify one generated answer against exactly the evidence used by RAG."""
    verification_start = time.perf_counter()

    # Supply the same retrieved passages used to generate the answer
    evidence_parts = []
    for i, doc in enumerate(retrieved_docs):
        metadata = retrieved_metadata[i]
        evidence_parts.append(
            f"SOURCE: {metadata['source']}\n"
            f"PAGE: {metadata['page']}\n"
            f"CONTENT:\n{doc}"
        )

    evidence = "\n\n--- EVIDENCE PASSAGE ---\n\n".join(evidence_parts)

    verification_prompt = f"""
You are an independent verifier for a question-answering system operating over governance documents.

Your task is NOT to answer the question again from general knowledge. Your task is to determine whether the GENERATED ANSWER is supported by the RETRIEVED EVIDENCE supplied below.

Use only the retrieved evidence. Do not use outside knowledge.

Evaluate the material claims needed for the generated answer to answer the question.

VERDICTS:

SUPPORTED
- The retrieved evidence directly establishes the material claims in the generated answer.
- Minor wording differences are acceptable when the meaning is the same.
- Do not mark a claim supported merely because the evidence is related, plausible, or consistent with it.
- Any inference must be logically necessary from the supplied evidence and must not require an unstated assumption.

CONTRADICTED
- The retrieved evidence directly conflicts with a material claim in the generated answer.

NOT_FOUND
- The retrieved evidence does not establish a material claim needed for the answer.
- Use NOT_FOUND when support would require an assumption, analogy, extrapolation, substitution of an entity/role/document/process, or strengthening/weakening the wording.
- Use NOT_FOUND when the answer may be plausible but the supplied evidence is insufficient to verify it.

IMPORTANT:
- Match the exact subject/entity, action or relationship, object, scope, conditions, modality, and timing asked by the question.
- Distinguish permissions from requirements, recommendations from approvals, and related responsibilities from the specific responsibility being asked about.
- A rule stated for one role, body, document, procedure, or time period must not automatically be transferred to another.
- If the answer contains multiple material claims, all claims required for the answer must be supported for an overall SUPPORTED verdict.
- Ignore harmless extra wording only when it does not add an unsupported material factual claim.

QUESTION:
{question}

GENERATED ANSWER:
{generated_answer}

RETRIEVED EVIDENCE:
{evidence}

Return ONLY this format:
VERDICT: <SUPPORTED, CONTRADICTED, or NOT_FOUND>
REASON: <one or two concise sentences explaining the evidence-based decision>
"""

    # Ask the verifier model for an evidence-based verdict
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": verification_prompt}],
        options={"temperature": VERIFIER_TEMPERATURE}
    )

    verification_latency = time.perf_counter() - verification_start
    text = response["message"]["content"].strip()
    # Extract the verifier verdict and explanation from its response
    verdict_match = re.search(
        r"VERDICT:\s*(SUPPORTED|CONTRADICTED|NOT_FOUND)",
        text,
        re.IGNORECASE
    )
    reason_match = re.search(
        r"REASON:\s*(.*)",
        text,
        re.IGNORECASE | re.DOTALL
    )

    verdict = verdict_match.group(1).upper() if verdict_match else "PARSE_ERROR"
    reason = reason_match.group(1).strip() if reason_match else text

    return {
        "verdict": verdict,
        "reason": reason,
        "raw": text,
        "computational_cost": {
            "latency_seconds": verification_latency,
            "llm_calls": 1,
            **_ollama_durations(response),
        },
    }


# Run all three verification models and combine their decisions
def multi_model_verify(question, generated_answer, retrieved_docs, retrieved_metadata):
    verification_start = time.perf_counter()
    individual = {}

    for label, model_name in VERIFIER_MODELS:
        try:
            print(f"  Verifying with {label} ({model_name})...")
            individual[label] = verify_with_model(
                model_name,
                question,
                generated_answer,
                retrieved_docs,
                retrieved_metadata
            )
        except Exception as e:
            print(f"  ERROR verifying with {label}: {e}")
            individual[label] = {
                "verdict": "ERROR",
                "reason": str(e),
                "raw": "",
                "computational_cost": {
                    "latency_seconds": 0.0,
                    "llm_calls": 0,
                    "ollama_total_duration_seconds": 0.0,
                    "ollama_load_duration_seconds": 0.0,
                    "ollama_prompt_eval_duration_seconds": 0.0,
                    "ollama_eval_duration_seconds": 0.0,
                },
            }

    # Count supported verdicts as PASS and contradicted or missing evidence as FAIL
    valid = [
        v["verdict"] for v in individual.values()
        if v["verdict"] in {"SUPPORTED", "CONTRADICTED", "NOT_FOUND"}
    ]

    pass_votes = sum(v == "SUPPORTED" for v in valid)
    fail_votes = sum(v in {"CONTRADICTED", "NOT_FOUND"} for v in valid)

    # Require all three valid responses before deciding the binary majority
    if len(valid) == len(VERIFIER_MODELS):
        majority_binary = "PASS" if pass_votes >= 2 else "FAIL"
    else:
        majority_binary = "UNRESOLVED"

# Preserve three-way consensus when at least two models agree
    counts = {name: valid.count(name) for name in {"SUPPORTED", "CONTRADICTED", "NOT_FOUND"}}
    majority_three_way = "MIXED"
    if counts:
        best = max(counts, key=counts.get)
        if counts[best] >= 2:
            majority_three_way = best

    # Measure agreement between individual verifiers
    unanimous = len(valid) == 3 and len(set(valid)) == 1
    binary_votes = ["PASS" if v == "SUPPORTED" else "FAIL" for v in valid]
    unanimous_binary = len(binary_votes) == 3 and len(set(binary_votes)) == 1

    # Aggregate the computational cost of all verification calls
    verification_cost = {
        "verification_latency_seconds": time.perf_counter() - verification_start,
        "llm_calls": 0,
        "ollama_total_duration_seconds": 0.0,
        "ollama_load_duration_seconds": 0.0,
        "ollama_prompt_eval_duration_seconds": 0.0,
        "ollama_eval_duration_seconds": 0.0,
    }
    for data in individual.values():
        cost = data.get("computational_cost", {})
        verification_cost["llm_calls"] += cost.get("llm_calls", 0)
        _add_duration_metrics(verification_cost, cost)

    return {
        "models": individual,
        "computational_cost": verification_cost,
        "majority_verdict": majority_three_way,
        "majority_binary": majority_binary,
        "pass_votes": pass_votes,
        "fail_votes": fail_votes,
        "unanimous_three_way": unanimous,
        "unanimous_binary": unanimous_binary
    }



# Evaluate the generated answer against its reference answer
def evaluate_answer(
    question,
    reference_answer,
    generated_answer
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


    evaluation_text = response[
        "message"
    ]["content"].strip()



    # Extract correctness, completeness, faithfulness, score, and reason from the evaluator response
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


# Calculate lexical and semantic similarity against the reference answer
def calculate_standard_metrics(reference_answer, generated_answer):


    # ROUGE measures word and sequence overlap
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



    # BLEU measures n-gram precision against the reference answer
    bleu = sacrebleu.sentence_bleu(
        generated_answer,
        [reference_answer]
    )

    bleu_score = bleu.score / 100.0



    # BERTScore measures semantic similarity using contextual embeddings
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



# Run the full benchmark over every question
print("\n============================================")
print("STARTING BENCHMARK")
print("============================================")

results = []


for i, item in enumerate(questions):


    print(
        f"Processing {i + 1}/{len(questions)} "
        f"({item['id']})"
    )



    # Generate the initial answer, retaining fallback values if generation fails
    model_pipeline_start = time.perf_counter()

    try:

        generated_answer, sources, retrieved_docs, generation_cost = generate_answer(
            item["question"]
        )

    except Exception as e:

        print(
            f"ERROR generating answer for "
            f"{item['id']}: {e}"
        )

        generated_answer = (
            "ERROR: Could not generate answer."
        )

        sources = []
        retrieved_docs = []
        generation_cost = {
            "retrieval_latency_seconds": 0.0,
            "generation_latency_seconds": 0.0,
            "rag_latency_seconds": 0.0,
            "llm_calls": 0,
            "embedding_calls": 0,
            "ollama_total_duration_seconds": 0.0,
            "ollama_load_duration_seconds": 0.0,
            "ollama_prompt_eval_duration_seconds": 0.0,
            "ollama_eval_duration_seconds": 0.0,
        }



    # Verify the generated answer using the three-model ensemble
    try:
        multi_verification = multi_model_verify(
            item["question"],
            generated_answer,
            retrieved_docs,
            sources
        )
    except Exception as e:
        print(f"ERROR in multi-model verification for {item['id']}: {e}")
        multi_verification = {
            "models": {},
            "majority_verdict": "ERROR",
            "majority_binary": "UNRESOLVED",
            "pass_votes": 0,
            "fail_votes": 0,
            "unanimous_three_way": False,
            "unanimous_binary": False,
            "computational_cost": {
                "verification_latency_seconds": 0.0,
                "llm_calls": 0,
                "ollama_total_duration_seconds": 0.0,
                "ollama_load_duration_seconds": 0.0,
                "ollama_prompt_eval_duration_seconds": 0.0,
                "ollama_eval_duration_seconds": 0.0,
            },
        }


    # Combine answer generation and verification computational costs
    verification_cost = multi_verification.get("computational_cost", {})
    computational_cost = {
        "end_to_end_latency_seconds": time.perf_counter() - model_pipeline_start,
        "retrieval_latency_seconds": generation_cost.get("retrieval_latency_seconds", 0.0),
        "generation_latency_seconds": generation_cost.get("generation_latency_seconds", 0.0),
        "verification_latency_seconds": verification_cost.get("verification_latency_seconds", 0.0),
        "llm_calls": generation_cost.get("llm_calls", 0) + verification_cost.get("llm_calls", 0),
        "embedding_calls": generation_cost.get("embedding_calls", 0),
        "ollama_total_duration_seconds": generation_cost.get("ollama_total_duration_seconds", 0.0) + verification_cost.get("ollama_total_duration_seconds", 0.0),
        "ollama_load_duration_seconds": generation_cost.get("ollama_load_duration_seconds", 0.0) + verification_cost.get("ollama_load_duration_seconds", 0.0),
        "ollama_prompt_eval_duration_seconds": generation_cost.get("ollama_prompt_eval_duration_seconds", 0.0) + verification_cost.get("ollama_prompt_eval_duration_seconds", 0.0),
        "ollama_eval_duration_seconds": generation_cost.get("ollama_eval_duration_seconds", 0.0) + verification_cost.get("ollama_eval_duration_seconds", 0.0),
    }



    # Evaluate the generated answer using the reference answer
    try:

        evaluation = evaluate_answer(

            item["question"],

            item["reference_answer"],

            generated_answer
        )

    except Exception as e:

        print(
            f"ERROR evaluating "
            f"{item['id']}: {e}"
        )

        evaluation = {

            "correctness": None,

            "completeness": None,

            "faithfulness": None,

            "score": None,

            "reason":
                f"Evaluation error: {e}"
        }


    # Calculate ROUGE, BLEU, and BERTScore metrics
    try:

        standard_metrics = calculate_standard_metrics(
            item["reference_answer"],
            generated_answer
        )

    except Exception as e:

        print(
            f"ERROR calculating standard metrics "
            f"for {item['id']}: {e}"
        )

        standard_metrics = {

            "rouge1_f1": None,
            "rouge2_f1": None,
            "rougeL_f1": None,

            "bleu": None,

            "bertscore_precision": None,
            "bertscore_recall": None,
            "bertscore_f1": None
        }



    # Store each question, answer, evaluation, verification, and computational result
    result = {

        "id":
            item["id"],

        "category":
            item["category"],

        "question":
            item["question"],

        "reference_answer":
            item["reference_answer"],

        "generated_answer":
            generated_answer,

        "retrieved_sources": [

            {
                "source":
                    source["source"],

                "page":
                    source["page"]
            }

            for source in sources
        ],

        "multi_model_verification":
            multi_verification,

        "computational_cost":
            computational_cost,

        "evaluation":
            evaluation,

        "standard_metrics": 
            standard_metrics
    }


    results.append(result)



    # Save results after each question so completed work survives interruptions
    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )


print("\n============================================")
print("BENCHMARK COMPLETE")
print("============================================")



# Calculate overall accuracy from questions with valid evaluation scores
total_questions = len(results)



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



# Calculate average latency, model calls, and Ollama durations
def _mean_cost(key):
    values = [r.get("computational_cost", {}).get(key) for r in results]
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.mean(values) if values else 0.0


latencies = [
    r.get("computational_cost", {}).get("end_to_end_latency_seconds")
    for r in results
]
latencies = [v for v in latencies if isinstance(v, (int, float))]

computational_summary = {
    "average_end_to_end_latency_seconds": _mean_cost("end_to_end_latency_seconds"),
    "median_end_to_end_latency_seconds": statistics.median(latencies) if latencies else 0.0,
    "std_end_to_end_latency_seconds": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
    "average_retrieval_latency_seconds": _mean_cost("retrieval_latency_seconds"),
    "average_generation_latency_seconds": _mean_cost("generation_latency_seconds"),
    "average_verification_latency_seconds": _mean_cost("verification_latency_seconds"),
    "average_llm_calls": _mean_cost("llm_calls"),
    "total_llm_calls": sum(r.get("computational_cost", {}).get("llm_calls", 0) for r in results),
    "average_embedding_calls": _mean_cost("embedding_calls"),
    "total_embedding_calls": sum(r.get("computational_cost", {}).get("embedding_calls", 0) for r in results),
    "average_ollama_total_duration_seconds": _mean_cost("ollama_total_duration_seconds"),
    "average_ollama_load_duration_seconds": _mean_cost("ollama_load_duration_seconds"),
    "average_ollama_prompt_eval_duration_seconds": _mean_cost("ollama_prompt_eval_duration_seconds"),
    "average_ollama_eval_duration_seconds": _mean_cost("ollama_eval_duration_seconds"),
}



# Initialise accuracy, evaluation, similarity, and computational statistics for each category
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

        "average_bertscore_f1": 0,
        "average_end_to_end_latency_seconds": 0,
        "average_retrieval_latency_seconds": 0,
        "average_generation_latency_seconds": 0,
        "average_verification_latency_seconds": 0,
        "average_llm_calls": 0,
        "average_embedding_calls": 0,
        "average_ollama_total_duration_seconds": 0,
        "average_ollama_load_duration_seconds": 0,
        "average_ollama_prompt_eval_duration_seconds": 0,
        "average_ollama_eval_duration_seconds": 0
    }
)


# Accumulate evaluation and metric totals by question category
for result in results:

    category = result["category"]

    evaluation = result["evaluation"]


    category_stats[category]["total"] += 1


# Count answers receiving the maximum evaluation score as correct

    if evaluation["score"] == 3:

        category_stats[category]["correct"] += 1

    else:

        category_stats[category]["incorrect"] += 1



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


    # Accumulate standard text similarity scores
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


    # Accumulate computational cost measurements for each category
    computational_cost = result.get("computational_cost", {})
    for key in (
        "end_to_end_latency_seconds",
        "retrieval_latency_seconds",
        "generation_latency_seconds",
        "verification_latency_seconds",
        "llm_calls",
        "embedding_calls",
        "ollama_total_duration_seconds",
        "ollama_load_duration_seconds",
        "ollama_prompt_eval_duration_seconds",
        "ollama_eval_duration_seconds",
    ):
        value = computational_cost.get(key)
        if isinstance(value, (int, float)):
            category_stats[category][f"average_{key}"] += value



# Convert category totals into averages and accuracy percentages
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


    for key in (
        "average_end_to_end_latency_seconds",
        "average_retrieval_latency_seconds",
        "average_generation_latency_seconds",
        "average_verification_latency_seconds",
        "average_llm_calls",
        "average_embedding_calls",
        "average_ollama_total_duration_seconds",
        "average_ollama_load_duration_seconds",
        "average_ollama_prompt_eval_duration_seconds",
        "average_ollama_eval_duration_seconds",
    ):
        stats[key] = stats[key] / total if total > 0 else 0



# Compare individual verifier and ensemble decisions with benchmark correctness
verifier_labels = [label for label, _ in VERIFIER_MODELS]

verification_stats = {
    label: {
        "evaluated": 0,
        "correct_classification": 0,
        "incorrect_classification": 0,
        "accuracy": 0,
        "true_positive": 0,
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0
    }
    for label in verifier_labels
}

ensemble_stats = {
    "evaluated": 0,
    "correct_classification": 0,
    "incorrect_classification": 0,
    "accuracy": 0,
    "true_positive": 0,
    "true_negative": 0,
    "false_positive": 0,
    "false_negative": 0,
    "unanimous_binary": 0,
    "two_to_one_binary_disagreement": 0,
    "unanimous_binary_rate": 0,
    "two_to_one_binary_disagreement_rate": 0,
    "llama_disagrees_with_both_other_models": 0,
    "llama_disagrees_with_both_other_models_rate": 0
}

verification_category_stats = defaultdict(lambda: {
    "total": 0,
    "ensemble_correct": 0,
    "ensemble_incorrect": 0,
    "ensemble_accuracy": 0,
    **{f"{label}_correct": 0 for label in verifier_labels},
    **{f"{label}_accuracy": 0 for label in verifier_labels}
})

for result in results:
    score = result["evaluation"]["score"]
    if score is None:
        continue

# Treat score 3 as PASS and scores 0–2 as FAIL for verification classification
    actual_binary = "PASS" if score == 3 else "FAIL"
    mmv = result.get("multi_model_verification", {})
    category = result["category"]
    verification_category_stats[category]["total"] += 1

    model_binary = {}
    for label in verifier_labels:
        data = mmv.get("models", {}).get(label, {})
        verdict = data.get("verdict")
        if verdict not in {"SUPPORTED", "CONTRADICTED", "NOT_FOUND"}:
            continue

        # Record verifier classification accuracy and confusion matrix counts
        predicted = "PASS" if verdict == "SUPPORTED" else "FAIL"
        model_binary[label] = predicted
        st = verification_stats[label]
        st["evaluated"] += 1
        is_correct = predicted == actual_binary
        st["correct_classification" if is_correct else "incorrect_classification"] += 1
        verification_category_stats[category][f"{label}_correct"] += int(is_correct)

        if actual_binary == "FAIL" and predicted == "FAIL":
            st["true_positive"] += 1
        elif actual_binary == "PASS" and predicted == "PASS":
            st["true_negative"] += 1
        elif actual_binary == "PASS" and predicted == "FAIL":
            st["false_positive"] += 1
        elif actual_binary == "FAIL" and predicted == "PASS":
            st["false_negative"] += 1

    # Evaluate the ensemble majority against the benchmark classification
    ensemble = mmv.get("majority_binary")
    if ensemble in {"PASS", "FAIL"}:
        ensemble_stats["evaluated"] += 1
        is_correct = ensemble == actual_binary
        ensemble_stats["correct_classification" if is_correct else "incorrect_classification"] += 1
        verification_category_stats[category]["ensemble_correct"] += int(is_correct)
        verification_category_stats[category]["ensemble_incorrect"] += int(not is_correct)

        if actual_binary == "FAIL" and ensemble == "FAIL":
            ensemble_stats["true_positive"] += 1
        elif actual_binary == "PASS" and ensemble == "PASS":
            ensemble_stats["true_negative"] += 1
        elif actual_binary == "PASS" and ensemble == "FAIL":
            ensemble_stats["false_positive"] += 1
        elif actual_binary == "FAIL" and ensemble == "PASS":
            ensemble_stats["false_negative"] += 1

    # Track unanimous agreement and two-to-one disagreements
    if mmv.get("unanimous_binary"):
        ensemble_stats["unanimous_binary"] += 1
    elif len(model_binary) == 3:
        ensemble_stats["two_to_one_binary_disagreement"] += 1

    # Check whether Llama disagrees with both other verifier models
    if len(model_binary) == 3:
        llama_vote = model_binary.get("llama3")
        qwen_vote = model_binary.get("qwen3")
        mistral_vote = model_binary.get("mistral")
        if llama_vote is not None and qwen_vote == mistral_vote and llama_vote != qwen_vote:
            ensemble_stats["llama_disagrees_with_both_other_models"] += 1

# Calculate individual and ensemble verification accuracy rates
for label, st in verification_stats.items():
    if st["evaluated"]:
        st["accuracy"] = round(st["correct_classification"] / st["evaluated"] * 100, 2)

if ensemble_stats["evaluated"]:
    ensemble_stats["accuracy"] = round(
        ensemble_stats["correct_classification"] / ensemble_stats["evaluated"] * 100, 2
    )
    ensemble_stats["unanimous_binary_rate"] = round(
        ensemble_stats["unanimous_binary"] / ensemble_stats["evaluated"] * 100, 2
    )
    ensemble_stats["two_to_one_binary_disagreement_rate"] = round(
        ensemble_stats["two_to_one_binary_disagreement"] / ensemble_stats["evaluated"] * 100, 2
    )
    ensemble_stats["llama_disagrees_with_both_other_models_rate"] = round(
        ensemble_stats["llama_disagrees_with_both_other_models"] / ensemble_stats["evaluated"] * 100, 2
    )

# Calculate verification accuracy for each benchmark category
for category, st in verification_category_stats.items():
    total = st["total"]
    if total:
        st["ensemble_accuracy"] = round(st["ensemble_correct"] / total * 100, 2)
        for label in verifier_labels:
            st[f"{label}_accuracy"] = round(st[f"{label}_correct"] / total * 100, 2)



# Organise overall, computational, verification, and category results for JSON export
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

    "computational_cost": {
        key: round(value, 4) if isinstance(value, float) else value
        for key, value in computational_summary.items()
    },

    "multi_model_verification": {
        "models": {label: model for label, model in VERIFIER_MODELS},
        "verifier_statistics": verification_stats,
        "ensemble_statistics": ensemble_stats,
        "category_statistics": dict(verification_category_stats)
    },


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
                ),

            "computational_cost": {
                "average_end_to_end_latency_seconds": round(stats["average_end_to_end_latency_seconds"], 4),
                "average_retrieval_latency_seconds": round(stats["average_retrieval_latency_seconds"], 4),
                "average_generation_latency_seconds": round(stats["average_generation_latency_seconds"], 4),
                "average_verification_latency_seconds": round(stats["average_verification_latency_seconds"], 4),
                "average_llm_calls": round(stats["average_llm_calls"], 2),
                "average_embedding_calls": round(stats["average_embedding_calls"], 2),
                "average_ollama_total_duration_seconds": round(stats["average_ollama_total_duration_seconds"], 4),
                "average_ollama_load_duration_seconds": round(stats["average_ollama_load_duration_seconds"], 4),
                "average_ollama_prompt_eval_duration_seconds": round(stats["average_ollama_prompt_eval_duration_seconds"], 4),
                "average_ollama_eval_duration_seconds": round(stats["average_ollama_eval_duration_seconds"], 4)
            }
        }

        for category, stats
        in category_stats.items()
    }
}



# Save the structured benchmark summary as JSON
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



# Build a readable text summary of accuracy, costs, and verification statistics
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


summary_lines.append("============================================")
summary_lines.append("COMPUTATIONAL COST")
summary_lines.append("============================================")
summary_lines.append("")
summary_lines.append(f"Average end-to-end latency: {computational_summary['average_end_to_end_latency_seconds']:.4f} s")
summary_lines.append(f"Median end-to-end latency: {computational_summary['median_end_to_end_latency_seconds']:.4f} s")
summary_lines.append(f"Latency standard deviation: {computational_summary['std_end_to_end_latency_seconds']:.4f} s")
summary_lines.append(f"Average retrieval latency: {computational_summary['average_retrieval_latency_seconds']:.4f} s")
summary_lines.append(f"Average generation latency: {computational_summary['average_generation_latency_seconds']:.4f} s")
summary_lines.append(f"Average verification latency: {computational_summary['average_verification_latency_seconds']:.4f} s")
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
summary_lines.append("MULTI-MODEL VERIFICATION")
summary_lines.append("============================================")
summary_lines.append("")
for label in verifier_labels:
    st = verification_stats[label]
    summary_lines.append(
        f"{label}: {st['accuracy']:.2f}% verification classification accuracy "
        f"({st['correct_classification']}/{st['evaluated']})"
    )
    summary_lines.append(
        f"  TP={st['true_positive']} TN={st['true_negative']} "
        f"FP={st['false_positive']} FN={st['false_negative']}"
    )
summary_lines.append("")
summary_lines.append(
    f"Ensemble: {ensemble_stats['accuracy']:.2f}% verification classification accuracy "
    f"({ensemble_stats['correct_classification']}/{ensemble_stats['evaluated']})"
)
summary_lines.append(
    f"  TP={ensemble_stats['true_positive']} TN={ensemble_stats['true_negative']} "
    f"FP={ensemble_stats['false_positive']} FN={ensemble_stats['false_negative']}"
)
summary_lines.append(
    f"  Unanimous binary agreement: {ensemble_stats['unanimous_binary_rate']:.2f}%"
)
summary_lines.append(
    f"  2-1 binary disagreement: {ensemble_stats['two_to_one_binary_disagreement_rate']:.2f}%"
)
summary_lines.append(
    f"  Llama disagrees with both independent verifiers: "
    f"{ensemble_stats['llama_disagrees_with_both_other_models_rate']:.2f}%"
)
summary_lines.append("")
summary_lines.append("Verification accuracy by category:")
for category, st in verification_category_stats.items():
    summary_lines.append(
        f"{category}: Ensemble {st['ensemble_accuracy']:.2f}% | "
        + " | ".join(f"{label} {st[f'{label}_accuracy']:.2f}%" for label in verifier_labels)
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



# Display the final benchmark summary and output filenames
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