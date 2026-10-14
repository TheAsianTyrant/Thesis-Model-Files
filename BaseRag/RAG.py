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

# Similarity and semantic metric libraries
from rouge_score import rouge_scorer
import sacrebleu
from bert_score import score as bertscore

# Naming all the files/giving names of files needed

DOCUMENT_FOLDER = "docs" # Folder with the governance corpus
QUESTIONS_FILE = "Questions.txt" # Text file containing the 110 benchmark questions and the associated reference answers

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "rag_collection"

EMBEDDING_MODEL = "nomic-embed-text" # Model for embedding
GENERATION_MODEL = "llama3" # Ollama model used
EVALUATION_MODEL = "llama3" # Ollama model used

# Files to save results in
OUTPUT_FILE = "evaluation_results.json"
SUMMARY_JSON_FILE = "evaluation_summary.json"
SUMMARY_TEXT_FILE = "evaluation_summary.txt"

# Retrieval settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 12


# Getting and loading all the documents for the governance corpus
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


# Splitting PDF page text into overlapping chunks while retaining source and page information

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


# Rebuilding ChromaDB to prevent previous benchmark embeddings from affecting retrieval

print("\n============================================")
print("REBUILDING VECTOR DATABASE")
print("============================================")


# Removing the existing database ensures embeddings from earlier runs are not reused.

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


# Embedding each document chunk and storing it in the vector database

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


# Reading the question IDs, categories, prompts and reference answers from the benchmark file

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


# Helper functions for extracting Ollama timing information and calculating statistics

def get_ollama_durations(response):
    """Extract timing metadata returned by Ollama chat calls.

    Ollama durations are reported in nanoseconds. Missing fields are handled
    safely so the benchmark still runs with older Ollama client versions.
    """
    eval_duration_ns = int(response.get("eval_duration", 0) or 0)
    prompt_eval_duration_ns = int(response.get("prompt_eval_duration", 0) or 0)
    load_duration_ns = int(response.get("load_duration", 0) or 0)
    total_duration_ns = int(response.get("total_duration", 0) or 0)

    return {
        "prompt_eval_duration_seconds": prompt_eval_duration_ns / 1_000_000_000,
        "generation_duration_seconds": eval_duration_ns / 1_000_000_000,
        "load_duration_seconds": load_duration_ns / 1_000_000_000,
        "ollama_total_duration_seconds": total_duration_ns / 1_000_000_000
    }


def safe_mean(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else 0


def safe_median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else 0


def safe_stdev(values):
    values = [value for value in values if value is not None]
    return statistics.stdev(values) if len(values) > 1 else 0


# Retrieving relevant governance passages and generating a document-grounded answer

def generate_answer(question):

    total_start = time.perf_counter()
    retrieval_start = total_start

# Convert the question into an embedding for similarity search
    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=question
    )

    question_embedding = response["embedding"]


# Retrieve the most relevant document chunks from ChromaDB
    results = collection.query(

        query_embeddings=[
            question_embedding
        ],

        n_results=RETRIEVAL_K
    )


    retrieved_docs = results["documents"][0]

    retrieved_metadata = results["metadatas"][0]

    retrieval_latency = time.perf_counter() - retrieval_start


# Combine retrieved passages with their document and page references
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


# Constructing a prompt that restricts the answer to the retrieved documents

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


    generation_latency = time.perf_counter() - generation_start
    total_latency = time.perf_counter() - total_start

    answer = response["message"]["content"].strip()

    durations = get_ollama_durations(response)

    computational_cost = {
        "total_latency_seconds": total_latency,
        "retrieval_latency_seconds": retrieval_latency,
        "generation_latency_seconds": generation_latency,
        "llm_calls": 1,
        "embedding_calls": 1,
        **durations
    }

    return answer, retrieved_metadata, computational_cost


# Evaluating generated answers against the reference answers using the LLM rubric

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


# Extract the individual rubric scores and explanation from the evaluator response
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

# Calculating ROUGE-1, ROUGE-2 and ROUGE-L F1 scores

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


# Calculating the sentence-level BLEU score

    bleu = sacrebleu.sentence_bleu(
        generated_answer,
        [reference_answer]
    )

    bleu_score = bleu.score / 100.0


# Calculating contextual semantic similarity with BERTScore

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


# Processing each benchmark question and recording its results

print("\n============================================")
print("STARTING BENCHMARK")
print("============================================")

results = []


for i, item in enumerate(questions):


    print(
        f"Processing {i + 1}/{len(questions)} "
        f"({item['id']})"
    )


# Generate an answer and record the retrieval and generation costs
    try:

        generated_answer, sources, computational_cost = generate_answer(
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

        computational_cost = {
            "total_latency_seconds": None,
            "retrieval_latency_seconds": None,
            "generation_latency_seconds": None,
            "llm_calls": 0,
            "embedding_calls": 0,
            "prompt_eval_duration_seconds": None,
            "generation_duration_seconds": None,
            "load_duration_seconds": None,
            "ollama_total_duration_seconds": None,
        }


# Assess the answer using the correctness, completeness and faithfulness rubric
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

# Calculate the reference-based text similarity scores
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


# Collect the answer, source references, evaluation and computational metrics
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

        "evaluation":
            evaluation,

        "standard_metrics": 
            standard_metrics,

        "computational_cost":
            computational_cost
    }


    results.append(result)


# Save after each question so completed results remain available if execution stops


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


# Calculating overall benchmark accuracy and computational statistics

total_questions = len(results)


# Count fully correct answers and calculate overall accuracy
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


# Summarise latency, model calls and Ollama durations across questions
latencies = [
    result["computational_cost"]["total_latency_seconds"]
    for result in results
    if result["computational_cost"]["total_latency_seconds"] is not None
]

retrieval_latencies = [
    result["computational_cost"]["retrieval_latency_seconds"]
    for result in results
    if result["computational_cost"]["retrieval_latency_seconds"] is not None
]

generation_latencies = [
    result["computational_cost"]["generation_latency_seconds"]
    for result in results
    if result["computational_cost"]["generation_latency_seconds"] is not None
]

llm_calls = [result["computational_cost"]["llm_calls"] for result in results]
embedding_calls = [result["computational_cost"]["embedding_calls"] for result in results]
prompt_eval_durations = [result["computational_cost"]["prompt_eval_duration_seconds"] for result in results]
generation_durations = [result["computational_cost"]["generation_duration_seconds"] for result in results]
load_durations = [result["computational_cost"]["load_duration_seconds"] for result in results]
ollama_total_durations = [result["computational_cost"]["ollama_total_duration_seconds"] for result in results]

computational_summary = {
    "average_latency_seconds": safe_mean(latencies),
    "median_latency_seconds": safe_median(latencies),
    "latency_std_seconds": safe_stdev(latencies),
    "average_retrieval_latency_seconds": safe_mean(retrieval_latencies),
    "average_generation_latency_seconds": safe_mean(generation_latencies),
    "average_llm_calls_per_question": safe_mean(llm_calls),
    "total_llm_calls": sum(llm_calls),
    "average_embedding_calls_per_question": safe_mean(embedding_calls),
    "total_embedding_calls": sum(embedding_calls),
    "average_prompt_eval_duration_seconds": safe_mean(prompt_eval_durations),
    "average_ollama_generation_duration_seconds": safe_mean(generation_durations),
    "average_load_duration_seconds": safe_mean(load_durations),
    "average_ollama_total_duration_seconds": safe_mean(ollama_total_durations)
}


# Aggregating accuracy, evaluation metrics and computational costs by question category

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

        "average_latency_seconds": 0,
        "average_retrieval_latency_seconds": 0,
        "average_generation_latency_seconds": 0,
        "average_llm_calls": 0,
        "average_embedding_calls": 0,
        "average_prompt_eval_duration_seconds": 0,
        "average_ollama_generation_duration_seconds": 0,
        "average_load_duration_seconds": 0,
        "average_ollama_total_duration_seconds": 0
    }
)


for result in results:

    category = result["category"]

    evaluation = result["evaluation"]


    category_stats[category]["total"] += 1


# Count fully correct answers separately from other evaluation scores
    if evaluation["score"] == 3:

        category_stats[category]["correct"] += 1

    else:

        category_stats[category]["incorrect"] += 1


# Accumulate the evaluator rubric scores for each category
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

# Accumulate ROUGE, BLEU and BERTScore values for each category
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


# Accumulate timing and call counts for each category
    cost = result["computational_cost"]

    if cost["total_latency_seconds"] is not None:
        category_stats[category]["average_latency_seconds"] += cost["total_latency_seconds"]
    if cost["retrieval_latency_seconds"] is not None:
        category_stats[category]["average_retrieval_latency_seconds"] += cost["retrieval_latency_seconds"]
    if cost["generation_latency_seconds"] is not None:
        category_stats[category]["average_generation_latency_seconds"] += cost["generation_latency_seconds"]

    category_stats[category]["average_llm_calls"] += cost["llm_calls"]
    category_stats[category]["average_embedding_calls"] += cost["embedding_calls"]
    category_stats[category]["average_prompt_eval_duration_seconds"] += cost["prompt_eval_duration_seconds"] or 0
    category_stats[category]["average_ollama_generation_duration_seconds"] += cost["generation_duration_seconds"] or 0
    category_stats[category]["average_load_duration_seconds"] += cost["load_duration_seconds"] or 0
    category_stats[category]["average_ollama_total_duration_seconds"] += cost["ollama_total_duration_seconds"] or 0


# Convert category totals into averages and percentages
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


    for cost_key in [
        "average_latency_seconds",
        "average_retrieval_latency_seconds",
        "average_generation_latency_seconds",
        "average_llm_calls",
        "average_embedding_calls",
        "average_prompt_eval_duration_seconds",
        "average_ollama_generation_duration_seconds",
        "average_load_duration_seconds",
        "average_ollama_total_duration_seconds"
    ]:
        stats[cost_key] = stats[cost_key] / total if total > 0 else 0


# Preparing the overall and category-level summary for JSON output

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
,

            "average_latency_seconds": round(stats["average_latency_seconds"], 4),
            "average_retrieval_latency_seconds": round(stats["average_retrieval_latency_seconds"], 4),
            "average_generation_latency_seconds": round(stats["average_generation_latency_seconds"], 4),
            "average_llm_calls": round(stats["average_llm_calls"], 4),
            "average_embedding_calls": round(stats["average_embedding_calls"], 4),
            "average_prompt_eval_duration_seconds": round(stats["average_prompt_eval_duration_seconds"], 4),
            "average_ollama_generation_duration_seconds": round(stats["average_ollama_generation_duration_seconds"], 4),
            "average_load_duration_seconds": round(stats["average_load_duration_seconds"], 4),
            "average_ollama_total_duration_seconds": round(stats["average_ollama_total_duration_seconds"], 4)
        }

        for category, stats
        in category_stats.items()
    }
}


# Saving the benchmark summary as JSON

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


# Preparing and saving the human-readable benchmark summary

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

summary_lines.append("COMPUTATIONAL COST (ANSWER PIPELINE ONLY)")
summary_lines.append("")
summary_lines.append(f"Average latency: {computational_summary['average_latency_seconds']:.4f} s/question")
summary_lines.append(f"Median latency: {computational_summary['median_latency_seconds']:.4f} s/question")
summary_lines.append(f"Latency standard deviation: {computational_summary['latency_std_seconds']:.4f} s")
summary_lines.append(f"Average retrieval latency: {computational_summary['average_retrieval_latency_seconds']:.4f} s/question")
summary_lines.append(f"Average generation latency: {computational_summary['average_generation_latency_seconds']:.4f} s/question")
summary_lines.append(f"Average LLM calls: {computational_summary['average_llm_calls_per_question']:.2f}/question")
summary_lines.append(f"Total LLM calls: {computational_summary['total_llm_calls']}")
summary_lines.append(f"Average embedding calls: {computational_summary['average_embedding_calls_per_question']:.2f}/question")
summary_lines.append(f"Total embedding calls: {computational_summary['total_embedding_calls']}")
summary_lines.append(f"Average Ollama prompt-evaluation duration: {computational_summary['average_prompt_eval_duration_seconds']:.4f} s/question")
summary_lines.append(f"Average Ollama generation duration: {computational_summary['average_ollama_generation_duration_seconds']:.4f} s/question")
summary_lines.append(f"Average Ollama model-load duration: {computational_summary['average_load_duration_seconds']:.4f} s/question")
summary_lines.append(f"Average Ollama total duration: {computational_summary['average_ollama_total_duration_seconds']:.4f} s/question")
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

    summary_lines.append(
        f"  Average Latency: {stats['average_latency_seconds']:.4f} s"
    )
    summary_lines.append(
        f"  Average LLM Calls: {stats['average_llm_calls']:.2f}"
    )
    summary_lines.append(f"  Average Retrieval Latency: {stats['average_retrieval_latency_seconds']:.4f} s")
    summary_lines.append(f"  Average Generation Latency: {stats['average_generation_latency_seconds']:.4f} s")
    summary_lines.append(f"  Average Embedding Calls: {stats['average_embedding_calls']:.2f}")
    summary_lines.append(f"  Average Ollama Prompt-Eval Duration: {stats['average_prompt_eval_duration_seconds']:.4f} s")
    summary_lines.append(f"  Average Ollama Generation Duration: {stats['average_ollama_generation_duration_seconds']:.4f} s")
    summary_lines.append(f"  Average Ollama Load Duration: {stats['average_load_duration_seconds']:.4f} s")
    summary_lines.append(f"  Average Ollama Total Duration: {stats['average_ollama_total_duration_seconds']:.4f} s")

    summary_lines.append("")


with open(
    SUMMARY_TEXT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(summary_lines)
    )


# Printing the final accuracy results and output file names

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