# Importing the libraries used for document processing, retrieval, evaluation and timing.
import os
import re
import json
import shutil
from collections import defaultdict
import math
import time
import statistics

from pypdf import PdfReader
import chromadb
import ollama

from rouge_score import rouge_scorer
import sacrebleu
from bert_score import score as bertscore



# Setting the document paths, models and output files used for the benchmark.
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

# Setting the chunk size, overlap and number of passages to retrieve.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 12

# Configuring the optional reward and verification checks used by supporting functions.
REWARD_THRESHOLD = 0.80
ADDITIONAL_CANDIDATES = 1
CANDIDATE_TEMPERATURE = 0.0
REWARD_WEIGHTS = {"factual": 0.30, "evidence": 0.25,
                  "completeness": 0.20, "question": 0.15,
                  "citation": 0.10}
# Model-generated verification scores are estimates, so minimum thresholds are used as safeguards.
MIN_FACTUAL = 0.70
MIN_EVIDENCE = 0.70
MIN_COMPLETENESS = 0.60
MIN_QUESTION = 0.70
MAX_VERIFIER_RETRIES = 2
MAX_PAIRWISE_RETRIES = 2
CHALLENGE_CONFIDENCE = 0.75



# Loading the PDF documents and extracting text with its original page number.
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



# Splitting document pages into overlapping chunks for retrieval.
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



# Rebuilding ChromaDB so earlier embeddings do not affect this benchmark run.
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



# Embedding each document chunk and storing its source information in ChromaDB.
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



# Reading the benchmark questions, categories and reference answers from the text file.
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



# Tracking model calls, embedding calls and Ollama durations during answer generation only.
_ACTIVE_COST = None

# Creating an empty tracker for the computational cost of each question.
def _new_cost_tracker():
    return {
        "llm_calls": 0,
        "embedding_calls": 0,
        "ollama_total_duration_seconds": 0.0,
        "ollama_load_duration_seconds": 0.0,
        "ollama_prompt_eval_duration_seconds": 0.0,
        "ollama_eval_duration_seconds": 0.0,
        "retrieval_latency_seconds": 0.0,
        "baseline_generation_latency_seconds": 0.0,
    }

def _duration_seconds(response, key):
    try:
        value = response.get(key, 0) if hasattr(response, "get") else 0
        return float(value or 0) / 1_000_000_000.0
    except (TypeError, ValueError):
        return 0.0

# Recording the duration reported by Ollama for each model request.
def _record_ollama_durations(response):
    global _ACTIVE_COST
    if _ACTIVE_COST is None:
        return
    _ACTIVE_COST["ollama_total_duration_seconds"] += _duration_seconds(response, "total_duration")
    _ACTIVE_COST["ollama_load_duration_seconds"] += _duration_seconds(response, "load_duration")
    _ACTIVE_COST["ollama_prompt_eval_duration_seconds"] += _duration_seconds(response, "prompt_eval_duration")
    _ACTIVE_COST["ollama_eval_duration_seconds"] += _duration_seconds(response, "eval_duration")

# Wrapping model calls so their counts and durations can be measured.
def tracked_chat(*args, **kwargs):
    global _ACTIVE_COST
    response = ollama.chat(*args, **kwargs)
    if _ACTIVE_COST is not None:
        _ACTIVE_COST["llm_calls"] += 1
        _record_ollama_durations(response)
    return response

# Tracking embedding requests separately from text generation.
def tracked_embedding(*args, **kwargs):
    global _ACTIVE_COST
    response = ollama.embeddings(*args, **kwargs)
    if _ACTIVE_COST is not None:
        _ACTIVE_COST["embedding_calls"] += 1
        _record_ollama_durations(response)
    return response


# Generating the baseline answer using retrieved governance document passages.
def generate_answer(question, return_retrieval=False):


    # Embedding the question before searching the vector database.
    retrieval_start = time.perf_counter()
    response = tracked_embedding(
        model=EMBEDDING_MODEL,
        prompt=question
    )

    question_embedding = response["embedding"]



    # Retrieving the most relevant document chunks for the question.
    results = collection.query(

        query_embeddings=[
            question_embedding
        ],

        n_results=RETRIEVAL_K
    )


    retrieved_docs = results["documents"][0]

    retrieved_metadata = results["metadatas"][0]

    if _ACTIVE_COST is not None:
        _ACTIVE_COST["retrieval_latency_seconds"] += time.perf_counter() - retrieval_start



    # Combining the retrieved passages with their source and page information.
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



    # Instructing the model to answer using only the supplied governance context.
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


    # Generating a deterministic baseline answer and measuring generation latency.
    generation_start = time.perf_counter()
    response = tracked_chat(

        model=GENERATION_MODEL,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        options={"temperature": 0.0}
    )


    answer = response["message"]["content"].strip()

    if _ACTIVE_COST is not None:
        _ACTIVE_COST["baseline_generation_latency_seconds"] += time.perf_counter() - generation_start


    # Returning the retrieved evidence as well when the verification pipeline needs it.
    if return_retrieval:
        evidence = [
            {"id": f"E{idx+1}", "text": doc,
             "source": meta["source"], "page": meta["page"]}
            for idx, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metadata))
        ]
        return answer, retrieved_metadata, evidence, prompt
    return answer, retrieved_metadata



# Evaluating answers against the benchmark reference using the same scoring rubric.
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



    # Extracting the correctness, completeness, faithfulness and overall score.
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


# Calculating ROUGE, BLEU and BERTScore for the generated answer.
def calculate_standard_metrics(reference_answer, generated_answer):


    # Measuring word overlap with ROUGE-1, ROUGE-2 and ROUGE-L.
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



    # Calculating sentence-level BLEU against the reference answer.
    bleu = sacrebleu.sentence_bleu(
        generated_answer,
        [reference_answer]
    )

    bleu_score = bleu.score / 100.0



    # Measuring semantic similarity using BERTScore.
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




# Calling the local generation model with a specified temperature.
def chat_text(prompt, temperature=0.0):
    response = tracked_chat(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature}
    )
    return response["message"]["content"].strip()


# Formatting the retrieved passages with evidence IDs for verification.
def evidence_context(evidence):
    return "\n\n".join(
        f'[{p["id"]}] SOURCE: {p["source"]} | PAGE: {p["page"]}\n{p["text"]}'
        for p in evidence
    )


# Parsing a JSON response even when the model includes code fences.
def parse_json_object(raw):
    raw = raw.strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw, flags=re.I).strip()
        decoder = json.JSONDecoder()
        start = raw.find('{')
        if start < 0:
            raise ValueError('Verifier returned no JSON object')
        obj, _ = decoder.raw_decode(raw[start:])
    if not isinstance(obj, dict):
        raise ValueError('Verifier response is not a JSON object')
    return obj


# Checking that each verifier score is a valid number between zero and one.
def bounded_score(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'Invalid {name}: expected number')
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'Invalid {name}: expected finite 0..1')
    return value


# Normalising whitespace and case when comparing supporting quotations.
def normalise_quote(value):
    return ' '.join(str(value).split()).casefold()


# Checking a candidate answer against retrieved passages without using the reference answer.
def verify_candidate(question, answer, evidence):
    '''Verify against retrieved passages only; never read reference answers.

    Factual, evidence and completeness ratings are model-assisted. Each
    claimed supporting quotation must ALSO match the cited retrieved chunk
    programmatically. Citation reward measures VERIFIED VERIFIER CITATIONS,
    not answer formatting, keeping the unchanged baseline prompt comparable.
    '''
    if not evidence or not answer or answer.startswith('ERROR:'):
        return {"reward": None, "passes": False, "error": "Missing answer/evidence",
                "components": None, "verified_quotes": [], "verifier": None}

    prompt = f'''You are a cautious, independent verifier of governance QA.
Use ONLY the supplied excerpts. Do NOT use outside knowledge. You do not
have a reference answer. Distinguish lack of evidence from proof of absence.
For ambiguous questions, a justified request for clarification may be valid.
IMPORTANT: If the question is genuinely underspecified (for example an unclear
pronoun, missing subject, missing committee/topic, or missing time/scope), and
the candidate correctly explains what must be clarified instead of guessing,
that IS a direct and complete response. Give high factual/completeness/question
scores when that clarification is justified by the question and evidence.
For unanswerable questions, abstention is appropriate only if excerpts cannot
support a substantive answer; retrieval might be incomplete, so be cautious.
Do not assume a citation proves a claim: inspect its actual words.

CRITICAL CONSISTENCY RULES:
- Semantic equivalence counts as support; exact wording is NOT required.
- If an excerpt directly states the same proposition as the candidate, you MUST
  treat that proposition as supported. Do not call a claim unsupported while
  simultaneously returning a quotation that directly states that claim.
- A paraphrase of a retrieved rule is supported when its meaning is preserved.
- Missing evidence is not contradiction. Contradiction requires affirmative
  conflict with the retrieved text.
- Preserve qualifiers such as not, only, may, must, before, after, up to, and
  conditional requirements when judging factual consistency.

QUESTION:\n{question}\n
CANDIDATE ANSWER:\n{answer}\n
RETRIEVED EVIDENCE:\n{evidence_context(evidence)}\n
Assess the answer, including incorrect extra claims and missing key points.
Return ONE valid JSON object and no other text, with EXACT fields:
{{
 "factual": 0.0,
 "evidence": 0.0,
 "completeness": 0.0,
 "question": 0.0,
 "contradiction": false,
 "unsupported_material_claim": false,
 "insufficient_retrieval": false,
 "reason": "brief justification",
 "support": [{{"id":"E1", "quote":"EXACT contiguous quotation from that excerpt"}}]
}}
Scores are 0..1. Factual measures whether claims are correct GIVEN THE
EVIDENCE, not presumed true. Evidence measures whether the retrieved passages
semantically support the material claims; exact wording is not required.
Completeness measures whether the answer covers the material information that
the evidence can establish. Question measures whether the candidate directly
answers the SPECIFIC question asked, including its temporal framing, negation,
scope, actor, conditions, exceptions, and requested relationship. A candidate
that states true and supported facts but answers a different issue MUST receive
a low question score. Include 1-5 shortest DIRECT
quotations supporting material claims, using valid excerpt IDs. Quotes should
be verbatim where possible, but quotation formatting is an audit aid and MUST
NOT determine semantic evidence support. If the answer is a justified
abstention or clarification, support may be empty; explain why. Set
insufficient_retrieval=true when excerpts cannot establish the requested fact.
Never invent supporting quotations. Set contradiction=true if a material claim
contradicts excerpts; unsupported_material_claim=true for material claims not
supported by excerpts. Use conservative scores when evidence is insufficient.'''

    # Retrying the verifier if its response is invalid or cannot be parsed.
    last_error = None
    for attempt in range(MAX_VERIFIER_RETRIES + 1):
        try:
            raw = chat_text(prompt, temperature=0.0)
            obj = parse_json_object(raw)
            factual = bounded_score(obj['factual'], 'factual')
            ev_model = bounded_score(obj['evidence'], 'evidence')
            completeness = bounded_score(obj['completeness'], 'completeness')
            question_score = bounded_score(obj['question'], 'question')
            for key in ('contradiction', 'unsupported_material_claim',
                        'insufficient_retrieval'):
                if not isinstance(obj.get(key), bool):
                    raise ValueError(f'Invalid {key}')
            supports = obj.get('support')
            if not isinstance(supports, list):
                raise ValueError('support must be a list')
            # Checking whether cited evidence IDs and quotations match retrieved passages.
            by_id = {p['id']: p for p in evidence}
            checked = []
            for item in supports:
                if not isinstance(item, dict):
                    raise ValueError('Invalid support item')
                sid, quote = item.get('id'), item.get('quote')
                valid_id = isinstance(sid, str) and sid in by_id
                valid_quote = (isinstance(quote, str) and len(quote.strip()) >= 8
                               and valid_id and normalise_quote(quote)
                               in normalise_quote(by_id[sid]['text']))
                checked.append({"id": sid, "quote": quote,
                                "valid_id": valid_id,
                                "valid_quote": bool(valid_quote),
                                "source": by_id[sid]['source'] if valid_id else None,
                                "page": by_id[sid]['page'] if valid_id else None})
            # Keeping semantic evidence scores separate from the formatting of quotations.
            valid_ids = sum(x['valid_id'] for x in checked)
            exact_quotes = sum(x['valid_quote'] for x in checked)

            # Allowing justified abstentions or clarification requests without a supporting quote.
            abstention_like = bool(re.search(
                r'cannot (?:be |determine|answer)|not enough|insufficient|'
                r'not (?:specified|provided|identified|stated)|'
                r'clarif|ambiguous|do not contain', answer, re.I))
            no_quote_exception = (not checked and abstention_like
                                  and not obj['contradiction']
                                  and not obj['unsupported_material_claim'])

            citation = (valid_ids / len(checked) if checked else
                        (1.0 if no_quote_exception else 0.0))

            evidence_score = ev_model

            # Combining verification scores and applying minimum thresholds to the candidate.
            components = {"factual": factual, "evidence": evidence_score,
                          "completeness": completeness, "question": question_score,
                          "citation": citation}
            reward = sum(REWARD_WEIGHTS[k] * components[k]
                         for k in REWARD_WEIGHTS)
            passes = (reward >= REWARD_THRESHOLD
                      and factual >= MIN_FACTUAL
                      and evidence_score >= MIN_EVIDENCE
                      and completeness >= MIN_COMPLETENESS
                      and question_score >= MIN_QUESTION
                      and not obj['contradiction']
                      and not obj['unsupported_material_claim'])
            return {"reward": round(reward, 5), "passes": passes,
                    "components": components, "verified_quotes": checked,
                    "verifier": obj, "no_quote_exception": no_quote_exception,
                    "exact_quote_match_rate": (exact_quotes / len(checked)
                                               if checked else None),
                    "verification_kind": "model_assisted_semantic_support_plus_valid_evidence_ids",
                    "error": None}
        except Exception as exc:
            last_error = f'{type(exc).__name__}: {exc}'
            if attempt < MAX_VERIFIER_RETRIES:
                time.sleep(0.2)
    return {"reward": None, "passes": False, "components": None,
            "verified_quotes": [], "verifier": None, "error": last_error}


# Using a second model check to look for material problems in an answer.
def challenge_answer(question, answer, evidence):
    """Adversarial second opinion that searches for concrete verifier blind spots."""
    prompt = f'''You are an adversarial governance QA auditor. Use ONLY the
retrieved excerpts. You do NOT have a reference answer. Try to FALSIFY the
candidate rather than rescore its writing quality.

Identify exactly what the QUESTION asks: governance object (Constitution,
Bylaw, Operational Manual, Society, IEEE, Board, committee), actor, temporal
status (proposed/adopted/approved/effective), negation, scope, and any missing
referent. Test whether the candidate answers THAT exact proposition. A rule
about a related governance object is not evidence for the requested object.
Do not infer an effective date from amendment, approval, distribution,
objection, or adoption unless the excerpt explicitly connects them. General
rights do not automatically imply a specific right such as voting. If the
question is genuinely underspecified, guessing a specific entity/outcome is a
material issue; requesting clarification is not.

QUESTION:\n{question}\n
CANDIDATE:\n{answer}\n
EVIDENCE:\n{evidence_context(evidence)}\n
Return ONE JSON object only:
{{
 "material_issue": false,
 "issue_type": "none",
 "confidence": 0.0,
 "question_answerable": true,
 "clarification_required": false,
 "issue": "brief concrete issue or none",
 "evidence_ids": ["E1"]
}}
material_issue=true only for a concrete error/omission that could make the
answer materially wrong or incomplete. confidence is 0..1.'''
    last_error = None
    for attempt in range(MAX_VERIFIER_RETRIES + 1):
        try:
            obj = parse_json_object(chat_text(prompt, temperature=0.0))
            for key in ('material_issue', 'question_answerable', 'clarification_required'):
                if not isinstance(obj.get(key), bool):
                    raise ValueError(f'{key} must be bool')
            conf = bounded_score(obj.get('confidence'), 'confidence')
            ids = obj.get('evidence_ids', [])
            if not isinstance(ids, list):
                raise ValueError('evidence_ids must be list')
            valid = {e['id'] for e in evidence}
            obj['evidence_ids'] = [x for x in ids if x in valid]
            obj['confidence'] = conf
            flagged = bool(obj['material_issue'] and conf >= CHALLENGE_CONFIDENCE)
            return {'flagged': flagged, 'audit': obj, 'error': None}
        except Exception as exc:
            last_error = f'{type(exc).__name__}: {exc}'
            if attempt < MAX_VERIFIER_RETRIES:
                time.sleep(0.2)
    return {'flagged': False, 'audit': None, 'error': last_error}


# Generating an alternative answer informed by any concrete audit concern.
def generate_alternative(question, evidence, candidate_number, audit=None):
    '''Independent sample targeted at any concrete audit concern.'''
    audit_text = ''
    if audit and audit.get('audit'):
        a = audit['audit']
        audit_text = f'''\nADVERSARIAL AUDIT OF INITIAL ANSWER:\nIssue type: {a.get("issue_type")}\nIssue: {a.get("issue")}\nClarification required: {a.get("clarification_required")}\nUse this only as a warning to re-check the excerpts; do not blindly accept it.\n'''
    prompt = f'''You answer questions about IEEE SMCS governance documents.
Use ONLY the excerpts. No external facts, no invented details. Combine passages
when necessary. Distinguish governance objects precisely: IEEE vs SMCS,
Constitution vs Bylaws vs Operational Manual, Board vs committee, and proposed,
adopted, approved and effective rules. Preserve negation, actors, scope,
conditions and exceptions. Do NOT infer one rule from a related rule unless the
excerpts explicitly connect them. If the question lacks an essential referent,
subject, time, committee/topic or other context, request the specific
clarification needed instead of guessing. If the excerpts do not establish the
answer, say that explicitly. Be concise and answer the exact question.

RETRIEVED CONTEXT:\n{evidence_context(evidence)}\n{audit_text}
QUESTION:\n{question}\n
ANSWER:'''
    return chat_text(prompt, temperature=CANDIDATE_TEMPERATURE)

# Comparing an alternative with the initial answer using retrieved evidence.
def pairwise_improvement_check(question, initial_answer, candidate_answer, evidence):
    """Conservative deployment-time comparison; never sees benchmark reference."""
    prompt = f'''You are a conservative governance QA replacement judge.
Use ONLY the retrieved excerpts. You do NOT have a reference answer.

Your task is NOT to choose the answer you like better. Decide whether the
PROPOSED REPLACEMENT is DEMONSTRABLY BETTER than the INITIAL ANSWER for the
exact question. Preserve a correct initial answer unless the replacement fixes
a specific material error, omission, ambiguity, or failure to answer.

Pay special attention to temporal status, negation, scope, actors, conditions,
exceptions, and effective-vs-proposed rules. A replacement that merely states
true supported facts about a related procedure is NOT better if it fails to
answer the question actually asked. Do not prefer an answer merely because it
is shorter, longer, more fluent, or has citations. If the initial answer guesses
a specific outcome for a genuinely underspecified question, and the replacement
correctly requests the missing clarification, that IS a material fix and may be
judged better. Likewise, if the initial answer applies a rule for the wrong
governance object or temporal state and the replacement correctly says the
requested fact is not established by the excerpts, that may be a material fix.

A replacement may be accepted only when ALL are true:
1. You can identify a concrete material weakness in the initial answer.
2. The replacement fixes that weakness.
3. The replacement directly answers the exact question at least as well.
4. The replacement introduces no new material error, contradiction, or
   unsupported claim.
5. The retrieved evidence is sufficient to justify the comparison.

QUESTION:
{question}

INITIAL ANSWER:
{initial_answer}

PROPOSED REPLACEMENT:
{candidate_answer}

RETRIEVED EVIDENCE:
{evidence_context(evidence)}

Return ONE valid JSON object and no other text:
{{
  "better": false,
  "initial_errors": ["specific material weakness, if any"],
  "candidate_fixes": ["specific fix, if any"],
  "candidate_new_errors": ["new material problem, if any"],
  "question_alignment_preserved": true,
  "evidence_sufficient": true,
  "reason": "brief comparison grounded in the excerpts"
}}

Use better=true ONLY when improvement is clear and demonstrable. If uncertain,
if both answers are comparably acceptable, or if the initial answer is already
adequate, return better=false.'''
    last_error = None
    for attempt in range(MAX_PAIRWISE_RETRIES + 1):
        try:
            raw = chat_text(prompt, temperature=0.0)
            obj = parse_json_object(raw)
            for key in ('better', 'question_alignment_preserved', 'evidence_sufficient'):
                if not isinstance(obj.get(key), bool):
                    raise ValueError(f'Invalid {key}')
            for key in ('initial_errors', 'candidate_fixes', 'candidate_new_errors'):
                if not isinstance(obj.get(key), list):
                    raise ValueError(f'{key} must be a list')
            reason = obj.get('reason')
            if not isinstance(reason, str):
                raise ValueError('reason must be a string')
            approved = (obj['better']
                        and obj['question_alignment_preserved']
                        and obj['evidence_sufficient']
                        and len(obj['initial_errors']) > 0
                        and len(obj['candidate_fixes']) > 0
                        and len(obj['candidate_new_errors']) == 0)
            return {"approved": approved, "comparison": obj, "error": None}
        except Exception as exc:
            last_error = f'{type(exc).__name__}: {exc}'
            if attempt < MAX_PAIRWISE_RETRIES:
                time.sleep(0.2)
    return {"approved": False, "comparison": None, "error": last_error}



# Splitting the answer into claims and checking each claim for direct document support.
def claim_level_verify(question, answer, evidence):
    """Verify atomic answer claims against retrieved evidence using discrete labels."""
    prompt = f'''You are a STRICT ENTAILMENT verifier for IEEE SMCS governance QA.
Use ONLY the retrieved excerpts. You do NOT have a benchmark/reference answer.

Your task is NOT to decide whether an answer sounds plausible, is related to the evidence, or can be inferred from governance knowledge. Your task is to decide whether the retrieved excerpts DIRECTLY ESTABLISH each exact material proposition in the candidate answer.

STEP 1 — QUESTION TARGET
Identify exactly what the question asks. Preserve every material distinction, including:
- actor / office / body / membership class
- governance object or document (e.g. Constitution vs Bylaws vs Operational Manual)
- action or relationship (e.g. approve vs sign vs recommend vs appoint)
- negation
- number, duration, frequency, threshold or deadline
- procedural stage (proposed vs adopted vs approved vs effective)
- conditions, exceptions and scope
- unresolved pronouns or missing context

STEP 2 — ATOMIC CLAIMS
Split the candidate answer into atomic MATERIAL claims. Do not combine independent propositions into one claim. Ignore stylistic filler and source labels.

STEP 3 — STRICT ENTAILMENT VERDICT
Assign exactly one verdict to every claim:

SUPPORTED
The retrieved excerpts directly establish the SAME material proposition. A faithful paraphrase is allowed, but all material entities, relationships, quantities, conditions and temporal states must match.

CONTRADICTED
The retrieved excerpts directly establish a materially incompatible proposition.

NOT_FOUND
The excerpts do not directly establish the exact proposition and do not directly contradict it. Use NOT_FOUND whenever support would require analogy, assumption, extrapolation, role substitution, temporal inference, or combining a related rule into a stronger claim.

NON-NEGOTIABLE RULES:
1. Related evidence is NOT sufficient. Semantic similarity is NOT entailment.
2. Never upgrade words such as "may", "can", "normally", "subject to", or "shall not conflict with" into a stronger proposition.
3. Never infer precedence merely because one document must not conflict with another. "A shall not conflict with B" does NOT by itself mean "A takes precedence over B".
4. Never equate a duration with a number of terms. "No more than four continuous years" does NOT establish "no more than two terms" unless term length and the resulting term count are explicitly established by the retrieved evidence.
5. General rights or privileges do NOT establish a specific right. Evidence that Society Affiliates have certain rights does NOT establish that they may vote unless voting is explicitly established.
6. Distinguish APPROVING an expenditure from SIGNING/AUTHORIZING a disbursement, cheque, payment, or transaction. One does NOT establish the other.
7. Distinguish governance objects. A rule for a Constitution or Constitutional Amendment does NOT establish the rule for a Bylaw, Operational Manual, policy, or other document.
8. Distinguish procedural stages. Proposed != adopted != approved != published/distributed != effective. Do not infer an effective date from an adoption, approval, distribution, or amendment rule unless the evidence explicitly connects them.
9. Distinguish bodies and roles. Board of Directors != Board of Governors; IEEE != SMCS; Society Member != Society Affiliate; President != President-Elect; committee != Technical Committee unless the evidence explicitly makes the connection.
10. Distinguish obligation levels: may != shall/must; recommendation != approval; consultation != authorization; nomination != appointment.
11. For yes/no questions, verify the proposition behind yes/no.
12. If the candidate says information is absent, that claim is CONTRADICTED only when the retrieved excerpts explicitly contain the requested information.
13. If the question is genuinely underspecified, a justified clarification request may satisfy the target. Do not invent the missing referent from retrieved passages.
14. When uncertain between SUPPORTED and NOT_FOUND, choose NOT_FOUND. SUPPORTED requires direct textual entailment.
15. Evidence IDs for SUPPORTED or CONTRADICTED claims must point to excerpts that directly establish the verdict. Do not cite merely related excerpts.

Before returning SUPPORTED, silently perform this substitution test:
"Could I quote/paraphrase the cited excerpt to assert this exact claim without adding a new assumption?"
If NO, verdict = NOT_FOUND.

QUESTION:
{question}

CANDIDATE ANSWER:
{answer}

EVIDENCE:
{evidence_context(evidence)}

Return ONE JSON object only:
{{
  "question_target": "the exact proposition being asked",
  "question_answerable": true,
  "clarification_required": false,
  "claims": [{{"claim": "one atomic material claim", "verdict": "SUPPORTED", "evidence_ids": ["E1"], "reason": "state exactly how the cited evidence entails, contradicts, or fails to establish this claim"}}],
  "answers_target": true,
  "target_reason": "explain whether the candidate answers the exact requested proposition without substituting a related entity, rule, quantity, action, or temporal state"
}}'''
    # Parsing the claim verdicts and counting supported, contradicted and missing claims.
    last_error = None
    for attempt in range(MAX_VERIFIER_RETRIES + 1):
        try:
            obj = parse_json_object(chat_text(prompt, temperature=0.0))
            if not isinstance(obj.get('question_target'), str): raise ValueError('question_target must be string')
            for k in ('question_answerable','clarification_required','answers_target'):
                if not isinstance(obj.get(k), bool): raise ValueError(f'{k} must be bool')
            claims = obj.get('claims')
            if not isinstance(claims, list) or not claims: raise ValueError('claims must be a non-empty list')
            valid_ids = {e['id'] for e in evidence}
            cleaned=[]
            for cl in claims:
                if not isinstance(cl, dict) or not isinstance(cl.get('claim'), str): raise ValueError('invalid claim')
                verdict=str(cl.get('verdict','')).upper()
                if verdict not in {'SUPPORTED','CONTRADICTED','NOT_FOUND'}: raise ValueError('invalid claim verdict')
                ids=cl.get('evidence_ids',[])
                if not isinstance(ids,list): ids=[]
                cleaned.append({'claim':cl['claim'],'verdict':verdict,'evidence_ids':[x for x in ids if x in valid_ids],'reason':str(cl.get('reason',''))})
            obj['claims']=cleaned
            counts={v:sum(1 for c in cleaned if c['verdict']==v) for v in ('SUPPORTED','CONTRADICTED','NOT_FOUND')}
            total=len(cleaned)
            support_ratio=counts['SUPPORTED']/total
            has_failure=(counts['CONTRADICTED']>0 or counts['NOT_FOUND']>0 or not obj['answers_target'])
            passes=(not has_failure and (obj['answers_target'] or obj['clarification_required']))
            return {'passes':passes,'has_failure':has_failure,'support_ratio':round(support_ratio,4),'counts':counts,'analysis':obj,'error':None}
        except Exception as exc:
            last_error=f'{type(exc).__name__}: {exc}'
            if attempt < MAX_VERIFIER_RETRIES: time.sleep(0.2)
    return {'passes':False,'has_failure':True,'support_ratio':0.0,'counts':None,'analysis':None,'error':last_error}


# Generating one targeted correction for claims that failed verification.
def generate_claim_correction(question, initial_answer, evidence, claim_report, candidate_number):
    """Generate a correction targeted specifically at failed atomic claims."""
    a=(claim_report or {}).get('analysis') or {}
    failed=[cl for cl in a.get('claims',[]) if cl.get('verdict') in ('CONTRADICTED','NOT_FOUND')]
    failed_text=json.dumps(failed, ensure_ascii=False, indent=2)
    target=a.get('question_target','')
    prompt=f'''You are correcting an IEEE SMCS governance QA answer.
Use ONLY the retrieved excerpts. Do not use a benchmark/reference answer.
The initial answer has been decomposed into atomic claims by a strict-entailment verifier. Fix ONLY the identified failed claims while producing a coherent final answer. Do not preserve a claim merely because it appeared in the initial answer. Use only propositions directly established by the retrieved excerpts. Do not substitute related governance objects, bodies, roles, actions, quantities, rights, or procedural stages. In particular: Constitution != Bylaw; approval != signature/disbursement authorization; years != number of terms; general rights != voting rights; proposed != adopted != approved != effective; and 'shall not conflict with' does not by itself establish precedence. Answer the exact question directly. Preserve negation, actor, governance object, time/effective status, quantities, conditions and exceptions. If the evidence cannot directly establish the requested fact, explicitly say that it is not established by the retrieved excerpts. If essential context is genuinely missing, request clarification.

QUESTION TARGET:
{target}

QUESTION:
{question}

INITIAL ANSWER:
{initial_answer}

FAILED CLAIMS:
{failed_text}

EVIDENCE:
{evidence_context(evidence)}

Write only the corrected answer.'''
    return chat_text(prompt, temperature=CANDIDATE_TEMPERATURE)


# Comparing the initial and corrected claim-verification outcomes.
def claim_pairwise_decision(initial_report, candidate_report):
    """Deterministic comparison of discrete claim-verification outcomes."""
    if not candidate_report or not candidate_report.get('passes'): return False, 'candidate_claim_verification_failed'
    if not initial_report or not initial_report.get('has_failure'): return False, 'initial_has_no_verified_claim_failure'
    ic=initial_report.get('counts') or {}; cc=candidate_report.get('counts') or {}
    initial_bad=ic.get('CONTRADICTED',0)+ic.get('NOT_FOUND',0)
    candidate_bad=cc.get('CONTRADICTED',0)+cc.get('NOT_FOUND',0)
    if candidate_bad != 0: return False, 'candidate_contains_failed_claim'
    if initial_bad <= 0 and (initial_report.get('analysis') or {}).get('answers_target',True): return False, 'no_discrete_improvement'
    return True, 'candidate_resolves_verified_claim_failure'


# Running the RLVR-inspired pipeline with one possible targeted correction.
def reward_guided_answer(question):
    """
    Simplified V3 inference:
      1. Generate one deterministic baseline answer.
      2. Run one claim-level verifier against the fixed retrieved evidence.
      3. If no material failure is detected, keep the baseline.
      4. If a material failure is detected, generate ONE targeted correction.
      5. Re-run the SAME claim verifier on the correction.
      6. Replace the baseline only if the correction fully passes.

    No scalar reward, requirement extractor, challenge critic, or pairwise LLM judge
    participates in selection. The benchmark reference answer remains isolated until
    the later evaluation stage.
    """
    # Generating the baseline answer and keeping the retrieved evidence for verification.
    initial, sources, evidence, baseline_prompt = generate_answer(
        question, return_retrieval=True
    )

    # Verifying the baseline answer at claim level before deciding whether to revise it.
    initial_claims = claim_level_verify(question, initial, evidence)

    candidates = [{
        "index": 0,
        "type": "deterministic_baseline",
        "answer": initial,
        "verification": {
            "reward": None,
            "passes": initial_claims.get("passes", False),
            "note": "Selection uses claim verification only; scalar reward disabled."
        },
        "claim_verification": initial_claims,
        "pairwise": None
    }]

    triggered = bool(initial_claims.get("has_failure", True))
    selected = candidates[0]
    reason = "initial_passed_claim_verification"

    # Attempting a correction only when the baseline has a claim-verification failure.
    if triggered:
        try:
            correction = generate_claim_correction(
                question, initial, evidence, initial_claims, 1
            )
            correction_claims = claim_level_verify(
                question, correction, evidence
            )

            # Accepting a correction only when it fully passes the same claim verifier.
            approved = bool(correction_claims.get("passes", False))

            candidate = {
                "index": 1,
                "type": "single_targeted_correction",
                "answer": correction,
                "verification": {
                    "reward": None,
                    "passes": approved,
                    "note": "Selection uses claim verification only; scalar reward disabled."
                },
                "claim_verification": correction_claims,
                "pairwise": {
                    "approved": approved,
                    "reason": (
                        "correction_fully_passed_same_claim_verifier"
                        if approved else
                        "correction_still_has_verified_failure"
                    )
                }
            }
            candidates.append(candidate)

            if approved:
                selected = candidate
                reason = "single_targeted_correction_verified"
            else:
                reason = "claim_failure_detected_but_single_correction_not_verified"

        except Exception as exc:
            candidates.append({
                "index": 1,
                "type": "single_targeted_correction",
                "answer": None,
                "verification": {
                    "reward": None,
                    "passes": False,
                    "error": f"{type(exc).__name__}: {exc}"
                },
                "claim_verification": {
                    "passes": False,
                    "has_failure": True,
                    "error": f"{type(exc).__name__}: {exc}"
                },
                "pairwise": {
                    "approved": False,
                    "reason": "generation_or_verification_error"
                }
            })
            reason = "claim_failure_detected_but_correction_errored"

    return {
        "initial_answer": initial,
        "final_answer": selected["answer"],
        "sources": sources,
        "evidence": evidence,
        "candidates": candidates,
        "selected_candidate": selected["index"],
        "triggered": triggered,
        "changed": selected["index"] != 0,
        "additional_generations": len(candidates) - 1,
        "initial_claim_verification": initial_claims,
        "selection_reason": reason
    }

# Evaluating the final answer against the reference after candidate selection.
def safe_benchmark_evaluation(question, reference, answer):
    try:
        return evaluate_answer(question, reference, answer)
    except Exception as exc:
        return {"correctness": None, "completeness": None, "faithfulness": None,
                "score": None, "reason": f'Evaluation error: {exc}'}


# Running the benchmark for each question and recording the results.
print("\n============================================")
print("STARTING BENCHMARK")
print("============================================")

results = []


for i, item in enumerate(questions):


    print(
        f"Processing {i + 1}/{len(questions)} "
        f"({item['id']})"
    )


    # Starting a fresh cost tracker for each question.
    _ACTIVE_COST = _new_cost_tracker()
    pipeline_start = time.perf_counter()
    try:
        run = reward_guided_answer(item["question"])
        generated_answer = run["final_answer"]
        sources = run["sources"]
    except Exception as exc:
        print(f"ERROR processing {item['id']}: {exc}")
        generated_answer = "ERROR: Could not generate answer."
        sources = []
        run = {"initial_answer": generated_answer,
               "final_answer": generated_answer, "candidates": [],
               "selected_candidate": 0, "triggered": False,
               "changed": False, "additional_generations": 0,
               "selection_reason": f"Generation error: {exc}"}

    pipeline_latency = time.perf_counter() - pipeline_start
    computational_cost = dict(_ACTIVE_COST)
    computational_cost["end_to_end_latency_seconds"] = pipeline_latency
    computational_cost["rlvr_overhead_latency_seconds"] = max(
        0.0,
        pipeline_latency
        - computational_cost["retrieval_latency_seconds"]
        - computational_cost["baseline_generation_latency_seconds"]
    )
    _ACTIVE_COST = None

    # Keeping benchmark evaluation separate from the answer selection process.
    initial_evaluation = safe_benchmark_evaluation(
        item["question"], item["reference_answer"], run["initial_answer"])
    if run["changed"]:
        evaluation = safe_benchmark_evaluation(
            item["question"], item["reference_answer"], generated_answer)
    else:
        evaluation = initial_evaluation


    try:

        # Calculating text similarity metrics for the selected answer.
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


    # Also calculating the initial-answer metrics when a correction was selected.
    initial_standard_metrics = standard_metrics
    if run["changed"]:
        try:
            initial_standard_metrics = calculate_standard_metrics(
                item["reference_answer"], run["initial_answer"])
        except Exception as exc:
            print(f"ERROR initial metrics {item['id']}: {exc}")
            initial_standard_metrics = {k: None for k in standard_metrics}


    # Storing the final answer, evaluation, verification details and computational cost.
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

        "computational_cost": computational_cost,

        "initial_answer": run["initial_answer"],
        "final_answer": generated_answer,
        "initial_evaluation": initial_evaluation,
        "initial_standard_metrics": initial_standard_metrics,
        "reward_guidance": {
            "threshold": REWARD_THRESHOLD,
            "weights": REWARD_WEIGHTS,
            "triggered": run["triggered"],
            "changed": run["changed"],
            "selected_candidate": run["selected_candidate"],
            "additional_generations": run["additional_generations"],
            "selection_reason": run["selection_reason"],
            "candidates": run["candidates"]
        }
    }


    results.append(result)



    # Saving results after each question so completed work is retained if execution stops.
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



# Calculating overall accuracy using questions with valid evaluation scores.
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



# Collecting accuracy and evaluation metrics for each question category.
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



# Converting category totals into average scores and percentages.
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


# Summarising how often claim verification triggered a correction and changed an answer.
initial_valid = [r for r in results if r["initial_evaluation"]["score"] is not None]
initial_correct = sum(r["initial_evaluation"]["score"] == 3 for r in initial_valid)
paired = [r for r in results if r["initial_evaluation"]["score"] is not None
          and r["evaluation"]["score"] is not None]
corrections = sum(r["initial_evaluation"]["score"] != 3
                  and r["evaluation"]["score"] == 3 for r in paired)
degradations = sum(r["initial_evaluation"]["score"] == 3
                   and r["evaluation"]["score"] != 3 for r in paired)
reward_values = [c["verification"].get("reward") for r in results
                 for c in r["reward_guidance"]["candidates"]
                 if c["verification"].get("reward") is not None]
reward_diagnostics = {
    "method": "RLVR-inspired simplified strict-entailment verification + single targeted correction; no training",
    "initial_accuracy": round(100*initial_correct/len(initial_valid), 2) if initial_valid else 0,
    "final_accuracy": round(overall_accuracy, 2),
    "paired_questions": len(paired),
    "corrections": corrections,
    "degradations": degradations,
    "correction_rate_percent": round(100*corrections/len(paired), 2) if paired else 0,
    "degradation_rate_percent": round(100*degradations/len(paired), 2) if paired else 0,
    "triggered_questions": sum(r["reward_guidance"]["triggered"] for r in results),
    "changed_answers": sum(r["reward_guidance"]["changed"] for r in results),
    "total_additional_generations": sum(r["reward_guidance"]["additional_generations"] for r in results),
    "mean_additional_generations": round(sum(r["reward_guidance"]["additional_generations"]
                                         for r in results)/len(results), 3) if results else 0,
    "mean_candidate_reward": round(sum(reward_values)/len(reward_values), 4)
                             if reward_values else None,
    "category_breakdown": {}
}
for cat in category_stats:
    group = [r for r in results if r["category"] == cat]
    before = [r for r in group if r["initial_evaluation"]["score"] is not None]
    reward_diagnostics["category_breakdown"][cat] = {
        "initial_accuracy": round(100*sum(r["initial_evaluation"]["score"] == 3
                                          for r in before)/len(before), 2) if before else 0,
        "final_accuracy": round(category_stats[cat]["accuracy"], 2),
        "changed": sum(r["reward_guidance"]["changed"] for r in group)
    }


# Summarising latency, model calls and Ollama durations across the benchmark.
def _mean_cost(key, rows=results):
    values = [r["computational_cost"].get(key, 0.0) for r in rows]
    return sum(values) / len(values) if values else 0.0

latencies = [r["computational_cost"]["end_to_end_latency_seconds"] for r in results]
computational_cost_summary = {
    "mean_end_to_end_latency_seconds": round(_mean_cost("end_to_end_latency_seconds"), 4),
    "median_end_to_end_latency_seconds": round(statistics.median(latencies), 4) if latencies else 0.0,
    "std_end_to_end_latency_seconds": round(statistics.pstdev(latencies), 4) if len(latencies) > 1 else 0.0,
    "mean_retrieval_latency_seconds": round(_mean_cost("retrieval_latency_seconds"), 4),
    "mean_baseline_generation_latency_seconds": round(_mean_cost("baseline_generation_latency_seconds"), 4),
    "mean_rlvr_overhead_latency_seconds": round(_mean_cost("rlvr_overhead_latency_seconds"), 4),
    "mean_ollama_total_duration_seconds": round(_mean_cost("ollama_total_duration_seconds"), 4),
    "mean_ollama_load_duration_seconds": round(_mean_cost("ollama_load_duration_seconds"), 4),
    "mean_ollama_prompt_eval_duration_seconds": round(_mean_cost("ollama_prompt_eval_duration_seconds"), 4),
    "mean_ollama_eval_duration_seconds": round(_mean_cost("ollama_eval_duration_seconds"), 4),
    "mean_llm_calls_per_question": round(_mean_cost("llm_calls"), 3),
    "total_llm_calls": sum(r["computational_cost"]["llm_calls"] for r in results),
    "mean_embedding_calls_per_question": round(_mean_cost("embedding_calls"), 3),
    "total_embedding_calls": sum(r["computational_cost"]["embedding_calls"] for r in results),
    "category_breakdown": {}
}
for cat in category_stats:
    group = [r for r in results if r["category"] == cat]
    glat = [r["computational_cost"]["end_to_end_latency_seconds"] for r in group]
    computational_cost_summary["category_breakdown"][cat] = {
        "mean_end_to_end_latency_seconds": round(_mean_cost("end_to_end_latency_seconds", group), 4),
        "median_end_to_end_latency_seconds": round(statistics.median(glat), 4) if glat else 0.0,
        "mean_retrieval_latency_seconds": round(_mean_cost("retrieval_latency_seconds", group), 4),
        "mean_baseline_generation_latency_seconds": round(_mean_cost("baseline_generation_latency_seconds", group), 4),
        "mean_rlvr_overhead_latency_seconds": round(_mean_cost("rlvr_overhead_latency_seconds", group), 4),
        "mean_llm_calls_per_question": round(_mean_cost("llm_calls", group), 3),
        "mean_embedding_calls_per_question": round(_mean_cost("embedding_calls", group), 3)
    }


# Preparing the overall and category results for the JSON summary.
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


    "reward_guidance": reward_diagnostics,

    "computational_cost": computational_cost_summary,

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



# Saving the machine-readable summary as JSON.
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



# Building a text summary of accuracy, metrics and computational cost.
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
summary_lines.append("REWARD-GUIDED INFERENCE")
summary_lines.append(f"Initial accuracy: {reward_diagnostics['initial_accuracy']:.2f}%")
summary_lines.append(f"Final accuracy: {reward_diagnostics['final_accuracy']:.2f}%")
summary_lines.append(f"Corrections: {corrections}; degradations: {degradations}")
summary_lines.append(f"Triggered: {reward_diagnostics['triggered_questions']}; "
                     f"changed: {reward_diagnostics['changed_answers']}")
summary_lines.append(f"Additional generations: {reward_diagnostics['total_additional_generations']}")
summary_lines.append("")

summary_lines.append("COMPUTATIONAL COST")
summary_lines.append(f"Mean end-to-end latency: {computational_cost_summary['mean_end_to_end_latency_seconds']:.4f} s")
summary_lines.append(f"Median end-to-end latency: {computational_cost_summary['median_end_to_end_latency_seconds']:.4f} s")
summary_lines.append(f"Latency standard deviation: {computational_cost_summary['std_end_to_end_latency_seconds']:.4f} s")
summary_lines.append(f"Mean retrieval latency: {computational_cost_summary['mean_retrieval_latency_seconds']:.4f} s")
summary_lines.append(f"Mean baseline generation latency: {computational_cost_summary['mean_baseline_generation_latency_seconds']:.4f} s")
summary_lines.append(f"Mean RLVR overhead latency: {computational_cost_summary['mean_rlvr_overhead_latency_seconds']:.4f} s")
summary_lines.append(f"Mean Ollama total duration: {computational_cost_summary['mean_ollama_total_duration_seconds']:.4f} s")
summary_lines.append(f"Mean Ollama load duration: {computational_cost_summary['mean_ollama_load_duration_seconds']:.4f} s")
summary_lines.append(f"Mean Ollama prompt-eval duration: {computational_cost_summary['mean_ollama_prompt_eval_duration_seconds']:.4f} s")
summary_lines.append(f"Mean Ollama eval duration: {computational_cost_summary['mean_ollama_eval_duration_seconds']:.4f} s")
summary_lines.append(f"Mean LLM calls/question: {computational_cost_summary['mean_llm_calls_per_question']:.3f}")
summary_lines.append(f"Total LLM calls: {computational_cost_summary['total_llm_calls']}")
summary_lines.append(f"Mean embedding calls/question: {computational_cost_summary['mean_embedding_calls_per_question']:.3f}")
summary_lines.append(f"Total embedding calls: {computational_cost_summary['total_embedding_calls']}")
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


with open(
    SUMMARY_TEXT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(summary_lines)
    )



# Printing the final benchmark results and output filenames.
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