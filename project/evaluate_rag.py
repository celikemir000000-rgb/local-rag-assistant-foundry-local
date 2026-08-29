import csv
import gc
import json
import math
import re
import sqlite3
import time
from pathlib import Path

from foundry_local_sdk import (
    Configuration,
    FoundryLocalManager,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent

DATABASE_PATH = (
    ROOT_DIR
    / "documents"
    / "rag_database.db"
)

QUESTIONS_PATH = (
    PROJECT_DIR
    / "evaluation_questions.json"
)

RESULTS_PATH = (
    PROJECT_DIR
    / "evaluation_results.csv"
)

MODEL_CACHE_DIR = str(
    ROOT_DIR
    / "model_cache"
)


# ============================================================
# SETTINGS
# ============================================================

EMBEDDING_MODEL = "qwen3-embedding-0.6b"
ANSWER_MODEL = "qwen3-4b"

TOP_K = 3
MIN_RETRIEVAL_SCORE = 0.35
MODEL_SWITCH_WAIT = 5

MAX_ANSWER_SENTENCES = 3

FALLBACK_MESSAGE = (
    "The information is not available in the provided sources."
)


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(vector_1, vector_2):

    dot_product = sum(
        a * b
        for a, b in zip(
            vector_1,
            vector_2,
        )
    )

    magnitude_1 = math.sqrt(
        sum(
            a * a
            for a in vector_1
        )
    )

    magnitude_2 = math.sqrt(
        sum(
            b * b
            for b in vector_2
        )
    )

    if magnitude_1 == 0 or magnitude_2 == 0:
        return 0.0

    return (
        dot_product
        / (magnitude_1 * magnitude_2)
    )


# ============================================================
# REMOVE THINK TAGS
# ============================================================

def remove_thinking_text(text):

    cleaned_text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    cleaned_text = re.sub(
        r"</?think>",
        "",
        cleaned_text,
        flags=re.IGNORECASE,
    )

    return cleaned_text.strip()


# ============================================================
# NORMALIZE FINAL ANSWER
# ============================================================

def normalize_answer(text):

    cleaned_text = remove_thinking_text(
        text
    )

    # --------------------------------------------------------
    # FIX MALFORMED SOURCE LABELS
    #
    # Example:
    # [Source 1]] -> [Source 1]
    # --------------------------------------------------------

    cleaned_text = re.sub(
        r"(\[Source\s+\d+\])\]+",
        r"\1",
        cleaned_text,
        flags=re.IGNORECASE,
    )

    cleaned_text = " ".join(
        cleaned_text.split()
    )


    # --------------------------------------------------------
    # NORMALIZE FALLBACK
    #
    # If the model writes:
    #
    # The information is not available in the provided
    # sources. [Source 3]
    #
    # return only the official fallback message.
    # --------------------------------------------------------

    fallback_pattern = (
        r"^"
        + re.escape(FALLBACK_MESSAGE)
        + r"\s*"
        + r"(?:\[Source\s+\d+\]\s*)*"
        + r"$"
    )

    if re.fullmatch(
        fallback_pattern,
        cleaned_text,
        flags=re.IGNORECASE,
    ):

        return FALLBACK_MESSAGE


    # --------------------------------------------------------
    # LIMIT ANSWER LENGTH
    # --------------------------------------------------------

    sentences = re.split(
        r"(?<=[.!?])\s+",
        cleaned_text,
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    if (
        len(sentences)
        > MAX_ANSWER_SENTENCES
    ):

        cleaned_text = " ".join(
            sentences[
                :MAX_ANSWER_SENTENCES
            ]
        )

    return cleaned_text.strip()


# ============================================================
# EXTRACT SOURCE NUMBERS
# ============================================================

def extract_source_numbers(answer):

    matches = re.findall(
        r"\[Source\s+(\d+)\]",
        answer,
        flags=re.IGNORECASE,
    )

    source_numbers = []

    for match in matches:

        number = int(
            match
        )

        if number not in source_numbers:

            source_numbers.append(
                number
            )

    return source_numbers


# ============================================================
# LOAD DATABASE CHUNKS
# ============================================================

def load_chunks():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            documents.file_name,
            chunks.page,
            chunks.chunk_number,
            chunks.text,
            chunks.embedding

        FROM chunks

        JOIN documents
        ON chunks.document_id = documents.id
        """
    )

    rows = cursor.fetchall()

    connection.close()

    chunks = []

    for row in rows:

        chunks.append(
            {
                "file_name": row[0],
                "page": row[1],
                "chunk_number": row[2],
                "text": row[3],
                "embedding": json.loads(
                    row[4]
                ),
            }
        )

    return chunks


# ============================================================
# LOAD EVALUATION QUESTIONS
# ============================================================

def load_questions():

    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    query_embedding,
    chunks,
):

    results = []

    for chunk in chunks:

        score = cosine_similarity(
            query_embedding,
            chunk["embedding"],
        )

        results.append(
            {
                **chunk,
                "score": score,
            }
        )

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results[:TOP_K]


# ============================================================
# BUILD SOURCE CONTEXT
# ============================================================

def build_source_context(
    retrieved_chunks,
):

    return "\n\n".join(
        (
            f"[Source {index}]\n"
            f"File: {chunk['file_name']}\n"
            f"Page: {chunk['page']}\n"
            f"Text: {chunk['text']}"
        )

        for index, chunk in enumerate(
            retrieved_chunks,
            start=1,
        )
    )


# ============================================================
# GENERATE ONE ANSWER
# ============================================================

def generate_answer(
    chat_client,
    question,
    retrieved_chunks,
):

    source_context = (
        build_source_context(
            retrieved_chunks
        )
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a local RAG question-answering assistant. "

                "Answer using only the provided sources. "
                "Do not use outside knowledge. "
                "Do not invent information. "

                "If the sources contain enough information, "
                "answer directly using that information. "

                "Every factual answer must cite only the "
                "source that actually supports it using "
                "the exact format [Source N]. "

                "Do not cite irrelevant sources. "

                "If none of the sources contain enough "
                "information, respond exactly with: "
                f"'{FALLBACK_MESSAGE}' "

                "Do not add a citation to the fallback message. "

                "Return only the final answer. "
                "Do not show reasoning, analysis, planning, "
                "or intermediate steps. "

                "Use no more than three short sentences. "
                "Do not repeat information. "
                "Answer in English."
            ),
        },

        {
            "role": "user",
            "content": (
                f"SOURCES:\n"
                f"{source_context}\n\n"

                f"QUESTION:\n"
                f"{question}\n\n"

                "Answer directly. "
                "Use at most three short sentences. "
                "Do not repeat yourself. "
                "Cite supporting information with [Source N]. "
                "/no_think"
            ),
        },
    ]


    answer_parts = []

    for stream_chunk in (
        chat_client
        .complete_streaming_chat(
            messages
        )
    ):

        if not stream_chunk.choices:
            continue

        content = (
            stream_chunk
            .choices[0]
            .delta
            .content
        )

        if content:

            answer_parts.append(
                content
            )


    raw_answer = "".join(
        answer_parts
    )

    if not raw_answer.strip():

        raise RuntimeError(
            "The answer model returned an empty response."
        )

    return normalize_answer(
        raw_answer
    )


# ============================================================
# FIND CITED DOCUMENTS
# ============================================================

def get_cited_documents(
    answer,
    retrieved_chunks,
):

    source_numbers = (
        extract_source_numbers(
            answer
        )
    )

    cited_documents = []

    for source_number in source_numbers:

        index = (
            source_number - 1
        )

        if (
            index < 0
            or index >= len(retrieved_chunks)
        ):

            continue

        file_name = (
            retrieved_chunks[index]
            ["file_name"]
        )

        if file_name not in cited_documents:

            cited_documents.append(
                file_name
            )

    return cited_documents


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    print()
    print(
        "END-TO-END RAG EVALUATION"
    )

    print(
        "========================="
    )


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    chunks = load_chunks()

    questions = load_questions()

    print()
    print(
        f"Database chunks: {len(chunks)}"
    )

    print(
        f"Evaluation questions: {len(questions)}"
    )


    # --------------------------------------------------------
    # INITIALIZE FOUNDRY LOCAL
    # --------------------------------------------------------

    configuration = Configuration(
        app_name="EndToEndRAGEvaluation",
        model_cache_dir=MODEL_CACHE_DIR,
    )

    FoundryLocalManager.initialize(
        configuration
    )

    manager = (
        FoundryLocalManager.instance
    )


    # ========================================================
    # STEP 1: RETRIEVE FOR ALL QUESTIONS
    # ========================================================

    embedding_model = (
        manager.catalog.get_model(
            EMBEDDING_MODEL
        )
    )

    print()
    print(
        "Loading embedding model..."
    )

    embedding_model.load()

    embedding_client = (
        embedding_model
        .get_embedding_client()
    )

    evaluation_items = []


    for item in questions:

        question = (
            item["question"]
        )

        print(
            f"Retrieving "
            f"[{item['id']}/{len(questions)}]..."
        )

        retrieval_query = (
            "Instruct: Retrieve passages from the "
            "document collection that directly answer "
            "the following question.\n"
            f"Query: {question}"
        )

        result = (
            embedding_client
            .generate_embedding(
                retrieval_query
            )
        )

        query_embedding = (
            result.data[0].embedding
        )

        retrieved_chunks = retrieve(
            query_embedding,
            chunks,
        )

        evaluation_items.append(
            {
                **item,
                "retrieved_chunks":
                    retrieved_chunks,
            }
        )


    print()
    print(
        "Unloading embedding model..."
    )

    embedding_model.unload()

    gc.collect()

    time.sleep(
        MODEL_SWITCH_WAIT
    )


    # ========================================================
    # STEP 2: GENERATE ANSWERS
    # ========================================================

    answer_model = (
        manager.catalog.get_model(
            ANSWER_MODEL
        )
    )

    print()
    print(
        f"Loading answer model: "
        f"{ANSWER_MODEL}"
    )

    answer_model.load()

    chat_client = (
        answer_model
        .get_chat_client()
    )

    chat_client.settings.temperature = 0.0

    chat_client.settings.max_tokens = 120


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    answerable_total = 0
    answerable_pass = 0

    fallback_total = 0
    fallback_pass = 0

    results_for_csv = []


    # ========================================================
    # TEST QUESTIONS
    # ========================================================

    for item in evaluation_items:

        question_id = (
            item["id"]
        )

        question = (
            item["question"]
        )

        expected_document = (
            item["expected_document"]
        )

        answerable = (
            item["answerable"]
        )

        retrieved_chunks = (
            item["retrieved_chunks"]
        )


        print()
        print(
            f"[{question_id}] "
            f"{question}"
        )

        print(
            "Generating answer..."
        )


        # ----------------------------------------------------
        # GENERATE
        # ----------------------------------------------------

        if (
            not retrieved_chunks
            or retrieved_chunks[0]["score"] < MIN_RETRIEVAL_SCORE
        ):
            answer = FALLBACK_MESSAGE
        else:
            answer = generate_answer(
                chat_client,
                question,
                retrieved_chunks,
            )


        # ----------------------------------------------------
        # CITATIONS
        # ----------------------------------------------------

        cited_documents = (
            get_cited_documents(
                answer,
                retrieved_chunks,
            )
        )

        retrieved_documents = [
            chunk["file_name"]
            for chunk in retrieved_chunks
        ]


        # ----------------------------------------------------
        # ANSWERABLE QUESTION
        # ----------------------------------------------------

        if answerable:

            answerable_total += 1

            retrieval_pass = (
                expected_document
                in retrieved_documents
            )

            answer_not_fallback = (
                answer.strip()
                != FALLBACK_MESSAGE
            )

            citation_pass = (
                expected_document
                in cited_documents
            )

            final_pass = (
                retrieval_pass
                and answer_not_fallback
                and citation_pass
            )

            if final_pass:

                answerable_pass += 1


        # ----------------------------------------------------
        # UNANSWERABLE QUESTION
        # ----------------------------------------------------

        else:

            fallback_total += 1

            retrieval_pass = True

            answer_not_fallback = False

            citation_pass = (
                len(cited_documents)
                == 0
            )

            final_pass = (
                answer.strip()
                == FALLBACK_MESSAGE
                and citation_pass
            )

            if final_pass:

                fallback_pass += 1


        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        print(
            f"Answer: {answer}"
        )

        if cited_documents:

            print(
                "Cited documents: "
                + ", ".join(
                    cited_documents
                )
            )

        else:

            print(
                "Cited documents: None"
            )

        print(
            f"Result: "
            f"{'PASS' if final_pass else 'FAIL'}"
        )


        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        results_for_csv.append(
            {
                "id":
                    question_id,

                "question":
                    question,

                "answerable":
                    answerable,

                "expected_document":
                    expected_document or "",

                "top_1_document":
                    retrieved_chunks[0]
                    ["file_name"],

                "top_1_score":
                    (
                        f"{retrieved_chunks[0]['score']:.4f}"
                    ),

                "retrieved_documents":
                    " | ".join(
                        retrieved_documents
                    ),

                "answer":
                    answer,

                "cited_documents":
                    " | ".join(
                        cited_documents
                    ),

                "result":
                    (
                        "PASS"
                        if final_pass
                        else "FAIL"
                    ),
            }
        )


    # ========================================================
    # UNLOAD ANSWER MODEL
    # ========================================================

    print()
    print(
        "Unloading answer model..."
    )

    answer_model.unload()

    gc.collect()


    # ========================================================
    # SAVE RESULTS TO CSV
    # ========================================================

    with open(
        RESULTS_PATH,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        field_names = [
            "id",
            "question",
            "answerable",
            "expected_document",
            "top_1_document",
            "top_1_score",
            "retrieved_documents",
            "answer",
            "cited_documents",
            "result",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=field_names,
        )

        writer.writeheader()

        writer.writerows(
            results_for_csv
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print(
        "END-TO-END EVALUATION SUMMARY"
    )

    print(
        "============================="
    )


    # --------------------------------------------------------
    # ANSWERABLE RESULTS
    # --------------------------------------------------------

    print(
        f"Answerable questions passed: "
        f"{answerable_pass}/{answerable_total}"
    )

    if answerable_total:

        answerable_rate = (
            answerable_pass
            / answerable_total
            * 100
        )

        print(
            f"Answerable success rate: "
            f"{answerable_rate:.1f}%"
        )


    # --------------------------------------------------------
    # FALLBACK RESULTS
    # --------------------------------------------------------

    print()

    print(
        f"Fallback questions passed: "
        f"{fallback_pass}/{fallback_total}"
    )

    if fallback_total:

        fallback_rate = (
            fallback_pass
            / fallback_total
            * 100
        )

        print(
            f"Fallback success rate: "
            f"{fallback_rate:.1f}%"
        )


    # --------------------------------------------------------
    # OVERALL RESULTS
    # --------------------------------------------------------

    total_pass = (
        answerable_pass
        + fallback_pass
    )

    total_questions = (
        answerable_total
        + fallback_total
    )

    print()

    print(
        f"Overall passed: "
        f"{total_pass}/{total_questions}"
    )

    if total_questions:

        overall_rate = (
            total_pass
            / total_questions
            * 100
        )

        print(
            f"Overall success rate: "
            f"{overall_rate:.1f}%"
        )


    # --------------------------------------------------------
    # RESULTS FILE
    # --------------------------------------------------------

    print()
    print(
        "Detailed results saved to:"
    )

    print(
        RESULTS_PATH
    )


if __name__ == "__main__":
    main()