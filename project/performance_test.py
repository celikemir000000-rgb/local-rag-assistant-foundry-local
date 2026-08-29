import gc
import json
import math
import sqlite3
import time
from pathlib import Path

from foundry_local_sdk import (
    Configuration,
    FoundryLocalManager,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent

DATABASE_PATH = (
    ROOT_DIR
    / "documents"
    / "rag_database.db"
)

MODEL_CACHE_DIR = (
    ROOT_DIR
    / "model_cache"
)

EMBEDDING_MODEL = "qwen3-embedding-0.6b"
ANSWER_MODEL = "qwen3-4b"

TOP_K = 3
MIN_RETRIEVAL_SCORE = 0.35

TEST_QUESTIONS = [
    "What is chunk overlap?",
    "What is an embedding?",
]


# ============================================================
# LOAD DATABASE
# ============================================================

def load_chunks():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            d.file_name,
            c.page,
            c.chunk_number,
            c.text,
            c.embedding
        FROM chunks AS c
        JOIN documents AS d
            ON c.document_id = d.id
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
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    vector_a,
    vector_b,
):

    dot_product = sum(
        a * b
        for a, b in zip(
            vector_a,
            vector_b,
        )
    )

    magnitude_a = math.sqrt(
        sum(
            value * value
            for value in vector_a
        )
    )

    magnitude_b = math.sqrt(
        sum(
            value * value
            for value in vector_b
        )
    )

    if (
        magnitude_a == 0
        or magnitude_b == 0
    ):
        return 0.0

    return (
        dot_product
        / (magnitude_a * magnitude_b)
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_chunks(
    query_embedding,
    chunks,
):

    results = []

    for chunk in chunks:

        score = cosine_similarity(
            query_embedding,
            chunk["embedding"],
        )

        result = dict(chunk)
        result["score"] = score

        results.append(
            result
        )

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results[:TOP_K]


# ============================================================
# QUERY EMBEDDING
# ============================================================

def create_query_embedding(
    embedding_client,
    question,
):

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

    if not result.data:

        raise RuntimeError(
            "Query embedding could not be created."
        )

    return result.data[0].embedding


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    chat_client,
    question,
    retrieved_chunks,
):

    source_context = "\n\n".join(
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

    messages = [
        {
            "role": "system",
            "content": (
                "You are a local RAG question-answering assistant. "
                "Answer using only the provided sources. "
                "Do not use outside knowledge. "
                "Do not invent information. "
                "Return only the final answer. "
                "Do not show reasoning or intermediate steps. "
                "Use no more than three short sentences. "
                "Cite supporting information with [Source N]. "
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

    return "".join(
        answer_parts
    )


# ============================================================
# TEST ONE QUESTION
# ============================================================

def test_question(
    number,
    question,
    embedding_client,
    chat_client,
    chunks,
):

    total_start = time.perf_counter()

    print()
    print(
        f"QUESTION {number}"
    )
    print(
        "=" * 40
    )
    print(
        question
    )


    # Query embedding

    start = time.perf_counter()

    query_embedding = (
        create_query_embedding(
            embedding_client,
            question,
        )
    )

    embedding_time = (
        time.perf_counter()
        - start
    )


    # Retrieval

    start = time.perf_counter()

    retrieved_chunks = (
        retrieve_chunks(
            query_embedding,
            chunks,
        )
    )

    retrieval_time = (
        time.perf_counter()
        - start
    )


    # Threshold

    top_score = (
        retrieved_chunks[0]["score"]
        if retrieved_chunks
        else 0.0
    )

    answer_generation_time = 0.0

    if (
        not retrieved_chunks
        or top_score < MIN_RETRIEVAL_SCORE
    ):

        answer_type = "Fallback"

    else:

        start = time.perf_counter()

        generate_answer(
            chat_client,
            question,
            retrieved_chunks,
        )

        answer_generation_time = (
            time.perf_counter()
            - start
        )

        answer_type = "Generated"


    total_time = (
        time.perf_counter()
        - total_start
    )


    print()
    print(
        "PERFORMANCE"
    )
    print(
        "==========="
    )

    print(
        f"Query embedding:       "
        f"{embedding_time:.3f} s"
    )

    print(
        f"Retrieval search:      "
        f"{retrieval_time:.6f} s"
    )

    print(
        f"Answer generation:     "
        f"{answer_generation_time:.3f} s"
    )

    print(
        f"TOTAL QUESTION TIME:   "
        f"{total_time:.3f} s"
    )

    print(
        f"Answer type:           "
        f"{answer_type}"
    )

    if retrieved_chunks:

        print(
            f"Top source:            "
            f"{retrieved_chunks[0]['file_name']}"
        )

        print(
            f"Top score:             "
            f"{top_score:.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "LOCAL RAG PERFORMANCE TEST"
    )

    print(
        "=========================="
    )

    startup_start = time.perf_counter()


    # Database

    start = time.perf_counter()

    chunks = load_chunks()

    database_time = (
        time.perf_counter()
        - start
    )


    # Foundry initialization

    start = time.perf_counter()

    configuration = Configuration(
        app_name="LocalRAGPerformanceTest",
        model_cache_dir=str(
            MODEL_CACHE_DIR
        ),
    )

    FoundryLocalManager.initialize(
        configuration
    )

    manager = (
        FoundryLocalManager.instance
    )

    foundry_time = (
        time.perf_counter()
        - start
    )


    embedding_model = (
        manager.catalog.get_model(
            EMBEDDING_MODEL
        )
    )

    answer_model = (
        manager.catalog.get_model(
            ANSWER_MODEL
        )
    )

    embedding_loaded = False
    answer_loaded = False

    try:

        # Load embedding model once

        start = time.perf_counter()

        embedding_model.load()
        embedding_loaded = True

        embedding_client = (
            embedding_model
            .get_embedding_client()
        )

        embedding_load_time = (
            time.perf_counter()
            - start
        )


        # Load answer model once

        start = time.perf_counter()

        answer_model.load()
        answer_loaded = True

        chat_client = (
            answer_model
            .get_chat_client()
        )

        chat_client.settings.temperature = 0.0
        chat_client.settings.max_tokens = 120

        answer_load_time = (
            time.perf_counter()
            - start
        )


        startup_total = (
            time.perf_counter()
            - startup_start
        )


        print()
        print(
            "STARTUP PERFORMANCE"
        )

        print(
            "==================="
        )

        print(
            f"Database load:         "
            f"{database_time:.3f} s"
        )

        print(
            f"Foundry initialization:"
            f" {foundry_time:.3f} s"
        )

        print(
            f"Embedding model load:  "
            f"{embedding_load_time:.3f} s"
        )

        print(
            f"Answer model load:     "
            f"{answer_load_time:.3f} s"
        )

        print(
            f"TOTAL STARTUP TIME:    "
            f"{startup_total:.3f} s"
        )


        # Run questions using already-loaded models

        for number, question in enumerate(
            TEST_QUESTIONS,
            start=1,
        ):

            test_question(
                number,
                question,
                embedding_client,
                chat_client,
                chunks,
            )

    finally:

        print()
        print(
            "Cleaning up..."
        )

        if answer_loaded:
            answer_model.unload()

        if embedding_loaded:
            embedding_model.unload()

        gc.collect()

        print(
            "Models unloaded."
        )


if __name__ == "__main__":
    main()

