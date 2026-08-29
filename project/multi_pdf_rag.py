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
# NORMALIZE ANSWER
# ============================================================

def normalize_answer(text):

    cleaned_text = remove_thinking_text(
        text
    )

    # Fix malformed citations:
    # [Source 1]] -> [Source 1]

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

        source_number = int(
            match
        )

        if source_number not in source_numbers:

            source_numbers.append(
                source_number
            )

    return source_numbers


# ============================================================
# LOAD CHUNKS FROM SQLITE
# ============================================================

def load_chunks():

    if not DATABASE_PATH.exists():

        raise FileNotFoundError(
            f"Database not found: {DATABASE_PATH}"
        )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            chunks.id,
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
                "id": row[0],
                "file_name": row[1],
                "page": row[2],
                "chunk_number": row[3],
                "text": row[4],
                "embedding": json.loads(
                    row[5]
                ),
            }
        )

    return chunks


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
# CREATE QUERY EMBEDDING
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

    print()
    print(
        "Creating query embedding..."
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

    return (
        result.data[0].embedding
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    chat_client,
    question,
    retrieved_chunks,
):

    chat_client.settings.temperature = 0.0
    chat_client.settings.max_tokens = 120


    # ----------------------------------------------------
    # BUILD SOURCE CONTEXT
    # ----------------------------------------------------

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


    # ----------------------------------------------------
    # PROMPT
    # ----------------------------------------------------

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


    # ----------------------------------------------------
    # STREAM RESPONSE
    # ----------------------------------------------------

    print()
    print(
        "Generating source-grounded answer..."
    )

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
# DISPLAY SUPPORTING SOURCES
# ============================================================

def display_supporting_sources(
    answer,
    retrieved_chunks,
):

    print()
    print(
        "SOURCES"
    )

    print(
        "======="
    )


    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if answer.strip() == FALLBACK_MESSAGE:

        print(
            "No supporting source found."
        )

        return


    # --------------------------------------------------------
    # FIND CITED SOURCES
    # --------------------------------------------------------

    source_numbers = (
        extract_source_numbers(
            answer
        )
    )

    if not source_numbers:

        print(
            "No explicit source citation was returned by the model."
        )

        return


    # --------------------------------------------------------
    # DISPLAY CITED SOURCES
    # --------------------------------------------------------

    seen_sources = set()

    for source_number in source_numbers:

        index = (
            source_number - 1
        )

        if (
            index < 0
            or index >= len(retrieved_chunks)
        ):

            continue

        chunk = (
            retrieved_chunks[index]
        )

        source_key = (
            chunk["file_name"],
            chunk["page"],
        )

        if source_key in seen_sources:

            continue

        seen_sources.add(
            source_key
        )

        print(
            f"- {chunk['file_name']}, "
            f"page {chunk['page']}"
        )


# ============================================================
# ANSWER ONE QUESTION
# ============================================================

def answer_question(
    embedding_client,
    chat_client,
    question,
    chunks,
):

    # --------------------------------------------------------
    # QUERY EMBEDDING
    # --------------------------------------------------------

    query_embedding = (
        create_query_embedding(
            embedding_client,
            question,
        )
    )


    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    print()
    print(
        "Searching the document collection..."
    )

    retrieved_chunks = (
        retrieve_chunks(
            query_embedding,
            chunks,
        )
    )


    # --------------------------------------------------------
    # DISPLAY RETRIEVAL RESULTS
    # --------------------------------------------------------

    print()
    print(
        "RETRIEVED SOURCES"
    )

    print(
        "================="
    )

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):

        print(
            f"{index}. "
            f"{chunk['file_name']} | "
            f"Page {chunk['page']} | "
            f"Chunk {chunk['chunk_number']} | "
            f"Score {chunk['score']:.4f}"
        )


    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    if (
        not retrieved_chunks
        or retrieved_chunks[0]["score"] < MIN_RETRIEVAL_SCORE
    ):

        print()
        print(
            "Retrieved sources are not relevant enough. "
            "Using fallback response."
        )

        answer = FALLBACK_MESSAGE

    else:

        answer = generate_answer(
            chat_client,
            question,
            retrieved_chunks,
        )


    # --------------------------------------------------------
    # DISPLAY ANSWER
    # --------------------------------------------------------

    print()
    print(
        "RAG ANSWER"
    )

    print(
        "=========="
    )

    print(
        answer
    )


    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    display_supporting_sources(
        answer,
        retrieved_chunks,
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    print()
    print(
        "LOCAL RAG ASSISTANT"
    )

    print(
        "==================="
    )


    # --------------------------------------------------------
    # LOAD DATABASE ONCE
    # --------------------------------------------------------

    print()
    print(
        "Loading local knowledge base..."
    )

    chunks = load_chunks()

    print(
        "Documents are ready."
    )

    print(
        f"Total chunks: {len(chunks)}"
    )


    # --------------------------------------------------------
    # INITIALIZE FOUNDRY LOCAL ONCE
    # --------------------------------------------------------

    configuration = Configuration(
        app_name="LocalRAGAssistant",
        model_cache_dir=MODEL_CACHE_DIR,
    )

    FoundryLocalManager.initialize(
        configuration
    )

    manager = (
        FoundryLocalManager.instance
    )


    # --------------------------------------------------------
    # LOAD MODELS ONCE
    # --------------------------------------------------------

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

        print()
        print(
            "Loading embedding model..."
        )

        embedding_model.load()
        embedding_loaded = True

        embedding_client = (
            embedding_model
            .get_embedding_client()
        )


        print(
            "Embedding model ready."
        )


        print()
        print(
            f"Loading answer model: "
            f"{ANSWER_MODEL}"
        )

        answer_model.load()
        answer_loaded = True

        chat_client = (
            answer_model
            .get_chat_client()
        )

        chat_client.settings.temperature = 0.0
        chat_client.settings.max_tokens = 120

        print(
            "Answer model ready."
        )


        print()
        print(
            "Both models will remain loaded "
            "while the application is running."
        )


        # ----------------------------------------------------
        # HELP MESSAGE
        # ----------------------------------------------------

        print()
        print(
            "Ask a question about the local documents."
        )

        print(
            "Type 'exit' or 'quit' to close the application."
        )


        # ====================================================
        # QUESTION LOOP
        # ====================================================

        while True:

            print()

            question = input(
                "Question: "
            ).strip()


            if not question:

                print(
                    "Please enter a question."
                )

                continue


            if question.lower() in {
                "exit",
                "quit",
            }:

                print()
                print(
                    "Goodbye."
                )

                break


            try:

                answer_question(
                    embedding_client,
                    chat_client,
                    question,
                    chunks,
                )

            except Exception as error:

                print()
                print(
                    "An error occurred:"
                )

                print(
                    error
                )

    finally:

        print()
        print(
            "Unloading models..."
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

