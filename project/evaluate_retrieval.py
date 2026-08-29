import gc
import json
import math
import sqlite3
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

MODEL_CACHE_DIR = str(
    ROOT_DIR
    / "model_cache"
)


# ============================================================
# SETTINGS
# ============================================================

EMBEDDING_MODEL = "qwen3-embedding-0.6b"

TOP_K = 3


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
# RETRIEVE TOP RESULTS
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
# MAIN EVALUATION
# ============================================================

def main():

    print()
    print(
        "RETRIEVAL EVALUATION"
    )

    print(
        "===================="
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
        app_name="RetrievalEvaluation",
        model_cache_dir=MODEL_CACHE_DIR,
    )

    FoundryLocalManager.initialize(
        configuration
    )

    manager = (
        FoundryLocalManager.instance
    )

    embedding_model = (
        manager.catalog.get_model(
            EMBEDDING_MODEL
        )
    )

    model_loaded = False


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    answerable_count = 0

    top_1_correct = 0
    top_3_correct = 0

    unanswerable_scores = []


    try:

        print()
        print(
            "Loading embedding model..."
        )

        embedding_model.load()

        model_loaded = True

        embedding_client = (
            embedding_model
            .get_embedding_client()
        )


        # ====================================================
        # EVALUATE QUESTIONS
        # ====================================================

        for item in questions:

            question_id = item["id"]
            question = item["question"]

            expected_document = (
                item["expected_document"]
            )

            answerable = (
                item["answerable"]
            )


            # ------------------------------------------------
            # CREATE QUERY EMBEDDING
            # ------------------------------------------------

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


            # ------------------------------------------------
            # RETRIEVE
            # ------------------------------------------------

            retrieved = retrieve(
                query_embedding,
                chunks,
            )

            top_1_document = (
                retrieved[0]["file_name"]
            )

            top_1_score = (
                retrieved[0]["score"]
            )

            retrieved_documents = [
                chunk["file_name"]
                for chunk in retrieved
            ]


            # ------------------------------------------------
            # ANSWERABLE QUESTION
            # ------------------------------------------------

            if answerable:

                answerable_count += 1

                top_1_pass = (
                    top_1_document
                    == expected_document
                )

                top_3_pass = (
                    expected_document
                    in retrieved_documents
                )

                if top_1_pass:
                    top_1_correct += 1

                if top_3_pass:
                    top_3_correct += 1

                print()
                print(
                    f"[{question_id}] {question}"
                )

                print(
                    f"Expected: {expected_document}"
                )

                print(
                    f"Top-1:    {top_1_document} "
                    f"({top_1_score:.4f})"
                )

                print(
                    "Top-3:"
                )

                for rank, chunk in enumerate(
                    retrieved,
                    start=1,
                ):

                    print(
                        f"    {rank}. "
                        f"{chunk['file_name']} | "
                        f"Chunk {chunk['chunk_number']} | "
                        f"{chunk['score']:.4f}"
                    )

                print(
                    f"Top-1 result: "
                    f"{'PASS' if top_1_pass else 'FAIL'}"
                )

                print(
                    f"Top-3 result: "
                    f"{'PASS' if top_3_pass else 'FAIL'}"
                )


            # ------------------------------------------------
            # UNANSWERABLE QUESTION
            # ------------------------------------------------

            else:

                unanswerable_scores.append(
                    top_1_score
                )

                print()
                print(
                    f"[{question_id}] {question}"
                )

                print(
                    "Expected: No supporting document"
                )

                print(
                    f"Highest retrieved result: "
                    f"{top_1_document} "
                    f"({top_1_score:.4f})"
                )

                print(
                    "Type: UNANSWERABLE TEST"
                )


    finally:

        if model_loaded:

            print()
            print(
                "Unloading embedding model..."
            )

            embedding_model.unload()

        gc.collect()


    # ========================================================
    # FINAL METRICS
    # ========================================================

    print()
    print(
        "EVALUATION SUMMARY"
    )

    print(
        "=================="
    )

    if answerable_count > 0:

        top_1_accuracy = (
            top_1_correct
            / answerable_count
            * 100
        )

        top_3_accuracy = (
            top_3_correct
            / answerable_count
            * 100
        )

        print(
            f"Answerable questions: "
            f"{answerable_count}"
        )

        print(
            f"Top-1 correct: "
            f"{top_1_correct}/{answerable_count}"
        )

        print(
            f"Top-1 accuracy: "
            f"{top_1_accuracy:.1f}%"
        )

        print(
            f"Top-3 correct: "
            f"{top_3_correct}/{answerable_count}"
        )

        print(
            f"Top-3 accuracy: "
            f"{top_3_accuracy:.1f}%"
        )


    if unanswerable_scores:

        average_unanswerable_score = (
            sum(unanswerable_scores)
            / len(unanswerable_scores)
        )

        highest_unanswerable_score = max(
            unanswerable_scores
        )

        print()
        print(
            f"Unanswerable questions: "
            f"{len(unanswerable_scores)}"
        )

        print(
            f"Average highest similarity score: "
            f"{average_unanswerable_score:.4f}"
        )

        print(
            f"Highest unanswerable similarity score: "
            f"{highest_unanswerable_score:.4f}"
        )


if __name__ == "__main__":
    main()