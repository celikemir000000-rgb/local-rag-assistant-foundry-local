import gc
import hashlib
import json
import sqlite3
from pathlib import Path

import pymupdf

from foundry_local_sdk import (
    Configuration,
    FoundryLocalManager,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

ROOT_DIR = PROJECT_DIR.parent

DOCUMENTS_DIR = (
    ROOT_DIR
    / "documents"
)

DATABASE_FILE = (
    DOCUMENTS_DIR
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

CHUNK_WORD_COUNT = 120

CHUNK_OVERLAP = 20

EMBEDDING_BATCH_SIZE = 2


# ============================================================
# CALCULATE PDF HASH
# ============================================================

def calculate_pdf_hash(file_path):

    sha256 = hashlib.sha256()

    with open(
        file_path,
        "rb",
    ) as file:

        while True:

            data = file.read(
                1024 * 1024
            )

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    return " ".join(
        text.split()
    )


# ============================================================
# SPLIT TEXT INTO CHUNKS
# ============================================================

def split_text_into_chunks(text):

    words = text.split()

    if not words:
        return []

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + CHUNK_WORD_COUNT,
            len(words),
        )

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():

            chunks.append(
                chunk.strip()
            )

        if end >= len(words):
            break

        start = (
            end
            - CHUNK_OVERLAP
        )

    return chunks


# ============================================================
# CREATE PDF CHUNKS
# ============================================================

def create_pdf_chunks(
    pdf_file,
):

    all_chunks = []

    with pymupdf.open(
        pdf_file
    ) as document:

        for page_index, page in enumerate(
            document
        ):

            page_number = (
                page_index + 1
            )

            text = page.get_text()

            text = clean_text(
                text
            )

            page_chunks = (
                split_text_into_chunks(
                    text
                )
            )

            for chunk_number, chunk in enumerate(
                page_chunks,
                start=1,
            ):

                all_chunks.append(
                    {
                        "page": page_number,
                        "chunk_number": chunk_number,
                        "text": chunk,
                    }
                )

    return all_chunks


# ============================================================
# PREPARE SQLITE DATABASE
# ============================================================

def prepare_database():

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS documents
        (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_hash TEXT NOT NULL UNIQUE
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks
        (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            page INTEGER NOT NULL,
            chunk_number INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding TEXT NOT NULL,

            FOREIGN KEY (document_id)
            REFERENCES documents(id)
            ON DELETE CASCADE
        )
        """
    )

    connection.commit()

    connection.close()


# ============================================================
# CHECK IF PDF ALREADY EXISTS
# ============================================================

def pdf_already_exists(
    pdf_hash,
):

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id
        FROM documents
        WHERE file_hash = ?
        """,
        (
            pdf_hash,
        ),
    )

    result = cursor.fetchone()

    connection.close()

    return result is not None


# ============================================================
# DELETE OLD VERSION OF SAME PDF
# ============================================================

def delete_old_version(
    pdf_file,
):

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id
        FROM documents
        WHERE file_name = ?
        """,
        (
            pdf_file.name,
        ),
    )

    result = cursor.fetchone()

    if result:

        document_id = result[0]

        cursor.execute(
            """
            DELETE FROM documents
            WHERE id = ?
            """,
            (
                document_id,
            ),
        )

    connection.commit()

    connection.close()


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(
    client,
    chunks,
):

    embeddings = []

    total = len(
        chunks
    )

    for start in range(
        0,
        total,
        EMBEDDING_BATCH_SIZE,
    ):

        batch = chunks[
            start:
            start + EMBEDDING_BATCH_SIZE
        ]

        texts = [
            chunk["text"]
            for chunk in batch
        ]

        first_item = start + 1

        last_item = min(
            start + len(batch),
            total,
        )

        print(
            f"    Embedding: "
            f"{first_item}-{last_item}/{total}"
        )

        try:

            result = (
                client.generate_embedding(
                    texts
                )
            )

            batch_embeddings = [
                item.embedding
                for item in result.data
            ]

            if (
                len(batch_embeddings)
                != len(batch)
            ):

                raise RuntimeError(
                    "Embedding count does not match chunk count."
                )

            embeddings.extend(
                batch_embeddings
            )

        except Exception:

            for text in texts:

                result = (
                    client.generate_embedding(
                        text
                    )
                )

                embeddings.append(
                    result.data[0].embedding
                )

    return embeddings


# ============================================================
# SAVE TO SQLITE
# ============================================================

def save_to_sqlite(
    pdf_file,
    pdf_hash,
    chunks,
    embeddings,
):

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO documents
        (
            file_name,
            file_path,
            file_hash
        )
        VALUES (?, ?, ?)
        """,
        (
            pdf_file.name,
            str(pdf_file),
            pdf_hash,
        ),
    )

    document_id = (
        cursor.lastrowid
    )

    for chunk, embedding in zip(
        chunks,
        embeddings,
    ):

        cursor.execute(
            """
            INSERT INTO chunks
            (
                document_id,
                page,
                chunk_number,
                text,
                embedding
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                document_id,
                chunk["page"],
                chunk["chunk_number"],
                chunk["text"],
                json.dumps(
                    embedding
                ),
            ),
        )

    connection.commit()

    connection.close()


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    print()
    print(
        "MULTI-PDF INGESTION"
    )

    print(
        "==================="
    )

    print()


    # --------------------------------------------------------
    # PREPARE DATABASE
    # --------------------------------------------------------

    prepare_database()


    # --------------------------------------------------------
    # FIND ALL PDF FILES
    # --------------------------------------------------------

    pdf_files = sorted(
        DOCUMENTS_DIR.glob(
            "*.pdf"
        )
    )

    print(
        f"PDF files found: "
        f"{len(pdf_files)}"
    )

    for pdf in pdf_files:

        print(
            f"- {pdf.name}"
        )

    if not pdf_files:

        print(
            "No PDF files found to process."
        )

        return


    # --------------------------------------------------------
    # INITIALIZE FOUNDRY LOCAL
    # --------------------------------------------------------

    configuration = Configuration(
        app_name="MultiPDFIngest",
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
        # PROCESS PDF FILES
        # ====================================================

        for index, pdf_file in enumerate(
            pdf_files,
            start=1,
        ):

            print()
            print(
                f"[{index}/{len(pdf_files)}] "
                f"{pdf_file.name}"
            )


            # ------------------------------------------------
            # CALCULATE HASH
            # ------------------------------------------------

            pdf_hash = (
                calculate_pdf_hash(
                    pdf_file
                )
            )


            # ------------------------------------------------
            # SKIP UNCHANGED PDF
            # ------------------------------------------------

            if pdf_already_exists(
                pdf_hash
            ):

                print(
                    "    Already stored in the database."
                )

                print(
                    "    Skipped."
                )

                continue


            # ------------------------------------------------
            # DELETE OLD VERSION IF FILE CHANGED
            # ------------------------------------------------

            delete_old_version(
                pdf_file
            )


            # ------------------------------------------------
            # READ PDF AND CREATE CHUNKS
            # ------------------------------------------------

            print(
                "    Reading PDF..."
            )

            chunks = (
                create_pdf_chunks(
                    pdf_file
                )
            )

            print(
                f"    Chunk count: "
                f"{len(chunks)}"
            )

            if not chunks:

                print(
                    "    No text found."
                )

                continue


            # ------------------------------------------------
            # CREATE EMBEDDINGS
            # ------------------------------------------------

            print(
                "    Creating embeddings..."
            )

            embeddings = (
                create_embeddings(
                    embedding_client,
                    chunks,
                )
            )


            # ------------------------------------------------
            # SAVE TO SQLITE
            # ------------------------------------------------

            print(
                "    Saving to SQLite..."
            )

            save_to_sqlite(
                pdf_file,
                pdf_hash,
                chunks,
                embeddings,
            )

            print(
                "    Completed."
            )


    finally:

        if model_loaded:

            print()
            print(
                "Unloading embedding model..."
            )

            embedding_model.unload()

        gc.collect()


    print()
    print(
        "ALL PDF FILES PROCESSED."
    )

    print(
        f"Database: "
        f"{DATABASE_FILE}"
    )


if __name__ == "__main__":
    main()