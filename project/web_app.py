"""Local-only web interface for the Microsoft Foundry Local RAG project.

The server binds to 127.0.0.1, serves every interface asset from disk, and
reuses the retrieval and generation functions from ``multi_pdf_rag.py``.
"""

from __future__ import annotations

import argparse
import atexit
import json
import mimetypes
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import multi_pdf_rag as rag
from foundry_local_sdk import Configuration, FoundryLocalManager


PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent
WEB_DIR = PROJECT_DIR / "web"
DOCUMENTS_DIR = ROOT_DIR / "documents"
MAX_REQUEST_BYTES = 32 * 1024

ANSWER_MODELS = {
    "fast": {
        "name": "qwen2.5-0.5b",
        "label": "Fast",
        "display_name": "Qwen2.5 0.5B",
    },
    "quality": {
        "name": rag.ANSWER_MODEL,
        "label": "Quality",
        "display_name": "Qwen3 4B",
    },
}
DEFAULT_ANSWER_MODEL = "fast"


def resolve_model_cache_dir() -> str:
    """Use packaged models first, then reuse a sibling development cache."""

    candidates = [
        ROOT_DIR / "model_cache",
        ROOT_DIR.parent / "model_cache",
    ]

    for candidate in candidates:
        if (candidate / "Microsoft").is_dir():
            return str(candidate)

    return rag.MODEL_CACHE_DIR


class ServiceUnavailableError(RuntimeError):
    """Raised when the models are not ready to answer yet."""


class BusyError(RuntimeError):
    """Raised when another question is already being processed."""


class LocalRAGService:
    """Owns the local models and exposes a thread-safe question API."""

    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self._question_lock = threading.Lock()
        self._status = "starting"
        self._stage = "Reading the local knowledge base"
        self._error: str | None = None

        self.chunks: list[dict[str, Any]] = []
        self.embedding_model: Any = None
        self.embedding_client: Any = None
        self.answer_models: dict[str, Any] = {}
        self.chat_clients: dict[str, Any] = {}
        self._embedding_loaded = False
        self._loaded_answer_models: set[str] = set()
        self._closed = False

    def start(self) -> None:
        threading.Thread(
            target=self._initialize,
            name="local-rag-model-loader",
            daemon=True,
        ).start()

    def _set_state(
        self,
        status: str,
        stage: str,
        *,
        error: str | None = None,
    ) -> None:
        with self._state_lock:
            self._status = status
            self._stage = stage
            self._error = error

    def _initialize(self) -> None:
        try:
            self.chunks = rag.load_chunks()

            self._set_state("loading", "Loading the embedding model")
            configuration = Configuration(
                app_name="LocalRAGWeb",
                model_cache_dir=resolve_model_cache_dir(),
            )
            FoundryLocalManager.initialize(configuration)
            manager = FoundryLocalManager.instance

            self.embedding_model = manager.catalog.get_model(
                rag.EMBEDDING_MODEL
            )

            self.embedding_model.load()
            self._embedding_loaded = True
            self.embedding_client = self.embedding_model.get_embedding_client()

            for model_id, model_config in ANSWER_MODELS.items():
                self._set_state(
                    "loading",
                    f"Loading the {model_config['label'].lower()} answer model",
                )
                answer_model = manager.catalog.get_model(model_config["name"])
                answer_model.load()
                chat_client = answer_model.get_chat_client()
                chat_client.settings.temperature = 0.0
                chat_client.settings.max_tokens = 120
                self.answer_models[model_id] = answer_model
                self.chat_clients[model_id] = chat_client
                self._loaded_answer_models.add(model_id)

            self._set_state("ready", "Ready")
        except Exception as error:  # surfaced in the local interface
            self._set_state(
                "error",
                "The local models could not be started",
                error=str(error),
            )

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            status = self._status
            stage = self._stage
            error = self._error

        return {
            "status": status,
            "stage": stage,
            "error": error,
            "chunk_count": len(self.chunks),
            "document_count": len(self.document_names()),
            "embedding_model": rag.EMBEDDING_MODEL,
            "answer_models": {
                model_id: {
                    **model_config,
                    "ready": model_id in self._loaded_answer_models,
                }
                for model_id, model_config in ANSWER_MODELS.items()
            },
            "default_answer_model": DEFAULT_ANSWER_MODEL,
            "top_k": rag.TOP_K,
            "threshold": rag.MIN_RETRIEVAL_SCORE,
        }

    def document_names(self) -> list[str]:
        return sorted({chunk["file_name"] for chunk in self.chunks})

    def documents(self) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for chunk in self.chunks:
            entry = grouped.setdefault(
                chunk["file_name"],
                {
                    "file_name": chunk["file_name"],
                    "chunk_count": 0,
                    "pages": set(),
                },
            )
            entry["chunk_count"] += 1
            if chunk["page"] is not None:
                entry["pages"].add(chunk["page"])

        return [
            {
                "file_name": item["file_name"],
                "chunk_count": item["chunk_count"],
                "page_count": len(item["pages"]),
            }
            for item in sorted(grouped.values(), key=lambda value: value["file_name"])
        ]

    def ask(self, question: str, model_id: str) -> dict[str, Any]:
        question = " ".join(question.split())
        if not question:
            raise ValueError("Please enter a question.")
        if len(question) > 1_000:
            raise ValueError("The question must be 1,000 characters or fewer.")
        if model_id not in ANSWER_MODELS:
            raise ValueError("Please select a valid answer model.")

        with self._state_lock:
            if self._status not in {"ready"}:
                raise ServiceUnavailableError(
                    "The local models are still loading. Please wait a moment."
                )

        if not self._question_lock.acquire(blocking=False):
            raise BusyError("Another question is already being processed.")

        try:
            with self._state_lock:
                self._status = "busy"
                self._stage = "Creating the query embedding"

            query_embedding = rag.create_query_embedding(
                self.embedding_client,
                question,
            )

            self._set_state("busy", "Searching the local documents")
            retrieved_chunks = rag.retrieve_chunks(query_embedding, self.chunks)

            is_fallback = (
                not retrieved_chunks
                or retrieved_chunks[0]["score"] < rag.MIN_RETRIEVAL_SCORE
            )

            if is_fallback:
                answer = rag.FALLBACK_MESSAGE
            else:
                self._set_state("busy", "Generating a grounded answer")
                answer = rag.generate_answer(
                    self.chat_clients[model_id],
                    question,
                    retrieved_chunks,
                )

                if not rag.extract_source_numbers(answer):
                    answer = f"{answer.rstrip()} [Source 1]"

            cited_numbers = set(rag.extract_source_numbers(answer))
            sources = []
            for index, chunk in enumerate(retrieved_chunks, start=1):
                sources.append(
                    {
                        "number": index,
                        "file_name": chunk["file_name"],
                        "page": chunk["page"],
                        "chunk_number": chunk["chunk_number"],
                        "score": round(float(chunk["score"]), 4),
                        "excerpt": " ".join(chunk["text"].split())[:500],
                        "cited": index in cited_numbers,
                    }
                )

            return {
                "question": question,
                "answer": answer,
                "fallback": is_fallback or answer == rag.FALLBACK_MESSAGE,
                "sources": sources,
                "model": {
                    "id": model_id,
                    **ANSWER_MODELS[model_id],
                },
            }
        finally:
            with self._state_lock:
                if self._status != "error":
                    self._status = "ready"
                    self._stage = "Ready"
            self._question_lock.release()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            for model_id in reversed(tuple(self._loaded_answer_models)):
                answer_model = self.answer_models.get(model_id)
                if answer_model is not None:
                    answer_model.unload()
            if self._embedding_loaded and self.embedding_model is not None:
                self.embedding_model.unload()
        except Exception:
            pass


class LocalRAGRequestHandler(BaseHTTPRequestHandler):
    server_version = "LocalRAGWeb/1.0"

    @property
    def app(self) -> "LocalRAGHTTPServer":
        return self.server  # type: ignore[return-value]

    def _security_headers(self, *, cache: bool = False) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; "
            "style-src 'self'; script-src 'self'; object-src 'self'",
        )
        self.send_header(
            "Cache-Control",
            "public, max-age=3600" if cache else "no-store",
        )

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._security_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, *, cache: bool = True) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self._security_headers(cache=cache)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        request_path = unquote(urlparse(self.path).path)

        if request_path == "/api/status":
            self._send_json(self.app.rag_service.status())
            return
        if request_path == "/api/documents":
            self._send_json({"documents": self.app.rag_service.documents()})
            return
        if request_path.startswith("/documents/"):
            file_name = request_path.removeprefix("/documents/")
            if Path(file_name).name != file_name:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            if file_name not in self.app.rag_service.document_names():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send_file(DOCUMENTS_DIR / file_name, cache=False)
            return

        static_files = {
            "/": "index.html",
            "/index.html": "index.html",
            "/styles.css": "styles.css",
            "/app.js": "app.js",
            "/favicon.svg": "favicon.svg",
            "/favicon.ico": "favicon.svg",
        }
        file_name = static_files.get(request_path)
        if file_name is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send_file(WEB_DIR / file_name, cache=False)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        request_path = urlparse(self.path).path
        if request_path != "/api/ask":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json({"error": "Invalid request size."}, 400)
            return

        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            self._send_json({"error": "Invalid request size."}, 400)
            return

        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            question = payload.get("question", "")
            model_id = payload.get("model", DEFAULT_ANSWER_MODEL)
            if not isinstance(question, str):
                raise ValueError("Question must be text.")
            if not isinstance(model_id, str):
                raise ValueError("Answer model must be text.")
            result = self.app.rag_service.ask(question, model_id)
            self._send_json(result)
        except json.JSONDecodeError:
            self._send_json({"error": "The request was not valid JSON."}, 400)
        except ValueError as error:
            self._send_json({"error": str(error)}, 400)
        except BusyError as error:
            self._send_json({"error": str(error)}, 409)
        except ServiceUnavailableError as error:
            self._send_json({"error": str(error)}, 503)
        except Exception as error:
            self._send_json({"error": f"The local request failed: {error}"}, 500)

    def log_message(self, format_string: str, *args: Any) -> None:
        if self.path.startswith("/api/status"):
            return
        print(f"[Local RAG] {self.address_string()} - {format_string % args}")


class LocalRAGHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        rag_service: LocalRAGService,
    ) -> None:
        self.rag_service = rag_service
        super().__init__(address, LocalRAGRequestHandler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the offline Local RAG site.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    host = "127.0.0.1"
    service = LocalRAGService()
    atexit.register(service.close)

    try:
        server = LocalRAGHTTPServer((host, args.port), service)
    except OSError as error:
        raise SystemExit(
            f"The local site could not use port {args.port}: {error}"
        ) from error

    service.start()
    local_url = f"http://localhost:{args.port}"
    print("\nLOCAL RAG WEB INTERFACE")
    print("=======================")
    print(f"Open: {local_url}")
    print("This site is available only on this computer.")
    print("Press Ctrl+C to stop it.\n")

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(local_url)).start()

    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nStopping the local site...")
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
