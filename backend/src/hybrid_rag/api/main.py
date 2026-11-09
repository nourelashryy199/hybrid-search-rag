
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from hybrid_rag.api.ask import router as ask_router
from hybrid_rag.api.documents import router as documents_router

from hybrid_rag.application.chunking import TextChunker
from hybrid_rag.application.document_upload import DocumentUploadService
from hybrid_rag.application.factory import create_rag_application
from hybrid_rag.application.ingestion import DocumentIngestionService

from hybrid_rag.infrastructure.config import AppSettings
from hybrid_rag.infrastructure.pdf_parser import PDFParser


def create_api(
    rag_service: Any = None,
    *,
    settings: AppSettings | None = None,
    initialize_rag: bool = False,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        application = None

        try:
            if initialize_rag:
                configuration = settings or AppSettings()

                application = create_rag_application(
                    settings=configuration,
                )

                ingestor = DocumentIngestionService(
                    parser=PDFParser(),
                    chunker=TextChunker(),
                )

                upload_service = DocumentUploadService(
                    ingestor=ingestor,
                    indexer=application.indexing,
                )

                app.state.rag_application = application
                app.state.rag_service = application.rag
                app.state.indexing_service = application.indexing
                app.state.document_ingestion_service = upload_service

            yield

        finally:
            if application is not None:
                try:
                    application.close()
                finally:
                    app.state.rag_application = None
                    app.state.rag_service = None
                    app.state.indexing_service = None
                    app.state.document_ingestion_service = None

    app = FastAPI(
        title="Hybrid Search RAG API",
        description=(
            "Persistent hybrid document retrieval and "
            "evidence-grounded question answering."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.rag_application = None
    app.state.rag_service = rag_service
    app.state.indexing_service = None
    app.state.document_ingestion_service = None

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    app.include_router(ask_router)
    app.include_router(documents_router)

    return app


app = create_api(initialize_rag=True)
