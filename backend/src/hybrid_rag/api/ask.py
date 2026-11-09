
from typing import Protocol

from fastapi import APIRouter, Depends, HTTPException, Request

from hybrid_rag.api.schemas import AskRequest, AskResponse
from hybrid_rag.domain.generation import GeneratedAnswer


class QuestionAnsweringService(Protocol):
    def ask(
        self,
        question: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> GeneratedAnswer:
        ...


router = APIRouter(tags=["Question Answering"])


def get_rag_service(request: Request) -> QuestionAnsweringService:
    service = getattr(request.app.state, "rag_service", None)

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Question answering is not configured",
        )

    return service


@router.post("/ask", response_model=AskResponse)
def ask_question(
    payload: AskRequest,
    service: QuestionAnsweringService = Depends(get_rag_service),
) -> AskResponse:
    if payload.candidate_k < payload.top_k:
        raise HTTPException(
            status_code=422,
            detail="candidate_k must be >= top_k",
        )

    try:
        result = service.ask(
            question=payload.question,
            top_k=payload.top_k,
            candidate_k=payload.candidate_k,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return AskResponse(
        answer=result.answer,
        citations=[
            {
                "chunk_id": citation.chunk_id,
                "document_id": citation.document_id,
                "page_number": citation.page_number,
            }
            for citation in result.citations
        ],
        abstained=result.abstained,
    )
