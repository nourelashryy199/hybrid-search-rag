
from hybrid_rag.domain.document import DocumentChunk


def make_chunk(
    chunk_id: str,
    document_id: str,
    text: str,
    page_number: int,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        chunk_index=0,
        page_number=page_number,
    )


def get_sample_chunks() -> list[DocumentChunk]:
    return [
        make_chunk(
            "leave-001",
            "employee-handbook",
            "Full-time employees receive 25 days of paid "
            "annual leave per year.",
            1,
        ),
        make_chunk(
            "leave-002",
            "employee-handbook",
            "Annual leave requests must be approved by "
            "the employee's manager.",
            2,
        ),
        make_chunk(
            "remote-001",
            "employee-handbook",
            "Employees may work remotely up to three "
            "days per week with manager approval.",
            3,
        ),
        make_chunk(
            "security-001",
            "security-policy",
            "Multi-factor authentication is required "
            "for all company accounts.",
            1,
        ),
        make_chunk(
            "expenses-001",
            "finance-policy",
            "Travel expenses must be submitted within "
            "30 days of returning from a business trip.",
            4,
        ),
        make_chunk(
            "training-001",
            "employee-handbook",
            "Employees can request funding for "
            "professional development courses.",
            5,
        ),

        make_chunk(
            "leave-003",
            "employee-handbook",
            "Employees may carry over a maximum of five "
            "unused annual leave days into the next year.",
            6,
        ),
        make_chunk(
            "security-002",
            "security-policy",
            "Policy SEC-204 requires employees to reset "
            "compromised passwords immediately.",
            2,
        ),
        make_chunk(
            "security-003",
            "security-policy",
            "Policy SEC-205 requires encryption of "
            "confidential files stored on laptops.",
            3,
        ),
        make_chunk(
            "expenses-002",
            "finance-policy",
            "Expense code FIN-301 applies to meals "
            "purchased during approved business travel.",
            5,
        ),
        make_chunk(
            "remote-002",
            "employee-handbook",
            "Employees working from home must remain "
            "available during their agreed working hours.",
            7,
        ),
        make_chunk(
            "training-002",
            "employee-handbook",
            "Professional development funding requests "
            "must be submitted before course registration.",
            8,
        ),

    ]
