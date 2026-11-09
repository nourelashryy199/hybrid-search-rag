
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hybrid_rag.domain.generation import EvidencePassage
from hybrid_rag.infrastructure.claude_generator import (
    ClaudeGenerator,
)


MODEL = "test-claude-model"


def make_evidence():
    return [
        EvidencePassage(
            chunk_id="leave-001",
            document_id="handbook",
            text="Employees receive 25 days of annual leave.",
            page_number=4,
        )
    ]


def make_answer(
    answer="Employees receive 25 days of annual leave.",
    citations=None,
    abstained=False,
):
    if citations is None:
        citations = [
            {
                "chunk_id": "leave-001",
                "document_id": "handbook",
                "page_number": 4,
            }
        ]

    return {
        "answer": answer,
        "citations": citations,
        "abstained": abstained,
    }


def make_client(payload, stop_reason="end_turn"):
    client = Mock()

    text = (
        payload
        if isinstance(payload, str)
        else json.dumps(payload)
    )

    client.messages.create.return_value = SimpleNamespace(
        stop_reason=stop_reason,
        content=[
            SimpleNamespace(
                type="text",
                text=text,
            )
        ],
    )

    return client


def make_generator(client):
    return ClaudeGenerator(
        client=client,
        model=MODEL,
    )


def test_generates_structured_answer():
    client = make_client(make_answer())
    generator = make_generator(client)

    result = generator.generate(
        "How many days of annual leave?",
        make_evidence(),
    )

    assert result.abstained is False
    assert "25 days" in result.answer
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "leave-001"

    client.messages.create.assert_called_once()


def test_sends_evidence_in_prompt():
    client = make_client(make_answer())
    generator = make_generator(client)

    generator.generate(
        "How many days of annual leave?",
        make_evidence(),
    )

    kwargs = client.messages.create.call_args.kwargs
    payload = json.loads(kwargs["messages"][0]["content"])

    assert payload["question"] == (
        "How many days of annual leave?"
    )
    assert payload["evidence"][0]["chunk_id"] == "leave-001"
    assert payload["evidence"][0]["page_number"] == 4
    assert kwargs["model"] == MODEL
    assert kwargs["temperature"] == 0


def test_empty_evidence_abstains_without_api_call():
    client = make_client(make_answer())
    generator = make_generator(client)

    result = generator.generate(
        "What is the vacation policy?",
        [],
    )

    assert result.abstained is True
    assert result.citations == ()
    client.messages.create.assert_not_called()


def test_empty_question_is_rejected():
    generator = make_generator(
        make_client(make_answer())
    )

    with pytest.raises(ValueError, match="Question"):
        generator.generate("   ", make_evidence())


def test_invalid_json_is_rejected():
    generator = make_generator(
        make_client("not valid JSON")
    )

    with pytest.raises(ValueError, match="invalid JSON"):
        generator.generate("Question?", make_evidence())


def test_missing_fields_are_rejected():
    generator = make_generator(
        make_client({"answer": "25 days"})
    )

    with pytest.raises(ValueError, match="fields"):
        generator.generate("Question?", make_evidence())


def test_unknown_citation_is_rejected():
    payload = make_answer(
        citations=[
            {
                "chunk_id": "invented",
                "document_id": "handbook",
                "page_number": 4,
            }
        ]
    )

    generator = make_generator(make_client(payload))

    with pytest.raises(ValueError, match="Unknown citation"):
        generator.generate("Question?", make_evidence())


def test_incorrect_page_number_is_rejected():
    payload = make_answer(
        citations=[
            {
                "chunk_id": "leave-001",
                "document_id": "handbook",
                "page_number": 99,
            }
        ]
    )

    generator = make_generator(make_client(payload))

    with pytest.raises(ValueError, match="metadata"):
        generator.generate("Question?", make_evidence())


def test_abstained_response_is_accepted():
    payload = make_answer(
        answer="Insufficient information.",
        citations=[],
        abstained=True,
    )

    generator = make_generator(make_client(payload))

    result = generator.generate(
        "What is the CEO's salary?",
        make_evidence(),
    )

    assert result.abstained is True
    assert result.citations == ()


def test_uncited_answer_is_rejected():
    payload = make_answer(
        citations=[],
        abstained=False,
    )

    generator = make_generator(make_client(payload))

    with pytest.raises(ValueError, match="require citations"):
        generator.generate("Question?", make_evidence())


def test_truncated_response_is_rejected():
    client = make_client(
        make_answer(),
        stop_reason="max_tokens",
    )

    generator = make_generator(client)

    with pytest.raises(ValueError, match="complete"):
        generator.generate("Question?", make_evidence())


def test_invalid_model_name_is_rejected():
    with pytest.raises(ValueError, match="Model"):
        ClaudeGenerator(
            client=Mock(),
            model="   ",
        )


def test_invalid_max_tokens_is_rejected():
    with pytest.raises(ValueError, match="max_tokens"):
        ClaudeGenerator(
            client=Mock(),
            model=MODEL,
            max_tokens=0,
        )


def test_duplicate_citations_are_rejected():
    citation = make_answer()["citations"][0]

    payload = make_answer(
        citations=[citation, citation]
    )

    generator = make_generator(make_client(payload))

    with pytest.raises(ValueError, match="Duplicate answer citation"):
        generator.generate("Question?", make_evidence())


def test_non_boolean_abstention_is_rejected():
    payload = make_answer()
    payload["abstained"] = "false"

    generator = make_generator(make_client(payload))

    with pytest.raises(ValueError, match="boolean"):
        generator.generate("Question?", make_evidence())
