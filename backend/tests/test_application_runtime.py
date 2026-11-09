
import pytest

from hybrid_rag.infrastructure.runtime import ApplicationRuntime


class FakeResource:
    def __init__(self, name, events, should_fail=False):
        self.name = name
        self.events = events
        self.should_fail = should_fail

    def close(self):
        self.events.append(self.name)

        if self.should_fail:
            raise RuntimeError(f"Failed: {self.name}")


def test_runtime_closes_resources():
    events = []
    runtime = ApplicationRuntime()

    runtime.register(FakeResource("qdrant", events))
    runtime.close()

    assert events == ["qdrant"]


def test_resources_close_in_reverse_order():
    events = []
    runtime = ApplicationRuntime()

    runtime.register(FakeResource("first", events))
    runtime.register(FakeResource("second", events))
    runtime.register(FakeResource("third", events))

    runtime.close()

    assert events == ["third", "second", "first"]


def test_close_is_idempotent():
    events = []
    runtime = ApplicationRuntime()

    runtime.register(FakeResource("qdrant", events))

    runtime.close()
    runtime.close()

    assert events == ["qdrant"]


def test_register_after_close_is_rejected():
    runtime = ApplicationRuntime()
    runtime.close()

    with pytest.raises(RuntimeError, match="already closed"):
        runtime.register(FakeResource("late", []))


def test_all_resources_close_even_if_one_fails():
    events = []
    runtime = ApplicationRuntime()

    runtime.register(FakeResource("first", events))
    runtime.register(
        FakeResource("second", events, should_fail=True)
    )
    runtime.register(FakeResource("third", events))

    with pytest.raises(ExceptionGroup):
        runtime.close()

    assert events == ["third", "second", "first"]


def test_context_manager_closes_resources():
    events = []

    with ApplicationRuntime() as runtime:
        runtime.register(FakeResource("qdrant", events))

    assert events == ["qdrant"]


def test_context_manager_does_not_suppress_errors():
    events = []

    with pytest.raises(ValueError, match="Application error"):
        with ApplicationRuntime() as runtime:
            runtime.register(FakeResource("qdrant", events))
            raise ValueError("Application error")

    assert events == ["qdrant"]
