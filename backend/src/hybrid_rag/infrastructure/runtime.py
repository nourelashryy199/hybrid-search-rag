
from typing import Protocol


class ClosableResource(Protocol):
    def close(self) -> None:
        ...


class ApplicationRuntime:
    """Own and clean up application resources."""

    def __init__(self) -> None:
        self._resources: list[ClosableResource] = []
        self._closed = False

    def register(self, resource: ClosableResource) -> None:
        if self._closed:
            raise RuntimeError("Runtime is already closed")

        self._resources.append(resource)

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        errors: list[Exception] = []

        for resource in reversed(self._resources):
            try:
                resource.close()
            except Exception as exc:
                errors.append(exc)

        self._resources.clear()

        if errors:
            raise ExceptionGroup(
                "Failed to close application resources",
                errors,
            )

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Runtime is already closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
