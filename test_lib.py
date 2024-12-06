from abc import ABC, abstractmethod
from collections.abc import Mapping


class AssistantInterface(ABC):
    """Interface for assistant processor implementations."""

    @abstractmethod
    def process(self, text: str) -> str:
        """Process the input text."""
        pass

    @abstractmethod
    def format_response(self, text: str) -> str:
        """Format the response text."""
        pass


class AssistantTestImpl(AssistantInterface):
    """Simple implementation of the AssistantInterface."""

    def process(self, text: str) -> str:
        return self.format_response("AssistantTestImpl: " + text)

    def format_response(self, text: str) -> str:
        return f"[Formatted] {text}"

    def additional_method(self, text: str) -> str:
        """Extra functionality not defined in the interface."""
        return f"[Extra Feature] {text.lower()}"


assistant_processor_implementations: dict[str, type[AssistantInterface]] = {
    "openai": AssistantTestImpl,  # Simple default implementation
}


class AssistantProcessor:
    """Base class for processing assistant responses."""

    assistant_processor_implementations: dict[str, type[AssistantInterface]] = (
        assistant_processor_implementations
    )

    def __init__(
        self,
        provider: str,
        model: str,
        custom_processors: Mapping[str, type[AssistantInterface]] | None = None,
    ):
        self.assistant_processor_implementations.update(custom_processors or {})
        self.processor = self.assistant_processor_implementations[provider]()
        self.model = model

    def __getattr__(self, name: str):
        """Allow calling any method from the implementation class."""
        return getattr(self.processor, name)

    def process(self, text: str) -> str:
        return self.processor.process(text=text)
