from openai.types.beta import Thread
from openai.types.beta.threads import Message, Run

from abstract_provider import AssistantInterface, AssistantProcessorConfig
from implementations.openai_provider import OpenAIAssistantImpl


OPENAI = "openai"


processor_implementations: dict[str, type[AssistantInterface]] = {
    OPENAI: OpenAIAssistantImpl,
}


def get_assistant_processor(config):
    """Get the appropriate assistant processor implementation based on the config.

    Args:
        config: Configuration object containing provider information.

    Returns:
        An instance of the appropriate AssistantInterface implementation.

    Raises:
        KeyError: If the requested provider is not implemented.
    """
    if config.provider not in processor_implementations:
        raise KeyError(
            f"AssistanProcessor for Provider: {config.provider} is not implemented."
        )
    return processor_implementations[config.provider](config)


class AssistantProcessor(AssistantInterface):
    """Base class for processing assistant responses.

    This class serves as a blueprint for implementing various assistant processors.
    It defines common attributes and methods that all processors should implement.

    Attributes:
        config (AssistantProcessorConfig): The configuration for the assistant
                                            processor.
        processor (AssistantInterface): The underlying processor implementation.
    """

    def __init__(
        self,
        config: AssistantProcessorConfig,
    ) -> None:
        """Initialize the AssistantProcessor with the given configuration.

        Args:
            config (AssistantProcessorConfig): The configuration for the assistant
                                                processor.
        """
        self.processor = get_assistant_processor(config)

    def __getattr__(self, name: str):
        """Allow calling any method from the implementation class."""
        return getattr(self.processor, name)

    async def start_run(self, thread_id: str, assistant_id: str) -> Run | None:
        """Start a new run for the assistant.

        Args:
            thread_id: The ID of the thread to run.
            assistant_id: The ID of the assistant to use.

        Returns:
            The run object if successful, None otherwise.
        """
        return await self.processor.start_run(
            thread_id,
            assistant_id,
        )

    async def create_message(
        self, thread_id: str, assistant_id: str, content: str
    ) -> Message | None:
        """Create a new message in the thread.

        Args:
            thread_id: The ID of the thread.
            assistant_id: The ID of the assistant.
            content: The message content.

        Returns:
            The created message if successful, None otherwise.
        """
        return await self.processor.create_message(
            thread_id,
            assistant_id,
            content,
        )

    async def get_assistant_response(
        self, thread_id: str, max_retries: int = 3
    ) -> Message | None:
        """Get the assistant's response from a thread.

        Args:
            thread_id: The ID of the thread.
            max_retries: Maximum number of retry attempts.

        Returns:
            The assistant's response message if successful, None otherwise.
        """
        return await self.processor.get_assistant_response(
            thread_id,
            max_retries,
        )

    async def execute_function_call(
        self, thread: Thread, function_name: str, arguments: dict
    ):
        """Execute a function call from the assistant.

        Args:
            thread: The thread object.
            function_name: Name of the function to execute.
            arguments: Function arguments.

        Returns:
            The result of the function execution.
        """
        return await self.processor.execute_function_call(
            thread,
            function_name,
            arguments,
        )

    async def get_thread_messages(self, thread_id: str) -> list[Message] | None:
        """Get all messages from a thread.

        Args:
            thread_id: The ID of the thread.

        Returns:
            List of messages if successful, None otherwise.
        """
        return await self.processor.get_thread_messages(
            thread_id,
        )

    async def setup_new_thread(self, thread_id: str, messages: list[Message]) -> None:
        """Set up a new thread with initial messages.

        Args:
            thread_id: The ID of the thread to set up.
            messages: List of messages to add to the thread.
        """
        return await self.processor.setup_new_thread(
            thread_id,
            messages,
        )

    async def get_or_create_thread(self, thread_id: str | None = None) -> Thread | None:
        """Get an existing thread or create a new one.

        Args:
            thread_id: Optional ID of an existing thread.

        Returns:
            The thread object if successful, None otherwise.
        """
        return await self.processor.get_or_create_thread(
            thread_id,
        )

    async def submit_tool_outputs(
        self, run: Run, thread: Thread, tool_calls: list
    ) -> Run:
        """Submit outputs from tool calls back to the assistant.

        Args:
            run: The current run object.
            thread: The thread object.
            tool_calls: List of tool calls to process.

        Returns:
            The updated run object.
        """
        return await self.processor.submit_tool_outputs(
            run,
            thread,
            tool_calls,
        )
