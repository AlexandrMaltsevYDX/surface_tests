from abc import ABC, abstractmethod
from collections.abc import Callable

from openai.types.beta import Thread
from openai.types.beta.threads import Message, Run
from pydantic import BaseModel


class AssistantProcessorConfig(BaseModel):
    """Base class for processing assistant responses."""

    provider: str
    model: str
    function_registry: dict[str, Callable] | None = None
    api_key: str
    assistant_id: str | None = None


class AssistantInterface(ABC):
    """Interface for assistant processor implementations."""

    @abstractmethod
    def __init__(
        self,
        config: AssistantProcessorConfig,
    ):
        """Initialize the assistant interface with the provided configuration.

        Args:
            config (AssistantProcessorConfig): The configuration for the assistant interface.
        """
        pass

    @abstractmethod
    async def start_run(self, thread_id: str, assistant_id: str) -> Run | None:
        """Start a new run for the specified thread and assistant.

        This method initiates a new run, which represents an execution of the assistant
        on the current thread. It continuously checks the run status and handles any
        required actions until the run is completed.

        Args:
            thread_id (str): The ID of the thread to start the run on.
            assistant_id (str): The ID of the assistant to use for the run.

        Returns:
            Run | None: The completed Run object if successful, None if an error occurs.

        Raises:
            Exception: If there's an error starting or processing the run.
        """
        pass

    @abstractmethod
    async def create_message(
        self,
        thread_id: str,
        assistant_id: str,
        content: str,
    ) -> Message | None:
        """Create a new message in the specified thread and start a run with the assistant.

        This method creates a new user message in the given thread and then initiates
        a run with the specified assistant. It handles any exceptions that may occur
        during this process.

        Args:
            thread_id (str): The ID of the thread to create the message in.
            assistant_id (str): The ID of the assistant to use for the run.
            content (str): The content of the message to be created.

        Returns:
            Message | None: The created Message object if successful,
            None if an error occurs.

        Raises:
            Exception: If there's an error creating the message or starting the run.
        """
        pass

    @abstractmethod
    async def get_assistant_response(
        self, thread_id: str, max_retries: int = 3
    ) -> Message | None:
        """Attempt to retrieve the assistant's response from the thread.

        This method will retry up to `max_retries` times if no assistant message is found.
        It introduces a delay between retries to allow time for the assistant to respond.

        Args:
            thread_id (str): The ID of the thread to retrieve messages from.
            max_retries (int, optional): The maximum number of retry attempts. Defaults to 3.

        Returns:
            Message | None: The latest message from the assistant if found, otherwise None.

        Raises:
            Exception: If unable to retrieve the assistant's response after all retry attempts.
        """
        pass

    @abstractmethod
    async def get_or_create_thread(self, thread_id: str | None = None) -> Thread | None:
        """Get an existing thread or create a new one.

        This method attempts to retrieve an existing thread using the provided thread_id.
        If no thread_id is provided, it creates a new thread.

        Args:
            thread_id (str | None): The ID of the thread to retrieve. If None, a new thread is created.

        Returns:
            Thread | None: The retrieved or newly created Thread object, or None if an error occurs.

        Raises:
            Exception: Any exception that occurs during thread retrieval or creation is caught
                       and logged, returning None in such cases.
        """
        pass

    @abstractmethod
    async def get_thread_messages(self, thread_id: str) -> list[Message] | None:
        """Get all messages from the specified thread.

        Args:
            thread_id (str): The ID of the thread to retrieve messages from.

        Returns:
            list[Message] | None: The list of messages from the thread if successful,
            None if an error occurs.
        """
        pass

    @abstractmethod
    async def setup_new_thread(self, thread_id: str, messages: list[Message]) -> None:
        """Put messages in the new thread.

        Args:
            thread_id (str): The ID of the thread to retrieve messages from.
            messages (list[Message]): The list of messages to put in the thread.

        """
        pass

    @abstractmethod
    async def execute_function_call(
        self, thread: Thread, function_name: str, arguments: dict
    ):
        """Execute a function call based on the provided function name and arguments.

        This method checks if the requested function exists in the function registry.
        If the function is not found, it returns an error message.

        Args:
            thread (Thread): The thread object associated with the function call.
            function_name (str): The name of the function to be executed.
            arguments (dict): A dictionary of arguments to be passed to the function.

        Returns:
            str: An error message or function result.

        Note:
            This is part of the error handling before attempting to execute the function.
            The actual function execution is handled in the subsequent code.
        """
        pass

    @abstractmethod
    async def submit_tool_outputs(
        self, run: Run, thread: Thread, tool_calls: list
    ) -> Run:
        """Initialize an empty list to store tool outputs.

        This list will be populated with the results of function calls
        made in response to the assistant's tool calls. Each output
        will be a dictionary containing the tool call ID and the
        function's response.

        Returns:
            list: An empty list that will be filled with tool outputs.
        """
        pass
