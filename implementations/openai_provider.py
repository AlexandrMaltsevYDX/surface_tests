import asyncio
from contextlib import asynccontextmanager

from openai import AsyncOpenAI
from openai.types.beta import Assistant, Thread
from openai.types.beta.threads import Message, Run

from abstract_provider import AssistantProcessorConfig
from abstract_provider.assistant import AssistantInterface


@asynccontextmanager
async def retry_manager(max_retries: int = 5, delay: int = 2):
    """Context manager for retrying operations with exponential backoff.

    Args:
        max_retries (int): Maximum number of retry attempts.
        delay (int, optional): Initial delay between retries in seconds. Defaults to 2.

    Yields:
        int: The current retry attempt number.

    Raises:
        Exception: If operation fails after max_retries attempts.
    """
    retries = 0
    while retries < max_retries:
        try:
            yield retries
            return
        except Exception as e:
            retries += 1
            if retries < max_retries:
                print(f"Retry attempt {retries}: {e}")
                await asyncio.sleep(delay)
            else:
                raise Exception(f"Operation failed after {max_retries} retries") from e


class OpenAIAssistantImpl(AssistantInterface):
    """OpenAI implementation of the AssistantInterface.qqqq"""

    def __init__(self, config: AssistantProcessorConfig) -> None:  # noqa: D107
        self.client = self.create_client(api_key=config.api_key)
        self.assistant_id = config.assistant_id

    async def start_run(self, thread_id: str, assistant_id: str) -> Run | None:
        try:
            run = await self.client.beta.threads.runs.create(
                thread_id=thread_id, assistant_id=assistant_id
            )
            # check Processing Status
            while run.status != "completed":
                run = await self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run.id,
                )

                if run.status == "requires_action":
                    tool_calls = run.required_action.submit_tool_outputs.tool_calls  # type: ignore
                    await self.submit_tool_outputs(run, thread_id, tool_calls)  # type: ignore

                await asyncio.sleep(1)

            return run
        except Exception as e:
            print(f"Error starting run: {e}")
            return None

    async def create_message(
        self, thread_id: str, assistant_id: str, content: str
    ) -> Message | None:
        """create_message using OpenAI API."""
        try:
            message = await self.client.beta.threads.messages.create(
                thread_id=thread_id, role="user", content=content
            )
            await self.start_run(thread_id, assistant_id)
            return message
        except Exception as e:
            print(f"Error creating message: {e}")
            return None

    async def _try_get_latest_assistant_message(self, thread_id: str) -> Message:
        """Get the latest message from the assistant in the messages list.

        Args:
            thread_id: The ID of the thread to get messages from.

        Returns:
            Message | None: The latest assistant message if found, None otherwise.
        """
        messages = await self.client.beta.threads.messages.list(thread_id=thread_id)

        if messages.data:
            # Get the latest message from the assistant
            assistant_message = next(
                (msg for msg in messages.data if msg.role == "assistant"), None
            )
            if assistant_message:
                return assistant_message
        raise Exception("No assistant message found, retrying...")

    async def get_assistant_response(
        self, thread_id: str, max_retries: int = 5
    ) -> Message | None:
        """Get the assistant's response from a thread.

        Args:
            thread_id (str): The ID of the thread to retrieve messages from.
            max_retries (int, optional): Maximum number of retry attempts. Defaults to 5.

        Returns:
            Message | None: The latest message from the assistant if found, None if an error occurs.

        Raises:
            Exception: If unable to retrieve the assistant's response after all retry attempts.
        """
        try:
            async with retry_manager(max_retries=max_retries):
                return await self._try_get_latest_assistant_message(thread_id)

        except Exception as e:
            print(f"Error getting assistant response: {e}")
            return None

    async def submit_tool_outputs(
        self, run: Run, thread: Thread, tool_calls: list
    ) -> Run:
        tool_outputs = []
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            arguments = tool_call.function.arguments
            function_id = tool_call.id

            function_response = await self.execute_function_call(
                thread, function_name, arguments
            )
            tool_outputs.append(
                {
                    "tool_call_id": function_id,
                    "output": str(function_response),
                }
            )

        run = await self.client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread.id, run_id=run.id, tool_outputs=tool_outputs
        )

        return run

    async def get_or_create_thread(self, thread_id: str | None = None) -> Thread | None:
        """Get an existing thread or create a new one.

        This method attempts to retrieve an existing thread using the provided
            thread_id.
        If no thread_id is provided, it creates a new thread.

        Args:
            thread_id (str | None): The ID of the thread to retrieve. If None, a new
                                    thread is created.

        Returns:
            Thread | None: The retrieved or newly created Thread object,
                            or None if an error occurs.

        Raises:
            Exception: Any exception that occurs during thread retrieval
                       or creation is caught
                       and logged, returning None in such cases.
        """
        try:
            if thread_id is None:
                # Create a new thread if no thread_id is provided
                return await self.client.beta.threads.create()
            # Try to retrieve the existing thread
            return await self.client.beta.threads.retrieve(thread_id=thread_id)
        except Exception as e:
            print(f"Error in get_or_create_thread: {e}")
            return None

    async def get_thread_messages(self, thread_id: str) -> list[Message] | None:
        """Get all messages from the specified thread.

        Args:
            thread_id (str): The ID of the thread to retrieve messages from.

        Returns:
            list[Message] | None: The list of messages from the thread if successful,
            None if an error occurs.
        """
        try:
            messages = await self.client.beta.threads.messages.list(thread_id=thread_id)
            if messages.data:
                return list(messages.data)
            return None
        except Exception as e:
            print(f"Error getting thread messages: {e}")
            return None

    async def setup_new_thread(self, thread_id: str, messages: list[Message]) -> None:
        """Put messages in the new thread.

        Args:
            thread_id (str): The ID of the thread to retrieve messages from.
            messages (list[Message]): The list of messages to put in the thread.
        """
        try:
            for message in messages:
                await self.client.beta.threads.messages.create(
                    thread_id=thread_id,
                    role=message.role,
                    content=message.content[0].text.value if message.content else "",  # type: ignore
                )
        except Exception as e:
            print(f"Error setting up new thread: {e}")

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
            This is part of the error handling before attempting to execute the
                function.
            The actual function execution is handled in the subsequent code.
        """
        pass

    async def create_external_assistant(
        self,
        name: str,
        instructions: str,
        model: str,
        tools: list[dict],
    ) -> Assistant:
        return await self.client.beta.assistants.create(
            name=name,
            instructions=instructions,
            model=model,
            tools=tools,  # type: ignore
        )

    @classmethod
    def create_client(cls, api_key: str) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=api_key)
