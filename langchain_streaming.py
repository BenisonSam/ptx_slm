import asyncio
# from src.logger import log
import logging
import os
import re
from threading import Thread
from typing import Any, Dict, List, Tuple, Optional, Iterator, AsyncIterator

import torch
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.schema.messages import AIMessage, BaseMessage
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import ToolMessage
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

log = logging.getLogger(__name__)


class TransformersModel(LLM):
    """A LangChain LLM implementation that streams responses from a HuggingFace model."""

    model_name: str  # Required field
    model_path: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    model: Optional[AutoModelForCausalLM] = None
    tokenizer: Optional[AutoTokenizer] = None

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    generation_kwargs: Dict[str, Any] = {
        "do_sample": False,
        "num_beams": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 50,
        "repetition_penalty": 1.0,
        "use_cache": True,
        "max_new_tokens": 512,
    }

    def __init__(
            self,
            model_name: str,
            model_path: Optional[str] = None,
            model_kwargs: Optional[Dict[str, Any]] = None,
            generation_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ):
        super().__init__(model_name=model_name, **kwargs)

        self.model_name = model_name
        self.model_path = model_path or os.path.join("models", re.sub(r'[^a-zA-Z0-9\-/]', '_', model_name))
        self.model_kwargs = {**self.model_kwargs, **(model_kwargs or {})}
        self.generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        self._initialize_model()

    def _initialize_model(self):
        # Set device-specific kwargs
        if self.device == "cpu":
            self.model_kwargs["torch_dtype"] = torch.float32
            self.model_kwargs["device_map"] = None

        # Download and save model if it doesn't exist
        get_model = not os.path.exists(self.model_path)

        # Create models directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)

        log.info(f"{'Downl' if get_model else 'L'}oading model {'to' if get_model else 'from'} {self.model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name if get_model else self.model_path,
            **self.model_kwargs,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name if get_model else self.model_path)
        if get_model:
            # noinspection PyUnresolvedReferences
            self.model.save_pretrained(self.model_path)
            # noinspection PyUnresolvedReferences
            self.tokenizer.save_pretrained(self.model_path)

    def unload(self):
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None

        # Force garbage collection
        import gc
        gc.collect()

        # Clear PyTorch's CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def _llm_type(self) -> str:
        return "streaming_transformers"

    def _convert_messages_to_text(self, messages: List[BaseMessage]) -> str:
        """Convert a list of messages to a single text string using the model's chat template."""
        formatted_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                formatted_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                formatted_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                formatted_messages.append({"role": "assistant", "content": message.content})
            elif isinstance(message, ToolMessage):
                formatted_messages.append({"role": "tool", "content": message.content})

        return self.tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _stream_generate(
            self,
            prompt: str,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Tuple[TextIteratorStreamer, Thread, str]:
        """Shared implementation for streaming generation used by both _call and stream methods.
        
        Args:
            prompt: The prompt to send to the model.
            run_manager: Callback manager for LLM.
            **kwargs: Additional arguments to pass to generation.
            
        Returns:
            A tuple containing (streamer, thread, eos_token) for the caller to use.
        """
        # Create inputs - ensure input is a string
        if not isinstance(prompt, str):
            raise ValueError(
                "Prompt must be a string. If you're using message objects, convert them with _convert_messages_to_text first.")

        # noinspection PyCallingNonCallable
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(torch.device(self.device))

        # Setup streamer
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)

        # Get special token
        # noinspection PyUnresolvedReferences
        eos_token = self.tokenizer.eos_token

        # Prepare generation kwargs
        generation_kwargs = {
            **model_inputs,
            "streamer": streamer,
            **self.generation_kwargs,
            **kwargs,
        }

        # Start generation in a separate thread
        # noinspection PyUnresolvedReferences
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        return streamer, thread, eos_token

    def stream(
            self,
            input: LanguageModelInput,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[str]:
        """Stream the tokens of the response as they are generated.
        
        Args:
            input: Either a string prompt or a list of messages.
            stop: A list of strings to stop generation when encountered.
            run_manager: Callback manager for LLM.
            **kwargs: Additional arguments to pass to call.
            
        Yields:
            The token strings as they are generated.
        """
        # Convert messages to text
        prompt = self._convert_input(input)
        prompt = self._convert_messages_to_text(prompt.to_messages())

        # Get stream components
        streamer, thread, eos_token = self._stream_generate(prompt, run_manager, **kwargs)

        # Stream the output
        for new_text in streamer:
            if new_text.endswith(eos_token):
                new_text = new_text[:-len(eos_token)]
            if new_text != eos_token:
                if run_manager:
                    run_manager.on_llm_new_token(new_text)
                yield new_text

        thread.join()

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        # Get stream components
        streamer, thread, eos_token = self._stream_generate(prompt, run_manager, **kwargs)

        # Process the streamed output
        generated_text = ""
        for new_text in streamer:
            if new_text.endswith(eos_token):
                new_text = new_text[:-len(eos_token)]
            if new_text != eos_token:
                if run_manager:
                    run_manager.on_llm_new_token(new_text)
                generated_text += new_text

        thread.join()
        return generated_text

    async def _astream_generate(
            self,
            prompt: str,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Asynchronous version of streaming text generation.
        
        Args:
            prompt: The prompt to send to the model.
            run_manager: Callback manager for LLM.
            **kwargs: Additional arguments to pass to generation.
            
        Yields:
            The token strings as they are generated.
        """
        # Create a queue to transfer data from the synchronous thread to the async context
        queue = asyncio.Queue()

        # Create a function that will put data into the queue
        def threaded_generation():
            try:
                streamer, thread, eos_token = self._stream_generate(prompt, run_manager, **kwargs)

                for new_text in streamer:
                    if new_text.endswith(eos_token):
                        new_text = new_text[:-len(eos_token)]
                    if new_text != eos_token:
                        if run_manager:
                            run_manager.on_llm_new_token(new_text)
                        asyncio.run_coroutine_threadsafe(queue.put(new_text), loop)

                thread.join()
                # Signal we're done
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)
            except Exception as e:
                asyncio.run_coroutine_threadsafe(queue.put(e), loop)

        # Get the event loop
        loop = asyncio.get_event_loop()

        # Start the generation thread
        Thread(target=threaded_generation).start()

        # Yield items from the queue as they become available
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    async def astream(
            self,
            input: LanguageModelInput,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Asynchronously stream the tokens of the response as they are generated.
        
        Args:
            input: Either a string prompt or a list of messages.
            stop: A list of strings to stop generation when encountered.
            run_manager: Callback manager for LLM.
            **kwargs: Additional arguments to pass to call.
            
        Yields:
            The token strings as they are generated.
        """
        # Convert messages to text
        prompt = self._convert_input(input)
        prompt = self._convert_messages_to_text(prompt.to_messages())

        async for token in self._astream_generate(prompt, run_manager, **kwargs):
            yield token


# Example usage
if __name__ == "__main__":
    from langchain.schema import HumanMessage, SystemMessage
    from langchain.callbacks import StreamingStdOutCallbackHandler

    # Initialize the model
    model = TransformersModel(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        # model_name="Qwen/Qwen2.5-1.5B-Instruct",
        # model_name="deepseek-ai/deepseek-coder-1.3b-instruct",
        # model_name="microsoft/DialoGPT-large",
    )

    # Example messages
    messages = [
        SystemMessage(
            content="You are a helpful assistant capable of helping with a variety of tasks and questions related to Bible translation. Always provide short and accurate response. If needed, format your responses in markdown for better readability"),
        HumanMessage(content="Can you explain what textual criticism is?"),
        AIMessage(
            content="Textual criticism is the study of manuscripts and their variations to determine the most accurate version of a text. It's particularly important in biblical studies."),
        HumanMessage(content="What are its main goals?"),
        AIMessage(
            content="The main goals of textual criticism are to identify and correct errors in texts, reconstruct the original text, and understand the history of its transmission."),
        HumanMessage(content="What is the role of textual criticism?"),
    ]

    print("\n--- Example 1: Using callbacks for streaming ---")

    # Generate with callbacks streaming
    response = model(
        model._convert_messages_to_text(messages),
        callbacks=[StreamingStdOutCallbackHandler()]
    )

    print("\n--- Example 2: Using custom stream method ---")
    # Use our custom streaming method
    full_response = ""
    for chunk in model.stream(messages):
        print(chunk, end="", flush=True)
        full_response += chunk

    # Example 3: Async streaming
    print("\n--- Example 3: Using astream method ---")


    async def run_async_example():
        full_response = ""
        async for chunk in model.astream(messages):
            print(chunk, end="", flush=True)
            full_response += chunk


    # Run the async example
    asyncio.run(run_async_example())
