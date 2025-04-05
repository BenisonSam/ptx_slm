from typing import Any, Dict, List, Optional
import os
import re
import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.schema.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage

class StreamingTransformersModel(LLM):
    """A LangChain LLM implementation that streams responses from a HuggingFace model."""
    
    model_name: str  # Required field
    cache_dir: Optional[str] = None
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        cache_dir: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, **kwargs)

        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.join("models", re.sub(r'[^a-zA-Z0-9\-/]', '_', model_name))
        self.model_kwargs = {**self.model_kwargs, **(model_kwargs or {})}
        self.generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        self._initialize_model()
        
    def _initialize_model(self):
        # Set device-specific kwargs
        if self.device.type == "cpu":
            self.model_kwargs["torch_dtype"] = torch.float32
            self.model_kwargs["device_map"] = None
            
        # Create cache directory if needed
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load or download model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            **self.model_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
    
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
        
        return self.tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # Create inputs
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        
        # Setup streamer
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        
        # Get special token
        eos_token = self.tokenizer.eos_token
        
        # Prepare generation kwargs
        generation_kwargs = {
            **model_inputs,
            "streamer": streamer,
            **self.generation_kwargs,
        }
        
        # Start generation in a separate thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
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

# Example usage
if __name__ == "__main__":
    from langchain.schema import HumanMessage, SystemMessage
    from langchain.callbacks import StreamingStdOutCallbackHandler
    
    # Initialize the model
    model = StreamingTransformersModel(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
    )
    
    # Example messages
    messages = [
        SystemMessage(content="You are a helpful assistant capable of helping with a variety of tasks and questions related to Bible translation. Always provide short and accurate response. If needed, format your responses in markdown for better readability"),
        HumanMessage(content="Can you explain what textual criticism is?"),
        AIMessage(content="Textual criticism is the study of manuscripts and their variations to determine the most accurate version of a text. It's particularly important in biblical studies."),
        HumanMessage(content="What are its main goals?"),
        AIMessage(content="The main goals of textual criticism are to identify and correct errors in texts, reconstruct the original text, and understand the history of its transmission."),
        HumanMessage(content="What is the role of textual criticism?"),
    ]
    
    # Convert messages to text
    prompt = model._convert_messages_to_text(messages)
    
    # Generate with streaming
    response = model(
        prompt,
        callbacks=[StreamingStdOutCallbackHandler()]
    )