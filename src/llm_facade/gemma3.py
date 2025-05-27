from typing import Any, final

from llama_index.core.llms import (
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from openai import OpenAI
from pydantic import Field, PrivateAttr
from structlog.stdlib import BoundLogger

from llm_facade.llm_config import LLMConfig


@final
class GemaVllm(CustomLLM):
    client: OpenAI
    config: LLMConfig
    last_log: str = Field(default="", description="Last log message")

    __logger: BoundLogger | None = PrivateAttr(default=None)

    def __init__(self, config: LLMConfig, logger: BoundLogger | None = None, *args: Any, **kwargs: Any) -> None:
        client = OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_api_base_url,
        )

        self.__logger = logger
        super().__init__(*args, config=config, client=client, **kwargs)

        print(f"VLLM client initialized {self.config.openai_api_base_url}")

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(model_name=self.config.llm_model)

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        completion = self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        output: str = completion.choices[0].message.content

        self.last_log = completion.choices[0].model_dump_json()

        return CompletionResponse(text=output, raw=completion)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        response = ""

        stream = self.client.completions.create(
            model=self.config.llm_model,
            prompt=prompt,
            max_tokens=100,
            temperature=0.1,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content.replace("ß", "ss")
                response += content
                yield CompletionResponse(text=response, delta=content)
