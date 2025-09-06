import openai
import os
from typing import Union, BinaryIO, Optional, AsyncGenerator
from aisuite.provider import Provider, LLMError, ASRError, Audio
from aisuite.providers.message_converter import OpenAICompliantMessageConverter
from aisuite.framework.message import (
    TranscriptionResult,
    Segment,
    Word,
    StreamingTranscriptionChunk,
    TranscriptionOptions,
)
from aisuite.framework.parameter_mapper import ParameterMapper


class OpenaiProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the OpenAI provider with the given configuration.
        Pass the entire configuration dictionary to the OpenAI client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("OPENAI_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "OpenAI API key is missing. Please provide it in the config or set the OPENAI_API_KEY environment variable."
            )

        # NOTE: We could choose to remove above lines for api_key since OpenAI will automatically
        # infer certain values from the environment variables.
        # Eg: OPENAI_API_KEY, OPENAI_ORG_ID, OPENAI_PROJECT_ID, OPENAI_BASE_URL, etc.

        # Pass the entire config to the OpenAI client constructor
        self.client = openai.OpenAI(**config)
        self.transformer = OpenAICompliantMessageConverter()

        # Initialize audio functionality
        super().__init__()
        self.audio = OpenAIAudio(self.client)

    def chat_completions_create(self, model, messages, **kwargs):
        # Any exception raised by OpenAI will be returned to the caller.
        # Maybe we should catch them and raise a custom LLMError.
        try:
            transformed_messages = self.transformer.convert_request(messages)
            response = self.client.chat.completions.create(
                model=model,
                messages=transformed_messages,
                **kwargs,  # Pass any additional arguments to the OpenAI API
            )
            return response
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")


# Audio Classes
class OpenAIAudio(Audio):
    """OpenAI Audio functionality container."""

    def __init__(self, client):
        super().__init__()
        self.transcriptions = self.Transcriptions(client)

    class Transcriptions(Audio.Transcription):
        """OpenAI Audio Transcriptions functionality."""

        def __init__(self, client):
            self.client = client

        def create(
            self,
            model: str,
            file: Union[str, BinaryIO],
            options: Optional[TranscriptionOptions] = None,
            **kwargs,
        ) -> TranscriptionResult:
            """Create audio transcription using OpenAI Whisper API."""
            try:
                # Determine parameters to use
                if options is not None:
                    api_params = ParameterMapper.map_to_openai(options)
                else:
                    api_params = kwargs

                # Use the model name directly (client already extracted it)
                model_name = model

                # Handle timestamp_granularities requirement
                if "timestamp_granularities" in api_params:
                    # OpenAI requires verbose_json format for timestamp_granularities
                    api_params["response_format"] = "verbose_json"

                # Handle file input
                if isinstance(file, str):
                    with open(file, "rb") as audio_file:
                        response = self.client.audio.transcriptions.create(
                            file=audio_file, model=model_name, **api_params
                        )
                else:
                    response = self.client.audio.transcriptions.create(
                        file=file, model=model_name, **api_params
                    )

                return self._parse_openai_response(response)

            except Exception as e:
                raise ASRError(f"OpenAI transcription error: {e}")

        async def create_stream_output(
            self,
            model: str,
            file: Union[str, BinaryIO],
            options: Optional[TranscriptionOptions] = None,
            **kwargs,
        ) -> AsyncGenerator[StreamingTranscriptionChunk, None]:
            """Create output streaming audio transcription using OpenAI Whisper API."""
            try:
                # Determine parameters to use
                if options is not None:
                    api_params = ParameterMapper.map_to_openai(options)
                else:
                    api_params = kwargs

                # Use the model name directly (client already extracted it)
                model_name = model
                api_params["stream"] = True

                # Handle timestamp_granularities requirement
                if "timestamp_granularities" in api_params:
                    # OpenAI requires verbose_json format for timestamp_granularities
                    api_params["response_format"] = "verbose_json"

                # Handle file input
                if isinstance(file, str):
                    with open(file, "rb") as audio_file:
                        response_stream = self.client.audio.transcriptions.create(
                            file=audio_file, model=model_name, **api_params
                        )
                else:
                    response_stream = self.client.audio.transcriptions.create(
                        file=file, model=model_name, **api_params
                    )

                # Process streaming response
                for chunk in response_stream:
                    if hasattr(chunk, "text") and chunk.text:
                        yield StreamingTranscriptionChunk(
                            text=chunk.text,
                            is_final=True,
                            confidence=getattr(chunk, "confidence", None),
                        )

            except Exception as e:
                raise ASRError(f"OpenAI streaming transcription error: {e}")

        def _parse_openai_response(self, response) -> TranscriptionResult:
            """Parse OpenAI API response into TranscriptionResult."""
            text = response.text if hasattr(response, "text") else ""
            language = getattr(response, "language", "unknown")

            # Parse segments if available
            segments = []
            if hasattr(response, "segments") and response.segments:
                for seg in response.segments:
                    words = []
                    if hasattr(seg, "words") and seg.words:
                        for word in seg.words:
                            words.append(
                                Word(
                                    word=word.word,
                                    start=word.start,
                                    end=word.end,
                                    confidence=getattr(word, "confidence", None),
                                )
                            )

                    segments.append(
                        Segment(
                            id=getattr(seg, "id", 0),
                            seek=getattr(seg, "seek", 0),
                            text=seg.text,
                            start=seg.start,
                            end=seg.end,
                            words=words,
                            confidence=getattr(seg, "avg_logprob", None),
                        )
                    )

            return TranscriptionResult(
                text=text,
                language=language,
                confidence=getattr(response, "confidence", None),
                segments=segments,
            )
