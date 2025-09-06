import os
import asyncio
from typing import Union, BinaryIO, Optional, AsyncGenerator

from aisuite.provider import Provider, ASRError, Audio
from aisuite.framework.message import (
    TranscriptionResult,
    Segment,
    Word,
    Alternative,
    Channel,
    TranscriptionOptions,
    StreamingTranscriptionChunk,
)
from aisuite.framework.parameter_mapper import ParameterMapper


class DeepgramProvider(Provider):
    """Deepgram ASR provider."""

    def __init__(self, **config):
        """Initialize the Deepgram provider with the given configuration."""
        super().__init__()

        # Ensure API key is provided either in config or via environment variable
        self.api_key = config.get("api_key") or os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Deepgram API key is missing. Please provide it in the config or set the DEEPGRAM_API_KEY environment variable."
            )

        # Initialize Deepgram client
        try:
            from deepgram import DeepgramClient, DeepgramClientOptions

            client_options = DeepgramClientOptions(
                api_key=self.api_key, options={"keepalive": "true"}
            )
            self.client = DeepgramClient(self.api_key, client_options)
        except ImportError:
            raise ImportError(
                "Deepgram SDK is required. Install it with: pip install deepgram-sdk"
            )

        # Initialize audio functionality
        self.audio = DeepgramAudio(self.client)

    def chat_completions_create(self, model, messages):
        """Deepgram does not support chat completions."""
        raise NotImplementedError(
            "Deepgram provider only supports audio transcription, not chat completions."
        )


# Audio Classes
class DeepgramAudio(Audio):
    """Deepgram Audio functionality container."""

    def __init__(self, client):
        super().__init__()
        self.transcriptions = self.Transcriptions(client)

    class Transcriptions(Audio.Transcription):
        """Deepgram Audio Transcriptions functionality."""

        def __init__(self, client):
            self.client = client

        def create(
            self,
            model: str,
            file: Union[str, BinaryIO],
            options: Optional[TranscriptionOptions] = None,
            **kwargs,
        ) -> TranscriptionResult:
            """Create audio transcription using Deepgram SDK."""
            try:
                from deepgram import PrerecordedOptions

                api_params = self._prepare_api_params(model, options, kwargs)
                deepgram_options = PrerecordedOptions(**api_params)
                payload = self._prepare_audio_payload(file)

                response = self.client.listen.rest.v("1").transcribe_file(
                    payload, deepgram_options
                )

                response_dict = (
                    response.to_dict() if hasattr(response, "to_dict") else response
                )

                return self._parse_deepgram_response(response_dict)

            except Exception as e:
                raise ASRError(f"Deepgram transcription error: {e}")

        async def create_stream_output(
            self,
            model: str,
            file: Union[str, BinaryIO],
            options: Optional[TranscriptionOptions] = None,
            **kwargs,
        ) -> AsyncGenerator[StreamingTranscriptionChunk, None]:
            """Create streaming audio transcription using Deepgram SDK."""
            try:
                from deepgram import LiveOptions

                api_params = self._prepare_api_params(model, options, kwargs)
                live_options = LiveOptions(**api_params)
                connection = self.client.listen.websocket.v("1")

                transcript_queue = asyncio.Queue()
                finished_event = asyncio.Event()

                def on_message(self, result, **kwargs):
                    sentence = result.channel.alternatives[0].transcript
                    if sentence:
                        chunk = StreamingTranscriptionChunk(
                            text=sentence,
                            is_final=result.is_final,
                            confidence=getattr(
                                result.channel.alternatives[0], "confidence", None
                            ),
                        )
                        asyncio.create_task(transcript_queue.put(chunk))

                def on_error(self, error, **kwargs):
                    asyncio.create_task(
                        transcript_queue.put(
                            ASRError(f"Deepgram streaming error: {error}")
                        )
                    )
                    finished_event.set()

                def on_close(self, close, **kwargs):
                    finished_event.set()

                connection.on(connection.event.TRANSCRIPT_RECEIVED, on_message)
                connection.on(connection.event.ERROR, on_error)
                connection.on(connection.event.CLOSE, on_close)

                if not connection.start(live_options):
                    raise ASRError("Failed to start Deepgram streaming connection")

                audio_data = self._read_audio_data(file)
                await self._send_audio_chunks(connection, audio_data)
                connection.finish()

                async for chunk in self._yield_transcription_chunks(
                    transcript_queue, finished_event
                ):
                    yield chunk

            except Exception as e:
                raise ASRError(f"Deepgram streaming transcription error: {e}")

        def _extract_model_name(self, model: str) -> str:
            """Use model name directly (client already extracted it)."""
            return model

        def _prepare_api_params(
            self, model: str, options: Optional[TranscriptionOptions], kwargs: dict
        ) -> dict:
            """Prepare API parameters for Deepgram."""
            if options is not None:
                api_params = ParameterMapper.map_to_deepgram(options)
            else:
                api_params = self._map_openai_to_deepgram_params(kwargs)

            model_name = self._extract_model_name(model)
            api_params.setdefault("smart_format", True)
            api_params.setdefault("punctuate", True)
            api_params.setdefault("language", "en")
            api_params["model"] = model_name
            return api_params

        def _prepare_audio_payload(self, file: Union[str, BinaryIO]) -> dict:
            """Prepare audio payload for Deepgram API."""
            if isinstance(file, str):
                with open(file, "rb") as audio_file:
                    buffer_data = audio_file.read()
            else:
                if hasattr(file, "read"):
                    buffer_data = file.read()
                else:
                    raise ValueError(
                        "File must be a file path string or file-like object"
                    )
            return {"buffer": buffer_data}

        def _read_audio_data(self, file: Union[str, BinaryIO]) -> bytes:
            """Read audio data from file or file-like object."""
            if isinstance(file, str):
                with open(file, "rb") as audio_file:
                    return audio_file.read()
            else:
                return file.read()

        async def _send_audio_chunks(self, connection, audio_data: bytes) -> None:
            """Send audio data in chunks to Deepgram connection."""
            chunk_size = 8192
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                connection.send(chunk)
                await asyncio.sleep(0.01)

        async def _yield_transcription_chunks(
            self, transcript_queue: asyncio.Queue, finished_event: asyncio.Event
        ) -> AsyncGenerator[StreamingTranscriptionChunk, None]:
            """Yield transcription chunks as they arrive."""
            while not finished_event.is_set():
                try:
                    chunk = await asyncio.wait_for(transcript_queue.get(), timeout=1.0)
                    if isinstance(chunk, Exception):
                        raise chunk
                    yield chunk
                except asyncio.TimeoutError:
                    continue

        def _map_openai_to_deepgram_params(self, openai_params: dict) -> dict:
            """Map OpenAI-style parameters to Deepgram parameters."""
            deepgram_params = {}

            if "language" in openai_params:
                deepgram_params["language"] = openai_params["language"]
            if "prompt" in openai_params:
                deepgram_params["keywords"] = openai_params["prompt"]
            if "timestamp_granularities" in openai_params:
                granularities = openai_params["timestamp_granularities"]
                if "word" in granularities:
                    deepgram_params["punctuate"] = True
                    deepgram_params["utterances"] = True

            return deepgram_params

        def _parse_deepgram_response(self, response_dict: dict) -> TranscriptionResult:
            """Convert Deepgram API response to unified TranscriptionResult."""
            try:
                results = response_dict.get("results", {})
                channels = results.get("channels", [])

                if not channels or not channels[0].get("alternatives"):
                    return TranscriptionResult(
                        text="", language=None, confidence=None, task="transcribe"
                    )

                best_alternative = channels[0]["alternatives"][0]
                text = best_alternative.get("transcript", "")
                confidence = best_alternative.get("confidence", None)

                words = [
                    Word(
                        word=word_data.get("word", ""),
                        start=word_data.get("start", None),
                        end=word_data.get("end", None),
                        confidence=word_data.get("confidence", None),
                    )
                    for word_data in best_alternative.get("words", [])
                ]

                segments = []
                paragraphs = results.get("paragraphs", {}).get("paragraphs", [])
                for para in paragraphs:
                    for sentence in para.get("sentences", []):
                        segments.append(
                            Segment(
                                id=len(segments),
                                seek=0,
                                start=sentence.get("start", None),
                                end=sentence.get("end", None),
                                text=sentence.get("text", ""),
                                tokens=[],
                                temperature=0.0,
                                avg_logprob=0.0,
                                compression_ratio=0.0,
                                no_speech_prob=0.0,
                            )
                        )

                alternatives_list = [
                    Alternative(
                        transcript=alt.get("transcript", ""),
                        confidence=alt.get("confidence", None),
                    )
                    for alt in channels[0]["alternatives"][1:]
                ]

                channels_list = [
                    Channel(
                        alternatives=[
                            Alternative(
                                transcript=alt.get("transcript", ""),
                                confidence=alt.get("confidence", None),
                            )
                            for alt in channel.get("alternatives", [])
                        ]
                    )
                    for channel in channels
                ]

                metadata = response_dict.get("metadata", {})

                return TranscriptionResult(
                    text=text,
                    language=results.get("language", None),
                    confidence=confidence,
                    task="transcribe",
                    duration=metadata.get("duration", None) if metadata else None,
                    segments=segments or None,
                    words=words or None,
                    channels=channels_list or None,
                    alternatives=alternatives_list or None,
                    utterances=results.get("utterances", []),
                    paragraphs=results.get("paragraphs", None),
                    topics=results.get("topics", []),
                    intents=results.get("intents", []),
                    sentiment=results.get("sentiment", None),
                    summary=results.get("summary", None),
                    metadata=metadata,
                )

            except (KeyError, TypeError, IndexError) as e:
                raise ASRError(f"Error parsing Deepgram response: {e}")
