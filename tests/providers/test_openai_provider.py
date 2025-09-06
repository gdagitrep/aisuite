"""Tests for OpenAI provider functionality."""

import io
from unittest.mock import MagicMock, mock_open, patch

import pytest

from aisuite.providers.openai_provider import OpenaiProvider
from aisuite.provider import ASRError
from aisuite.framework.message import (
    TranscriptionResult,
    TranscriptionOptions,
    StreamingTranscriptionChunk,
    Segment,
    Word,
)


@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")


@pytest.fixture
def openai_provider():
    """Create an OpenAI provider instance for testing."""
    return OpenaiProvider()


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI API response for ASR."""
    mock_response = MagicMock()
    mock_response.text = "Hello, this is a test transcription."
    mock_response.language = "en"
    mock_response.segments = None
    return mock_response


class TestOpenAIProvider:
    """Test suite for OpenAI provider functionality."""

    def test_provider_initialization(self, openai_provider):
        """Test that OpenAI provider initializes correctly."""
        assert openai_provider is not None
        assert hasattr(openai_provider, "client")
        assert hasattr(openai_provider, "audio")
        assert hasattr(openai_provider.audio, "transcriptions")


class TestOpenAIASR:
    """Test suite for OpenAI ASR functionality."""

    def test_audio_transcriptions_create_success(
        self, openai_provider, mock_openai_response
    ):
        """Test successful audio transcription."""
        with patch(
            "builtins.open", mock_open(read_data=b"fake audio data")
        ), patch.object(
            openai_provider.client.audio.transcriptions,
            "create",
            return_value=mock_openai_response,
        ):
            result = openai_provider.audio.transcriptions.create(
                model="openai:whisper-1", file="test_audio.mp3"
            )

            assert isinstance(result, TranscriptionResult)
            assert result.text == "Hello, this is a test transcription."
            assert result.language == "en"

    def test_audio_transcriptions_create_with_file_object(
        self, openai_provider, mock_openai_response
    ):
        """Test audio transcription with file-like object."""
        audio_data = io.BytesIO(b"fake audio data")

        with patch.object(
            openai_provider.client.audio.transcriptions,
            "create",
            return_value=mock_openai_response,
        ):
            result = openai_provider.audio.transcriptions.create(
                model="openai:whisper-1", file=audio_data
            )

            assert isinstance(result, TranscriptionResult)
            assert result.text == "Hello, this is a test transcription."

    def test_audio_transcriptions_create_with_kwargs(
        self, openai_provider, mock_openai_response
    ):
        """Test audio transcription with additional parameters."""
        with patch(
            "builtins.open", mock_open(read_data=b"fake audio data")
        ), patch.object(
            openai_provider.client.audio.transcriptions,
            "create",
            return_value=mock_openai_response,
        ) as mock_create:
            result = openai_provider.audio.transcriptions.create(
                model="openai:whisper-1",
                file="test_audio.mp3",
                language="en",
                temperature=0.5,
            )

            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["language"] == "en"
            assert call_kwargs["temperature"] == 0.5
            assert isinstance(result, TranscriptionResult)
            assert result.text == "Hello, this is a test transcription."

    def test_audio_transcriptions_create_with_options(
        self, openai_provider, mock_openai_response
    ):
        """Test audio transcription with TranscriptionOptions."""
        options = TranscriptionOptions(
            language="en",
            include_word_timestamps=True,
            enable_automatic_punctuation=True,
        )

        with patch(
            "builtins.open", mock_open(read_data=b"fake audio data")
        ), patch.object(
            openai_provider.client.audio.transcriptions,
            "create",
            return_value=mock_openai_response,
        ) as mock_create:
            result = openai_provider.audio.transcriptions.create(
                model="openai:whisper-1", file="test_audio.mp3", options=options
            )

            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs
            assert "language" in call_kwargs
            assert isinstance(result, TranscriptionResult)
            assert result.text == "Hello, this is a test transcription."

    def test_audio_transcriptions_create_error_handling(self, openai_provider):
        """Test error handling for API failures."""
        with patch(
            "builtins.open", mock_open(read_data=b"fake audio data")
        ), patch.object(
            openai_provider.client.audio.transcriptions,
            "create",
            side_effect=Exception("API Error"),
        ):
            with pytest.raises(ASRError, match="OpenAI transcription error: API Error"):
                openai_provider.audio.transcriptions.create(
                    model="openai:whisper-1", file="test_audio.mp3"
                )

    @pytest.mark.asyncio
    async def test_audio_transcriptions_create_stream_output(self, openai_provider):
        """Test streaming audio transcription."""
        mock_chunk = MagicMock()
        mock_chunk.text = "Hello"
        mock_chunk.confidence = 0.95

        with patch(
            "builtins.open", mock_open(read_data=b"fake audio data")
        ), patch.object(
            openai_provider.client.audio.transcriptions,
            "create",
            return_value=[mock_chunk],
        ):
            result = openai_provider.audio.transcriptions.create_stream_output(
                model="openai:whisper-1", file="test_audio.mp3"
            )

            chunks = []
            async for chunk in result:
                chunks.append(chunk)
                break

            assert len(chunks) == 1
            assert isinstance(chunks[0], StreamingTranscriptionChunk)
            assert chunks[0].text == "Hello"

    def test_parse_openai_response_with_segments_and_words(self, openai_provider):
        """Test parsing OpenAI response with segments and words."""
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_response.language = "en"

        mock_segment = MagicMock()
        mock_segment.id = 0
        mock_segment.seek = 0
        mock_segment.start = 0.0
        mock_segment.end = 2.5
        mock_segment.text = "Hello world"
        mock_segment.words = []
        mock_response.segments = [mock_segment]

        result = openai_provider.audio.transcriptions._parse_openai_response(
            mock_response
        )

        assert result.text == "Hello world"
        assert len(result.segments) == 1
        assert isinstance(result.segments[0], Segment)

    def test_parse_openai_response_empty(self, openai_provider):
        """Test parsing response with minimal data."""
        mock_response = MagicMock()
        mock_response.text = "Test"
        mock_response.language = "en"
        mock_response.segments = None

        result = openai_provider.audio.transcriptions._parse_openai_response(
            mock_response
        )

        assert result.text == "Test"
        assert result.language == "en"
