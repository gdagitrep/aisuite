"""Tests for client functionality."""

import io
from unittest.mock import Mock, MagicMock, patch
import pytest
from aisuite import Client
from aisuite.framework.message import TranscriptionResult
from aisuite.provider import ASRError


@pytest.fixture(scope="module")
def provider_configs():
    return {
        "openai": {"api_key": "test_openai_api_key"},
        "mistral": {"api_key": "test_mistral_api_key"},
        "groq": {"api_key": "test_groq_api_key"},
        "aws": {
            "access_key_id": "test_access_key_id",
            "secret_access_key": "test_secret_access_key",
            "region_name": "us-east-1",
        },
        "azure": {
            "api_key": "test_azure_api_key",
            "azure_endpoint": "https://test.openai.azure.com/",
            "api_version": "2024-02-01",
        },
        "anthropic": {"api_key": "test_anthropic_api_key"},
        "google": {
            "project_id": "test-project",
            "location": "us-central1",
            "credentials": "test_credentials.json",
        },
        "fireworks": {"api_key": "test_fireworks_api_key"},
        "nebius": {"api_key": "test_nebius_api_key"},
        "inception": {"api_key": "test_inception_api_key"},
        "deepgram": {"api_key": "deepgram-api-key"},
    }


@pytest.fixture
def client_with_config(provider_configs):
    client = Client()
    client.configure(provider_configs)
    return client


@pytest.fixture
def configured_client():
    """Create a configured client for testing."""
    client = Client()
    client.configure(
        {
            "openai": {"api_key": "test-openai-key"},
            "deepgram": {"api_key": "test-deepgram-key"},
        }
    )
    return client


@pytest.fixture
def mock_transcription_result():
    """Create a mock transcription result."""
    return TranscriptionResult(
        text="Hello, this is a test transcription.",
        language="en",
        confidence=0.95,
        task="transcribe",
    )


# Existing chat completion tests
@pytest.mark.parametrize(
    "provider_method_path,provider_key,model",
    [
        (
            "aisuite.providers.openai_provider.OpenaiProvider.chat_completions_create",
            "openai",
            "gpt-4o",
        ),
        (
            "aisuite.providers.mistral_provider.MistralProvider.chat_completions_create",
            "mistral",
            "mistral-model",
        ),
        (
            "aisuite.providers.groq_provider.GroqProvider.chat_completions_create",
            "groq",
            "groq-model",
        ),
        (
            "aisuite.providers.aws_provider.AwsProvider.chat_completions_create",
            "aws",
            "claude-v3",
        ),
        (
            "aisuite.providers.azure_provider.AzureProvider.chat_completions_create",
            "azure",
            "azure-model",
        ),
        (
            "aisuite.providers.anthropic_provider.AnthropicProvider.chat_completions_create",
            "anthropic",
            "anthropic-model",
        ),
        (
            "aisuite.providers.google_provider.GoogleProvider.chat_completions_create",
            "google",
            "google-model",
        ),
        (
            "aisuite.providers.fireworks_provider.FireworksProvider.chat_completions_create",
            "fireworks",
            "fireworks-model",
        ),
        (
            "aisuite.providers.nebius_provider.NebiusProvider.chat_completions_create",
            "nebius",
            "nebius-model",
        ),
        (
            "aisuite.providers.inception_provider.InceptionProvider.chat_completions_create",
            "inception",
            "mercury",
        ),
    ],
)
def test_client_chat_completions(
    client_with_config, provider_method_path, provider_key, model
):
    user_greeting = "Hello!"
    message_history = [{"role": "user", "content": user_greeting}]
    response_text_content = "mocked-text-response-from-model"

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = response_text_content

    with patch(provider_method_path) as mock_create:
        mock_create.return_value = mock_response

        response = client_with_config.chat_completions_create(
            model=f"{provider_key}:{model}",
            messages=message_history,
            temperature=0.75,
        )

        mock_create.assert_called_once_with(
            model=model, messages=message_history, temperature=0.75
        )

        assert response.choices[0].message.content == response_text_content


def test_invalid_provider_in_client_config():
    client = Client()
    invalid_config = {"invalid_provider": {"api_key": "test"}}

    client.configure(invalid_config)
    # Should not raise an error during configuration


def test_invalid_model_format_in_create(client_with_config):
    with pytest.raises(ValueError, match="Invalid model format"):
        client_with_config.chat_completions_create(
            model="invalid_format", messages=[{"role": "user", "content": "Hello"}]
        )


class TestClientASR:
    """Test suite for Client ASR functionality."""

    def test_audio_interface_initialization(self):
        """Test that Audio interface is properly initialized."""
        client = Client()
        assert hasattr(client, "audio")
        assert hasattr(client.audio, "transcriptions")
        assert client.audio.transcriptions.client == client

    def test_transcriptions_create_openai_success(
        self, configured_client, mock_transcription_result
    ):
        """Test successful transcription with OpenAI provider."""
        with patch(
            "aisuite.providers.openai_provider.OpenaiProvider.audio_transcriptions_create",
            return_value=mock_transcription_result,
        ) as mock_create:
            result = configured_client.audio.transcriptions.create(
                model="openai:whisper-1", file="test.mp3", language="en"
            )

            mock_create.assert_called_once_with("whisper-1", "test.mp3", language="en")
            assert result == mock_transcription_result

    def test_transcriptions_create_deepgram_success(
        self, configured_client, mock_transcription_result
    ):
        """Test successful transcription with Deepgram provider."""
        with patch(
            "aisuite.providers.deepgram_provider.DeepgramProvider.audio_transcriptions_create",
            return_value=mock_transcription_result,
        ) as mock_create:
            result = configured_client.audio.transcriptions.create(
                model="deepgram:nova-2", file="test.mp3", diarize=True
            )

            mock_create.assert_called_once_with("nova-2", "test.mp3", diarize=True)
            assert result == mock_transcription_result

    def test_transcriptions_create_with_file_object(
        self, configured_client, mock_transcription_result
    ):
        """Test transcription with file-like object."""
        audio_file = io.BytesIO(b"fake audio data")

        with patch(
            "aisuite.providers.openai_provider.OpenaiProvider.audio_transcriptions_create",
            return_value=mock_transcription_result,
        ) as mock_create:
            result = configured_client.audio.transcriptions.create(
                model="openai:whisper-1", file=audio_file
            )

            mock_create.assert_called_once_with("whisper-1", audio_file)
            assert result == mock_transcription_result

    def test_transcriptions_create_invalid_model_format(self, configured_client):
        """Test error handling for invalid model format."""
        with pytest.raises(
            ValueError, match="Invalid model format. Expected 'provider:model'"
        ):
            configured_client.audio.transcriptions.create(
                model="invalid-model-format", file="test.mp3"
            )

    def test_transcriptions_create_unsupported_provider(self, configured_client):
        """Test error handling for unsupported provider."""
        with pytest.raises(ValueError, match="Invalid provider key 'unsupported'"):
            configured_client.audio.transcriptions.create(
                model="unsupported:model", file="test.mp3"
            )

    def test_transcriptions_create_asr_error_propagation(self, configured_client):
        """Test that ASR errors are properly propagated."""
        with patch(
            "aisuite.providers.openai_provider.OpenaiProvider.audio_transcriptions_create",
            side_effect=ASRError("Test ASR error"),
        ):
            with pytest.raises(ASRError, match="Test ASR error"):
                configured_client.audio.transcriptions.create(
                    model="openai:whisper-1", file="test.mp3"
                )
