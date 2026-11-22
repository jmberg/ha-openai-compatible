"""Text to speech support for OpenAI."""

from collections.abc import Mapping
import logging
from typing import TYPE_CHECKING, Any, Literal, override
from urllib.parse import urlsplit, urlunsplit
from httpx import HTTPError

from openai import OpenAIError
from propcache.api import cached_property

from homeassistant.components.tts import (
    ATTR_PREFERRED_FORMAT,
    ATTR_VOICE,
    TextToSpeechEntity,
    TtsAudioType,
    Voice,
)
from homeassistant.config_entries import ConfigSubentry
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.const import CONF_PROMPT
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import (
    CONF_CHAT_MODEL,
    CONF_TTS_SPEED,
    RECOMMENDED_TTS_SPEED,
    DEFAULT_TTS_RESPONSE_FORMAT,
    CONF_TTS_RESPONSE_FORMAT,
    CONF_BASE_URL,
    CONF_TTS_BASE_URL,
    PLACEHOLDER_API_KEY,
)
from .entity import OpenAIBaseLLMEntity

if TYPE_CHECKING:
    from . import OpenAIConfigEntry

_LOGGER = logging.getLogger(__name__)

PARALLEL_UPDATES = 0


def _get_speakers_url(base_url: str) -> str:
    """Build the speaker discovery URL from the configured API URL."""
    parsed = urlsplit(base_url.rstrip("/"))
    path = parsed.path.rstrip("/")

    if path.endswith("/v1"):
        path = path[:-3]

    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            f"{path}/v2/speakers",
            "",
            "",
        )
    )


async def _async_get_supported_voices(
    hass: HomeAssistant,
    base_url: str,
    tts_base_url: str | None,
    api_key: str | None,
) -> list[Voice] | None:
    """Discover supported voices from an OpenAI-compatible TTS server."""
    url = _get_speakers_url(tts_base_url or base_url)
    headers = {}

    if api_key and api_key != PLACEHOLDER_API_KEY:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        session = get_async_client(hass)
        response = await session.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        speakers = response.json().get("speakers", [])

        voices = [
            Voice(speaker["id"], speaker["label"])
            for speaker in speakers
            if isinstance(speaker, dict)
            and isinstance(speaker.get("id"), str)
            and isinstance(speaker.get("label"), str)
        ]

        if voices:
            _LOGGER.info("Discovered %d TTS voices", len(voices))
            return voices

    except (HTTPError, ValueError, TypeError, AttributeError):
        _LOGGER.debug("TTS voice discovery failed for %s", url, exc_info=True)

    return None


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: OpenAIConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up TTS entities."""

    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "tts":
            continue

        supported_voices = await _async_get_supported_voices(
            hass,
            config_entry.data[CONF_BASE_URL],
            subentry.data.get(CONF_TTS_BASE_URL),
            config_entry.data.get(CONF_API_KEY),
        )

        async_add_entities(
            [
                OpenAITTSEntity(
                    config_entry,
                    subentry,
                    supported_voices,
                )
            ],
            config_subentry_id=subentry.subentry_id,
        )

class OpenAITTSEntity(TextToSpeechEntity, OpenAIBaseLLMEntity):
    """OpenAI TTS entity."""

    _attr_supported_options = [ATTR_VOICE, ATTR_PREFERRED_FORMAT]
    # https://platform.openai.com/docs/guides/text-to-speech#supported-languages
    # The model may also generate the audio in different
    # languages but with lower quality
    _attr_supported_languages = [
        "af-ZA",  # Afrikaans
        "ar-SA",  # Arabic
        "hy-AM",  # Armenian
        "az-AZ",  # Azerbaijani
        "be-BY",  # Belarusian
        "bs-BA",  # Bosnian
        "bg-BG",  # Bulgarian
        "ca-ES",  # Catalan
        "zh-CN",  # Chinese (Mandarin)
        "hr-HR",  # Croatian
        "cs-CZ",  # Czech
        "da-DK",  # Danish
        "nl-NL",  # Dutch
        "en-US",  # English
        "et-EE",  # Estonian
        "fi-FI",  # Finnish
        "fr-FR",  # French
        "gl-ES",  # Galician
        "de-DE",  # German
        "el-GR",  # Greek
        "he-IL",  # Hebrew
        "hi-IN",  # Hindi
        "hu-HU",  # Hungarian
        "is-IS",  # Icelandic
        "id-ID",  # Indonesian
        "it-IT",  # Italian
        "ja-JP",  # Japanese
        "kn-IN",  # Kannada
        "kk-KZ",  # Kazakh
        "ko-KR",  # Korean
        "lv-LV",  # Latvian
        "lt-LT",  # Lithuanian
        "mk-MK",  # Macedonian
        "ms-MY",  # Malay
        "mr-IN",  # Marathi
        "mi-NZ",  # Maori
        "ne-NP",  # Nepali
        "no-NO",  # Norwegian
        "fa-IR",  # Persian
        "pl-PL",  # Polish
        "pt-PT",  # Portuguese
        "ro-RO",  # Romanian
        "ru-RU",  # Russian
        "sr-RS",  # Serbian
        "sk-SK",  # Slovak
        "sl-SI",  # Slovenian
        "es-ES",  # Spanish
        "sw-KE",  # Swahili
        "sv-SE",  # Swedish
        "fil-PH",  # Tagalog (Filipino)
        "ta-IN",  # Tamil
        "th-TH",  # Thai
        "tr-TR",  # Turkish
        "uk-UA",  # Ukrainian
        "ur-PK",  # Urdu
        "vi-VN",  # Vietnamese
        "cy-GB",  # Welsh
    ]
    # Unused, but required by base class.
    # The models detect the input language automatically.
    _attr_default_language = "en-US"

    # https://platform.openai.com/docs/guides/text-to-speech#voice-options
    _default_supported_voices = [
        Voice(voice.lower(), voice)
        for voice in (
            "Marin",
            "Cedar",
            "Alloy",
            "Ash",
            "Ballad",
            "Coral",
            "Echo",
            "Fable",
            "Nova",
            "Onyx",
            "Sage",
            "Shimmer",
            "Verse",
        )
    ]

    _supported_formats = ["mp3", "opus", "aac", "flac", "wav", "pcm"]

    _attr_has_entity_name = False

    def __init__(
        self,
        entry: OpenAIConfigEntry,
        subentry: ConfigSubentry,
        supported_voices: list[Voice] | None = None,
    ) -> None:
        """Initialize the entity."""
        super().__init__(entry, subentry)

        self._attr_name = subentry.title
        self._supported_voices = (
            supported_voices or self._default_supported_voices
        )

    @callback
    @override
    def async_get_supported_voices(self, language: str) -> list[Voice]:
        """Return a list of supported voices for a language."""
        return self._supported_voices

    @cached_property
    @override
    def default_options(self) -> Mapping[str, Any]:
        """Return a mapping with the default options."""
        return {
            ATTR_VOICE: self._supported_voices[0].voice_id,
            ATTR_PREFERRED_FORMAT: self.subentry.data.get(
                CONF_TTS_RESPONSE_FORMAT,
                DEFAULT_TTS_RESPONSE_FORMAT
            ),
        }

    @override
    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any]
    ) -> TtsAudioType:
        """Load tts audio file from the engine."""

        options = {**self.subentry.data, **options}
        client = self.entry.runtime_data

        response_format = options[ATTR_PREFERRED_FORMAT]
        if response_format in ("ogg", "oga"):
            codec: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "opus"
        elif response_format == "raw":
            response_format = codec = "pcm"
        elif response_format not in self._supported_formats:
            response_format = self.default_options[ATTR_PREFERRED_FORMAT]
            codec = response_format
        else:
            codec = response_format

        try:
            async with client.audio.speech.with_streaming_response.create(
                model=options[CONF_CHAT_MODEL],
                voice=options[ATTR_VOICE],
                input=message,
#                instructions=str(options.get(CONF_PROMPT)),
#                speed=options.get(CONF_TTS_SPEED, RECOMMENDED_TTS_SPEED),
                response_format=codec,
            ) as response:
                response_data = bytearray()
                async for chunk in response.iter_bytes():
                    response_data.extend(chunk)
        except OpenAIError as exc:
            _LOGGER.exception("Error during TTS")
            raise HomeAssistantError(exc) from exc

        return response_format, bytes(response_data)
