"""Assist satellite entity for VoIP integration."""

import asyncio
from datetime import timedelta
from enum import IntFlag
from functools import partial
import logging
from pathlib import Path
import socket
import threading
import time
from typing import Any, Final, override

from voip_utils import SIP_PORT, RtpDatagramProtocol
from voip_utils.sip import SipEndpoint, get_sip_endpoint
from voip_utils.voip import AudioOutputChunk

from homeassistant.components import intent, tts
from homeassistant.components.assist_pipeline import PipelineEvent, PipelineEventType
from homeassistant.components.assist_satellite import (
    AssistSatelliteAnnouncement,
    AssistSatelliteConfiguration,
    AssistSatelliteEntity,
    AssistSatelliteEntityDescription,
    AssistSatelliteEntityFeature,
)
from homeassistant.components.intent import TimerEventType, TimerInfo
from homeassistant.components.network import async_get_source_ip
from homeassistant.core import Context, HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import VoipConfigEntry
from .const import (
    CHANNELS,
    CONF_SIP_PORT,
    CONF_SIP_USER,
    DOMAIN,
    RATE,
    RTP_AUDIO_SETTINGS,
    WIDTH,
)
from .devices import VoIPDevice
from .entity import VoIPEntity
from .wav_stream import WavPcmStream

_LOGGER = logging.getLogger(__name__)

_PIPELINE_TIMEOUT_SEC: Final = 30
_HANGUP_SEC: Final = 0.5
_ANNOUNCEMENT_BEFORE_DELAY: Final = 0.5
_ANNOUNCEMENT_AFTER_DELAY: Final = 1.0
_ANNOUNCEMENT_RING_TIMEOUT: Final = 30
# Streaming fork: give up on a TTS stream that stays silent this long
_TTS_CHUNK_TIMEOUT_SEC: Final = 30
# One 20 ms frame of silence, sent last so the final partial frame is flushed
_FLUSH_BYTES: Final = bytes(RATE * WIDTH * CHANNELS // 50)


class Tones(IntFlag):
    """Feedback tones for specific events."""

    LISTENING = 1
    PROCESSING = 2
    ERROR = 4


_TONE_FILENAMES: dict[Tones, str] = {
    Tones.LISTENING: "tone.pcm",
    Tones.PROCESSING: "processing.pcm",
    Tones.ERROR: "error.pcm",
}


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: VoipConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up VoIP Assist satellite entity."""
    domain_data = config_entry.runtime_data.domain_data

    @callback
    def async_add_device(device: VoIPDevice) -> None:
        """Add device."""
        async_add_entities([VoipAssistSatellite(hass, device, config_entry)])

    domain_data.devices.async_add_new_device_listener(async_add_device)

    entities: list[VoIPEntity] = [
        VoipAssistSatellite(hass, device, config_entry)
        for device in domain_data.devices
    ]

    async_add_entities(entities)


class VoipAssistSatellite(VoIPEntity, AssistSatelliteEntity, RtpDatagramProtocol):
    """Assist satellite for VoIP devices."""

    entity_description = AssistSatelliteEntityDescription(key="assist_satellite")
    _attr_translation_key = "assist_satellite"
    _attr_name = None
    _attr_icon = "mdi:phone-classic"
    _attr_supported_features = (
        AssistSatelliteEntityFeature.ANNOUNCE
        | AssistSatelliteEntityFeature.START_CONVERSATION
    )

    def __init__(
        self,
        hass: HomeAssistant,
        voip_device: VoIPDevice,
        config_entry: VoipConfigEntry,
        tones=Tones.LISTENING | Tones.PROCESSING | Tones.ERROR,
    ) -> None:
        """Initialize an Assist satellite."""
        VoIPEntity.__init__(self, voip_device)
        AssistSatelliteEntity.__init__(self)
        RtpDatagramProtocol.__init__(self)

        _LOGGER.debug("Assist satellite with device: %s", voip_device)

        self.config_entry = config_entry

        self._audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._audio_chunk_timeout: float = 2.0
        self._run_pipeline_task: asyncio.Task | None = None
        self._pipeline_had_error: bool = False
        # Streaming fork: the last run heard something but recognized no
        # words (line noise, a cut-off syllable). Listen again without tones.
        self._nothing_recognized: bool = False
        self._quiet_restart: bool = False
        self._tts_done = asyncio.Event()
        self._tts_extra_timeout: float = 1.0
        self._tone_bytes: dict[Tones, bytes] = {}
        self._tones = tones
        self._processing_tone_done = asyncio.Event()
        # Streaming fork: TTS stream of the current pipeline run, and whether
        # it is already being sent to the caller
        self._tts_token: str | None = None
        self._tts_started = False

        self._announcement: AssistSatelliteAnnouncement | None = None
        self._announcment_start_time: float = 0.0
        self._check_announcement_pickup_task: asyncio.Task | None = None
        self._check_hangup_task: asyncio.Task | None = None
        self._call_end_future: asyncio.Future[Any] = asyncio.Future()
        self._last_chunk_time: float | None = None
        self._rtp_port: int | None = None
        self._run_pipeline_after_announce: bool = False

    @property
    @override
    def pipeline_entity_id(self) -> str | None:
        """Return the entity ID of the pipeline to use for the next conversation."""
        return self.voip_device.get_pipeline_entity_id(self.hass)

    @property
    @override
    def vad_sensitivity_entity_id(self) -> str | None:
        """Return the VAD sensitivity entity ID for next conversation."""
        return self.voip_device.get_vad_sensitivity_entity_id(self.hass)

    @property
    @override
    def tts_options(self) -> dict[str, Any] | None:
        """Options passed for text-to-speech."""
        return {
            tts.ATTR_PREFERRED_FORMAT: "wav",
            tts.ATTR_PREFERRED_SAMPLE_RATE: 16000,
            tts.ATTR_PREFERRED_SAMPLE_CHANNELS: 1,
            tts.ATTR_PREFERRED_SAMPLE_BYTES: 2,
        }

    @override
    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass."""
        await super().async_added_to_hass()
        self.voip_device.protocol = self

        assert self.device_entry is not None
        self.async_on_remove(
            intent.async_register_timer_handler(
                self.hass, self.device_entry.id, self.async_handle_timer_event
            )
        )

    @override
    async def async_will_remove_from_hass(self) -> None:
        """Run when entity will be removed from hass."""
        await super().async_will_remove_from_hass()
        assert self.voip_device.protocol == self
        self.voip_device.protocol = None

    @callback
    @override
    def async_get_configuration(
        self,
    ) -> AssistSatelliteConfiguration:
        """Get the current satellite configuration."""
        raise NotImplementedError

    @callback
    def async_handle_timer_event(
        self,
        event_type: TimerEventType,
        timer_info: TimerInfo,
    ) -> None:
        """Handle timer event."""
        if event_type != TimerEventType.FINISHED:
            return

        if timer_info.name:
            message = f"{timer_info.name} finished"
        else:
            message = f"{timedelta(seconds=timer_info.created_seconds)} timer finished"

        async def announce_message():
            announcement = await self._resolve_announcement_media_id(message, None)
            await self.async_announce(announcement)

        self.config_entry.async_create_background_task(
            self.hass, announce_message(), "voip_announce_timer"
        )

    @override
    async def async_set_configuration(
        self, config: AssistSatelliteConfiguration
    ) -> None:
        """Set the current satellite configuration."""
        raise NotImplementedError

    @override
    async def async_announce(self, announcement: AssistSatelliteAnnouncement) -> None:
        """Announce media on the satellite.

        Plays announcement in a loop, blocking until the caller hangs up.
        """
        await self._do_announce(announcement, run_pipeline_after=False)

    async def _do_announce(
        self, announcement: AssistSatelliteAnnouncement, run_pipeline_after: bool
    ) -> None:
        """Announce media on the satellite.

        Optionally run a voice pipeline after the announcement has finished.
        """
        if announcement.media_id_source != "tts":
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="non_tts_announcement",
            )

        self._call_end_future = asyncio.Future()
        self._run_pipeline_after_announce = run_pipeline_after

        if self._rtp_port is None:
            # Choose random port for RTP
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setblocking(False)
            sock.bind(("", 0))
            _rtp_ip, self._rtp_port = sock.getsockname()
            sock.close()

        # HA SIP server
        source_ip = await async_get_source_ip(self.hass)
        sip_port = self.config_entry.options.get(CONF_SIP_PORT, SIP_PORT)
        sip_user = self.config_entry.options.get(CONF_SIP_USER)
        source_endpoint = get_sip_endpoint(
            host=source_ip, port=sip_port, username=sip_user
        )

        try:
            # VoIP ID is SIP header - This represents what is set as the To header
            destination_endpoint = SipEndpoint(self.voip_device.voip_id)
        except ValueError:
            # VoIP ID is IP address
            destination_endpoint = get_sip_endpoint(
                host=self.voip_device.voip_id, port=SIP_PORT
            )

        # Reset state so we can time out if needed
        self._last_chunk_time = None
        self._announcment_start_time = time.monotonic()
        self._announcement = announcement

        # Make the call
        sip_protocol = self.config_entry.runtime_data.domain_data.protocol
        _LOGGER.debug("Outgoing call to contact %s", self.voip_device.contact)
        call_info = sip_protocol.outgoing_call(
            source=source_endpoint,
            destination=destination_endpoint,
            rtp_port=self._rtp_port,
            contact=self.voip_device.contact,
        )

        # Check if caller didn't pick up
        self._check_announcement_pickup_task = (
            self.config_entry.async_create_background_task(
                self.hass,
                self._check_announcement_pickup(),
                "voip_announcement_pickup",
            )
        )

        try:
            await self._call_end_future
        except TimeoutError:
            # Stop ringing
            _LOGGER.debug("Caller did not pick up in time")
            sip_protocol.cancel_call(call_info)
            self.disconnect()
            raise

    async def _check_announcement_pickup(self) -> None:
        """Continuously checks if an audio chunk was received within a time limit.

        If not, the caller is presumed to have not picked
        up the phone and the announcement is ended.
        """
        while True:
            current_time = time.monotonic()
            if (self._last_chunk_time is None) and (
                (current_time - self._announcment_start_time)
                > _ANNOUNCEMENT_RING_TIMEOUT
            ):
                # Ring timeout
                _LOGGER.debug("Ring timeout")
                self._announcement = None
                self._check_announcement_pickup_task = None
                self._call_end_future.set_exception(
                    TimeoutError("User did not pick up in time")
                )
                _LOGGER.debug("Timed out waiting for the user to pick up the phone")
                break
            if self._last_chunk_time is not None:
                _LOGGER.debug("Picked up the phone")
                self._check_announcement_pickup_task = None
                break

            await asyncio.sleep(_HANGUP_SEC / 2)

    async def _check_hangup(self) -> None:
        """Continuously checks if an audio chunk was received within a time limit.

        If not, the caller is presumed to have hung up and the call is ended.
        """
        try:
            while True:
                current_time = time.monotonic()
                if (self._last_chunk_time is not None) and (
                    (current_time - self._last_chunk_time) > _HANGUP_SEC
                ):
                    # Caller hung up
                    _LOGGER.debug("Hang up")
                    self.disconnect()
                    break

                await asyncio.sleep(_HANGUP_SEC / 2)
        except asyncio.CancelledError:
            # Don't swallow cancellation
            if (current_task := asyncio.current_task()) and current_task.cancelling():
                raise
            _LOGGER.debug("Check hangup cancelled")

    @override
    async def async_start_conversation(
        self, start_announcement: AssistSatelliteAnnouncement
    ) -> None:
        """Start a conversation from the satellite."""
        await self._do_announce(start_announcement, run_pipeline_after=True)

    # -------------------------------------------------------------------------
    # VoIP
    # -------------------------------------------------------------------------

    def disconnect(self):
        """Server disconnected."""
        super().disconnect()
        if self._check_announcement_pickup_task is not None:
            self._check_announcement_pickup_task.cancel()
            self._check_announcement_pickup_task = None
        if self._check_hangup_task is not None:
            self._check_hangup_task.cancel()
            self._check_hangup_task = None
        self.voip_device.set_is_active(False)
        self._announcement = None
        if self._run_pipeline_task is not None:
            _LOGGER.debug("Cancelling running pipeline")
            self._run_pipeline_task.cancel()
        if not self._call_end_future.done():
            self._call_end_future.set_result(None)
        self._last_chunk_time = None
        self._rtp_port = None
        _LOGGER.debug("VoIP disconnected")

    def connection_made(self, transport):
        """Server is ready."""
        super().connection_made(transport)
        self.voip_device.set_is_active(True)
        # If announcement isn't set, assume it was an incoming call and
        # initialize the chunk time for checking if the caller hung up
        if self._announcement is None:
            self._last_chunk_time = time.monotonic()
        # Check if caller hung up
        self._check_hangup_task = self.config_entry.async_create_background_task(
            self.hass,
            self._check_hangup(),
            "voip_hangup",
        )

    def on_chunk(self, audio_bytes: bytes) -> None:
        """Handle raw audio chunk."""
        self._last_chunk_time = time.monotonic()

        if self._announcement is None:
            # Pipeline with STT
            if self._run_pipeline_task is None:
                # Run pipeline until voice command finishes, then start over
                self._clear_audio_queue()
                self._tts_done.clear()
                self._run_pipeline_task = (
                    self.config_entry.async_create_background_task(
                        self.hass,
                        self._run_pipeline(),
                        "voip_pipeline_run",
                    )
                )

            self._audio_queue.put_nowait(audio_bytes)
        elif self._run_pipeline_task is None:
            # Announcement only
            # Play announcement (will repeat)
            self._run_pipeline_task = self.config_entry.async_create_background_task(
                self.hass,
                self._play_announcement(self._announcement),
                "voip_play_announcement",
            )

    async def _run_pipeline(self) -> None:
        """Run a pipeline with STT input and TTS output."""
        _LOGGER.debug("Starting pipeline")

        async def stt_stream():
            retry: bool = True
            while True:
                try:
                    async with asyncio.timeout(self._audio_chunk_timeout):
                        chunk = await self._audio_queue.get()
                        if not chunk:
                            _LOGGER.debug("STT stream got None")
                            break

                    yield chunk
                except TimeoutError:
                    _LOGGER.debug("STT Stream timed out")
                    if not retry:
                        _LOGGER.debug("No more retries, ending STT stream")
                        break
                    retry = False

        # Play listening tone at the start of each cycle
        if self._quiet_restart:
            # Streaming fork: not after a run in which nothing was recognized
            self._quiet_restart = False
        else:
            await self._play_tone(Tones.LISTENING, silence_before=0.2)

        try:
            await self.async_accept_pipeline_from_satellite(
                audio_stream=stt_stream(),
                context=Context(user_id=self.config_entry.data["user"]),
            )

            if self._pipeline_had_error:
                _LOGGER.debug("Pipeline error")
                self._pipeline_had_error = False
                if self._nothing_recognized:
                    # Streaming fork: on a phone line this happens often and
                    # is not worth two seconds of error tone during which
                    # the caller cannot be heard.
                    self._nothing_recognized = False
                    self._quiet_restart = True
                else:
                    await self._play_tone(Tones.ERROR)
            else:
                # Block until TTS is done speaking.
                #
                # This is set in _send_tts and has a timeout that's based on the
                # length of the TTS audio.
                await self._tts_done.wait()
        except TimeoutError:
            # This shouldn't happen anymore, we are detecting
            # hang ups with a separate task
            _LOGGER.exception("Timeout error")
            self.disconnect()  # caller hung up
        except asyncio.CancelledError:
            _LOGGER.debug("Pipeline cancelled")
            # Don't swallow cancellation
            if (current_task := asyncio.current_task()) and current_task.cancelling():
                raise
        finally:
            # Stop audio stream
            await self._audio_queue.put(None)

            self._run_pipeline_task = None
            _LOGGER.debug("Pipeline finished")

    async def _play_announcement(
        self, announcement: AssistSatelliteAnnouncement
    ) -> None:
        """Play an announcement once."""
        _LOGGER.debug("Playing announcement")

        if announcement.tts_token is None:
            _LOGGER.error("Only TTS announcements are supported")
            return

        await asyncio.sleep(_ANNOUNCEMENT_BEFORE_DELAY)
        stream = tts.async_get_stream(self.hass, announcement.tts_token)
        if stream is None:
            _LOGGER.error("TTS stream no longer available")
            return

        try:
            await self._send_tts(stream, wait_for_tone=False)
            if not self._run_pipeline_after_announce:
                # Delay before looping announcement
                await asyncio.sleep(_ANNOUNCEMENT_AFTER_DELAY)
        except Exception:
            _LOGGER.exception("Unexpected error while playing announcement")
            self._announcement = None
            raise
        finally:
            self._run_pipeline_task = None
            _LOGGER.debug("Announcement finished")

            if self._run_pipeline_after_announce:
                # Clear announcement to allow pipeline to run
                _LOGGER.debug("Clearing announcement")
                self._announcement = None

    def _clear_audio_queue(self) -> None:
        """Ensure audio queue is empty."""
        while not self._audio_queue.empty():
            self._audio_queue.get_nowait()

    @override
    def on_pipeline_event(self, event: PipelineEvent) -> None:
        """Set state based on pipeline stage."""
        if event.type == PipelineEventType.RUN_START:
            # Streaming fork: remember the TTS stream of this run so playback
            # can start while the response is still being generated
            self._tts_started = False
            self._tts_token = None
            if event.data and (tts_output := event.data.get("tts_output")):
                self._tts_token = tts_output.get("token")
        elif event.type == PipelineEventType.INTENT_PROGRESS:
            # Streaming fork: the response is long enough to be spoken while it
            # is written. Start sending now instead of waiting for TTS_END.
            if (
                event.data
                and event.data.get("tts_start_streaming")
                and not self._tts_started
                and self._tts_token
                and (stream := tts.async_get_stream(self.hass, self._tts_token))
            ):
                self._tts_started = True
                self.config_entry.async_create_background_task(
                    self.hass,
                    self._send_tts(tts_stream=stream),
                    "voip_pipeline_tts",
                )
        elif event.type == PipelineEventType.STT_END:
            if (self._tones & Tones.PROCESSING) == Tones.PROCESSING:
                self._processing_tone_done.clear()
                self.config_entry.async_create_background_task(
                    self.hass, self._play_tone(Tones.PROCESSING), "voip_process_tone"
                )
        elif event.type == PipelineEventType.TTS_END:
            # Send TTS audio to caller over RTP
            if self._tts_started:
                # Streaming fork: this response is already being sent
                _LOGGER.debug("TTS already streaming")
            elif (
                event.data
                and (tts_output := event.data["tts_output"])
                and (stream := tts.async_get_stream(self.hass, tts_output["token"]))
            ):
                self.config_entry.async_create_background_task(
                    self.hass,
                    self._send_tts(tts_stream=stream),
                    "voip_pipeline_tts",
                )
            else:
                # Empty TTS response
                _LOGGER.debug("Empty TTS response")
                self._tts_done.set()
        elif event.type == PipelineEventType.ERROR:
            # Play error tone instead of wait for TTS when pipeline is finished.
            self._pipeline_had_error = True
            if event.data and event.data.get("code") == "stt-no-text-recognized":
                # Streaming fork: handled quietly, see _run_pipeline
                self._nothing_recognized = True
                _LOGGER.debug(event)
            else:
                self._nothing_recognized = False
                _LOGGER.warning(event)

    async def _send_tts(
        self,
        tts_stream: tts.ResultStream,
        wait_for_tone: bool = True,
    ) -> None:
        """Send TTS audio to caller via RTP while it is being generated.

        Streaming fork: upstream reads the whole response before sending it.
        Here every chunk is queued for the RTP output loop as soon as it
        arrives, so the caller hears the first words while the rest is still
        being synthesized.
        """
        try:
            if self.transport is None:
                return  # not connected

            if tts_stream.extension != "wav":
                raise ValueError(
                    f"Only TTS WAV audio can be streamed, got {tts_stream.extension}"
                )

            loop = asyncio.get_running_loop()
            wav = WavPcmStream(RATE, WIDTH, CHANNELS)
            queued_bytes = 0
            first_audio_time: float | None = None

            chunks = tts_stream.async_stream_result()
            while True:
                try:
                    async with asyncio.timeout(_TTS_CHUNK_TIMEOUT_SEC):
                        chunk = await anext(chunks)
                except StopAsyncIteration:
                    break

                if self.transport is None:
                    _LOGGER.debug("Caller hung up during TTS")
                    return

                # TTS audio is 16Khz 16-bit mono
                audio_bytes = wav.feed(chunk)
                if not audio_bytes:
                    continue

                if first_audio_time is None:
                    if wait_for_tone and (
                        (self._tones & Tones.PROCESSING) == Tones.PROCESSING
                    ):
                        # Don't overlap TTS and processing beep
                        _LOGGER.debug("Waiting for processing tone")
                        await self._processing_tone_done.wait()
                    first_audio_time = loop.time()

                self._queue_audio(audio_bytes)
                queued_bytes += len(audio_bytes)

            if first_audio_time is None:
                _LOGGER.debug("TTS stream contained no audio")
                return

            # Flush the last partial frame and wait until everything was sent
            finished = threading.Event()
            self._pending_audio_events.add(finished)
            self._queue_audio(_FLUSH_BYTES, is_end=True, finished=finished)

            tts_seconds = queued_bytes / (RATE * WIDTH * CHANNELS)
            remaining = max(0.0, tts_seconds - (loop.time() - first_audio_time))
            _LOGGER.debug(
                "Queued %.1f second(s) of audio, %.1f second(s) left to send",
                tts_seconds,
                remaining,
            )

            # Time out 1 second after TTS audio should be finished
            if not await self.hass.async_add_executor_job(
                finished.wait, remaining + self._tts_extra_timeout
            ):
                raise TimeoutError
        except TimeoutError:
            _LOGGER.warning("TTS timeout")
            raise
        finally:
            # Update satellite state
            self.tts_response_finished()

            # Signal pipeline to restart
            self._tts_done.set()

    def _queue_audio(
        self,
        audio_bytes: bytes,
        is_end: bool = False,
        finished: threading.Event | None = None,
    ) -> None:
        """Queue audio for the RTP output loop without waiting until it is sent.

        Streaming fork: send_audio() blocks until its audio has been played,
        which leaves a gap before the next chunk. Queued chunks play back to
        back.
        """
        self._output_audio_queue.put_nowait(
            AudioOutputChunk(audio_bytes, RATE, WIDTH, CHANNELS, is_end, finished)
        )

    async def _async_send_audio(self, audio_bytes: bytes, **kwargs):
        """Send audio in executor."""
        await self.hass.async_add_executor_job(
            partial(self.send_audio, audio_bytes, **RTP_AUDIO_SETTINGS, **kwargs)
        )

    async def _play_tone(self, tone: Tones, silence_before: float = 0.0) -> None:
        """Play a tone as feedback to the user if it's enabled."""
        if (self._tones & tone) != tone:
            return  # not enabled

        if tone not in self._tone_bytes:
            # Do I/O in executor
            self._tone_bytes[tone] = await self.hass.async_add_executor_job(
                self._load_pcm,
                _TONE_FILENAMES[tone],
            )

        await self._async_send_audio(
            self._tone_bytes[tone],
            silence_before=silence_before,
        )

        if tone == Tones.PROCESSING:
            self._processing_tone_done.set()

    def _load_pcm(self, file_name: str) -> bytes:
        """Load raw audio (16Khz, 16-bit mono)."""
        return (Path(__file__).parent / file_name).read_bytes()
