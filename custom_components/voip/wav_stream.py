"""Incremental WAV parsing for streamed TTS audio.

Not part of the upstream integration. TTS audio is generated while it is being
read; parsing the header from the first bytes lets the caller forward audio
immediately instead of waiting for the whole file.
"""

_PCM_FORMATS = (1, 0xFFFE)  # PCM, WAVE_FORMAT_EXTENSIBLE
_UNKNOWN_SIZES = (0, 0xFFFFFFFF)  # written by encoders that cannot seek back


class WavPcmStream:
    """Extract PCM sample frames from a WAV byte stream as it arrives."""

    def __init__(self, rate: int, width: int, channels: int) -> None:
        """Expect audio with the given sample rate, sample width and channels."""
        self._expected = (rate, width, channels)
        self._frame_bytes = width * channels
        self._buffer = bytearray()
        self._in_data = False
        # Bytes left in the data chunk; None if the header does not say
        self._remaining: int | None = None

    def feed(self, data: bytes) -> bytes:
        """Add stream bytes and return the whole sample frames that are ready.

        Returns nothing until the header is complete. A partial sample at the
        end of a chunk is kept for the next call.
        """
        self._buffer += data
        if not self._in_data and not self._read_header():
            return b""

        usable = len(self._buffer)
        if self._remaining is not None:
            usable = min(usable, self._remaining)
        usable -= usable % self._frame_bytes

        pcm = bytes(self._buffer[:usable])
        del self._buffer[:usable]
        if self._remaining is not None:
            self._remaining -= usable
            if self._remaining < self._frame_bytes:
                # Anything after the audio data is not audio
                self._buffer.clear()
        return pcm

    def _read_header(self) -> bool:
        """Parse RIFF chunks up to the audio data; False if more bytes are needed."""
        buffer = self._buffer
        if len(buffer) < 12:
            return False
        if buffer[0:4] != b"RIFF" or buffer[8:12] != b"WAVE":
            raise ValueError("TTS audio is not a WAV stream")

        position = 12
        audio_format: tuple[int, int, int] | None = None
        while True:
            if len(buffer) < position + 8:
                return False
            chunk_id = bytes(buffer[position : position + 4])
            size = int.from_bytes(buffer[position + 4 : position + 8], "little")
            body = position + 8

            if chunk_id == b"data":
                if audio_format is None:
                    raise ValueError("WAV stream has no format before its audio data")
                if audio_format != self._expected:
                    rate, width, channels = self._expected
                    raise ValueError(
                        f"Expected rate/width/channels as {rate}/{width}/{channels},"
                        f" got {audio_format[0]}/{audio_format[1]}/{audio_format[2]}"
                    )
                self._remaining = None if size in _UNKNOWN_SIZES else size
                self._in_data = True
                del buffer[:body]
                return True

            if len(buffer) < body + size:
                return False
            if chunk_id == b"fmt ":
                if size < 16:
                    raise ValueError("WAV format chunk is too short")
                format_code = int.from_bytes(buffer[body : body + 2], "little")
                if format_code not in _PCM_FORMATS:
                    raise ValueError(f"Only PCM WAV audio is supported, got format {format_code}")
                channels = int.from_bytes(buffer[body + 2 : body + 4], "little")
                rate = int.from_bytes(buffer[body + 4 : body + 8], "little")
                bits = int.from_bytes(buffer[body + 14 : body + 16], "little")
                audio_format = (rate, bits // 8, channels)
            # Chunks are padded to an even length
            position = body + size + (size & 1)
