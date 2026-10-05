"""Tests for the WAV stream parser. Run with: python -m pytest ha-voip-streaming"""
import importlib.util
import io
import struct
import wave
from pathlib import Path

import pytest

# Load the module by path: importing the package would pull in Home Assistant.
_spec = importlib.util.spec_from_file_location(
    "wav_stream", Path(__file__).resolve().parent.parent / "custom_components" / "voip" / "wav_stream.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
WavPcmStream = _module.WavPcmStream

PCM = bytes(range(256)) * 10  # 2560 bytes, 1280 samples


def header(rate=16000, width=2, channels=1, data_size=0xFFFFFFFF, extra=b"", format_code=1):
    fmt = struct.pack("<HHIIHH", format_code, channels, rate, rate * width * channels, width * channels, width * 8)
    return b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt + extra + b"data" + struct.pack("<I", data_size)


def test_audio_after_the_header_comes_out_unchanged():
    stream = WavPcmStream(16000, 2, 1)

    assert stream.feed(header() + PCM) == PCM


def test_a_file_written_by_the_wave_module_is_read():
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav:
            wav.setframerate(16000)
            wav.setsampwidth(2)
            wav.setnchannels(1)
            wav.writeframes(PCM)
        data = buffer.getvalue()

    assert WavPcmStream(16000, 2, 1).feed(data) == PCM


def test_nothing_comes_out_until_the_header_is_complete():
    stream = WavPcmStream(16000, 2, 1)
    data = header() + PCM

    assert stream.feed(data[:20]) == b""
    assert stream.feed(data[20:43]) == b""
    assert stream.feed(data[43:]) == PCM


def test_a_stream_fed_byte_by_byte_yields_the_same_audio():
    stream = WavPcmStream(16000, 2, 1)
    data = header() + PCM

    out = b"".join(stream.feed(data[i : i + 1]) for i in range(len(data)))

    assert out == PCM


def test_half_a_sample_waits_for_its_other_half():
    stream = WavPcmStream(16000, 2, 1)
    stream.feed(header())

    assert stream.feed(PCM[:5]) == PCM[:4]
    assert stream.feed(PCM[5:9]) == PCM[4:8]
    assert stream.feed(PCM[9:10]) == PCM[8:10]


def test_other_chunks_before_the_audio_are_skipped():
    # ffmpeg writes a LIST chunk with its name before the audio
    info = b"LIST" + struct.pack("<I", 26) + b"INFOISFT" + struct.pack("<I", 14) + b"Lavf61.7.100\x00\x00"
    stream = WavPcmStream(16000, 2, 1)

    assert stream.feed(header(extra=info) + PCM) == PCM


def test_a_chunk_of_odd_length_is_followed_by_a_padding_byte():
    odd = b"note" + struct.pack("<I", 3) + b"abc" + b"\x00"
    stream = WavPcmStream(16000, 2, 1)

    assert stream.feed(header(extra=odd) + PCM) == PCM


def test_a_known_data_size_ends_the_audio():
    stream = WavPcmStream(16000, 2, 1)

    assert stream.feed(header(data_size=100) + PCM[:100] + b"LIST" + bytes(40)) == PCM[:100]
    assert stream.feed(bytes(64)) == b""


def test_a_data_size_of_zero_means_unknown():
    stream = WavPcmStream(16000, 2, 1)

    assert stream.feed(header(data_size=0) + PCM) == PCM


def test_audio_in_another_format_is_refused_with_both_formats_named():
    stream = WavPcmStream(16000, 2, 1)

    with pytest.raises(ValueError, match="16000/2/1.*48000/2/1"):
        stream.feed(header(rate=48000) + PCM)


def test_something_that_is_not_wav_is_refused():
    with pytest.raises(ValueError, match="not a WAV"):
        WavPcmStream(16000, 2, 1).feed(b"ID3\x04" + bytes(40))


def test_compressed_wav_is_refused():
    with pytest.raises(ValueError, match="Only PCM"):
        WavPcmStream(16000, 2, 1).feed(header(format_code=85) + PCM)
