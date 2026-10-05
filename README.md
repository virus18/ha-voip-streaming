# Voice over IP with streaming TTS

A copy of Home Assistant's built-in [Voice over IP](https://www.home-assistant.io/integrations/voip) integration (core 2026.9.4) with one change: the spoken response is sent to the caller **while it is being generated** instead of after it has been generated completely.

Installed as a custom integration it takes the place of the built-in one. Devices, entities and settings stay the same.

## What is different

The built-in integration reads the whole TTS result into memory and only then starts sending audio. With a TTS engine that needs about as long to synthesize as the audio lasts, the caller waits for the full length of the answer before hearing the first word.

This copy

- starts sending as soon as the pipeline reports that the response is being streamed (`tts_start_streaming`), which can be before the language model has finished writing,
- forwards every audio chunk to the RTP output queue as it arrives, so playback begins with the first chunk and the chunks play back to back.

Everything else is the upstream code. The changes are in `assist_satellite.py` (marked "Streaming fork") and the new `wav_stream.py`.

## Requirements

- Home Assistant 2026.9.x. The copy is taken from 2026.9.4 and pins `voip-utils==0.4.3`, the version that release ships. It uses two private attributes of that library (`_output_audio_queue`, `_pending_audio_events`).
- A TTS engine that streams its output. Without one nothing gets worse, it just is not faster.

After a Home Assistant update that changes the built-in integration, this copy has to be updated as well or removed.

## Install

HACS → three-dot menu → Custom repositories → add this repository as type "Integration" → download "Voice over IP (streaming TTS)" → restart Home Assistant.

## Remove

Remove it in HACS and restart Home Assistant. The built-in integration takes over again.

## License

Apache License 2.0, like Home Assistant. Original code © the Home Assistant authors; see `LICENSE`. This project is not affiliated with or endorsed by Home Assistant.
