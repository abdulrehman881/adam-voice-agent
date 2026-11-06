# Adam Voice Agent

A streaming AI voice agent you can talk to in the browser: press the mic,
speak, and Adam listens, thinks, and answers with a natural voice.

- **Speech-to-text:** Groq `whisper-large-v3-turbo`
- **Reasoning:** Groq `openai/gpt-oss-20b` (streaming)
- **Speech synthesis:** ElevenLabs `eleven_flash_v2_5`
- **Backend:** FastAPI with server-sent events for low-latency audio streaming

The whole conversation history is kept in memory per browser session and
expires after one hour of inactivity. No database, no external calls to
phone networks: it works purely microphone to speaker.

## Live demo flow

1. Open `/ui`
2. Press the microphone button and speak (or type and press **Ask Adam**)
3. The browser shows your transcript while Adam streams back sentence-sized
   audio chunks, so the reply starts before the full answer is generated

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/ui` | Voice interface page |
| GET | `/health` | Readiness check |
| POST | `/voice` | Text message to spoken MP3 reply |
| POST | `/voice-chat` | Uploaded audio to transcript + spoken MP3 reply |
| POST | `/stream-voice` | Text message streamed as sentence-sized audio chunks (SSE) |

`/stream-voice` emits these SSE events: `{"type": "audio", "text": ...,
"audio": "<base64 mp3>"}` repeated per sentence, then `{"type": "done"}`,
with `{"type": "error", "message": ...}` on failure.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file (see `.env.example`) with:

- `GROQ_API_KEY` - from console.groq.com
- `ELEVENLABS_API_KEY` - from elevenlabs.io
- `ADAM_VOICE_ID` - optional, defaults to the included public voice

## Run

```bash
uvicorn main:app --port 8000
```

Then open http://127.0.0.1:8000/ui

## Notes

- Voice recording uses the browser MediaRecorder API; microphone permission
  is required.
- The `/next-carrier` endpoint is a stand-alone compatibility stub so the
  original dispatch-edition UI works unchanged; in this project it only
  returns an empty carrier selection.
- Audio files are returned from temporary storage and deleted immediately
  after each response.
