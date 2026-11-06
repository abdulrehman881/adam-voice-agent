import os
import time
import re
import json
import base64
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from groq import Groq
from elevenlabs.client import ElevenLabs
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing from the project .env file")

groq_client = Groq(api_key=GROQ_API_KEY)

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not ELEVENLABS_API_KEY:
    raise RuntimeError(
        "ELEVENLABS_API_KEY is missing from the project .env file"
    )

elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

ADAM_VOICE_ID = os.getenv("ADAM_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
ADAM_SYSTEM_PROMPT = (
    "You are Adam, a friendly AI voice assistant. Keep responses "
    "concise and natural: at most two short sentences (under 30 "
    "words) per reply. Ask one question at a time. Use conversation "
    "history to understand follow-up questions. Never invent "
    "information or claims and never promise real-world actions."
)

app = FastAPI(
    title="Adam Voice Agent API",
    description="Streaming AI voice agent: Groq speech-to-text and "
    "LLM reasoning plus ElevenLabs speech synthesis.",
    version="1.0.0",
)


class TextVoiceRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class StreamVoiceRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str = Field(min_length=1, max_length=100)


HISTORY_MAX_MESSAGES = 20
HISTORY_TTL_SECONDS = 60 * 60

conversation_histories: dict[str, dict] = {}


def get_history(conversation_id: str) -> list[dict[str, str]]:
    """Return a valid, recently active history list for this conversation."""
    entry = conversation_histories.get(conversation_id)
    now = time.monotonic()

    if entry is None or now - entry["last_active"] > HISTORY_TTL_SECONDS:
        entry = {"messages": [], "last_active": now}
        conversation_histories[conversation_id] = entry

        expired = [
            key
            for key, item in conversation_histories.items()
            if now - item["last_active"] > HISTORY_TTL_SECONDS
        ]
        for key in expired:
            conversation_histories.pop(key, None)

    entry["last_active"] = now
    return entry["messages"]


def save_history(conversation_id: str, messages: list[dict[str, str]]):
    """Persist bounded history for this conversation."""
    entry = conversation_histories.get(conversation_id)

    if entry is None:
        entry = {"messages": [], "last_active": time.monotonic()}
        conversation_histories[conversation_id] = entry

    entry["messages"] = messages[-HISTORY_MAX_MESSAGES:]
    entry["last_active"] = time.monotonic()


def make_sse_event(data: dict) -> str:
    """Format one server-sent event as JSON."""
    return f"data: {json.dumps(data)}\n\n"


def extract_sentence(text: str):
    """Release short phrases early to reduce first-audio latency."""
    match = re.search(r'[.!?](?:["\']?)(?:\s|$)', text)
    if match:
        return match.end()

    if len(text) >= 32:
        split_at = text.rfind(" ", 0, 32)
        if split_at > 20:
            return split_at

    return None


@app.get("/health")
def health():
    """Simple readiness check."""
    return {
        "status": "healthy",
        "service": "AdamVoiceAgent",
    }


@app.post("/voice")
def voice(request: TextVoiceRequest):
    """Turn one text message into a spoken MP3 reply."""
    try:
        result = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": ADAM_SYSTEM_PROMPT},
                {"role": "user", "content": request.message},
            ],
            temperature=0.3,
        )

        reply = result.choices[0].message.content
        if not reply:
            raise HTTPException(
                status_code=502,
                detail="The AI returned an empty response.",
            )

        reply = reply.strip()

        audio = elevenlabs_client.text_to_speech.convert(
            text=reply,
            voice_id=ADAM_VOICE_ID,
            model_id="eleven_flash_v2_5",
            output_format="mp3_44100_128",
        )

        with tempfile.NamedTemporaryFile(
            suffix=".mp3", delete=False
        ) as temp_file:
            for chunk in audio:
                temp_file.write(chunk)
            audio_path = temp_file.name

        return FileResponse(
            audio_path,
            media_type="audio/mpeg",
            filename="adam_response.mp3",
            background=BackgroundTask(os.remove, audio_path),
        )

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Voice generation failed. Check your API keys, "
            "provider limits, and network connection.",
        )


@app.post("/voice-chat")
async def voice_chat(audio_file: UploadFile = File(...)):
    """Full voice turn: transcribe audio, reason, and speak back."""
    audio_path = None

    try:
        audio_bytes = await audio_file.read()

        if not audio_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded audio file is empty.",
            )

        if len(audio_bytes) > 15 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Audio file must be 15 MB or smaller.",
            )

        filename = audio_file.filename or "recording.webm"

        transcription = groq_client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model="whisper-large-v3-turbo",
            response_format="text",
        )

        transcript = str(transcription).strip()
        print("STEP 1 OK: Whisper transcription:", transcript)

        if not transcript:
            raise HTTPException(
                status_code=400,
                detail="No speech could be detected in the recording.",
            )

        result = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": ADAM_SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            temperature=0.3,
        )

        reply = result.choices[0].message.content

        if not reply:
            raise HTTPException(
                status_code=502,
                detail="The AI returned an empty response.",
            )

        reply = reply.strip()
        print("STEP 2 OK: Groq generated reply:", reply)

        print("STEP 3: Calling ElevenLabs...")
        audio = elevenlabs_client.text_to_speech.convert(
            text=reply,
            voice_id=ADAM_VOICE_ID,
            model_id="eleven_flash_v2_5",
            output_format="mp3_44100_128",
        )

        with tempfile.NamedTemporaryFile(
            suffix=".mp3", delete=False
        ) as temp_file:
            for chunk in audio:
                temp_file.write(chunk)

            audio_path = temp_file.name

        print("STEP 4 OK: ElevenLabs returned audio.")

        return FileResponse(
            audio_path,
            media_type="audio/mpeg",
            filename="adam_voice_chat.mp3",
            headers={
                "X-Transcript": quote(transcript, safe=""),
                "X-Assistant-Reply": quote(reply, safe=""),
            },
            background=BackgroundTask(os.remove, audio_path),
        )

    except HTTPException:
        raise

    except Exception as exc:
        print(f"Voice chat error: {exc}")
        raise HTTPException(
            status_code=502,
            detail=(
                "Voice chat failed. Check the terminal for the "
                "underlying error and verify provider access and limits."
            ),
        )

    finally:
        await audio_file.close()


@app.post("/stream-voice")
def stream_voice(request: StreamVoiceRequest):
    """Stream a conversational reply sentence by sentence with audio.

    Emits server-sent events: audio chunks (base64 MP3) followed by a
    final done event. Conversation history is in-memory per
    conversation_id and expires after one hour of inactivity.
    """

    def generate():
        buffer = ""

        try:
            history = get_history(request.conversation_id)

            messages = [
                {
                    "role": "system",
                    "content": ADAM_SYSTEM_PROMPT,
                },
                *history[-HISTORY_MAX_MESSAGES:],
                {"role": "user", "content": request.message},
            ]

            assistant_reply_parts = []

            llm_stream = groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                temperature=0.3,
                reasoning_effort="low",
                max_completion_tokens=300,
                stream=True,
                messages=messages,
            )

            def speak_chunk(text: str):
                """Generate one playable audio chunk."""
                audio_stream = elevenlabs_client.text_to_speech.stream(
                    text=text,
                    voice_id=ADAM_VOICE_ID,
                    model_id="eleven_flash_v2_5",
                    output_format="mp3_44100_128",
                    optimize_streaming_latency=3,
                )

                audio_bytes = b"".join(audio_stream)

                if audio_bytes:
                    return {
                        "type": "audio",
                        "text": text,
                        "audio": base64.b64encode(
                            audio_bytes
                        ).decode("ascii"),
                    }

                return None

            for event in llm_stream:
                if not event.choices:
                    continue

                delta = event.choices[0].delta.content

                if not delta:
                    continue

                assistant_reply_parts.append(delta)

                buffer += delta

                while True:
                    cut = extract_sentence(buffer)
                    if cut is None:
                        break

                    sentence = buffer[:cut].strip()
                    buffer = buffer[cut:].lstrip()

                    if sentence:
                        audio_event = speak_chunk(sentence)
                        if audio_event:
                            yield make_sse_event(audio_event)

            if buffer.strip():
                audio_event = speak_chunk(buffer.strip())
                if audio_event:
                    yield make_sse_event(audio_event)

            assistant_reply = "".join(assistant_reply_parts).strip()

            if assistant_reply:
                history.append({
                    "role": "user",
                    "content": request.message,
                })
                history.append({
                    "role": "assistant",
                    "content": assistant_reply,
                })

                save_history(request.conversation_id, history)

            print("STREAM COMPLETED")

            yield make_sse_event({"type": "done"})

        except Exception as exc:
            print(f"Streaming voice error: {exc}")
            yield make_sse_event({
                "type": "error",
                "message": (
                    "Adam could not finish the response. "
                    "Check the server terminal."
                ),
            })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/next-carrier", include_in_schema=False)
def next_carrier(conversation_id: str = ""):
    """Compatibility stub: the standalone agent has no dispatch queue.

    The voice UI calls this once per session; returning an empty
    selection keeps the original UI code working unchanged.
    """
    return {
        "carrier": None,
        "source": "standalone",
        "conversation_id": conversation_id,
    }


@app.get("/ui", include_in_schema=False)
def voice_interface():
    ui_path = PROJECT_ROOT / "static" / "voice.html"

    if not ui_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Voice interface file was not found.",
        )

    return FileResponse(ui_path, media_type="text/html")
