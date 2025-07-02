
# ai_call_agent.py

import sounddevice as sd
import whisper
import openai
import pyttsx3
import numpy as np
import scipy.io.wavfile

# SETTINGS
DURATION = 5  # seconds to record
SAMPLERATE = 16000
openai.api_key = "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"
openai.api_base = "https://openrouter.ai/api/v1"

# Initialize TTS
tts = pyttsx3.init()

def record_audio(filename="input.wav"):
    print("🎙️ Listening... Speak now.")
    recording = sd.rec(int(DURATION * SAMPLERATE), samplerate=SAMPLERATE, channels=1)
    sd.wait()
    audio_data = np.int16(recording * 32767)
    scipy.io.wavfile.write(filename, SAMPLERATE, audio_data)
    print("✅ Recorded")

def transcribe_audio(filename="input.wav"):
    model = whisper.load_model("base")
    result = model.transcribe(filename)
    print("📝 Transcribed:", result["text"])
    return result["text"]

def get_gpt_response(prompt):
    response = openai.ChatCompletion.create(
        model="openai/gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a friendly and persuasive marketing assistant from TrendHive. Your job is to help businesses understand how AI marketing can grow their brand."},
            {"role": "user", "content": prompt}
        ]
    )
    reply = response.choices[0].message.content
    print("🤖 GPT says:", reply)
    return reply

def speak_text(text):
    print("🔊 Speaking...")
    tts.say(text)
    tts.runAndWait()

def main():
    while True:
        record_audio()
        user_input = transcribe_audio()
        if user_input.lower() in ["exit", "quit", "stop"]:
            print("👋 Exiting...")
            break
        response = get_gpt_response(user_input)
        speak_text(response)

if __name__ == "__main__":
    main()
