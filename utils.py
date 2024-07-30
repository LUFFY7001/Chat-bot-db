import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
import os

def record_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please say something...")
        audio_data = recognizer.listen(source)
        print("Recording complete.")
        with open(file_path, "wb") as audio_file:
            audio_file.write(audio_data.get_wav_data())

def play_audio(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        play(audio)
    except Exception as e:
        print(f"Error playing audio: {e}")
