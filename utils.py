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
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.set_volume(1.0)  # Ensure volume is set to max
        pygame.mixer.music.play()
        st.write("Playing audio...")

        # Wait until the audio is finished playing
        while pygame.mixer.music.get_busy():
            time.sleep(1)
        st.write("Audio playback finished.")
    except Exception as e:
        st.error(f"An error occurred during audio playback: {e}")

