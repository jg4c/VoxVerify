import sounddevice as sd
import numpy as np


def list_audio_devices():
    print(sd.query_devices())


def capture_audio(duration=5, fs=22050, device=None):
    """Capture audio from the microphone."""
    print("Capturing audio... Press Ctrl+C to stop.")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, device=device, dtype='float32')
    sd.wait()  # Wait's until recording is finished
    return np.squeeze(recording)  # Removes the channel dimension for mono audio


if __name__ == "__main__":
    list_audio_devices()  # Call function to see all devices
    fs = 22050  # The Sample rate
    duration = 10  # Time of the recording in seconds
    device = 1  # Device index from the list
    audio_data = capture_audio(duration, fs, device=device)
    print("Audio captured successfully.")
