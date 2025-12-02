import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
import librosa
import sounddevice as sd
import numpy as np
from threading import Thread
import time
import torch
import torch.nn.functional as F
from model.voice_classifier import VoiceClassifier
from utils.audio_processor import extract_features


def load_model():
    model = VoiceClassifier()
    model.load_state_dict(torch.load('model/model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model


class RecorderApp:
    def __init__(self, master):
        self.master = master
        master.title("Vox Verify")
        master.configure(bg='#306998')  # Setting the background to a Python-like blue color

        # Title Label
        self.title_label = tk.Label(master, text="VoxVerify", font=("Helvetica", 16, "bold"), bg='#306998', fg='Yellow',
                                    underline=True)
        self.title_label.pack(pady=10)

        # Instructions Label
        self.instructions_label = tk.Label(master,
                                           text="Press 'Start Recording' to record your voice or drag and drop an audio file to analyze.",
                                           wraplength=400, justify="center", bg='#306998', fg='white')
        self.instructions_label.pack(pady=10)

        self.model = load_model()  # Load the model
        self.sample_rate = 22050  # Sample rate to match my model's expected input
        self.duration = 10  # Duration of recording in seconds
        self.recording = False
        self.audio_data = None

        # Recording controls
        self.start_button = tk.Button(master, text="Start Recording", command=self.start_recording, bg='green',
                                      fg='black')
        self.start_button.pack()

        self.stop_button = tk.Button(master, text="Stop Recording", command=self.stop_recording, state="disabled",
                                     bg='red', fg='black')
        self.stop_button.pack()

        self.timer_label = tk.Label(master, text="Timer: 0.0s", bg='#306998', fg='white')
        self.timer_label.pack()

        self.result_label = tk.Label(master, text="Result: None", bg='#306998', fg='white')
        self.result_label.pack()

        # Drag and drop file functionality
        self.drop_frame = tk.Frame(master, height=100, bd=2, relief="sunken", bg='#306998')
        self.drop_frame.pack(fill="x", padx=10, pady=10)
        self.drop_label = tk.Label(self.drop_frame, text="Drop audio files here", bg='#306998', fg='white')
        self.drop_label.pack(expand=True, fill="both")

        self.analyze_button = tk.Button(master, text="Analyse Audio", command=self.analyze_dropped_audio,
                                        state="disabled", bg='green', fg='black')
        self.analyze_button.pack()

        # Setup drag and drop
        self.master.drop_target_register(DND_FILES)
        self.master.dnd_bind('<<Drop>>', self.on_drop)

    def start_recording(self):
        self.audio_data = np.zeros((int(self.duration * self.sample_rate),), dtype=np.float32)
        self.recording = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        start_time = time.time()
        self.update_timer(start_time)
        Thread(target=self.record_audio).start()

    def record_audio(self):
        self.audio_data = sd.rec(int(self.duration * self.sample_rate), samplerate=self.sample_rate, channels=1,
                                 dtype='float32')
        sd.wait()

    def stop_recording(self):
        if self.recording:
            sd.stop()
            self.recording = False
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            result_text, confidence = self.analyse_audio(self.audio_data.flatten())
            self.result_label.config(text=f"Result: {result_text} (Confidence: {confidence:.2f}%)")

    def analyse_audio(self, audio_data):
        features = extract_features(audio_data, self.sample_rate)
        features = torch.tensor([features], dtype=torch.float)
        with torch.no_grad():
            logits = self.model(features)
            probabilities = F.softmax(logits, dim=1)
            predicted_label = probabilities.argmax(dim=1)
            confidence = probabilities.max().item()
        return "Human" if predicted_label.item() == 0 else "AI", confidence * 100

    def update_timer(self, start_time):
        if self.recording:
            elapsed_time = time.time() - start_time
            self.timer_label.config(text=f"Timer: {elapsed_time:.2f}s", bg='#306998', fg='white')
            if elapsed_time < self.duration:
                self.master.after(100, self.update_timer, start_time)
            else:
                self.stop_recording()

    def on_drop(self, event):
        # Handle file drop
        self.dropped_file_path = event.data.replace('{', '').replace('}', '')  # Clean up file path
        self.drop_label.config(text=f"File dropped: {self.dropped_file_path}")
        self.analyze_button.config(state="normal")  # Enable the analyze button once a file is dropped

    def analyze_dropped_audio(self):
        try:
            # Load and resample the audio file
            audio_data, _ = librosa.load(self.dropped_file_path, sr=self.sample_rate, mono=True)
            # Normalize audio data if necessary
            audio_data = librosa.util.normalize(audio_data)

            result_text, confidence = self.analyse_audio(audio_data)
            self.result_label.config(text=f"Result: {result_text} (Confidence: {confidence:.2f}%)")
        except Exception as e:
            self.result_label.config(text="Error processing file: " + str(e))
        finally:
            self.analyze_button.config(state="disabled")  # Disable analyze button after processing


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = RecorderApp(root)
    root.mainloop()