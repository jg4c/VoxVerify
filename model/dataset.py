import torch
import pandas as pd
import os
from torch.utils.data import Dataset
from utils.audio_processor import load_audio_file, extract_features
import json


def filepath():
    if os.path.isfile('config.json'):
        path = 'config.json'
    else:
        path = './../config.json'
    return path


def load_config(path=filepath()):
    with open(path, 'r') as file:
        return json.load(file)


config = load_config()


class VoiceDataset(Dataset):
    def __init__(self, annotations_file, audio_dirs):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dirs = audio_dirs
        print("Columns in DataFrame:", self.annotations.columns)
        print("First few rows of DataFrame:", self.annotations.head())

    def __getitem__(self, index):
        audio_filename = self.annotations.iloc[index]['Filename '].strip()  # To strip any accidental whitespace
        audio_type = self.annotations.iloc[index]['Type']
        label = int(self.annotations.iloc[index]['Label'])
        file_path = os.path.join(self.audio_dirs[audio_type], audio_filename)
        audio, sample_rate = load_audio_file(file_path)
        features = extract_features(audio, sample_rate)
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return features, label

    def __len__(self):
        return len(self.annotations)


annotations_file = config['data']['annotations_file']
audio_dirs = config['data']['audio_dirs']
