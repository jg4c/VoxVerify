import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.dataset import VoiceDataset
import matplotlib.pyplot as plt
import seaborn as sns


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


class VoiceClassifier(nn.Module):
    def __init__(self):
        super(VoiceClassifier, self).__init__()
        self.fc1 = nn.Linear(13, 256)  # 13 MFCC features as input
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)  # Output layer: 2 classes (Human, AI generated)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    annotations_file = config['data']['annotations_file']
    audio_dirs = config['data']['audio_dirs']

    # Create dataset and dataloader
    voice_dataset = VoiceDataset(annotations_file, audio_dirs)
    dataloader = DataLoader(voice_dataset, batch_size=config['model']['batch_size'], shuffle=True)

    # Initialize model
    model = VoiceClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['model']['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # Training loop
    num_epochs = config['model']['num_epochs']
    for epoch in range(num_epochs):
        for features, labels in dataloader:
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    # Save the model
    model_path = config['model']['path']
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')
    print('Training Complete')

    # Feature importance visualization (new addition)
    # Select a sample from the dataset
    features, _ = next(iter(dataloader))
    features.requires_grad = True

    # Forward pass
    model.eval()
    output = model(features)
    model.train()

    # Compute the loss for the most confident class
    score_max_index = output.argmax(1)
    score_max = output[torch.arange(output.shape[0]), score_max_index]
    score_max.sum().backward()

    # Extract the gradients
    gradients = features.grad.data.abs().mean(dim=0)  # Average over the batch

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(gradients.numpy().reshape(1, -1), annot=True, cmap='viridis', yticklabels=False)
    plt.title('Feature Importance')
    plt.xlabel('MFCC Features')
    plt.show()
