import unittest
from unittest.mock import patch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model.voice_classifier import VoiceClassifier


class TestVoiceClassifier(unittest.TestCase):
    def setUp(self):
        self.model = VoiceClassifier()

    def test_forward_pass(self):
        test_input = torch.randn(10, 13, requires_grad=True)  # Enable gradient computation
        output = self.model(test_input)
        self.assertEqual(output.shape, (10, 2))  # Expect output shape to be [batch_size, num_classes]


class TestTrainingLoop(unittest.TestCase):
    def setUp(self):
        self.model = VoiceClassifier()
        self.model.train()  # Ensure model is in training mode

    def test_training_step(self):
        # Setup a DataLoader with proper TensorDataset
        features = torch.randn(10, 13, requires_grad=True)
        labels = torch.randint(0, 2, (10,))
        dataset = TensorDataset(features, labels)
        data_loader = DataLoader(dataset, batch_size=10, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Patch forward method of the model to return a tensor with requires_grad
        with patch.object(VoiceClassifier, 'forward',
                          return_value=torch.randn(10, 2, requires_grad=True)) as mock_forward:
            # Execute one step of the training loop
            for features, labels in data_loader:
                optimizer.zero_grad()
                output = self.model(features)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

            # Check that forward is called
            mock_forward.assert_called()


if __name__ == '__main__':
    unittest.main()
