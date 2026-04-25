import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseClassifier(nn.Module):
    def __init__(self, input_shape: tuple, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        
        # Dropout para evitar overfitting (lição da V2!)
        self.dropout_conv = nn.Dropout2d(0.2)
        self.dropout_fc = nn.Dropout(0.5)

        self._to_linear = None
        self._get_flat_shape(input_shape)
        
        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def _get_flat_shape(self, shape):
        x = torch.randn(1, 1, shape[0], shape[1])
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        self._to_linear = x.numel()

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout_conv(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout_fc(F.relu(self.fc1(x)))
        return self.fc2(x)

class MFCC_Model(BaseClassifier):
    def __init__(self): super().__init__(input_shape=(130, 13))

class Mel_Model(BaseClassifier):
    def __init__(self): super().__init__(input_shape=(130, 64))

class Chroma_Model(BaseClassifier):
    def __init__(self): super().__init__(input_shape=(130, 12))