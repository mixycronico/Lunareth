import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
import os

def create_dummy_model():
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 3)
    os.makedirs("corec/models/mobilev3", exist_ok=True)
    torch.save(model.state_dict(), "corec/models/mobilev3/model.pth")

if __name__ == "__main__":
    create_dummy_model()
