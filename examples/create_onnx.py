import torch
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3,16,3)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(16,32,3)
        self.relu2 = nn.ReLU()

        self.fc = None

    def _init_fc(self, x):
        if self.fc is None:
            dummy = x.detach().clone()
            with torch.no_grad():
                dummy = self.conv1(dummy)
                dummy = self.relu1(dummy)
                dummy = self.conv2(dummy)
                dummy = self.relu2(dummy)
                self.fc = nn.Linear(dummy.flatten(1).shape[1], 10)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = x.flatten(1)

        x = self.fc(x)

        return x


model = TestModel().eval()

dummy = torch.randn(
    1,3,224,224
)

model._init_fc(dummy)

torch.onnx.export(
    model,
    dummy,
    "test.onnx",
    opset_version=17
)