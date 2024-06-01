import torch
import torch.nn as nn
import torch.optim as optim

import math
import time

# Using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out

if __name__ == '__main__' :
    x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]], device=device)
    y_train = torch.tensor([[0], [0], [1], [1]], dtype=torch.float, device=device)

    model = LogisticRegressionModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    num_epochs = 1000
    start = time.time()
    for epoch in range(num_epochs):
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    end = time.time()
    print(f"{end - start: .5f}")
    with torch.no_grad():
        predicted = model(torch.tensor([[1.0], [2.0], [3.0], [4.0]], device=device)).round()
        print(f'Predicted: {predicted.squeeze().cpu().numpy()}, Actual: [0, 0, 1, 1]')
