import torch
import torch.nn as nn
import torch.optim as optim

# Using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

if __name__ == '__main__' :    
    x_train = torch.tensor([[1.0], [2.0], [3.0]]).to(device)
    y_train = torch.tensor([[2.0], [4.0], [6.0]]).to(device)

    model = LinearRegressionModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    num_epochs = 1000
    for epoch in range(num_epochs):
        inputs = x_train.to(device)
        labels = y_train.to(device)

        predictions = model(inputs)
        loss = criterion(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
