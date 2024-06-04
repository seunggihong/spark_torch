from pyspark.sql import SparkSession
import torch
import time

spark = SparkSession.builder.appName("RDD to PyTorch").getOrCreate()

data = [x for x in range(1,6)]
rdd = spark.sparkContext.parallelize(data)

data_list = rdd.collect()

device = torch.device("cuda" if torch.cuda.is_available() else"mps")
print(device)

data_tensor = torch.tensor(data_list)

model = torch.nn.Linear(1, 1).to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

inputs = data_tensor.float().unsqueeze(1).to(device)
targets = data_tensor.float().unsqueeze(1).to(device)

for epoch in range(100):
    outputs = model(inputs)
    
    loss = criterion(outputs, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item()}')
