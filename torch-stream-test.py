import torch
from torch import nn

import numpy as np
from tqdm import tqdm

model1 = nn.Sequential(
    nn.Linear(10, 100),
    nn.Linear(100, 1000),
    nn.Linear(1000, 1)
).to('cuda')

model2 = nn.Sequential(
    nn.Linear(10, 100),
    nn.Linear(100, 1000),
    nn.Linear(1000, 1)
).to('cuda')

model3 = nn.Sequential(
    nn.Linear(10, 100),
    nn.Linear(100, 1000),
    nn.Linear(1000, 1)
).to('cuda')

def train(model: nn.Module, iterations):
  loss_fn = nn.MSELoss(reduction='mean')
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

  device = next(model.parameters()).device

  for iter in range(iterations):
    X = torch.rand((64, 10))
    y = torch.rand((64, 1))

    X = X.to(device)
    y = y.to(device)

    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_fn(y, y_pred)

    loss.backward()
    optimizer.step()

def train_parallel(models: list[nn.Module], iterations):
  loss_fn = nn.MSELoss(reduction='mean')
  optimizers = [torch.optim.SGD(model.parameters(), lr=1e-5) for model in models]
  device = next(models[0].parameters()).device

  streams = [torch.cuda.Stream() for _ in models]

  for iter in range(iterations):
    X = torch.rand((64, 10))
    y = torch.rand((64, 1))

    X = X.to(device)
    y = y.to(device)

    for model, optimizer, stream in zip(models, optimizers, streams):

      with torch.cuda.stream(stream):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y, y_pred)

        loss.backward()
        optimizer.step()

elapsed_times = []
for _ in tqdm(range(100)):

  model1 = nn.Sequential(
      nn.Linear(10, 100),
      nn.Linear(100, 1000),
      nn.Linear(1000, 1)
  ).to('cuda')

  model2 = nn.Sequential(
      nn.Linear(10, 100),
      nn.Linear(100, 1000),
      nn.Linear(1000, 1)
  ).to('cuda')

  model3 = nn.Sequential(
      nn.Linear(10, 100),
      nn.Linear(100, 1000),
      nn.Linear(1000, 1)
  ).to('cuda')

  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)
  start_event.record()

  train(model1, 1000)
  train(model2, 1000)
  train(model3, 1000)

  end_event.record()
  torch.cuda.synchronize()  # Wait for the events to be recorded!
  elapsed_time_ms = start_event.elapsed_time(end_event)
  elapsed_times.append(elapsed_time_ms)

print()
print('Fully sequential:')
print(np.mean(elapsed_times))
print(np.std(elapsed_times))

elapsed_times = []
for _ in tqdm(range(100)):
  s1 = torch.cuda.Stream()
  s2 = torch.cuda.Stream()
  s3 = torch.cuda.Stream()

  model1 = nn.Sequential(
      nn.Linear(10, 100),
      nn.Linear(100, 1000),
      nn.Linear(1000, 1)
  ).to('cuda')

  model2 = nn.Sequential(
      nn.Linear(10, 100),
      nn.Linear(100, 1000),
      nn.Linear(1000, 1)
  ).to('cuda')

  model3 = nn.Sequential(
      nn.Linear(10, 100),
      nn.Linear(100, 1000),
      nn.Linear(1000, 1)
  ).to('cuda')

  torch.cuda.synchronize()

  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)
  start_event.record()

  with torch.cuda.stream(s1):
    train(model1, 1000)

  with torch.cuda.stream(s2):
    train(model2, 1000)

  with torch.cuda.stream(s3):
    train(model3, 1000)

  end_event.record()
  torch.cuda.synchronize()  # Wait for the events to be recorded!
  elapsed_time_ms = start_event.elapsed_time(end_event)
  elapsed_times.append(elapsed_time_ms)

  del s1
  del s2
  del s3

print()
print('Coarse parallel:')
print(np.mean(elapsed_times))
print(np.std(elapsed_times))

elapsed_times = []
for _ in tqdm(range(100)):

  model1 = nn.Sequential(
      nn.Linear(10, 100),
      nn.Linear(100, 1000),
      nn.Linear(1000, 1)
  ).to('cuda')

  model2 = nn.Sequential(
      nn.Linear(10, 100),
      nn.Linear(100, 1000),
      nn.Linear(1000, 1)
  ).to('cuda')

  model3 = nn.Sequential(
      nn.Linear(10, 100),
      nn.Linear(100, 1000),
      nn.Linear(1000, 1)
  ).to('cuda')

  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)
  start_event.record()

  train_parallel([model1, model2, model3], 1000)

  end_event.record()
  torch.cuda.synchronize()  # Wait for the events to be recorded!
  elapsed_time_ms = start_event.elapsed_time(end_event)
  elapsed_times.append(elapsed_time_ms)

print()
print('Fine parallel:')
print(np.mean(elapsed_times))
print(np.std(elapsed_times))

