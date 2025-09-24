import os
from torch.nn import CrossEntropyLoss

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECIMAL = 2
SEEDS = list(range(5))
TASK_LOSS = CrossEntropyLoss()

API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjYjlhZjMxMS1mZjgyLTQ4Y2YtYmY5ZC1mMjVjOWU2YmI4YWMifQ=="  # insert your token to use neptune
