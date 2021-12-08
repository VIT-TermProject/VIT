import torch
import torch.nn as nn
import numpy as np

class Trainer():
    def __init__(self, args, model, train_loader, valid_loader):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def train(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            print('Device name:', torch.cuda.get_device_name(0))

        else:
            print('No GPU available, using the CPU instead.')
            device = torch.device("cpu")

        self.model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)