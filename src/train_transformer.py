from IPython.display import clear_output
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

from torch.utils.data import DataLoader
from torch.optim import Adam

from denoising_transformer import DenoisingTransformer
from datasets import RMSequenceDataset

def train_epoch(model, train_dataloader, optimizer, criterion, device):

    for input_seq, output_seq in train_dataloader:
        input_seq, output_seq = input_seq.to(device), output_seq.to(device)

        optimizer.zero_grad()

        logits = model(input_seq, output_seq)

        loss = criterion(logits[:, :-1].reshape(-1, logits.size(-1)), output_seq[:, 1:].reshape(-1))

        loss.backward()

        optimizer.step()

        losses.append(loss.item())


def plot_losses(losses):
    clear_output(wait=True)
    plt.plot(losses)
    plt.grid()
    plt.show()

def evaluate(model):
    pass

def train(model, train_dataloader, val_loader, optimizer, criterion, device, num_epochs):

    for epoch in range(num_epochs):
        train_epoch(model, train_dataloader, optimizer, criterion, device)
        evaluate(model)
        plot_losses(losses)


batch_size = 50

TrainData = RMSequenceDataset(10000, 4, 1, 0.15)
ValData = RMSequenceDataset(10000, 4, 1, 0.15)

train_dataloader = DataLoader(TrainData, batch_size=batch_size, drop_last=True)
val_dataloader = DataLoader(ValData, batch_size=batch_size, drop_last=True)

lr = 5e-4
num_epochs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DenoisingTransformer(128, 4, 2, 2, max_len=200, pos_encoding_embedding='random')
model = model.to(device)

optimizer = Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

losses = []

train(model, train_dataloader, val_dataloader, optimizer, criterion, device, num_epochs)

model.save('train_model.pt')