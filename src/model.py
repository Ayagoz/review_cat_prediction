import torch
import logging
from typing import Tuple, Optional

import argparse
from datetime import datetime
from torch import nn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from load_data import LMDBDataset

logger = logging.getLogger(__name__)

# Define your model architecture


class SmallModel(nn.Module):
    def __init__(self, params: dict = {}):
        super(SmallModel, self).__init__()
        # Define your layers here
        self.params = params
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Linear(768, 2454),
            nn.Sigmoid()

        )

    def get_params(self) -> dict:
        return {
            'model_name': self.__class__.__name__,
            'params': self.params
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Define the forward pass of your model
        x = self.model(x)
        return x

# Train loop


def train(model: nn.Module, criterion: nn.Module, optimizer: Optimizer,
          train_loader: DataLoader, val_loader: DataLoader, tb_writer: SummaryWriter, epochs: int = 10) -> None:

    model.train()
    for epoch in range(epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            train_accuracy, train_loss = evaluate(
                model, criterion, train_loader, tb_writer, epoch=epoch, label="train")
            val_accuracy, val_loss = evaluate(
                model, criterion, val_loader, tb_writer, epoch=epoch, label="val")
            print(f"Epoch: {epoch + 1}/{epochs}")
            print(f"Train Accuracy: {train_accuracy:.2f}%, Train Loss: {train_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.2f}%, Val Loss: {val_loss:.4f}")


# Evaluation function
def evaluate(model: nn.Module, criterion: nn.Module, test_loader: DataLoader, tb_writer: SummaryWriter,
             epoch: int, label: str = "test") -> Tuple[float, float]:
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            y_pred = (outputs > 0.5).numpy()
            labels = labels.numpy()
            acc = accuracy_score(labels, y_pred)
            total += len(outputs)
            correct += acc * len(outputs)

    accuracy = correct / total
    average_loss = total_loss / len(test_loader)
    tb_x = epoch * len(test_loader) + i + 1
    tb_writer.add_scalar(f'Loss/{label}', average_loss, tb_x)
    tb_writer.add_scalar(f'Accuracy/{label}', accuracy, tb_x)
    return accuracy, average_loss


def create_data_loader(data_dir: str, target_dir: str, batch_size: int, shuffle: bool = False) -> DataLoader:
    dataset = LMDBDataset(data_dir, target_dir)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> torch.optim.Optimizer:
    optim_state = optimizer.state_dict()
    # Convert to CPU
    for state in optim_state["state"].values():
        for k, v in state.items():
            state[k] = v.to(device)
    return optim_state


def save_model(model: nn.Module, path: str, optimizer: Optional[torch.optim.Optimizer] = None) -> None:
    state = {
        'model_name': model.__class__.__name__,
        'params': model.get_params(),
        'nn_state_dict': model.to('cpu').state_dict()
    }
    if optimizer is not None:
        state['optimizer_state_dict'] = optimizer_to_device(optimizer, torch.device('cpu'))

    torch.save(state, path)
    print(f"Model saved to '{path}'")


def load_model(path: str, load_optimizer: bool = False) -> Tuple[nn.Module, torch.optim.Optimizer]:
    state = torch.load(path)
    model = SmallModel()
    model.load_state_dict(state['nn_state_dict'])
    if load_optimizer:
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(state['optimizer_state_dict'])
        return model, optimizer
    return model


def main():
    # Create argument parser

    args = parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    writer = SummaryWriter(f'{args.logs_path}/exp_{timestamp}')
    # Load the data and create data loaders
    train_loader = create_data_loader(args.data_path, args.target_path, args.batch_size)

    # Create the model
    model = SmallModel()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train the model
    train(model, criterion, optimizer, train_loader,
          train_loader, tb_writer=writer, epochs=args.epochs)

    # Evaluate the model
    test_accuracy, test_loss = evaluate(
        model, criterion, train_loader, writer, epoch=1, label="test")
    print(f"Test Accuracy: {test_accuracy:.2f}%, Test Loss: {test_loss:.4f}")
    save_model(model, f'{args.model_path}/model_{timestamp}.pt', optimizer=optimizer)


def parse_args():
    parser = argparse.ArgumentParser(description='Training and Evaluation')
    parser.add_argument('--data_path', type=str,
                        help='Path to the data directory')
    parser.add_argument('--target_path', type=str)
    parser.add_argument('--logs_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    return parser.parse_args()


if __name__ == '__main__':
    main()
