import argparse
from functools import partial

import matplotlib.pyplot as plt

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import CIFARDataset
from network.vgg import VGG
from network.resnet import ResNet


# ���� CNN �� �� �ϳ��� �����Ͽ� �ҷ����� ���� �ڵ�
def get_model(model_name: str, bn_flag: bool, drop_prob: float, device):
    if "VGG" in model_name:
        net = VGG(model_name, bn_flag, drop_prob).to(device)
        return net
    elif "ResNet" in model_name:
        net = ResNet(model_name).to(device)


# ��Ƽ������ �����ϱ� ���� �ڵ�
def get_optimizer(name: str):
    if name == "SGD":
        return partial(SGD, weight_decay=0.005, momentum=0.9)
    elif name == "Adam":
        return partial(Adam, weight_decay=0.005)
    else:
        raise NotImplementedError(name)


def train(args):
    # Deice
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else \
        torch.device('cpu')
    print("\nDevice : ", DEVICE)

    # Dataset & DataLoader
    print('\n<Load CIFAR-10 Dataset>')
    train_dataset, val_dataset = CIFARDataset(args.data_path, True), \
        CIFARDataset(args.data_path, False)
    train_loader, val_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8), \
        DataLoader(val_dataset, args.batch_size, shuffle=True, num_workers=8)

    print('\n<Train CIFAR Dataset>')

    # Model, Criterion, Optimizer
    net = get_model(args.model, args.bn_flag, args.drop_prob, DEVICE)
    criterion = CrossEntropyLoss()
    optimizer_type = get_optimizer(args.optimizer)
    optimizer = optimizer_type(net.parameters(), lr=args.learning_rate)

    if args.resume:
        ckpt = torch.load(args.resume)
        net.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
    else:
        start_epoch = 1

    # losses
    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        net.train()

        # Training phase
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(train_loss)

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        net.eval()

        # Validation phase
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = net(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)

        # print info. about current epoch
        print(f"Epoch [{epoch}/{args.epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, f'ckpt/{args.model}.{epoch:04d}.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # �ҷ��� ������ ��� & ckpt�κ��� �н����� ����
    parser.add_argument("--data_path", type=str, default="./datasets/CIFAR-10")
    parser.add_argument('--resume', type=str, default=None)

    # model ���� �������Ķ����
    parser.add_argument("--model", type=str, default="VGG16")
    parser.add_argument("--bn_flag", type=bool, default=False)
    parser.add_argument("--drop_prob", type=float, default=0.1)

    # �Ʒ� ���� �������Ķ����
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="Adam")

    args = parser.parse_args()
    train(args)