import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import googlenet
import os
from fire import Fire
from sklearn.model_selection import train_test_split

class Runner:
    def __init__(self, model: nn.Module, device='cuda') -> None:
        self.model = model
        self.best_metric = 0
        self.device = device

    def train(self, dataloader, loss_func, optimizer, epoch):
        '''train model.'''
        self.model.train()
        data_size = len(dataloader.dataset)
        correct = 0
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            y_hat = self.model(X)
            loss = loss_func(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (y_hat.argmax(1) == y).type(torch.float).sum().item()

        loss, current = loss.item(), batch * len(X)
        correct /= data_size

        print(f'EPOCH{epoch+1}\tloss: {loss:>7f} Accuracy: {(100 * correct):>0.1f}%, ', end='\t')

    def validation(self, dataloader, loss_fn):
        '''valid model.'''
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        if correct > self.best_metric:
            self.best_metric = correct
            if not os.path.exists('./results/'):
                os.makedirs('./results')
            torch.save(self.model.state_dict(), './results/model_best.pkl')
        print(f'Test Error: Accuracy: {(100 * correct):>0.1f}%, Average loss: {test_loss:>8f}\n')
    
    def test(self, dataloader, loss_fn):
        '''valid model.'''
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f'Test Error: Accuracy: {(100 * correct):>0.1f}%, Average loss: {test_loss:>8f}\n')

def train():
    # the main key is to transform origin data
    transform_train = transforms.Compose(
        [
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(  (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)   )])
    transform = transforms.Compose(
        [ transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize( (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  )])
    train_data = datasets.CIFAR10(
        root='./data',
        train=True, 
        download=True,
        transform=transform_train
    )
    train_data, valid_data = train_test_split(train_data, test_size=0.2)
    test_data = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    batch_size = 256
    print('batch size: ', batch_size)
    lr = 1e-1
    print('train_data.size: ', train_data.__len__())
    print('test_data.size: ', test_data.__len__())

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    for X, y in train_dataloader:
        print(X.shape)		# torch.Size([256, 1, 28, 28])
        print(y.shape)		# torch.Size([256])
        break
    # MODEL
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    model = googlenet().to(device)
    # TRAIN MODEL
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    epoches = 50
    runner = Runner(model, device=device)
    for epoch in range(epoches):
        runner.train(train_dataloader, loss_func, optimizer, epoch)
        runner.validation(valid_dataloader, loss_func)
    runner.test(test_dataloader, loss_func)

    # Save models
    torch.save(model.state_dict(), './results/model.pth')

def inference():
    device = 'cuda:1'
    batch_size = 256
    transform = transforms.Compose(
        [ transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize( (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  )])
    test_data = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    model = googlenet().to(device)
    runner = Runner(model, device)
    loss_func = nn.CrossEntropyLoss()
    if os.path.exists('./results/model_best.pkl'):
        model_dict = torch.load('./results/model_best.pkl')
        model.load_state_dict(model_dict)
    runner.test(test_dataloader, loss_func)

def main(is_train):
    if is_train:
        print('start training model...')
        return train()
    else:
        print('model testing...')
        return inference()


if __name__ == '__main__':
    Fire(main)