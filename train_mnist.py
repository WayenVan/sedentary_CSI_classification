import torch
from torchinfo import summary
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from models_tc.BvP import BvP
from models_tc.base import ImgGRU, CNN  
from common import print_parameters, print_parameters_grad
import torch.nn.functional as F

use_cuda = True   
batch_size = 64
test_batch_size = 512
n_epoch = 20
log_interval = 10
dry_run = False

device = torch.device("cuda" if use_cuda else "cpu")

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.double()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        # print_parameters(model)
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.double()
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

train_data = torchvision.datasets.MNIST(r'dataset', transform=transforms.Compose([transforms.ToTensor()]), train=True)
test_data = torchvision.datasets.MNIST(r'dataset', transform=transforms.Compose([transforms.ToTensor()]), train=False)

train_loader = DataLoader(train_data, batch_size=batch_size)
test_loader = DataLoader(train_data, batch_size=test_batch_size)

model = ImgGRU(10, (1, 28, 28), 2).to(device)
# model = CNN(10, channel=1).to(device)
summary(model, input_data=torch.rand([32, 1, 28, 28], dtype=torch.float64), device=device)

optimizer = torch.optim.SGD(model.parameters(), lr=2.)
lossfunc = torch.nn.CrossEntropyLoss()

for epoch in range(1, n_epoch + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
