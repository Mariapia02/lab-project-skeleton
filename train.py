import torch
from torch import nn
from data.Dataset_Dataloader import MyDataloader
from models.customnet import CustomNet
from eval import validate

def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda() 

        optimizer.zero_grad()
        outputs = model(inputs) 
        loss = criterion(outputs, targets) 
        loss.backward()
        optimizer.step() 

        running_loss += loss.item() 
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')


def main():

    if torch.cuda.is_available():
        model = CustomNet().cuda()
    else:
        model = CustomNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    train_loader = MyDataloader("/tiny-imagenet/tiny-imagenet-200/train").getDataloader()
    val_loader = MyDataloader("/tiny-imagenet/tiny-imagenet-200/val").getDataloader()

    best_acc = 0

    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        train(epoch, model, train_loader, criterion, optimizer)
        val_accuracy = validate(model, val_loader, criterion)
        best_acc = max(best_acc, val_accuracy)

    return


main()

