import torch
from torch import nn
from data.Dataset_Dataloader import MyDataloader
from models.customnet import CustomNet

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0

    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_accuracy


def main():
    model = torch.load('checkpoints/my_model.pth', weights_only=False)
    if torch.cuda.is_available():
        model = model.to_device('cuda')
    criterion = nn.CrossEntropyLoss()
     
    val_loader = MyDataloader("/tiny-imagenet/tiny-imagenet-200/val").getDataLoader()

    best_acc = 0

    val_accuracy = validate(model, val_loader, criterion)

    best_acc = max(best_acc, val_accuracy)

    return
# Here we are using only the best model that has been saved, without updating the weights
if __name__ == "__main__":
    main()

