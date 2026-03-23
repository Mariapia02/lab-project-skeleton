import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

class MyDataloader():
     
    def __init__(self, root, transformations=None):

        self.root = root

        # Apply transformations to images
        if transformations is None:
            self.transformations = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0), ratio=(0.75, 1.33), interpolation=2),
                transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
            ])
        else:
            self.transformations = transformations

        # Read images from different folders and assign a label to each class
        self.imageFolder = ImageFolder(root, transform = self.transformations)
        
    # Returns ready batches of images to give to the NN we want to train
    def getDataloader(self, batch_size:int = 32, shuffle:bool = True, num_workers:int=8):
        return DataLoader(self.imageFolder, batch_size = batch_size, shuffle = shuffle, num_workers=num_workers)
    
    