from efficientnet_pytorch import EfficientNet
from PIL import Image
import torch
from torchvision import transforms
import os
from tqdm import *
import os
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json


model = EfficientNet.from_pretrained('efficientnet-b0')

class RGBTransforms():
     def __call__(self, img):
         if img.mode !='RGB':
            return img.convert('RGB')
         else:
            return img
            
tfms = transforms.Compose([RGBTransforms(),transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = json.load(open(annotations_file))
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, list(self.img_labels.keys())[idx])
        image = Image.open(img_path)
        label = list(self.img_labels.values())[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

transformed_dataset = CustomImageDataset(annotations_file='labels.json',img_dir='./dataset',transform=tfms)
dataloader =  DataLoader(transformed_dataset, batch_size = 32, shuffle=False, num_workers=2)

if __name__=='__main__':
    actual=[]
    predicted=[]
    for img,label in tqdm(dataloader):
        with torch.no_grad():
            outputs = model(img)
            pred_label = torch.argmax(outputs).item()
            actual.append(label)
            predicted.append(pred_label)

    results = {actual[i]: predicted[i] for i in range(len(actual))}

    with open('results.json', 'w') as fp:
        json.dump(results, fp,  indent=4)