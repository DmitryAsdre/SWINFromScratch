import os
from typing import Tuple, Optional, List

import fire
import numpy as np
import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import json

from SWIN.architecture import SWINTransformer
from SWIN.utils import train_one_epoch, valid_one_epoch

def save_metrics(train_losses : List[float],
                 valid_losses : List[float],
                 valid_accuracies : List[float],
                 path_to_save : str):
    os.makedirs(path_to_save, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))    
    fig.suptitle('Metrics')
    
    ax1.plot(train_losses, label='train cross entropy')
    ax1.plot(valid_losses, label='valid cross entropy')
    ax1.set_title('CrossEntropyLoss')
    ax1.legend()
    
    ax2.plot(valid_accuracies, label='valid accuracy')
    ax2.set_title('Accuracy Valid')
    ax2.legend()

    metrics_path = os.path.join(path_to_save, 'metrics.jpg')
    fig.savefig(metrics_path, bbox_inches='tight')
    
def save_to_json(path_to_save : str,
                 **kwargs):
    json_path = os.path.join(path_to_save, 'hyperparams.json')
    with open(json_path, 'w') as w:
        json.dump(kwargs, w)
    

def train(metrics_path : str,
          image_size : int = 32,
          num_classes : int = 10,
          emb_size : int = 96,
          batch_size : int = 256,
          learning_rate : float = 3e-4,
          weight_decay : float = 1e-1,
          n_epochs : int = 100,
          dataset_path : str = '../data/',
          device : str = 'cuda:0'):
    vit = SWINTransformer(emb_size=emb_size,
                          num_class=num_classes).to(device)

    train_transform = transforms.Compose(
        [transforms.TrivialAugmentWide(interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.RandomErasing(p=0.1),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    test_transform = transforms.Compose(
        [transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=train_transform)
    val_dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=test_transform)
    
    dl_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    dl_valid = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    
    optimizer = AdamW(vit.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    
    train_losses = []
    valid_losses = []
    valid_accuracies = []
    
    for _ in (pbar:=tqdm.tqdm(range(n_epochs))):
        train_loss = train_one_epoch(vit,
                                     optimizer,
                                     loss_fn,
                                     dl_train,
                                     device)

        valid_loss, valid_accuracy = valid_one_epoch(vit, 
                                                     loss_fn, 
                                                     dl_valid,
                                                     device)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)
        
        pbar.set_postfix_str(f'{train_losses[-1]}, {valid_losses[-1]}, {valid_accuracies[-1]}')

    print(train_losses)
    print(valid_losses)
    print(valid_accuracies)
    
    save_metrics(train_losses,
                 valid_losses,
                 valid_accuracies,
                 metrics_path)
    
    save_to_json(metrics_path,
                 best_valid_ce = np.min(valid_losses),
                 best_valid_accuracy = np.max(valid_accuracies),
                 image_size = image_size,
                 num_classes = num_classes,
                 batch_size = batch_size,
                 learning_rate = learning_rate,
                 weight_decay = weight_decay,
                 n_epochs = n_epochs)
    
if __name__ == '__main__':
    fire.Fire(train)