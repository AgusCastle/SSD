from fileinput import filename
import torch
from preferences.detect.engine import train_one_epoch, evaluate
from create_dataset import VOCDataset, PascalVOCDataset, PennFudanDataset2
from transformation import get_transform

import json
import time

from preferences.detect.utils import collate_fn
from instance_segmentation import get_model_instance_segmentation

def save_model(epoch, model, optim):
    filename = './bin/modelo.pth.rar'
    state = {
        'epoch': epoch,
        'model': model,
        'optimizer': optim,
    }

    torch.save(state, filename)

from PIL import Image
import numpy as np
array = ["loss_value"]

filename = './results/losses.json'
# 1. Read file contents
with open(filename, "r") as file:
    datos = json.load(file)
# 2. Update json object
for i in array:
    datos[i].clear()
# 3. Write json file
with open(filename, "w") as file:
    json.dump(datos, file)


def main():
    
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # our dataset has two classes only - background and person
    num_classes = 20

    # use our dataset and defined transformations
    dataset = VOCDataset('/home/fp/Escritorio/transfer-learning-ssd/JSONfiles', 'TRAIN', get_transform(True))
    dataset_test = VOCDataset('/home/fp/Escritorio/transfer-learning-ssd/JSONfiles', 'TEST',get_transform(False))

    #dataset = PennFudanDataset2('./../PennFudanPed', get_transform(True))
    #dataset_test = PennFudanDataset2('./../PennFudanPed', get_transform(False))

    print(dataset[0])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=4,
        collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=8, shuffle=False, num_workers=4,
        collate_fn=collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.002,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    tiempo_entrenamiento = time.time()
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=200)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        #evaluate(model, data_loader_test, device=device)

    tiempo_entrenamiento = time.time() - tiempo_entrenamiento
    print("That's it!")
 
    print("Tiempo_entrenamiento: ", tiempo_entrenamiento)

    save_model(epoch, model, optimizer)

main()