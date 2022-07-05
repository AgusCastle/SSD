from fileinput import filename
import torch
from preferences.detect.engine import train_one_epoch, evaluate
from create_dataset import VOCDataset, PascalVOCDataset, PennFudanDataset2
from transformation import get_transform

import json
import time

from preferences.detect.utils import collate_fn
from instance_segmentation import  get_model_retinanet_tf, get_model_instance_segmentation_transfer

def save_model(epoch, model, optim, name = None):
    if name is None:
        filename = '/home/bringascastle/Documentos/repos/SSD/checkpoints/RetinaNet_FT.pth.rar'
    else:
        filename = '/home/bringascastle/Documentos/repos/SSD/checkpoints/RetinaNet_FT_epoch_{}.pth.rar'.format(name)
    
    state = {
        'epoch': epoch,
        'model': model,
        'optimizer': optim,
        }

    torch.save(state, filename)

# array = ["NMS"]


# filename = './results/losses.json'
# with open(filename, "r") as file:
#     datos = json.load(file)
# for i in array:
#     datos[i].clear()
# with open(filename, "w") as file:
#     json.dump(datos, file)


lr = 1e-4
momentum = 0.9
weight_decay = 5e-4

decay_lr_at = [40, 154, 193]

iterations = 120000
batch_size = 4

def main(checkpoint = None):
    
    global lr, momentum, weight_decay, start_epoch, decay_lr_at

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 21

    dataset = VOCDataset('/home/bringascastle/Documentos/repos/SSD/JSONfiles', 'TRAIN', get_transform(True))
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4,
        collate_fn=collate_fn)

    if checkpoint is None:
        
        start_epoch = 0
        model = get_model_retinanet_tf(num_classes)
        model.to(device)

        biases = []
        not_biases = []

        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)

        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 1 * lr}, {'params': not_biases}], 
                                    lr=lr,
                                    momentum=momentum, 
                                    weight_decay=weight_decay)
    else:

        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # let's train it for 10 epochs
    print(start_epoch)


    num_epochs = 60
    #decay_lr_at = [it // (len(dataset) // 32) for it in decay_lr_at]

    tiempo_entrenamiento = time.time()

    for epoch in range(start_epoch ,num_epochs):
        # train for one epoch, printing every 200 iterations

        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, 0.1)

        train_one_epoch(model, optimizer, data_loader, device, epoch, batch_size= len(data_loader) ,print_freq=200)
        # update the learning rate

        if epoch == 5:
            save_model(epoch, model, optimizer, "5")
        
        if epoch == 10:
            save_model(epoch, model, optimizer, "10")

        if epoch == 15:
            save_model(epoch, model, optimizer, "15")

        if epoch == 20:
            save_model(epoch, model, optimizer, "20")

        if epoch == 25:
            save_model(epoch, model, optimizer, "25")

        if epoch == 30:
            save_model(epoch, model, optimizer, "30")
        
        if epoch == 35:
            save_model(epoch, model, optimizer, "35")

        if epoch == 40:
            save_model(epoch, model, optimizer, "40")

        if epoch == 45:
            save_model(epoch, model, optimizer, "45")

        if epoch == 50:
            save_model(epoch, model, optimizer, "50")

        if epoch == 55:
            save_model(epoch, model, optimizer, "55")

        save_model(epoch, model, optimizer)       

    tiempo_entrenamiento = time.time() - tiempo_entrenamiento
    print("That's it!")
 
    print("Tiempo_entrenamiento: ", tiempo_entrenamiento)


def adjust_learning_rate(optimizer, scale):
    """
    Scale learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param scale: factor to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
    print("DECAYING learning rate.\n The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))

main()