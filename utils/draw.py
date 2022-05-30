
from cv2 import imshow
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
import torchvision
from torchvision.ops import nms

plt.rcParams["savefig.bbox"] = 'tight'

from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path
from torchvision.transforms.functional import convert_image_dtype


image_prob = read_image("/home/fp/FPM/DataBase/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/001325.jpg")

chk = torch.load('/home/fp/Escritorio/transfer-learning-ssd/bin/modelo.pth.rar')

star = chk['epoch'] + 1

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

device = torch.device('cpu')
model = chk['model']
model.to(device)

batch = torch.stack([image_prob.to(device)])

batch = convert_image_dtype(batch, dtype=torch.float)
batch.to(device)

times = time.time()
output = model(batch)    # Returns predictions
print(time.time()- times)

score_threshold = .3


# read input image

# bounding box in (xmin, ymin, xmax, ymax) format
# top-left point=(xmin, ymin), bottom-right point = (xmax, ymax)
bbox = output[0]['boxes']
print(output)
a = []
for i in range(len(bbox)):
    if output[0]['scores'][i] > score_threshold:
        a.append(bbox[i].tolist())


a = torch.tensor(a, dtype=torch.float)
times = time.time()
# draw bounding box on the input image
img=draw_bounding_boxes(image_prob, a , width=3, colors=(255,255,0))

print(time.time()- times)
#| transform it to PIL image and display
img = torchvision.transforms.ToPILImage()(img)
img.show()