from pickle import TRUE
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from functools import partial
from torch import nn 
import torch
import math

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    
    model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
    #get number of input features for the classifier
    in_features = det_utils.retrieve_out_channels(model.backbone, (300, 300))

    #replace the pre-trained head with a new one
    anchors = model.anchor_generator.num_anchors_per_location()

    #SSD
    model.head.classification_head = SSDClassificationHead(in_features, anchors, num_classes)


    print(model)
    return model

def get_model_instance_segmentation_transfer(num_classes):
    # load an instance segmentation model pre-trained on COCO
    
    model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
    #get number of input features for the classifier

    for p in model.parameters():
        p.requires_grad = False

    in_features = det_utils.retrieve_out_channels(model.backbone, (300, 300))

    #replace the pre-trained head with a new one
    anchors = model.anchor_generator.num_anchors_per_location()
    
    #SSD
    model.head.classification_head = SSDClassificationHead(in_features, anchors, num_classes)


    print(model)
    return model

def get_model_instance_segmentation_fromS(num_classes):
    # load an instance segmentation model pre-trained on COCO
    
    model = torchvision.models.detection.ssd300_vgg16(pretrained=False)
    #get number of input features for the classifier

    in_features = det_utils.retrieve_out_channels(model.backbone, (300, 300))

    #replace the pre-trained head with a new one
    anchors = model.anchor_generator.num_anchors_per_location()
    
    #SSDlite
    #norm_layer = partial(nn.BatchNorm2d, eps= 0.001, momentum=0.03)
    #model.head.classification_head = SSDLiteClassificationHead(in_features, anchors, num_classes, norm_layer)

    #SSD
    model.head.classification_head = SSDClassificationHead(in_features, anchors, num_classes)


    print(model)
    return model

#
#   SSD LITE 
#
#

def get_model_instance_segmentation_SSD_LITE(num_classes):
    # load an instance segmentation model pre-trained on COCO
    
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
    
    #get number of input features for the classifier
    in_features = det_utils.retrieve_out_channels(model.backbone, (320, 320))
    anchors = model.anchor_generator.num_anchors_per_location()
    norm_layer = partial(nn.BatchNorm2d, eps= 0.001, momentum=0.03)
    model.head.classification_head = SSDLiteClassificationHead(in_features, anchors, num_classes, norm_layer)


    print(model)
    return model

def get_model_instance_segmentation_transfer_lite(num_classes):
    # load an instance segmentation model pre-trained on COCO
    
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
    #get number of input features for the classifier

    for p in model.parameters():
        p.requires_grad = False

    in_features = det_utils.retrieve_out_channels(model.backbone, (320, 320))

    #replace the pre-trained head with a new one
    anchors = model.anchor_generator.num_anchors_per_location()
    norm_layer = partial(nn.BatchNorm2d, eps= 0.001, momentum=0.03)
    model.head.classification_head = SSDLiteClassificationHead(in_features, anchors, num_classes, norm_layer)

    print(model)
    return model

def get_model_instance_segmentation_fromS_lite(num_classes):
    # load an instance segmentation model pre-trained on COCO
    
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=False) ###cAMBIAR ESTA MIERDA
    #get number of input features for the classifier

    in_features = det_utils.retrieve_out_channels(model.backbone, (320, 320))

    #replace the pre-trained head with a new one
    anchors = model.anchor_generator.num_anchors_per_location()
    norm_layer = partial(nn.BatchNorm2d, eps= 0.001, momentum=0.03)
    model.head.classification_head = SSDLiteClassificationHead(in_features, anchors, num_classes, norm_layer)


    print(model)
    return model

# RetinaNet
def get_model_retinanet_tf(num_classes):
    # load an instance segmentation model pre-trained on COCO

    #model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True) #ResNet50
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True) #RetinaNet_ResNet50
    #Transfer Learning
    # for p in model.parameters():
    #     p.requires_grad = False

    # get the number of input features for the classifier

    #in_features = model.roi_heads.box_predictor.cls_score.in_features
    in_channels = model.head.classification_head.conv[0].in_channels

    # replace the pre-trained head with a new one
    #model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    num_anchors = model.head.classification_head.num_anchors

    # now get the number of input features for the mask classifier
    #in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    #hidden_layer = 256
    model.head.classification_head.num_classes = num_classes

    # and replace the mask predictor with a new one
    #model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
    #                                                   hidden_layer,
    #                                                   num_classes)
    out_channels = 256
    cls_logits = torch.nn.Conv2d(out_channels, num_anchors * num_classes, kernel_size = 3, stride=1, padding=1)
    torch.nn.init.normal_(cls_logits.weight, std=0.01)  # as per pytorch code
    torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))  # as per pytorcch code 
    # assign cls head to model
    model.head.classification_head.cls_logits = cls_logits

    print(model)
    return model
'''
# RetinaNet
Optimización: RetinaNet se entrena con descenso de gradiente estocástico (SGD). 
Utilizamos SGD sincronizado en 8 GPUs con un total de 16 imágenes por minibatch (2 imágenes por GPU)
. A menos que se especifique lo contrario, todos los modelos se entrenan durante 90.000
 iteraciones con una tasa de aprendizaje inicial de 0,01, que se divide por 10 a las 60.000 y
  de nuevo a las 80.000 iteraciones. Utilizamos el volteo horizontal de la imagen como única
   forma de aumento de datos, a menos que se indique lo contrario. Se utiliza un decaimiento 
   del peso de 0,0001 y un impulso de 0,9. La pérdida de entrenamiento es la suma de la pérdida 
   focal y la pérdida L1 suave estándar utilizada para la regresión de caja regresión de caja [10].
    El tiempo de entrenamiento oscila entre 10 y 35 horas para los modelos de la Tabla 1e.
'''