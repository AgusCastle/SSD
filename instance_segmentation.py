from pickle import TRUE
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from functools import partial
from torch import nn 

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
    in_features = det_utils.retrieve_out_channels(model.backbone, (300, 300))
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

    in_features = det_utils.retrieve_out_channels(model.backbone, (300, 300))

    #replace the pre-trained head with a new one
    anchors = model.anchor_generator.num_anchors_per_location()
    norm_layer = partial(nn.BatchNorm2d, eps= 0.001, momentum=0.03)
    model.head.classification_head = SSDLiteClassificationHead(in_features, anchors, num_classes, norm_layer)

    print(model)
    return model

def get_model_instance_segmentation_fromS_lite(num_classes):
    # load an instance segmentation model pre-trained on COCO
    
    model = torchvision.models.detection.ssd300_vgg16(pretrained=False)
    #get number of input features for the classifier

    in_features = det_utils.retrieve_out_channels(model.backbone, (300, 300))

    #replace the pre-trained head with a new one
    anchors = model.anchor_generator.num_anchors_per_location()
    norm_layer = partial(nn.BatchNorm2d, eps= 0.001, momentum=0.03)
    model.head.classification_head = SSDLiteClassificationHead(in_features, anchors, num_classes, norm_layer)


    print(model)
    return model