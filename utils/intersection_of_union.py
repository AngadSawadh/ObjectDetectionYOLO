import torch

def IOU(bbox_coordinate_predictions,bbox_coordinate_labels,representation="midpoint"):
    
    #note the coordinates in the parameter list are using tensor data structure
    
    if representation=="midpoint":
        #each point is of shape [n,1]
        pred_x1 = bbox_coordinate_predictions[...,0:1] - bbox_coordinate_predictions[...,2:3]/2
        pred_y1 = bbox_coordinate_predictions[...,1:2] - bbox_coordinate_predictions[...,3:4]/2
        pred_x2 = bbox_coordinate_predictions[...,0:1] + bbox_coordinate_predictions[...,2:3]/2
        pred_y2 = bbox_coordinate_predictions[...,1:2] + bbox_coordinate_predictions[...,3:4]/2
        
        label_x1 = bbox_coordinate_labels[...,0:1] - bbox_coordinate_labels[...,2:3]/2
        label_y1 = bbox_coordinate_labels[...,1:2] - bbox_coordinate_labels[...,3:4]/2
        label_x2 = bbox_coordinate_labels[...,0:1] + bbox_coordinate_labels[...,2:3]/2
        label_y2 = bbox_coordinate_labels[...,1:2] + bbox_coordinate_labels[...,3:4]/2
        
    elif representation=="corner_points":
        #each point is of shape [n,1]
        pred_x1 = bbox_coordinate_predictions[...,0:1]
        pred_y1 = bbox_coordinate_predictions[...,1:2]
        pred_x2 = bbox_coordinate_predictions[...,2:3]
        pred_y2 = bbox_coordinate_predictions[...,3:4]
        
        label_x1 = bbox_coordinate_labels[...,0:1]
        label_y1 = bbox_coordinate_labels[...,1:2]
        label_x2 = bbox_coordinate_labels[...,2:3]
        label_y2 = bbox_coordinate_labels[...,3:4]

    #of shape [n,1]
    inter_x1 = torch.max(pred_x1,label_x1)
    inter_y1 = torch.max(pred_y1,label_y1)
    inter_x2 = torch.min(pred_x2,label_x2)
    inter_y2 = torch.min(pred_y2,label_y2)
    
    #of shape [n,1]
    inter_area = (inter_x2-inter_x1).clamp(min = 0)*(inter_y2-inter_y1).clamp(min = 0)
    pred_area = abs((pred_x2-pred_x1)*(pred_y2-pred_y1))
    label_area = abs((label_x2-label_x1)*(label_y2-label_y1))
    
    iou = inter_area/(pred_area+label_area-inter_area + 1e-6)
    return iou
    