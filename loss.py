import torch
import torch.nn as nn
from utils.intersection_of_union import IOU

class YOLOloss(nn.Module):
    def __init__(self,grid_size,num_bbox_per_cell,num_classes):
        super(YOLOloss,self).__init__()
        self.grid_size = grid_size
        self.num_bbox_per_cell = num_bbox_per_cell
        self.num_classes = num_classes
        self.MSE = nn.MSELoss(reduction="sum")
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
        
    def forward(self,predictions,targets):
        predictions.reshape(-1,self.grid_size,self.grid_size,self.num_bbox_per_cell*5+self.num_classes)
        iou_bbox1 = IOU(predictions[...,21:25],targets[...,21:25])
        iou_bbox2 = IOU(predictions[...,26:30],targets[...,21:25])
        ious = torch.cat([iou_bbox1.unsqueeze(0), iou_bbox2.unsqueeze(0)], dim=0)#2,n,s,s,1
        
        iou_maxes, bestbox = torch.max(ious, dim=0)#iou max for bbox1 or bbox2 with target
        box_exists = targets[...,20].unsqueeze(3)
        
        
        #box coordinates
        box_predictions = box_exists*(bestbox*(predictions[...,26:30])+(1-bestbox)*(predictions[...,21:25]))
        box_targets = box_exists*(targets[...,21:25])
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[:,2:4])+ 1e-6)
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        box_loss = self.MSE(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )
        
        #object 
        pred_box = (bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21])
        object_loss = self.MSE(torch.flatten(box_exists * pred_box,end_dim=-2),torch.flatten(box_exists * targets[..., 20:21],end_dim=-2))
        
        #no_object_loss
        no_object_loss = self.MSE(torch.flatten((1-box_exists)*predictions[...,20:21],end_dim=-2),torch.flatten((1-box_exists)*targets[...,20:21],end_dim=-2))
        no_object_loss += self.MSE(torch.flatten((1-box_exists)*predictions[...,25:26],end_dim=-2),torch.flatten((1-box_exists)*targets[...,20:21],end_dim=-2))
        
        #class loss
        class_loss = self.MSE(torch.flatten(box_exists*predictions[...,:20],end_dim=-2),torch.flatten(box_exists*targets[...,:20],end_dim=-2))

        #total_loss
        total_loss = (self.lambda_coord*box_loss+
                      object_loss+
                      self.lambda_noobj*no_object_loss+
                      class_loss
                        )
        
        return total_loss
# a = torch.rand(2,2,2,2)
# print(a.unsqueeze(0).shape)