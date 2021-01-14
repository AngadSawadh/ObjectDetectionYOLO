import torch
from intersection_of_union import IOU

def NMS(predicted_bbox,iou_thresh,object_score_thresh,representation="corner_points"):
   
    #predicted_box will be a list of dimension [N*N,(5+C)]
    #[x,y,w,h,o,classes]
    assert(predicted_bbox) == list
    
    #we need to chose those bboxes which have object score greater than the threshold
    bboxes = [box for box in predicted_bbox if box[4]>object_score_thresh]
    bboxes = sorted(bboxes,key=lambda x: x[4], reverse=True)
    nms_bboxes = []
    
    while bboxes:
        chosen_bbox = bboxes.pop(0)
        
        bboxes=[box
                for box in bboxes
                    if(
                    chosen_bbox[5:]!=box[5:]
                    or
                    IOU(torch.tensor(chosen_bbox[:4]),
                        torch.tensor(box[:4]),
                        representation=representation)<iou_thresh
                    )
                ]
        
        nms_bboxes.append(chosen_bbox)
    
    return nms_bboxes
        
            
        

