import torch
import os
import pandas as pd
from PIL import Image

class TextDetectionDataset(torch.utils.data.Dataset):
    def __init__(self,csv_file,img_dir,label_dir,grid_size=7,num_bbox=2,num_classes=20,transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.grid_size = grid_size
        self.num_bbox = num_bbox
        self.num_classes = num_classes
        
    def __len__(self):
        return len(self.annotations)
        
    def __getitem__(self, index):
        label_path = self.annotations.iloc[index, 1]
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                labels = label.split(" ")[:8]
                labels = [float(item) for item in labels]
                boxes.append(labels)

        img_path = self.annotations.iloc[index, 0]
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)
        
        #we have data in format [x1,x2,x3,x4,y1,y2,y3,y4] which needs to be changed to [x,y,w,h]
        updated_boxes = torch.empty(boxes.shape[0],4)
        updated_boxes[:,0] = (boxes[:,1]+boxes[:,0])/2
        updated_boxes[:,1] = (boxes[:,5]+boxes[:,4])/2
        updated_boxes[:,2] = (boxes[:,1]-boxes[:,0])
        updated_boxes[:,3] = (boxes[:,6]+boxes[:,5])
        
        #also we need to convert them into ratio from absolute values
        updated_boxes[:,0] = updated_boxes[:,0]/image.size[0]
        updated_boxes[:,1] = updated_boxes[:,0]/image.size[1]
        updated_boxes[:,2] = updated_boxes[:,0]/image.size[0]
        updated_boxes[:,3] = updated_boxes[:,0]/image.size[1]
        
        boxes = updated_boxes

        if self.transform:
            # image = self.transform(image)
            image, boxes = self.transform(image,boxes)

        # Convert To Cells
        label_matrix = torch.zeros((self.grid_size, self.grid_size, self.num_classes + 5 * self.num_bbox))
        for box in boxes:
            x, y, width, height = box.tolist()

            # i,j represents the cell row and cell column
            i, j = int(self.grid_size * y), int(self.grid_size * x)
            x_cell, y_cell = self.grid_size * x - j, self.grid_size * y - i

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
            
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (
                width * self.grid_size,
                height * self.grid_size,
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, self.num_classes] == 0:
                # Set that there exists an object
                label_matrix[i, j, self.num_classes] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j,self.num_classes+1:self.num_classes+5] = box_coordinates

                # Set one hot encoding for class_label
                # label_matrix[i, j, class_label] = 1

        return image, label_matrix

    
    

