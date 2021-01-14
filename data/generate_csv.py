import os
import csv

image_path = "G:/Project/Synthetic Train Set - Detection & Recognition/Image"
annotation_path = "G:/Project/Synthetic Train Set - Detection & Recognition/Annotation"
folders = os.listdir(image_path)
folder  = folders[0]

with open("train.csv", mode="w", newline="") as train_file:
    for folder in folders:
        new_image_path = image_path+"/{f}".format(f=folder)
        image_files = [new_image_path+"/{f}".format(f=element) for element in os.listdir(new_image_path)]
        new_annotation_path = annotation_path+"/{f}".format(f=folder)
        annotation_files = [new_annotation_path+"/{f}".format(f=element) for element in os.listdir(new_image_path)]
        updated_annotation_files = [af.replace(".jpg",".txt") for af in annotation_files]
        writer = csv.writer(train_file)
        for item0,item1 in zip(image_files,updated_annotation_files):
            writer.writerow([item0,item1])

test_image_path = "G:/Project/real_Image_dataset_Detection/real_Image_dataset_Detection/Image"
test_annotation_path = "G:/Project/real_Image_dataset_Detection/real_Image_dataset_Detection/Annotation"

with open("test.csv", mode="w", newline="") as test_file:
    test_image_files = [test_image_path+"/{f}".format(f=element) for element in os.listdir(test_image_path)]
    test_annotation_files = [test_annotation_path+"/{f}".format(f=element) for element in os.listdir(test_image_path)]
    updated_test_annotation_files = [af.replace(".jpg",".txt") for af in test_annotation_files]
    writer = csv.writer(test_file)
    for item0,item1 in zip(test_image_files,updated_test_annotation_files):
        writer.writerow([item0,item1])
