import pathlib
import os
import shutil
import sys

def create_folder(dirName: str):
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        
main_directory = "./"
        
dir_name = main_directory + "data/test/negative"
create_folder(dir_name)
dir_name = main_directory + "data/test/positive"
create_folder(dir_name)

dir_name = main_directory + "data/train/negative"
create_folder(dir_name)
dir_name = main_directory + "data/train/positive"
create_folder(dir_name)


def split_images_into_folders(main_directory: str, train_or_test: str):
    file_split = open(main_directory + train_or_test + "_split.txt", "r")
    file_split_lines = file_split.readlines()


    dir_to_images = main_directory + "data/" + train_or_test + "/"
    for line in file_split_lines:
        splited_line = line.split()
        file_name = splited_line[1]
        label = splited_line[2] # positive/negative
        if os.path.isfile(dir_to_images + file_name):
            shutil.move(dir_to_images + file_name, dir_to_images + label + "/" + file_name)



if __name__ == "__main__":      
    split_images_into_folders(main_directory, "test")
    split_images_into_folders(main_directory, "train")