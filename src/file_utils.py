from os.path import join
from os import listdir

import pandas as pd


def get_class_dir(data_dir, class_dir):
    return join(data_dir, str(class_dir))

def get_file(data_dir, class_dir, image):
    return join(data_dir, str(class_dir), image)

def get_files(data_dir, class_dir):
    return listdir(join(data_dir, str(class_dir)))

def count_images(directory):
    classes = {}
    
    for i in range(1, 129):
        folder = str(i)
        classes[i] = len(listdir(join(directory, folder)))
        
    return pd.Series(classes)