import os
import tensorflow as tf
import matplotlib as plt
import pandas as pd

# Cat File name change
# file_list = os.listdir('/home/ephemera/Project/Meow/Data/cat/')
# prefix = 'cat_'

# for index, filename in enumerate(file_list):
#     new_filename = f"{prefix}{index}" + os.path.splitext(filename)[1]
#     old_filepath = os.path.join('/home/ephemera/Project/Meow/Data/cat/', filename)
#     new_filepath = os.path.join('/home/ephemera/Project/Meow/Data/cat/', new_filename)
    
#     os.rename(old_filepath, new_filepath)
#     print(f"Renamed {filename} to {new_filename}")
    

# Loaf file name change
# file_list = os.listdir('/home/ephemera/Project/Meow/Data/loaf/')

# for i,j in enumerate(file_list):
#     os.rename('/home/ephemera/Project/Meow/Data/loaf/'+str(j), '/home/ephemera/Project/Meow/Data/loaf/'+f'loaf_{i}.jpg')



    
# Image Pre-Processing

        
# Convolution Block
def conv_block():
    input_layer = None