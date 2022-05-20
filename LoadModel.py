#used for evaluating the model on images
from contextlib import nullcontext
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tkinter as tk
from pathlib import Path
import pathlib
import cv2 
from tensorflow.keras.models import model_from_json



batch_size = 32
img_height = 250
img_width = 250


# load json and create model
json_file = open('F:\Python\A.I\Project\Class\model.json', 'r') #path to trained model
loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
json_file.close()

# load weights into new model
loaded_model.load_weights("F:\Python\A.I\Project\Class\model.h5") #path to weights
print("Loaded model from disk")
import os
folder="F:\Python\A.I\Project\Class\People" #path to folder with names
subfolders = [ f.name for f in os.scandir(folder) if f.is_dir() ]


loaded_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])



# evaluate loaded model on test data

img_height=250
img_width=250
Students=[]
image_in='imageinput'
  
while (image_in)!='exit':
  
  
  my_file = Path(image_in)
  
  if my_file.is_file():
      
      
      img = keras.preprocessing.image.load_img(image_in,target_size=(img_height, img_width)) 
      
      img_array = keras.preprocessing.image.img_to_array(img)
      
      img_array = tf.expand_dims(img_array, 0)      
      
      
      predictions = loaded_model.predict(img_array)
      
      score = tf.nn.softmax(predictions[0])
      print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(subfolders[np.argmax(score)], 100 * np.max(score))
      ) 
      Students.append(subfolders[np.argmax(score)])
  else:
    print("image does not exist")
  print("Please give path to image for the prediction (press exit to stop giving images)")
  image_in=input()


res = []
for i in Students:
    if i not in res:
        res.append(i)




print(str(res))


add=len(subfolders)-len(res)

if(add==0):
  print("All students are attending the Class")
else:
  print(add," students are not present")
   
