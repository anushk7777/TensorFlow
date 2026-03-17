# # intro to CNNs
## Layer 1: convolution layer
 # converts images into an array of numbers and does dimensionality reduction of array , that doesnt let the model lose important info as well as complexity is reduced
import inline
import matplotlib
## Layer 2: Relu - regularization rectified unit
# converts all negatives into 0 and positives as is , because it doesnt make sense to consider negative values since they are already dark/dead pixels , so doesnt matter if it all a  negative or a zero since it is all dark , Range is [0,infinity]

## Layer 3: Pooling layer
# Reduce the dimensionality , helps control overfitting , filters over 2x2 matrix

# Layer 4: Fully Connected Layer
# Flatten the output of the previous layer and feed it to a fully connected layer , which is a regular neural network layer

import tensorflow as tf
import urllib.request

from matplotlib import pylab
from tensorflow.python.keras.saving.saved_model_experimental import sequential

# Download datasets from Dropbox

urllib.request.urlretrieve(
    "https://www.dropbox.com/s/t4pzwpvrzneb190/training_set.zip?dl=1",
    "training_set.zip"
)


urllib.request.urlretrieve(
    "https://www.dropbox.com/s/i37jfni3d29raoc/test_set.zip?dl=1",
    "test_set.zip"
)

## Extracted the zip files and created training_set and test_set folders


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('training_set/cats/cat.4001.jpg')
imgplot = plt.imshow(img)
plt.show()


import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Dropout


print (tensorflow.__version__)

img_widht , img_height = 150, 150
train_data_dir = r"training_set/training_set/training_set"
validation_data_dir = r"test_set/test_set/test_set"
batch_size = 20
epochs = 20
nb_train_samples = 100
nb_validation_samples = 100


# data represented as rows x cols x channels , if channel is 1 then it is a greyscale and if the channel is 3 then we are using RGB

import tensorflow.keras.backend as K
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_widht, img_height)
else:
    input_shape = (img_widht, img_height, 3)


## image data generator is used to generate batches of tensor image data with real-time data augmentation. The data will be looped over (in batches) indefinitely. So we dont have to load the whole dataset in memory at once


plt.figure(figsize=(12,12))
for i in range (0,15):
    plt.subplot(5,3,i+1)
   for X_batch , Y_batch in train_generator:
        image = X_batch[0]
        plt.imshow(image)
        break

plt.tight_layout()
plt.show()

model = sequential()
model.add(Conv2D(64, (3,3), activation = 'relu', input_shape = input_shape))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(flatten())
model.add(Dense(64, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))
model.summary()





