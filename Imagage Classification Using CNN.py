import os
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # intro to CNNs
## Layer 1: convolution layer
# converts images into an array of numbers and does dimensionality reduction of array,
# that doesn't let the model lose important info as well as complexity is reduced

## Layer 2: Relu - rectified linear unit
# converts all negatives into 0 and positives as is

## Layer 3: Pooling layer
# Reduce the dimensionality, helps control overfitting

## Layer 4: Fully Connected Layer
# Flatten the output of the previous layer and feed it to a fully connected layer


def download_file(url: str, output_path: str) -> None:
    if not os.path.exists(output_path):
        print(f"Downloading {output_path}...")
        urllib.request.urlretrieve(url, output_path)


def extract_zip(zip_path: str, target_dir: str) -> None:
    if not os.path.isdir(target_dir):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)


# Download datasets only if needed
download_file(
    "https://www.dropbox.com/s/t4pzwpvrzneb190/training_set.zip?dl=1",
    "training_set.zip",
)

download_file(
    "https://www.dropbox.com/s/i37jfni3d29raoc/test_set.zip?dl=1",
    "test_set.zip",
)

# Extract datasets only if needed
extract_zip("training_set.zip", "training_set")
extract_zip("test_set.zip", "test_set")

print(tf.__version__)

img_width, img_height = 150, 150
train_data_dir = os.path.join("training_set", "training_set")
validation_data_dir = os.path.join("test_set", "test_set")
batch_size = 20
epochs = 20


if K.image_data_format() == "channels_first":
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


train_datagen = ImageDataGenerator(rescale=1.0 / 255)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="binary",
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="binary",
)


x_batch, y_batch = next(train_generator)

plt.figure(figsize=(12, 12))
for i in range(min(9, batch_size)):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_batch[i])
    plt.title(f"Label: {y_batch[i]}")
    plt.axis("off")
plt.tight_layout()
plt.show()


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
steps_per_epoch = train_generator.samples // batch_size
validation_steps = validation_generator.samples // batch_size








# Train the model
steps_per_epoch = train_generator.samples // batch_size
validation_steps = validation_generator.samples // batch_size

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
)

val_loss, val_acc = model.evaluate(validation_generator, steps=validation_steps)
print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")
