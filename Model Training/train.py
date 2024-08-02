import pandas as pd
from sklearn.model_selection import train_test_split
import random
import numpy as np
import tensorflow as tf
import cv2
import os
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.utils import np_utils
from tensorflow.keras.initializers import GlorotUniform
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping


random.seed(42)
#Reading trainning and validation data in pandas dataframe
data = pd.read_csv('train_labels.csv')
data = data.sample(frac=1)

train_data, test_data = train_test_split(data,test_size=0.1,random_state=42)
validation_data = pd.read_csv('val_labels.csv')
validation_data = validation_data.sample(frac=1)

train_images = train_data['img_name'].tolist()
train_labels = train_data['label'].tolist()

validation_images = validation_data['img_name'].tolist()
validation_labels = validation_data['label'].tolist()

test_images = test_data['img_name'].tolist()
test_labels = test_data['label'].tolist()

num_classes = 13  # Number of classes


trainLabels_class = train_labels

class_order = range(num_classes)

# Convert labels to one-hot encoded format
train_labels = to_categorical(train_data['label'], num_classes=num_classes)
validation_labels = to_categorical(validation_data['label'], num_classes=num_classes)
test_labels = to_categorical(test_data['label'], num_classes=num_classes)

# Reorder the columns of one-hot encoded vectors according to the true class order
train_labels = train_labels[:, class_order]
validation_labels = validation_labels[:, class_order]
test_labels = test_labels[:, class_order]


print("Shape after one-hot encoding: ", train_labels.shape)

initializer = GlorotUniform()

#Building the model
model = tf.keras.Sequential([
                            tf.keras.layers.Conv2D(32, (1, 1), activation='relu', input_shape=(224, 224, 3)),
                            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                            tf.keras.layers.MaxPooling2D((2, 2)),
                            tf.keras.layers.Conv2D(32, (3, 3), activation='relu') ,
                            tf.keras.layers.Conv2D(32, (3, 3), activation='relu') ,
                            tf.keras.layers.Conv2D(32, (3, 3), activation='relu') ,
                            tf.keras.layers.MaxPooling2D((2, 2)),
                            tf.keras.layers.Conv2D(64, (3, 3), activation='relu') ,
                            tf.keras.layers.Conv2D(64, (3, 3), activation='relu') ,
                            tf.keras.layers.Conv2D(64, (3, 3), activation='relu') ,
                            tf.keras.layers.MaxPooling2D((2, 2)),
                            tf.keras.layers.Conv2D(128, (3, 3), activation='relu') ,
                            tf.keras.layers.Conv2D(128, (3, 3), activation='relu') ,
                            tf.keras.layers.Conv2D(128, (3, 3), activation='relu') ,
                            tf.keras.layers.MaxPooling2D((2, 2)),
                            tf.keras.layers.Flatten(),
                            tf.keras.layers.Dense(512, activation='relu'),
                            tf.keras.layers.Dense(512, activation='relu'),
                            tf.keras.layers.Dense(31, activation='softmax'),
                            ])

#USE THE FOLLOWING BLOCK OF CODE FOR RESNETS
# import tensorflow as tf
# model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# model = tf.keras.Sequential([
#     model,
#     tf.keras.layers.GlobalAveragePooling2D(),  # Add a global average pooling layer
#     tf.keras.layers.Dense(64, activation='relu'),  # Add a dense fully connected layer
#     tf.keras.layers.Dropout(0.5),  # Add dropout for regularization
#     tf.keras.layers.Dense(num_classes, activation='softmax')  # Add a dense layer with softmax activation
# ])

model.summary()


images=[]
valid_images = []
test_images_arr = []
#Generating training images
for image_name in train_images:
    image_path = os.path.join('train_set', image_name)
    if os.path.exists(image_path):
        # Read the image using OpenCV
        image = cv2.imread(image_path)
        # Optionally, you can resize the image if needed
        # image = cv2.resize(image, (desired_width, desired_height))
        image = cv2.resize(image, (224, 224))
        # Normalize pixel values to [0, 1]
        image = image / 255.0
        images.append(image)
    else:
        print(f"Image {image_name} not found.")

#Generating validation images
for image_name in validation_images:
    image_path = os.path.join('val_set', image_name)
    if os.path.exists(image_path):
        # Read the image using OpenCV
        image = cv2.imread(image_path)
        # Optionally, you can resize the image if needed
        # image = cv2.resize(image, (desired_width, desired_height))
        image = cv2.resize(image, (224, 224))
        # Normalize pixel values to [0, 1]
        image = image / 255.0
        valid_images.append(image)
    else:
        print(f"Image {image_name} not found.")

#Generating test images
for image_name in test_images:
    image_path = os.path.join('train_set', image_name)
    if os.path.exists(image_path):
        # Read the image using OpenCV
        image = cv2.imread(image_path)
        # Optionally, you can resize the image if needed
        # image = cv2.resize(image, (desired_width, desired_height))
        image = cv2.resize(image, (224, 224))
        # Normalize pixel values to [0, 1]
        image = image / 255.0
        test_images_arr.append(image)
    else:
        print(f"Image {image_name} not found.")

train_image_input = np.array(images)
#train_labels = np.array(train_labels)

validation_image_input = np.array(valid_images)
#validation_labels = np.array(validation_labels)

test_image_input = np.array(test_images_arr)
#test_labels = np.array(test_labels)


#early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Define custom accuracy metric based on maximum argument prediction
def max_arg_accuracy(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))
    return accuracy

# Compile model with custom accuracy metric
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=[max_arg_accuracy])

# Train the model
history = model.fit(train_image_input, train_labels, epochs=10, batch_size=100,
                     validation_data=(validation_image_input, validation_labels))

#Save the model
model.save(os.path.join('model3.keras'))

test_loss, test_acc = model.evaluate(test_image_input, test_labels)
print('Test accuracy:', test_acc)

#Testing model performance
predictions = model.predict(test_image_input)
print(predictions)