import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the path to your training data folders
train_data_dir = r'D:\6TH SEM\bio optimized\english\english\data'
img_height = 90
img_width = 60
num_classes = 5

# Load and preprocess the training data
def load_data(data_dir):
    images = []
    labels = []
    for label, folder in enumerate(sorted(os.listdir(data_dir))):
        for file in os.listdir(os.path.join(data_dir, folder)):
            img_path = os.path.join(data_dir, folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_width, img_height))
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

train_images, train_labels = load_data(train_data_dir)
train_images = train_images.astype('float32') / 255.0

# Define and train the MLP model
model = models.Sequential([
    layers.Flatten(input_shape=(img_height, img_width)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=100, validation_split=0.2)

# Calculate accuracy
_, accuracy = model.evaluate(train_images, train_labels)
print(f'Training Accuracy: {accuracy * 100:.2f}%')

# Save the trained model
model.save('handwritten_alphabet_model.h5')

# Example of making inferences (predictions)
def predict_image(model, img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_width, img_height))
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img.astype('float32') / 255.0
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    return predicted_label

# Test inference on a new image
test_image_path = r'D:\6TH SEM\bio optimized\english\english\data\C\c1.png.jpg'
predicted_label = predict_image(model, test_image_path)

d=["A","B","C","D","E"]

print(f'Predicted Label: {d[predicted_label]}')