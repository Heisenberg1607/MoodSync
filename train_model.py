import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the images.csv file
data_path = '/Users/janyajaiswal/Desktop/ADBMS/MoodSync/MoodSync/images.csv'  # Update with your file path
data = pd.read_csv(data_path)

# Map emotion labels to integers (if necessary)
emotion_labels = {label: i for i, label in enumerate(data['emotion'].unique())}
data['emotion'] = data['emotion'].map(emotion_labels)

# Preprocess the pixel data
def preprocess_data(data):
    """
    Converts the 'pixels' column into a numpy array of images and normalizes them.
    """
    # Convert 'pixels' strings to numpy arrays
    images = data['pixels'].apply(lambda x: np.fromstring(x, sep=' ').reshape(48, 48, 1))
    X = np.stack(images).astype('float32') / 255.0  # Normalize pixel values
    y = data['emotion'].values  # Emotion labels
    return X, y

X, y = preprocess_data(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encode the labels
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes=len(emotion_labels))
y_test = to_categorical(y_test, num_classes=len(emotion_labels))

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(emotion_labels), activation='softmax')  # Number of emotion categories
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=64
)

# Save the trained model
model.save('emotion_model.h5')
print("Model saved as 'emotion_model.h5'.")

# Plot training results
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.show()
