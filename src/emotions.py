import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import argparse
import sys

# To avoid Jupyter notebook argument errors, exclude kernel arguments
sys.argv = sys.argv[:1]  # Keeps only the script name, discards the rest

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="train/display/image", default="image")
mode = ap.parse_args().mode
mode="display"
print(mode)
import matplotlib.pyplot as plt
import numpy as np

def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'])
    axs[0].plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    
    # Fix for set_xticks to create a proper sequence
    xticks_range = np.arange(1, len(model_history.history['accuracy']) + 1, len(model_history.history['accuracy']) // 10)
    axs[0].set_xticks(xticks_range)
    
    axs[0].legend(['train', 'val'], loc='best')
    
    # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    
    # Fix for set_xticks to create a proper sequence for loss
    xticks_range_loss = np.arange(1, len(model_history.history['loss']) + 1, len(model_history.history['loss']) // 10)
    axs[1].set_xticks(xticks_range_loss)
    
    axs[1].legend(['train', 'val'], loc='best')
    
    fig.savefig('plot.png')
    plt.show()
# Define data generators
train_dir = 'data/train'
val_dir = 'data/test'

num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 50

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')
# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to handle emotion prediction on a provided image
def process_image(image_path, model, emotion_dict):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read the image.")
        return
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load haar cascade for face detection
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # If faces are detected, process each face
    if len(faces) == 0:
        print("No faces detected in the image.")

    for (x, y, w, h) in faces:
        # Extract the region of interest (face) from the grayscale image
        roi_gray = gray[y:y + h, x:x + w]
        
        # Resize and preprocess the image for prediction (assuming model expects 48x48)
        cropped_img = cv2.resize(roi_gray, (48, 48))  # Resize the face region to 48x48
        cropped_img = np.expand_dims(cropped_img, axis=-1)  # Add channel dimension (grayscale)
        cropped_img = np.expand_dims(cropped_img, axis=0)  # Add batch dimension

        # Make the prediction
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))

        # Put the emotion label on the image
        cv2.putText(img, emotion_dict[maxindex], (x + 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0, 255), 2, cv2.LINE_AA)
        
        # Draw the bounding box around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 10)

        # Print the detected emotion in the console
        print(f"Detected emotion: {emotion_dict[maxindex]}")

    # Convert the BGR image to RGB for displaying with matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image with the bounding box and emotion text using plt
    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.axis('off')  # Turn off axis
    plt.show()
    print("Done")
import cv2
import numpy as np
from keras.models import load_model
# If you want to train the same model or try other models, go for this
if mode == "train":
   # Compile the model before training
    model.compile(
        optimizer='adam',               # You can change the optimizer as needed
        loss='categorical_crossentropy', # Change this based on your problem (e.g., binary_crossentropy for binary classification)
        metrics=['accuracy']            # You can include other metrics as needed
    )

    # Now you can train the model
    model_info = model.fit(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size
    )
    # Plot model history (assuming you have a function `plot_model_history` defined)
    plot_model_history(model_info)

    # Save model weights
    # model.save_weights('model.h5')
    model.save_weights('model_h5.weights.h5')

# emotions will be displayed on your face from the webcam feed
elif mode == "display":
    model.load_weights('model_h5.weights.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # start the webcam feed
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():  # Check if webcam is opened
        print("Error: Could not open video stream.")
        exit()

    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break

        # Convert image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Load haar cascade for face detection
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            
            # Resize and preprocess the image for prediction (assuming model expects 48x48)
            cropped_img = cv2.resize(roi_gray, (48, 48))  # Resize the face region to 48x48
            cropped_img = np.expand_dims(cropped_img, axis=-1)  # Add channel dimension (grayscale)
            cropped_img = np.expand_dims(cropped_img, axis=0)  # Add batch dimension

            # Make the prediction
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the result in a window
        cv2.imshow('Video', cv2.resize(frame, (1600,900), interpolation=cv2.INTER_CUBIC))

        # Exit on pressing the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
elif mode == "image":
    image_path = "img10.jpg"  # Replace with the path to the image you want to process
    model.load_weights('model_h5.weights.h5')
    # Emotion dictionary
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # Call the function to process the image
    process_image(image_path, model, emotion_dict)
