'''
Filename: webcam_drawpad.py
This is the main 'runner' for the QuickDraw webcam model. To run this
file, you must have a webcam plugged in via USB to your computer that's able to see
a strip of paper that you can hold up to it. I've already trained a model that's able
to recognize drawings of fruit and of shapes, they can be found in the 'models' folder
in this directory. Simply change the path on line 15 if you need to use a different
model.

To use the QuickDraw webcam model, simply run this code, draw your shape or fruit on a piece
of paper (preferably black marker on white paper), and hold it up to the webcam. The popup window
will then recognize the shape or the fruit with a confidence level.
NOTE: You may need to use a thick marker like a Sharpie in order for the webcam to recognize the
object properly. For best reults, hold the paper 3-5 inches away from the camera.
'''

import cv2
import numpy as np
import tensorflow as tf
import keras
import os

model_path = os.path.abspath("models/quickdraw_model_fruit_v2.keras") # change path in models/ if using the fruit model
model = keras.models.load_model(model_path)
CLASS_NAMES = ['square', 'circle', 'triangle'] # change class names if using a different model

def predict(image):
    norm = image / 255.0
    reshaped = norm.reshape(1, 28, 28, 1)
    prediction = model.predict(reshaped, verbose=0)[0]
    label = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)
    return label, confidence

def find_camera_indices():
    available_indices = []
    # Test indices 0 through 5
    for i in range(6):
        # We test with CAP_DSHOW as it's often more stable for discovery on Windows
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Index {i}: Found a working camera!")
                available_indices.append(i)
            cap.release()
    return available_indices

indices = find_camera_indices()
print(f"All available indices: {indices}")

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) # if using a laptop internal webcam, set this to 0. otherwise, set it to 1
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))


if not cap.isOpened():
    print("Cannot access webcam.") # appears if you don't have a webcam plugged in
    exit()

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert the webcam to grayscale
    small_img = cv2.resize(gray, (28,28)) # convert image dimensions to 28 x 28
    _, bw_img = cv2.threshold(small_img, 127, 255, cv2.THRESH_BINARY_INV) # change to THRESH_BINARY_INV if you need black drawings on white (e.g., Sharpie on paper)

    label, confidence = predict(bw_img)

    display_img = cv2.resize(bw_img, (280,280), interpolation=cv2.INTER_NEAREST) # scale image to 280 x 280
    cv2.rectangle(display_img, (0,0), (280,40), 0, -1) # add a black rectangle to top of screen
    text = f"Confidence: {label} ({confidence:.2f})" # add confidence label
    cv2.putText(display_img, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2) # set font
    cv2.imshow("QuickDraw Webcam", display_img) # set title of window

    if cv2.waitKey(1) & 0xFF == ord('q'): # if q is pressed, quit out
        break

    if cv2.getWindowProperty("QuickDraw Webcam", cv2.WND_PROP_VISIBLE) < 1: # if window is closed, quit out
        break 
    
cap.release()
cv2.destroyAllWindows()