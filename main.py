from preprocessing.image_preprocessor import ImagePreprocessor
from sklearn.model_selection import train_test_split
from models.model import initialize_cnn_model
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
import tensorflow as tf
from configurations.config import LABEL_MAP
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import traceback
from visualization.visualize import visualize_multiple_images
import cv2
import time

MODEL_FILEPATH = 'models/image_emotion_classification.pkl'
EPOCHS = 10

image_preprocessor = ImagePreprocessor(debug_mode=False)
le = LabelEncoder()
le.fit(np.array(list(LABEL_MAP.keys())))

def train():
    try:
        with open(MODEL_FILEPATH, 'rb') as file:
            model = pickle.load(file)

        print('Model loaded')

        tests, labels = image_preprocessor.process(label_type='image_path', folder_dir = 'tests')

        predictions = model.predict(tests)
        result = [le.inverse_transform([prediction.argmax()])[0] for prediction in predictions]
            
        visualize_multiple_images(image_paths=labels, texts=result)
        
    except Exception as e:
        traceback.print_exc()

        print('No pre-trained model found. Attempting to train a new model...')

        features, labels = image_preprocessor.process()
        le = LabelEncoder()

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        y_train = le.transform(y_train)
        y_test = le.transform(y_test)
        res = le.inverse_transform(np.array([0, 1, 2, 3, 4, 5, 6, 7]))

        X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

        y_train = tf.keras.utils.to_categorical(y_train, num_classes=8)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=8)

        model = initialize_cnn_model()

        history = model.fit(x=X_train, y=y_train, epochs=100, batch_size=128, validation_data=(X_test, y_test), verbose=2)

        with open(MODEL_FILEPATH, 'wb') as file:
            pickle.dump(model, file)

def train_webcam():
    with open(MODEL_FILEPATH, 'rb') as file:
        model = pickle.load(file)
    next_prediction_time = time.time()
    cap = cv2.VideoCapture(0)
    emotion = ''
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if time.time() >= next_prediction_time:
            image = image_preprocessor.preprocess_image(image=frame)
            predictions = model.predict(np.array([image]))
            for prediction in predictions:
                emotion = le.inverse_transform([prediction.argmax()])[0]
            next_prediction_time = time.time() + 5

        cv2.putText(frame, f'Emotion: {emotion}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

train_webcam()