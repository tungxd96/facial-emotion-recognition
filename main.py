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

MODEL_FILEPATH = 'models/image_emotion_classification.pkl'
EPOCHS = 10

image_preprocessor = ImagePreprocessor(debug_mode=False)
le = LabelEncoder()
le.fit(np.array(list(LABEL_MAP.keys())))

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