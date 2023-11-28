import os
import concurrent.futures
import cv2
import configurations.config as config
import time
from PIL import Image
import numpy as np
import dlib
import matplotlib.pyplot as plt

class ImagePreprocessor:

    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode

    def process(
        self, 
        label_type: str = 'label' in ('label', 'image_path'), 
        folder_dir: str = 'datasets') -> (list, list):

        start_time = time.time()

        try:
            image_files = []
            for foldername, subfolders, filenames in os.walk(folder_dir):
                for filename in filenames:
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(foldername, filename)
                        label = foldername.split('/')[1] if label_type == 'label' else image_path
                        image_files.append((label, image_path))

            features, labels = [], []

            if self.debug_mode:
                for image_info in image_files:
                    result = self.preprocess_image(image_info=image_info)
                    label, normalized_image = result
                    features.append(normalized_image)
                    labels.append(label)
            else:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = {executor.submit(self.preprocess_image, image_info): image_info for image_info in image_files}
                    concurrent.futures.wait(futures)

                    for future in concurrent.futures.as_completed(futures):
                        label, normalized_image = future.result()
                        features.append(normalized_image)
                        labels.append(label)
                    
            return np.array(features), np.array(labels)
        finally:
            end_time = time.time()
            time_taken_ms = (end_time - start_time) * 1000
            print(f"Time taken for ImagePreprocessor.process: {time_taken_ms:.2f} ms")

    
    def preprocess_image(self, image_info: tuple) -> tuple:
        label, image_path = image_info
        image = cv2.imread(image_path)
        gray_face_cropped_image = self.crop_facial_image(image=image)
        resized_image = cv2.resize(gray_face_cropped_image, (128, 128))
        normalized_image = resized_image / 255.0
        return label, np.array(normalized_image)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        gray_face_cropped_image = self.crop_facial_image(image=image)
        resized_image = cv2.resize(gray_face_cropped_image, (128, 128))
        normalized_image = resized_image / 255.0
        return np.array(normalized_image)

    def crop_facial_image(self, image: str):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

        resized_image = cv2.resize(image, (900, 900), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        padding = 50
        cropped_face = gray
        cur_w, cur_h = 0, 0
        choose = ''
        for (x, y, w, h) in faces:
            if w <= cur_w or h <= cur_h:
                continue
            cur_w, cur_h = w, h
            cropped_face = gray[y-padding//2-10:y + h + padding-10, x + 20:x + w - 20]
            if self.debug_mode:
                cv2.rectangle(gray, (x, y - padding // 2), (x+w, y+h+padding), (255, 0, 0), 2)
                cv2.imshow('Cropped face' + str(w) + ', ' + str(h), cropped_face)
                choose = 'Cropped face' + str(w) + ', ' + str(h)
        
        if self.debug_mode:
            print('Debug choose:', choose)
            cv2.imshow('Detected Face', gray)
            cv2.waitKey(0)

        return cropped_face