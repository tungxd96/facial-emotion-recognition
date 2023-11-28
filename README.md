# Facial Emotion Recognition

Facial Emotion Recognition is a project that aims to detect and recognize facial expressions in images or video frames. This project utilizes convolutional neural network (CNN) and computer vision techniques to analyze facial features and predict the emotion expressed by the person.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Result](#result)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

Facial Emotion Recognition is a crucial component in human-computer interaction and can find applications in areas like virtual reality, customer feedback analysis, and more. This project provides an open-source solution for implementing facial emotion recognition using state-of-the-art convolutional neural network (CNN) models.

## Features

- Emotion detection in real-time or from images
- Support for 8 emotions (e.g., anger, contempt, disgust, fear, happiness, neutrality, sadness, surprise)

## Getting Started

### Prerequisites

Before running the project, ensure you have the following prerequisites:

- Python 3.x (e.g., 3.9.6)
- Additional dependencies: numpy, tensorflow, sklearn, dlib, matplotlib, cv2

### Installation

1. Install Python3:

    Follow the instructions [here](https://www.python.org/downloads/)

2. Install the dependencies:

    ```
    pip install numpy tensorflow scikit-learn dlib matplotlib opencv-python
    ```

3. Clone the repository:

    ```
    git clone https://github.com/tungxd96/facial-emotion-recognition.git
    cd facial-emotion-recognition
    ```

### Usage

#### Input Data

The input data for emotion classification is stored in the `tests/` folder. To evaluate the model's performance and classify emotions accurately, please add the images you want the model to analyze within this directory.

#### Guidelines for Adding Images

1. File Location:
    - Place your images in the `tests/` folder.
2. Supported File Extensions:
    - Supported image file extensions include `.png`, `.jpg`, and `.jpeg`.
3. File Structure:
    ```
    facial-emotion-recognition/
    │
    ├── tests/
    │   ├── image1.jpg
    │   ├── image2.png
    │   ├── image3.jpeg
    │   └── ...
    └── ...
    ```
3. Notes:
    - Ensure that the images accurately represent the scenarios you want the model to handle.
    - Maintain a diverse set of images to cover a broad range of emotions and facial expressions.
    - Feel free to add or modify any additional instructions based on your project's specific requirements. This provides users and contributors with clear guidance on how to structure and contribute input data for emotion classification.

#### Model Training

1. First-Time Training:

    - For first-time users, run the following command in your terminal:
        ```
        python3 main.py
        ```
    - This command will initiate the training process and save the trained model into a file for future use.
    - *Notes: The number of epochs for this training process is 100. Each epoch takes 3 - 3.5 minutes to complete. The whole training process takes around 5 to 6 hours.*

2. Using the Pre-trained Model:
    - For subsequent runs, you can simply execute:
        ```
        python3 main.py
        ```
    - This will utilize the pre-trained model to classify your test data without the need for retraining.

### Result

The accuracy rate based on 12 test images is 92% (11 out of 12). Below is the result of our model:

![Facial Emotion Recognition Result](https://github.com/tungxd96/facial-emotion-recognition/blob/main/results/result_1.png)

### Contributing

We welcome contributions!

### License

The project is licensed under [Tung Dinh License](https://github.com/tungxd96/facial-emotion-recognition/blob/main/LICENSE.md)

### Acknowledgments

#### Datasets
This project uses the **"Random Images for Face Emotion Recognition"** dataset from Kaggle. The dataset was created by Sudarshan Vaidya and can be found at:

[Kaggle - Random Images for Face Emotion Recognition](https://www.kaggle.com/datasets/sudarshanvaidya/random-images-for-face-emotion-recognition)

If you use this dataset in your research or work, please cite it appropriately. Here is an example citation:

```plaintext
@dataset{vaidya2023randomimages,
  author = {Sudarshan Vaidya},
  title = {Random Images for Face Emotion Recognition},
  year = {2023},
  publisher = {Kaggle},
  journal = {Kaggle Datasets},
  howpublished = {\url{https://www.kaggle.com/datasets/sudarshanvaidya/random-images-for-face-emotion-recognition}},
}
```

#### Images

The images utilized in this project have been obtained from freely accessible sources on the internet, such as Google, Unsplash, Flickr, and similar platforms where they are made available for download.