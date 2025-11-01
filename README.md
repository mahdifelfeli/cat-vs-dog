# Cat vs. Dog Image Classification 🐶🐱

A Convolutional Neural Network (CNN) built with TensorFlow and Keras to classify images as either cats or dogs. This model is trained on the classic Kaggle "Cats and Dogs" dataset.



---

## Overview

This project implements a standard CNN architecture from scratch to tackle a binary image classification problem. The model learns to distinguish between images of cats and dogs.

### Key Features
* **Model:** Sequential CNN built with Keras.
* **Data Augmentation:** Uses `ImageDataGenerator` to prevent overfitting and improve generalization by applying random shears, zooms, and horizontal flips.
* **Training:** Trained on the Kaggle dataset, utilizing generators to feed data in batches.
* **Prediction:** Includes a script to load the trained model and predict a single new image.

---

## Tech Stack

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3776AB?style=for-the-badge&logo=matplotlib&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

---

## Model Architecture

The model is a simple sequential CNN composed of:
1.  **Conv2D** (32 filters, 3x3 kernel) + ReLU Activation
2.  **MaxPooling2D** (2x2)
3.  **Conv2D** (32 filters, 3x3 kernel) + ReLU Activation
4.  **MaxPooling2D** (2x2)
5.  **Flatten**
6.  **Dense** (128 units) + ReLU Activation
7.  **Dense** (1 unit) + **Sigmoid** Activation (for binary classification)

---

## How to Use

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/mahdifelfeli/cat-vs-dog.git](https://github.com/mahdifelfeli/cat-vs-dog.git)
    cd cat-vs-dog
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    # Create environment
    python -m venv venv
    # Activate (Windows)
    .\venv\Scripts\activate
    # Activate (macOS/Linux)
    source venv/bin/activate
    
    # Install libraries
    pip install -r requirements.txt
    ```

3.  **Download the Dataset:**
    * This model was trained on the [Kaggle Cats and Dogs Dataset](https://www.kaggle.com/c/dogs-vs-cats/data).
    * You must download the dataset and place the `training_set` and `test_set` folders in the project directory, or update the paths in the `dog_and_cat.ipynb` notebook.

4.  **Run the Notebook:**
    * Open and run the `dog_and_cat.ipynb` notebook using Jupyter:
    ```bash
    jupyter notebook
    ```
