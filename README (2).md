# MNIST Handwritten Digit Classifier

A deep learning project that uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify handwritten digits from the MNIST dataset.

## Live Demo

Deployed with Streamlit Community Cloud.

## Features

- CNN-based handwritten digit classification
- MNIST training and evaluation
- Confusion matrix and training curves
- Upload an image of a handwritten digit
- Displays predicted digit and confidence score
- Interactive probability visualization

## Project Structure

```text
MNIST-Digit-Classifier/
├── app.py
├── mnist_cnn.keras
├── MNIST_CNN_Digit_Classifier.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

The application is designed for deployment on Streamlit Community Cloud using the GitHub repository as the source.

## Model

The model is a CNN with two convolutional/max-pooling blocks followed by dense layers and a 10-class softmax output.
