# Handwritten-Digit-Detection Using ANN
This repository contains code and resources for detecting handwritten digits using Artificial Neural Networks (ANN).

# Table of Contents
- Introduction
- Installation
- Usage
- Dataset
- Preprocessing
- Model Training
- Evaluation
  
# Introduction
Handwritten digit detection is a popular problem in the field of [image recognition](https://medium.com/dataman-in-ai/module-6-image-recognition-for-insurance-claim-handling-part-i-a338d16c9de0). This project focuses on using Artificial Neural Networks (ANN) to train a model capable of recognizing and classifying handwritten digits accurately. The repository provides code and instructions to preprocess the data, train an ANN model, and evaluate its performance.

# Installation
To use the code in this repository, perform the following steps:
1. Clone the repository: git clone https://github.com/edilauxillea/Handwritten-Digit-Detection.git
2. Install the required dependencies: pip install -r requirements.txt

# Usage
Follow these steps to perform handwritten digit detection using ANN:
1. Prepare your dataset by following the instructions in the Dataset section.
2. Preprocess the data as described in the Preprocessing section.
3. Train the ANN model using the steps provided in the Model Training section.
4. Evaluate the performance of the model using the instructions in the Evaluation section.

# Dataset
The dataset used for this project can be downloaded from MNIST. It consists of a large collection of handwritten digits from 0 to 9, along with their corresponding labels.

# Preprocessing
Before training the ANN model, the dataset needs to be preprocessed. Follow these steps for preprocessing:
1. Load the images and labels from the dataset.
2. Normalize the pixel values of the images to a suitable range.
3. Flatten the images to convert them into a 1D array.
4. Split the dataset into training and testing sets.

# Model Training
This project utilizes Artificial Neural Networks (ANN) for handwritten digit detection. The steps for training an ANN model are as follows:
1. Design the architecture of the ANN, including the number and configuration of layers, activation functions, and optimizer.
2. Initialize the ANN model with suitable hyperparameters.
3. Train the model using the preprocessed training data.
4. Optimize the model by adjusting hyperparameters, such as the learning rate or number of epochs.
5. Save the trained model for future use.

# Evaluation
To evaluate the performance of the trained ANN model, follow these steps:
1. Load the saved model.
2. Preprocess the testing data using the same steps as mentioned in the Preprocessing section.
3. Use the model to predict labels for the testing images.
4. Calculate relevant metrics such as accuracy, precision, recall, and F1 score.
5. Analyze the results and interpret the model's performance.
