# FashionMNIST FeedForward Neural Network

This project implements a feedforward neural network (FNN) for image classification on the FashionMNIST dataset. The goal is to classify images of clothing articles into their respective categories using a deep neural network.

## Project Overview

- **Feedforward Neural Network (FNN)**: A deep neural network is used to classify the FashionMNIST dataset. The model is implemented using PyTorch.
- **FashionMNIST Dataset**: The dataset consists of grayscale images of clothing items, divided into various categories such as T-shirts, shoes, and bags.

## Project Structure

- **FashionMNIST.py**: The main Python script that implements the feedforward neural network for classification.
- **data/FashionMNIST**: Contains the FashionMNIST dataset in both raw and processed formats.

## Installation

### Prerequisites

- **Python 3.x**: Ensure Python 3.x is installed on your machine.
- **Required Libraries**: Install the required libraries listed in `requirements.txt`.

### Setup Instructions

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/YourUsername/FashionMNIST-FeedForwardNN.git
    cd FashionMNIST-FeedForwardNN
    ```

2. **Install Dependencies**:
    Install the necessary dependencies for running the project:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Project

1. **Run the FeedForward Neural Network**:
    Execute the script to train the feedforward neural network on the FashionMNIST dataset:
    ```bash
    python FashionMNIST.py
    ```

2. **Evaluate the Model**:
    The script will output evaluation results on the test dataset.

## Project Workflow

1. **Data Loading**: Load the FashionMNIST dataset from the `data/FashionMNIST/` directory.
2. **Model Training**: Train the feedforward neural network on the FashionMNIST dataset.
3. **Evaluation**: Evaluate the model on the test set and output the classification results.
