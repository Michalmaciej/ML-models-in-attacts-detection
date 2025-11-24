# ML-models-in-attacks-detection

This repository contains implementations of various machine learning models for network intrusion detection. It includes implementations of both classical ML algorithms and more recent generative models. Google Colab was used to write code and run the code.

## Features and Functionality

*   **Classical Machine Learning Models:** Implementation of common classification algorithms for attack category and label prediction.
    *   Decision Tree Classifier
    *   K-Nearest Neighbors Classifier
    *   Logistic Regression
    *   Multilayer Perceptron Classifier
    *   Gaussian Naive Bayes Classifier
    *   Random Forest Classifier

*   **Generative Adversarial Network (GAN):** Implements a GAN to generate synthetic network traffic data.
*   **Autoencoder:** Implements an autoencoder to learn compressed representations of network traffic data and detect anomalies based on reconstruction error.
*   **VAE and OpenAI API Integration**:  Leverages the OpenAI API to classify network traffic data using prompts and few-shot learning.

*   **Data Preprocessing:** Includes data loading, categorical feature encoding (one-hot encoding), scaling, and splitting into training and testing sets.

## Technology Stack

*   Python 3.x
*   pandas
*   NumPy
*   scikit-learn
*   TensorFlow/Keras
*   fastai
*   openai

## Prerequisites

Before running the code, ensure you have the following installed:

*   Python (>=3.6)
*   pip (Python package installer)

Install the required Python packages using pip:

```bash
pip install pandas numpy scikit-learn tensorflow fastai openai
```

You will also need:

*   An OpenAI API key for `VAE_s_and_Openai_API.py`.  Replace `"chatgpt-openai-api-key"` with your actual API key in the script.
*   Kaggle account and download of the UNSW-NB15 dataset (if executing `data_preprocessing.py` outside of Kaggle)

## Installation Instructions

1.  Clone the repository:

    ```bash
    git clone https://github.com/Michalmaciej/ML-models-in-attacts-detection.git
    cd ML-models-in-attacts-detection
    ```

2.  (Optional) Create a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    ```

3.  Install the dependencies (if not already installed):

    ```bash
    pip install pandas numpy scikit-learn tensorflow fastai openai
    ```

## Usage Guide

Each Python file in the repository implements a specific model or functionality.  Here's how to run each script:

*   **Classical ML Models (Decision_tree.py, KNeighbors.py, Logistic_regression.py, MLPC.py, Naive_Bayes_Classifier.py, chatgpt_randomforest.py, modified_chatgpt_randomforest.py):**

    ```bash
    python Decision_tree.py  # Example
    python KNeighbors.py
    python Logistic_regression.py
    python MLPC.py
    python Naive_Bayes_Classifier.py
    python chatgpt_randomforest.py
    python modified_chatgpt_randomforest.py
    ```

    These scripts load data, train the specified model, and evaluate its performance using confusion matrices and classification reports. They expect the relevant data to be available in the locations hardcoded in the files. Note that some models are trained twice, once with data split into `X_trainf` and `X_testf`, and again with data split into `X_trainff` and `X_testff`. You'll need to inspect these files to properly configure them to read data and point them to your datasets

*   **GAN (GANS.py):**

    ```bash
    python GANS.py
    ```

    This script trains a Generative Adversarial Network (GAN) on a subset of the UNSW-NB15 dataset (50,000 rows). It prints the discriminator and generator loss for each epoch and the mean squared error (MSE) of the generated data. This script assumes data is located in `/content/drive/MyDrive/ColabNotebooks/UNSW_NB15_testing-set.parquet` and requires modification if you have a different file path.

*   **Autoencoder (autoencoder.py):**

    ```bash
    python autoencoder.py
    ```

    This script trains an autoencoder on the UNSW-NB15 dataset. It prints the train and test mean squared error (MSE) to evaluate the reconstruction quality.  This script assumes data is located in `/content/drive/MyDrive/ColabNotebooks/UNSW_NB15_training-set.parquet` and requires modification if you have a different file path.

*   **VAE and OpenAI API (VAE\_s\_and\_Openai\_API.py):**

    ```bash
    python VAE_s_and_Openai_API.py
    ```

    This script uses the OpenAI API to classify network traffic data based on a prompt.  Ensure you replace `"chatgpt-openai-api-key"` with your actual API key.  This script assumes data is located in `/content/drive/MyDrive/ColabNotebooks/UNSW_NB15_training-set.parquet` and requires modification if you have a different file path.

*   **Data Preprocessing (data\_preprocessing.py):**

    This script is designed to be run within a Kaggle environment.

    ```bash
    python data_preprocessing.py
    ```

    If you wish to run it outside of Kaggle, ensure the UNSW-NB15 dataset is available in the `/kaggle/input/unswnb15/` directory or modify the file path accordingly. This script reduces the memory footprint of the dataframe and converts the CSV file into Parquet format for faster loading.

**Important Notes:**

*   File paths: Several scripts use hardcoded file paths (e.g., `/content/drive/MyDrive/ColabNotebooks/UNSW_NB15_training-set.parquet`).  You **must** modify these paths to point to the correct location of your UNSW-NB15 dataset.
*   Data: The code is designed to work with the UNSW-NB15 dataset. Ensure you have downloaded and prepared the data as needed.
*   Hardware: Training deep learning models (GAN, Autoencoder) can be computationally intensive.  Consider using a GPU for faster training.
*   Experimentation: The provided scripts offer starting points.  Experiment with different model parameters, architectures, and data preprocessing techniques to improve performance.

## API Documentation (OpenAI)

The `VAE_s_and_Openai_API.py` script uses the OpenAI API for classifying network traffic data. To use this functionality, you will need to:

1.  Sign up for an OpenAI account: [https://openai.com/](https://openai.com/)
2.  Obtain an API key.
3.  Replace the placeholder `"chatgpt-openai-api-key"` in the script with your actual API key.

The script uses the `openai.ChatCompletion.create()` endpoint. Refer to the OpenAI API documentation for detailed information on parameters and usage: [https://platform.openai.com/docs/api-reference/chat](https://platform.openai.com/docs/api-reference/chat)

## Contributing Guidelines

Contributions to this project are welcome!  Here's how you can contribute:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix:

    ```bash
    git checkout -b feature/your-feature-name
    ```

3.  Make your changes and commit them with descriptive commit messages.
4.  Push your changes to your forked repository.
5.  Submit a pull request to the `main` branch of the original repository.

Please ensure your code follows the existing coding style and includes appropriate tests.

## License Information

No license specified. All rights reserved.

## Contact/Support Information

For questions or support, please contact Michalmaciej via GitHub.
