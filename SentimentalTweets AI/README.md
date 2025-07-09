# 🐦❤️‍🩹 TweetPulse: RNN & LSTM for Sentiment Analysis 

## Overview

This repository features **TweetPulse**, a deep learning project dedicated to **sentiment analysis of tweets**. We explore and compare the effectiveness of **Recurrent Neural Networks (RNNs)** and **Long Short-Term Memory (LSTM) networks** in classifying tweet sentiment as positive or negative. Using a large-scale dataset of 1.6 million processed tweets, this project demonstrates a complete pipeline from data preprocessing and balancing to model training, evaluation, and real-time prediction. Our goal is to provide insights into how different recurrent architectures perform on text classification tasks. 💡

## Features

* **Large-Scale Dataset:** Utilizes the comprehensive [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) (`training.1600000.processed.noemoticon.csv`) containing 1.6 million tweets for robust model training. 📊
* **Advanced Text Preprocessing:** Implements a series of text cleaning steps, including removal of URLs, mentions, hashtags, and special characters, to prepare raw tweets for neural network input. ✂️
* **Tokenization & Padding:** Converts text into numerical sequences using `Tokenizer` and ensures uniform input length with `pad_sequences` for efficient model processing. 📏
* **Class Imbalance Handling:** Calculates and applies `class weights` during training to mitigate the effects of an imbalanced dataset, ensuring fair learning across sentiment classes. ⚖️
* **RNN & LSTM Comparison:** Builds and trains both a `SimpleRNN` model and an `LSTM` model, allowing for a direct comparison of their performance in capturing long-range dependencies in text. 🧠
* **Model Persistence:** Integrates functionality to save and load trained models to/from Google Drive, enabling persistent storage and easy deployment. 💾
* **Real-time Prediction:** Includes a user-friendly function to predict the sentiment of any new input tweet, demonstrating the models' practical application. ✅
* **Performance Visualization:** (If applicable, add if the notebook has plots for loss/accuracy over epochs. Otherwise, remove/adjust) Visualizes training and validation metrics to provide insights into model learning curves and convergence. 📉📈
* **TensorFlow & Keras:** Developed using the powerful TensorFlow 2.x and Keras API for efficient deep learning development. 🐍

## Project Structure

* `RNN2.ipynb`: The main Jupyter Notebook containing all the code for data loading, preprocessing, model building (RNN & LSTM), training, evaluation, and prediction. 📝

## Getting Started

### Prerequisites

* Python 3.x
* TensorFlow 2.x
* pandas
* numpy
* matplotlib
* scikit-learn

### Installation

1.  **Clone this repository:**
    ```bash
    git clone [https://github.com/Basmala-M-Farouk/Deep-Learning-Projects.git](https://github.com/Basmala-M-Farouk/Deep-Learning-Projects.git)
    ```
    * **Note:** If you are cloning a specific subfolder, you first clone the main repository, then navigate.
2.  **Navigate into the RNN2 project directory:**
    ```bash
    cd Deep-Learning-Projects/SentimentalTweets AI
    ```
    
3.  **Install the required libraries:**
    ```bash
    pip install tensorflow pandas numpy matplotlib scikit-learn
    ```

### Usage

1.  **Mount Google Drive:** The notebook assumes your dataset (`training.1600000.processed.noemoticon.csv`) will be accessed from Google Drive. Ensure it's located at `/content/drive/MyDrive/tweets_data/`. You will be prompted to mount your Google Drive within the notebook.
2.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook RNN2.ipynb
    ```
3.  Run all cells in the notebook to execute the entire project pipeline, from data loading to model training and testing with custom inputs. ▶️

## Results

This project provides a comparative analysis of RNN and LSTM models for tweet sentiment classification. Key performance metrics (e.g., accuracy, loss) for both models are captured during training and evaluation, demonstrating their capabilities on this large dataset. The notebook also showcases the ability to predict sentiment on new, unseen tweets. 🏆

*(Please refer to the `RNN2.ipynb` notebook for detailed training logs, evaluation metrics, and comparative plots of RNN vs. LSTM performance.)*

## Contributing

Contributions are welcome! Whether it's improving model architecture, optimizing preprocessing, or adding new features, feel free to fork this repository, open issues, and submit pull requests. 🤝

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. 📄

## Contact 📧

https://www.linkedin.com/in/basmala-mohamed-farouk-079588223/ 
