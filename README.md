# SMS-Classifier
Overview
The SMS Classifier is a machine learning model designed to automatically classify SMS messages as either spam (unwanted or junk messages) or non-spam (ham). This kind of classification is crucial for filtering out unwanted messages, improving user experience, and enhancing security.

Dataset
The dataset used for this task typically consists of SMS messages labeled as either spam or ham. Each message is analyzed and processed to train the model. For this project, we use a publicly available dataset from UCI Machine Learning Repository.

Methodology
Data Preprocessing:

Load the dataset and preprocess it by mapping labels ('ham' to 0 and 'spam' to 1).
Clean and normalize the text data to make it suitable for model training.
Feature Extraction:

Use the CountVectorizer from the sklearn.feature_extraction.text module to convert text data into a matrix of token counts (bag-of-words model).
Model Training:

Split the dataset into training and testing sets to evaluate model performance.
Train a Multinomial Naive Bayes model using the training data. This algorithm is well-suited for text classification tasks.
Prediction and Evaluation:

Predict the labels of the test set using the trained model.
Evaluate the model performance using metrics like accuracy.
Code Implementation
The code implementation involves:

Importing necessary libraries.
Loading and preprocessing the dataset.
Extracting features using CountVectorizer.
Splitting the dataset into training and testing sets.
Training the Multinomial Naive Bayes model.
Predicting and evaluating the model's performance.
