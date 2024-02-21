# Sentiment Analysis on Movie Reviews using Naive Bayes Classifier

## Overview

Welcome to the Sentiment Analysis on Movie Reviews project repository! This project explores the implementation and evaluation of a Naive Bayes classifier for sentiment analysis on a movie review dataset. The primary objectives include data preprocessing, vocabulary building, model implementation, training, evaluation, and analysis of the classifier's performance.

## Data Source

The movie review dataset used in this project was obtained from [Denny Britz's GitHub repository](https://github.com/dennybritz/cnn-text-classification-tf/tree/master/data/rt-polaritydata). It contains both negative and positive reviews, making it suitable for sentiment analysis tasks.

## Project Structure

- **Data Preprocessing:** Raw text data is preprocessed through tokenization, lowercase conversion, and stopwords removal to prepare it for classification.
- **Vocabulary Building:** A vocabulary of 3000 most frequent words is built from the training data to represent features for the classifier.
- **Data Splitting:** The dataset is split into training (70%), development (15%), and test (15%) sets for training, tuning, and evaluation purposes.
- **Naive Bayes Classifier Implementation:** The classifier is implemented from scratch, adhering to Naive Bayes principles for feature extraction, likelihood computation, and prediction.
- **Model Training and Evaluation:** The classifier is trained on the training set, tuned on the development set, and evaluated on both development and test sets to assess its performance.
- **Analysis:** Confident and uncertain examples are analyzed to understand the classifier's behavior, and the most useful features for sentiment classification are identified.

## Results

- **Development Set Accuracy:** 73.97%
- **Test Set Accuracy:** 76.03%

## Conclusion

The project provides valuable insights into sentiment analysis on movie reviews using a Naive Bayes classifier. Through meticulous preprocessing, model implementation, and evaluation, an effective classifier is developed and analyzed comprehensively. Further experimentation and exploration of advanced techniques could lead to enhancements in classifier performance and deeper insights into sentiment analysis tasks.

## Future Work

Future work may involve exploring advanced techniques such as lemmatization, feature selection, and model optimization to improve classifier performance. Additionally, investigating other machine learning algorithms and ensembling techniques could provide further insights and potentially enhance classification accuracy.

## Contact

For any questions, feedback, or collaboration opportunities, feel free to reach out to [bbofori90@gmail.com].

Happy exploring sentiment analysis in movie reviews! 🎬📝
