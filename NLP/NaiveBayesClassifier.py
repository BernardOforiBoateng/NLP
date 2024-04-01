import os
import random
from collections import defaultdict
import math
from nltk.corpus import stopwords

# Set random seed for reproducibility
random.seed(42)

# Function to load reviews from files
def load_reviews(file_path):
    with open(file_path, encoding="latin-1") as f:
        return f.readlines()

# Function to preprocess data
def preprocess_data(data):
    preprocessed_data = []
    for line in data:
        tokens = line.lower().split()
        preprocessed_data.append(tokens)
    return preprocessed_data

# Function to split data into train, dev, and test sets
def split_data(data, train_percent=0.7, dev_percent=0.15, test_percent=0.15):
    total_size = len(data)
    train_size = int(total_size * train_percent)
    dev_size = int(total_size * dev_percent)
    
    train_data = data[:train_size]
    dev_data = data[train_size:train_size+dev_size]
    test_data = data[train_size+dev_size:]
    
    return train_data, dev_data, test_data

# Function to build vocabulary
def build_vocab(data, vocab_size=3000):
    all_words = [word for line in data for word in line]
    freq_dist = defaultdict(int)
    for word in all_words:
        freq_dist[word] += 1
    sorted_vocab = sorted(freq_dist.items(), key=lambda x: x[1], reverse=True)
    vocab = [word for word, _ in sorted_vocab[:vocab_size]]
    return vocab

# Function to extract features
def extract_features(review, vocab):
    features = {}
    for word in vocab:
        features[word] = review.count(word) > 0
    return features

# Function to train Naive Bayes classifier
def train_naive_bayes(train_data, vocab):
    class_counts = defaultdict(int)
    word_counts = defaultdict(lambda: defaultdict(int))
    total_count = 0
    for review, label in train_data:
        class_counts[label] += 1
        for word in review:
            if word not in stopwords.words('english'):  # Exclude stopwords
                word_counts[label][word] += 1
                total_count += 1
 
    prior_probabilities = {label: count / sum(class_counts.values()) for label, count in class_counts.items()}

    conditional_probabilities = {}
    for label in class_counts:
        conditional_probabilities[label] = {}
        for word in vocab:
            if word not in stopwords.words('english'):  # Exclude stopwords
                count_word_given_label = word_counts[label][word]
                conditional_probabilities[label][word] = (count_word_given_label + 1) / (class_counts[label] + len(vocab))
    
    return prior_probabilities, conditional_probabilities

# Function to predict label for a single review
def predict(review, prior_probabilities, conditional_probabilities):
    scores = defaultdict(float)
    for label, prior in prior_probabilities.items():
        scores[label] = math.log(prior)
        for word, present in review.items():
            if present:
                scores[label] += math.log(conditional_probabilities[label].get(word, 1e-10))
    return max(scores, key=scores.get)

# Function to evaluate classifier
def evaluate_classifier(classifier, data, vocab):
    prior_probabilities, conditional_probabilities = classifier
    correct_predictions = 0
    total_predictions = 0
    for review, label in data:
        total_predictions += 1
        predicted_label = predict(extract_features(review, vocab), prior_probabilities, conditional_probabilities)
        if predicted_label == label:
            correct_predictions += 1
    accuracy = correct_predictions / total_predictions
    return accuracy

# Function to find confident and uncertain examples
def find_examples(classifier, data, vocab):
    confident_examples = []
    uncertain_examples = []
    for review, label in data:
        features = extract_features(review, vocab)
        prob_positive = 1.0
        prob_negative = 1.0
        for word, present in features.items():
            if present:
                prob_positive *= classifier[1]['positive'].get(word, 1e-10)
                prob_negative *= classifier[1]['negative'].get(word, 1e-10)
        confidence = prob_positive / (prob_positive + prob_negative)
        if confidence > 0.9:
            confident_examples.append((review, label, confidence))
        elif confidence < 0.5:
            uncertain_examples.append((review, label, confidence))
    return confident_examples, uncertain_examples

# Function to print examples
def print_examples(examples, title):
    print(title)
    for review, label, confidence in examples:
        print(f"Review: {' '.join(review)}")
        print(f"Label: {label}, Confidence: {confidence}")
        print()

# Function to find the most useful words in each class
def most_useful_features(classifier, vocab, n=5):
    conditional_probabilities = classifier[1]
    useful_features = {'positive': [], 'negative': []}
    english_stopwords = stopwords.words('english')  # Load stopwords

    for label in useful_features.keys():
        # Calculate log odds for each word-label pair
        log_odds = {}
        for word in vocab:
            if word not in english_stopwords:  # Exclude stopwords
                prob_pos = conditional_probabilities['positive'].get(word, 0.0)
                prob_neg = conditional_probabilities['negative'].get(word, 0.0)
                log_odds[word] = math.log2(prob_pos / (prob_neg + 1e-10))  # Add smoothing

        # Sort by log odds and select top n features
        sorted_features = sorted(log_odds.items(), key=lambda x: x[1], reverse=True)[:n] if label == 'positive' else sorted(log_odds.items(), key=lambda x: x[1], reverse=False)[:n]
        useful_features[label] = sorted_features

    return useful_features
 
# Main function
def main():
    # Get the current directory of the script
    current_directory = os.path.dirname(os.path.realpath(__file__))

    # Define file paths to positive and negative reviews
    positive_file_path = os.path.join(current_directory, 'rt-polarity.pos')
    negative_file_path = os.path.join(current_directory, 'rt-polarity.neg')

    # Load positive and negative reviews from files
    positive_reviews = load_reviews(positive_file_path)
    negative_reviews = load_reviews(negative_file_path)

    random.shuffle(positive_reviews)
    random.shuffle(negative_reviews)

    positive_reviews = preprocess_data(positive_reviews)
    negative_reviews = preprocess_data(negative_reviews)

    # Step 1: Split data into train, dev, and test sets
    positive_train, positive_dev, positive_test = split_data(positive_reviews)
    negative_train, negative_dev, negative_test = split_data(negative_reviews)

    train_data = [(review, 'positive') for review in positive_train] + [(review, 'negative') for review in negative_train]
    dev_data = [(review, 'positive') for review in positive_dev] + [(review, 'negative') for review in negative_dev]
    test_data = [(review, 'positive') for review in positive_test] + [(review, 'negative') for review in negative_test]

    # Step 2: Build vocabulary
    vocab = build_vocab(positive_train + negative_train)

    # Step 3: Train Naive Bayes classifier and evaluate on development set
    classifier = train_naive_bayes(train_data, vocab)
    dev_accuracy = evaluate_classifier(classifier, dev_data, vocab)
    print(f"Accuracy on Development Set: {dev_accuracy}")

    # Step 4: Train best model on concatenation of training and development sets and evaluate on test set
    concatenated_train_data = train_data + dev_data
    classifier = train_naive_bayes(concatenated_train_data, vocab)
    test_accuracy = evaluate_classifier(classifier, test_data, vocab)
    print(f"Accuracy on Test Set: {test_accuracy}")

    # Step 5: Find confident and uncertain examples
    confident_examples, uncertain_examples = find_examples(classifier, test_data, vocab)

    # Step 6: Print examples
    print_examples(confident_examples[:5], "Top 5 Confident Examples:")
    print_examples(uncertain_examples[:5], "Top 5 Uncertain Examples:")

    useful_features = most_useful_features(classifier, vocab)
    print("Most useful features for positive class:")
    for feature, prob in useful_features['positive']:
        print(f"Feature: {feature}, Probability: {prob}")

    print("\nMost useful features for negative class:")
    for feature, prob in useful_features['negative']:
        print(f"Feature: {feature}, Probability: {prob}")
            
if __name__ == "__main__":
    main()

