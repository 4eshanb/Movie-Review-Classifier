"""
Created on Apr 15, 2014
@author: Reid Swanson

Modified on: April 14, 2015
Modified by: Kendall Lewis

Modified on: Oct 24, 2019
Modified by: Wen Cui
"""

import nltk, random
from nltk.corpus import movie_reviews

# This will be used for feature selection later on
# Initialize it to None so that we use all the features the first time around
selected_features = None


# Adds unigram based lexical features
def add_lexical_features(fdist, feature_vector):
    for word, freq in fdist.items():
        fname = "unigram:{0}".format(word)

        # If we haven't selected any features yet then add the feature to
        # our feature vector
        # Otherwise make sure the feature is one of the ones we want
        # Note we use a Set for the selected features for O(1) lookup
        if selected_features == None or fname in selected_features:
            feature_vector[fname] = 1

        
# Adds all our features and returns the vector
def features(review_text, review_words):
    feature_vector = {}

    uni_dist = nltk.FreqDist(review_words)
    
    add_lexical_features(uni_dist, feature_vector)
    #print(len(feature_vector))
    #raise NotImplemented
    return feature_vector

def evaluate(classifier, data, reviews, output_file):
    fh = open(output_file, 'w', encoding='utf-8')

    # test on the data
    accuracy = nltk.classify.accuracy(classifier, data)
    fh.write("{0:10s} {1:8.5f}\n\n".format("Accuracy", accuracy))

    features_only = [example[0] for example in data]

    reference_labels = [example[1] for example in data]
    predicted_labels = classifier.classify_many(features_only)
    reference_text = [review[0] for review in reviews]
    confusion_matrix = nltk.ConfusionMatrix(reference_labels, predicted_labels)

    fh.write(str(confusion_matrix))
    fh.write('\n\n')

    for reference, predicted, text in zip(
                                          reference_labels,
                                          predicted_labels,
                                          reference_text
                                          ):
        if reference != predicted:
            fh.write("{0} {1}\n{2}\n\n".format(reference, predicted, text))

    fh.close()
    
if __name__ == '__main__':
    # You have to download the movie reviews first
    #nltk.download("movie_reviews")
    reviews = [
         (movie_reviews.raw(fid), list(movie_reviews.words(fid)), category) 
         for category in movie_reviews.categories() 
         for fid in movie_reviews.fileids(category)
         ]
    
    # Make sure we split the same way every time for the live coding
    random.seed(0)
    
    # Make sure to randomize the reviews first!
    random.shuffle(reviews)
    
    # Convert the data into feature vectors
    featuresets = [
        (features(review_text, review_words), label)
        for (review_text, review_words, label) in reviews
    ]
    
    # Split into the different datasets
    train_data      = featuresets[:1700]
    develop_data    = featuresets[1700:1800]
    test_data       = featuresets[1800:]

    print(train_data[0])
    # Train on the training data
    classifier = nltk.NaiveBayesClassifier.train(train_data)
    
    # Test on the development and test data
    dev_accuracy = nltk.classify.accuracy(classifier, develop_data)
    test_accuracy = nltk.classify.accuracy(classifier, test_data)

    print("{0:6s} {1:8.5f}".format("Dev", dev_accuracy))
    print("{0:6s} {1:8.5f}".format("Test", test_accuracy))
    

    # # Feature selection
    best = (0.0, 0)
    best_features = classifier.most_informative_features(10000)
    
    for i in [2**i for i in range(5, 15)]:
        selected_features = set([fname for fname, value in best_features[:i]])

        #print(selected_features)
        #print(i)
        #print(len(selected_features))
        
        featuresets = [
            (features(review_text, review_words), label)
            for (review_text, review_words, label) in reviews
        ]
        
        #print(featuresets)

        train_data      = featuresets[:1700]
        develop_data    = featuresets[1700:1800]
        test_data       = featuresets[1800:]

        classifier = nltk.NaiveBayesClassifier.train(train_data)
        accuracy = nltk.classify.accuracy(classifier, develop_data)
        print("{0:6d} {1:8.5f}".format(i, accuracy))

        if accuracy > best[0]:
            best = (accuracy, i)

    # Now train on the best features
    selected_features = set([fname for fname, value in best_features[:best[1]]])
    
    featuresets = [
        (features(review_text, review_words), label)
        for (review_text, review_words, label) in reviews
    ]

    train_data      = featuresets[:1700]
    develop_data    = featuresets[1700:1800]
    test_data       = featuresets[1800:]

    classifier = nltk.NaiveBayesClassifier.train(train_data)
    accuracy = nltk.classify.accuracy(classifier, test_data)
    print("{0:6s} {1:8.5f}".format("Test", accuracy))
    evaluate(classifier, test_data, reviews, "output.txt")
    