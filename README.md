# Movie-Review-Classifier

Given the text of a movie review, multiple classifiers are used to know whether it is expressing a positive or
a negative opinion about a movie.

### Data  
There is a total of 500 movie reviews: 250 that have been given a positive rating (an overall score
\> 5 out of 10) and 250 that have been given a negative rating (an overall score < 5). 
These reviews havebeen split into training, development, and testing files. 

Each dataset contains a list of reviews. Each review has two facets, Overall, which specifies whether the
review is positive or negative, based on the score from 1-10 and Text, which gives the textual review. Each
review is separated by a period character (.)

### Binning 
First the raw counts of values of each
feature are used. Then the binned value of raw counts are used. For example, imagine the word the occurs 10 times, food
occurs 2 times, and the bigram the food occurs 1 time.
The string vectors the same as before except now using the raw counts.   
Hence the new feature vector from the previous example would now be:  
> UNI_the:10 UNI_food:2 BIGRAM_the_food:1

If you apply binning, with results in bins of 0, 1, 2, or 3, the new feature vector from the previous example
now becomes:  
> UNI_the:3 UNI_food:2 BIGRAM_the_food:1  

In the implementation I use, bins of 10 are used if the raw count of a feature is less than 30, else bins of
60 are used.

### Naive Bayes Feature Selection

Using the model that performed best in my Naive Bayes Classifier repository, feature selection is perfoemed to
determine what number of features that return the highest accuracy. 
This was using the word frequencies rather than the 
relative frequencies with ngram features. It recieved 73 percent accuracy on the testing data and 78 percent accuracy 
in development data.
In this case, weare only interested in using the word features (unigram bigram and trigram), and can disregard the POS,
LIWC, opinion features in this case. 
One example of how to do feature selection is given to you in the main of
movie_reviews.py. It loops through the 10,000 best features returned by most informative features,
and for each loop it looks at some subset of the features. For every loop, it returns the accuracy of that
subset of features on the development data. Once it has determined which subset of features produces the
highest accuracy, it uses those same features to evaluate the testing data. 

You can see results in all-tables.pdf and all_results.txt

### Scikit-Learn Classifiers

The current machine learning model for Naive Bayes is used from NLTK. Now, I will use
different Machine Learning library for Python called scikit-learn.
Instructions for installing scikit-learn can be found at scikit-learn.org/stable/install .

http://scikit-learn.org/

### Naive Bayes and Decision Tree Classifiers

The scikit-learn versions of Naive Bayes (BernoulliNB is equivalent to nltk
version) and Decision Tree classifiers are also used to classify the IMDB movie reviews.
The single best set of features extracted earlier is used.

Refer to all-tables.pdf and all_results.txt to see results.

### Support Vector Machine
A new type of Machine Learning model called the Support Vector Machine (SVM) is used to classify
the reviews. Word embeddings are used to see how they compare against our best feature set.

### Word Embeddings
A feature set is created by embedding the review text using pre-trained word embeddings. 
A custom version of Word2Vec pre-trained word embeddings is used. The code file
word2vec_extractor.py implements some helper functions useful for turning words, sentences and
documents into vectors.  
> The Glove embedding under total-data/glove-w2v.txt.  
Gensim must also be installed.  

/radimrehurek.com/gensim/install  

Two SVMs are created:
1. An SVM using the single best feature set extracted earlier   
2. Train an SVM using only the word embedding features  

Refer to all-results.txt and all-tables.pdf for results


### Neural Network
The Multi-layer Perceptron (MLP) classifier is used to classify the reviews.  
The MLP classifier using only the word embedding features.

Refer to all-results.txt and all-tables.pdf for results



This was for an assignment in CSE 143 at UCSC with Professor Dilek Hakkani-Tür.


