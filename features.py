
from nltk.corpus import opinion_lexicon
import re
import nltk
import word_category_counter
import data_helper
import os, sys
from nltk.util import ngrams
from word2vec_extractor import Word2vecExtractor
DATA_DIR = "asg4-data/data"
LIWC_DIR = "asg4-data/liwc"

word_category_counter.load_dictionary(LIWC_DIR)

w2vecmodel = "asg4-data/glove-w2v.txt"
w2v = None

# TODO: You can expand this feature set, i.e. word_bin_features
#  best_features is for competition
FEATURE_SETS = {"word_features", "word_pos_features", "word_pos_liwc_features", "word_pos_opinion_features",
               "word_embedding", "best_features"}

## changed binning function name
def binn(count):
    """
    Results in bins of  0, 1, 2, 3 >=
    :param count:
    :return:
    """
    # Just a wild guess on the cutoff
    # you can experiment with different bin size
    #return count if count < 2 else 3
    return 10 if count < 30 else 60



def normalize(token, should_normalize=True):
    """
    This function performs text normalization.

    If should_normalize is False then we return the original token unchanged.
    Otherwise, we return a normalized version of the token, or None.

    For some tokens (like stopwords) we might not want to keep the token. In
    this case we return None.

    :param token: str: the word to normalize
    :param should_normalize: bool
    :return: None or str
    """
    if not should_normalize:
        normalized_token = token

    else:

        ###     YOUR CODE GOES HERE
        token_lower = token.lower()
        stop_words = nltk.corpus.stopwords.words('english')
        if token_lower in stop_words:
            return None
        if re.findall(r'[\w]', token_lower) == []:
            return None
        normalized_token = token_lower
        #raise NotImplemented


    return normalized_token



def get_words_tags(text, should_normalize=True):
    """
    This function performs part of speech tagging and extracts the words
    from the review text.

    You need to :
        - tokenize the text into sentences
        - word tokenize each sentence
        - part of speech tag the words of each sentence

    Return a list containing all the words of the review and another list
    containing all the part-of-speech tags for those words.

    :param text:
    :param should_normalize:
    :return:
    """
    words = []
    tags = []

    # tokenization for each sentence

    ###     YOUR CODE GOES HERE
    #raise NotImplemented
    sent_tokens = nltk.sent_tokenize(text)
    words_in_sent = [nltk.pos_tag(nltk.word_tokenize(sentence)) for sentence in sent_tokens]

    word_tokens = [tup[0] for ls in words_in_sent for tup in ls]

    tups_tokens = [tup for ls in words_in_sent for tup in ls if normalize(tup[0], should_normalize) != None]

    words = [normalize(tup[0], should_normalize) for tup in tups_tokens if normalize(tup[0], should_normalize) != None]
    tags = [tup[1] for tup in tups_tokens if normalize(tup[0], should_normalize) != None]
 
    #print(words)
    #print(tags)
    
    return words, tags

def feature_ngram_count(tokens, ngrams_count, feature_vectors, binning):
    tmp_dict = {}
    grams = ngrams(tokens, ngrams_count)
    #print(binning + "  "+ str(ngrams_count))
    if ngrams_count == 1:
        if binning is True:
            keys = ['UNI_'+tup[0] + '_BIN' for tup in grams]
        else:
            keys = ['UNI_'+tup[0] for tup in grams]

    if ngrams_count == 2:
        if binning is True:
            keys = ['BIGRAM_' + tup[0] + "_"+tup[1] + "_BIN" for tup in grams]
        else:
            keys = ['BIGRAM_' + tup[0] + "_"+tup[1] for tup in grams]
    
    if ngrams_count == 3:
        if binning is True:
            keys = ['TRIGRAM_' + tup[0] + "_"+tup[1] + "_" + tup[2] +"_BIN" for tup in grams]
        else:
            keys = ['TRIGRAM_' + tup[0] + "_"+tup[1] + "_"+tup[2] for tup in grams]

    token_len = len(keys)
    for key in keys:
        if key in tmp_dict:
            tmp_dict[key] += 1
        if key not in tmp_dict:
            tmp_dict[key] = 1
    #print(feature_vectors)
    #print(tmp_dict)
    if binning is True:
        [feature_vectors.update( {key : binn(tmp_dict[key])} ) for key in tmp_dict]
    else:
        [feature_vectors.update( {tup : tmp_dict[tup]} ) for tup in tmp_dict]
    return feature_vectors

def get_ngram_features(tokens, binning = False):
    """
    This function creates the unigram and bigram features as described in
    the assignment3 handout.

    :param tokens:
    :return: feature_vectors: a dictionary values for each ngram feature
    """
    feature_vectors = {}
    feature_ngram_count(tokens, 1, feature_vectors, binning = binning)
    feature_ngram_count(tokens, 2, feature_vectors, binning = binning)
    feature_ngram_count(tokens, 3, feature_vectors, binning = binning)
    #print(feature_vectors)
    #print(len(feature_vectors))

    ###     YOUR CODE GOES HERE
    #raise NotImplemented

    return feature_vectors


def get_pos_features(tags, binning):
    """
    This function creates the unigram and bigram part-of-speech features
    as described in the assignment3 handout.

    :param tags: list of POS tags
    :return: feature_vectors: a dictionary values for each ngram-pos feature
    """
    feature_vectors = {}
    feature_ngram_count(tags, 1, feature_vectors, binning )
    feature_ngram_count(tags, 2, feature_vectors, binning )
    feature_ngram_count(tags, 3, feature_vectors, binning )
    ###     YOUR CODE GOES HERE
    #print(feature_vectors)
    #print(len(feature_vectors))
    #raise NotImplemented

    return feature_vectors



def get_liwc_features(words):
    """
    Adds a simple LIWC derived feature

    :param words:
    :return:
    """

    feature_vectors = {}
    text = " ".join(words)
    liwc_scores = word_category_counter.score_text(text)

    # All possible keys to the scores start on line 269
    # of the word_category_counter.py script
    negative_score = liwc_scores["Negative Emotion"]
    positive_score = liwc_scores["Positive Emotion"]

    negations = liwc_scores["Negations"]
    sad = liwc_scores["Sadness"]
    positive_feeling = liwc_scores["Positive feelings"]
    assent = liwc_scores["Assent"]
    leisure = liwc_scores['Leisure']

    ### 2 GIVEN FEATURES
    feature_vectors["Negative Emotion"] = negative_score
    feature_vectors["Positive Emotion"] = positive_score
    ### 5 ADDED FEATURES
    feature_vectors["Negations"] = negations
    feature_vectors["Sadness"] = sad
    feature_vectors["Positive feelings"] = positive_feeling
    feature_vectors["Assent"] = assent
    feature_vectors["Leisure"] = leisure

    if negations > assent:
        feature_vectors['liwc:negations'] = 1
    else:
        feature_vectors['liwc:assent'] = 1

    if sad > positive_feeling:
        feature_vectors['liwc:sadness'] = 1
    else:
        feature_vectors['liwc:positive feelings'] = 1
    
    if leisure > 0:
        feature_vectors['liwc:leisure'] = 1
    else:
         feature_vectors['liwc:leisure'] = 0

    if positive_score > negative_score:
        feature_vectors["liwc:positive"] = 1
    else:
        feature_vectors["liwc:negative"] = 1

    #print(feature_vectors)
    #raise NotImplemented
    return feature_vectors

def get_word_embedding_features(text):
    feature_vectors = {}
    print(text)
    #print(w2vecmodel)
    global w2v
    if w2v is None:
        print("loading word vectors...", w2vecmodel)
        w2v = Word2vecExtractor(w2vecmodel)
    feature_vectors = w2v.get_doc2vec_feature_dict(text)
    #print(feature_vectors)
    # TODO: NEWLY ADDED, YOUR CODE GOES HERE
    #raise NotImplemented
    return feature_vectors


def get_opinion_features(tags):
    """
    This function creates the opinion lexicon features
    as described in the assignment3 handout.

    the negative and positive data has been read into the following lists:
    * neg_opinion
    * pos_opinion

    if you haven't downloaded the opinion lexicon, run the following commands:
    *  import nltk
    *  nltk.download('opinion_lexicon')

    :param tags: tokens
    :return: feature_vectors: a dictionary values for each opinion feature
    """
    neg_opinion = opinion_lexicon.negative()
    pos_opinion = opinion_lexicon.positive()
    feature_vectors = {}

    for word in neg_opinion:
        if word in tags:
            feature_vectors.update({"NEG_OP_"+ word: 1})
        else:
            feature_vectors.update({"NEG_OP_"+ word: 0})

    for word in pos_opinion:
        if word in tags:
            feature_vectors.update({"POS_OP_"+ word: 1})
        else:
            feature_vectors.update({"POS_OP_"+ word: 0})
    ###     YOUR CODE GOES HERE
    #print(feature_vectors)
    #raise NotImplemented

    return feature_vectors

#FEATURE_SETS = {"word_features", "word_pos_features", "word_pos_liwc_features", "word_pos_opinion_features",
#               "word_embedding", "best_features"}
def get_features_category_tuples(category_text_dict, feature_set, binning = False):
    """

    You will might want to update the code here for the competition part.

    :param category_text_dict:
    :param feature_set:
    :return:
    """
    #print(binning + " in features.py")
    features_category_tuples = []
    all_texts = []

    assert feature_set in FEATURE_SETS, "unrecognized feature set:{}, Accepted values:{}".format(feature_set, FEATURE_SETS)

    for category in category_text_dict:
        for text in category_text_dict[category]:

            words, tags = get_words_tags(text)
            feature_vectors = {}
            if feature_set == "word_features":
                feature_vectors.update(get_ngram_features(words, binning = bool(binning)))

            elif feature_set == "word_pos_features":
                feature_vectors.update(get_ngram_features(words, binning = bool(binning)))
                feature_vectors.update(get_pos_features(tags, binning = bool(binning)))

            elif feature_set == "word_pos_liwc_features":
                feature_vectors.update(get_ngram_features(words, binning = bool(binning)))
                feature_vectors.update(get_pos_features(tags, binning = bool(binning)))
                feature_vectors.update(get_liwc_features(words))

            elif feature_set == "word_pos_opinion_features":
                feature_vectors.update(get_ngram_features(words, binning = bool(binning)))
                feature_vectors.update(get_pos_features(tags, binning = bool(binning)))
                feature_vectors.update(get_opinion_features(words))

            elif feature_set == "word_embedding":
                feature_vectors.update(get_word_embedding_features(text))

            elif feature_set == "best_features":
                ### best feature is word_embeddings for comp
                feature_vectors.update(get_word_embedding_features(text))

            ###     YOUR CODE GOES HERE
            # TODO: best_features is for competition
            #raise NotImplemented

            features_category_tuples.append((feature_vectors, category))
            all_texts.append(text)

    return features_category_tuples, all_texts


def write_features_category(features_category_tuples, outfile_name):
    """
    Save the feature values to file.

    :param features_category_tuples:
    :param outfile_name:
    :return:
    """
    with open(outfile_name, "w", encoding="utf-8") as fout:
        for (features, category) in features_category_tuples:
            fout.write("{0:<10s}\t{1}\n".format(category, features))


def features_stub():
    # changed datafile path
    datafile = DATA_DIR + "/" + "imdb-training.data"
    raw_data = data_helper.read_file(datafile)
    positive_texts, negative_texts = data_helper.get_reviews(raw_data)

    category_texts = {"positive": positive_texts, "negative": negative_texts}
    feature_set = "word_embedding"

    features_category_tuples, texts = get_features_category_tuples(category_texts, feature_set)

    #raise NotImplemented
    filename = "???"
    write_features_category(features_category_tuples, filename)



if __name__ == "__main__":
    features_stub()

