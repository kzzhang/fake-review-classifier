import nltk
nltk.download('sentiwordnet')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import PorterStemmer
import numpy as np


def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def get_sentiment(word,tag):
    """ returns list of pos neg and objective score. But returns empty list if not present in senti wordnet. """

    wn_tag = penn_to_wn(tag)
    if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
        return []

    lemma = lemmatizer.lemmatize(word, pos=wn_tag)
    if not lemma:
        return []

    synsets = wn.synsets(word, pos=wn_tag)
    if not synsets:
        return []

    # Take the first sense, the most common
    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())

    return [swn_synset.pos_score(),swn_synset.neg_score(),swn_synset.obj_score()]


def wordnet_senti(sentence):
    pos_sentiments = []
    neg_sentiments = []
    words_data = sentence.split(' ')
    words_data = [word for word in words_data if len(word)>0]
    pos_val = nltk.pos_tag(words_data)
    senti_val = [get_sentiment(x,y) for (x,y) in pos_val]
    for word_sentiment in senti_val:
        if word_sentiment:
            pos_sentiments.append(word_sentiment[0])
            neg_sentiments.append(word_sentiment[1])
    
    if not pos_sentiments:
        return 0,0

    return np.mean(pos_sentiments), np.mean(neg_sentiments)