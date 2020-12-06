from experiment_design import training
from model import Model
from nltk import word_tokenize
from nltk.corpus import stopwords
from prepare_data import remove_noise
from nltk.tokenize.treebank import TreebankWordDetokenizer
from feature_conversion import convert_to_features
import numpy as np

# Training
print("Training full model")
print("===================")
print("\tloading features")
vader_features = np.load("data/training_vader.npy")
print("\ttraining model")
model = Model()
model.train(vader_features, 100, True, 85)
labels = model.label_map
print("\ttraining model finished")

print("")
while(True):
    print("Rating review (1 to 5):")
    rating = float(input())
    print("Days from first review:")
    date = int(input())
    print("Rating body text:")
    body = str(input())
    body = word_tokenize(body)
    body = remove_noise(body)
    detokenizer = TreebankWordDetokenizer()
    body = detokenizer.detokenize(body)
    length = len(body)
    review = [0, 0, body, length, rating, date]
    data = np.asarray([review])
    features = convert_to_features(data, "VADER")
    label = model.classify(features[0])
    print("")
    if label is True:
        print("This review is real.")
    else:
        print("This review is fake.")
    print("============")


    
