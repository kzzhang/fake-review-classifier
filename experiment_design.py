from feature_conversion import convert_to_features
from naive_model import NaiveModel
from model import Model
from sklearn.metrics import matthews_corrcoef,f1_score,accuracy_score
import numpy as np

def methods():
    return ['VADER', 'TextBlob', 'SentiWordNet']

def calculate_scores(model, data, labels):
    if(len(data)!=len(labels)):
        print("Error! Need enough labels")
        return
    
    predicted_labels = []
    for i in range(len(data)):
        predicted_labels.append(model.classify(data[i]))

    computed_accuracy  = accuracy_score(labels, predicted_labels)
    computed_f1score = f1_score(labels, predicted_labels)
    computed_mcc = matthews_corrcoef(labels, predicted_labels)
    return computed_accuracy, computed_f1score, computed_mcc

# Input datasets, notice, we actually need to do this 9 times, 3 times on each set
sample_review = [
    "ar3434", #review ID
    "fefo4214", # product ID
    "This product is amazing. This product sucks. I like this product", #review text
    11, #review text length
    3, #review rating
    123, #review post date - date of first product review
    False,
]

sample_review_2 = [
    "fewinof", #review ID
    "294j", # product ID
    "Amazing. Best", #review text
    2, #review text length
    5, #review rating
    23, #review post date - date of first product review
    True,
]

sample_review_3 = [
    "3r432r", #review ID
    "f3f3ff", # product ID
    "Horrible. Bad", #review text
    2, #review text length
    5, #review rating
    22, #review post date - date of first product review
    False,
]

sample_review_4 = [
    "42", #review ID
    "523532", # product ID
    "I really enjoyed using this product. It was really really good, but could be better.", #review text
    14, #review text length
    4, #review rating
    100, #review post date - date of first product review
    True, #review is real
]

sample_review_5 = [
    "253", #review ID
    "rkm2", # product ID
    "Ugly. Bad", #review text
    2, #review text length
    1, #review rating
    1, #review post date - date of first product review
    False, #review is fake
]

def get_labels(data):
    labels = data["label"]
    return np.asarray(labels)

# Training
def training(features_map, use_naive_model, use_sentiment_variability): #return 3 candidate models, one for VADER, TextBlob and SentiNet
    training_models = []
    for method in methods():
        method_models = []
        training_data = features_map.get(method)
        model = None
        if use_naive_model:
            model = NaiveModel()
            model.train(training_data)
        else:
            model = Model()
            model.train(training_data, 100, use_sentiment_variability, 70.0)
        training_models.append(model)
    return training_models

def validate(candidate_models, features_map, labels): #take in three models and return 1 based on MCC
    best_mcc_abs = 0
    best_mcc = 0
    best_mcc_index = 0
    for i in range(len(methods())):
        validate_data = features_map.get(methods()[i])
        _,_, mcc = calculate_scores(candidate_models[i], validate_data, labels)
        if abs(mcc)>best_mcc_abs:
            best_mcc_abs = abs(mcc)
            best_mcc = mcc
            best_mcc_index = i
        print(f'VALIDATION: MCC using {methods()[i]} is {mcc}')

    if(best_mcc<0):
        print("Need to switch labels!")
 #       candidate_models[i].switch_labels()
        # Need to switch labels, flag, don't expect this happens, in case it does,
        # will need to handle
    return best_mcc_index


def test(model, features, labels):
    accuracy,f1score,mcc = calculate_scores(model, features, labels)
    print(f"Test model accuracy:{accuracy}, f1score: {f1score}, mcc: {mcc}")
