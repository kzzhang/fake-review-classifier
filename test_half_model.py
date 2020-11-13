from experiment_design import training
from experiment_design import validate
from experiment_design import test
from experiment_design import methods
from datetime import datetime
import numpy as np


startTime = datetime.now()

# Training
print("Training half model")
print("===================")
print("\tloading features")
swn_features = np.load("data/training_swn.npy")
textblob_features = np.load("data/training_textblob.npy")
vader_features = np.load("data/training_vader.npy")
features_map = {}
for method in methods():
    if method == 'SentiWordNet':
        features_map[method] = swn_features
    elif method == 'TextBlob':
        features_map[method] = textblob_features
    else:
        features_map[method] = vader_features
print("\ttraining models")
models = training(features_map, False, False)

# Validate
print("Validating half model")
print("===================")
print("\tloading features")
swn_features = np.load("data/validation_swn.npy")
textblob_features = np.load("data/validation_textblob.npy")
vader_features = np.load("data/validation_vader.npy")
labels = np.load("data/validation_labels.npy")

features_map = {}
for method in methods():
    if method == 'SentiWordNet':
        features_map[method] = swn_features
    elif method == 'TextBlob':
        features_map[method] = textblob_features
    else:
        features_map[method] = vader_features
print("\tvalidating model parameters")
best_mcc_method, best_model_idx = validate(models, features_map, labels)
print("Best model uses: " + methods()[best_mcc_method] + " and real percentage: " + str(70 + 0.5*best_model_idx[best_mcc_method]))

# Test
print("Testing half model")
print("===================")
print("\tloading features")
method = methods()[best_mcc_method]
features = None
if method == 'SentiWordNet':
    features = np.load("data/testing_swn.npy")
elif method == 'TextBlob':
    features = np.load("data/testing_textblob.npy")
else:
    features = np.load("data/testing_vader.npy")
labels = np.load("data/testing_labels.npy")
print("\ttesting using test dataset")
test(models[best_mcc_method][best_model_idx[best_mcc_method]], features, labels)

print("Total execution time: " + str(datetime.now() - startTime))
