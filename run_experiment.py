from prepare_data import get_training_dataset
from prepare_data import get_validation_dataset
from prepare_data import get_testing_dataset
from prepare_data import preprocess
from feature_conversion import convert_to_features
from model import Model
from naive_model import NaiveModel
from experiment_design import training
from experiment_design import validate
from experiment_design import test
from experiment_design import methods
from experiment_design import get_labels
from datetime import datetime

startTime = datetime.now()
print("Parsing datasets")
training_dataset = get_training_dataset()
print("Training dataset size: " + str(training_dataset.size))
#validation_dataset = get_validation_dataset()
#testing_dataset = get_testing_dataset()

# Get best full model
# Train
print("Training full model")
print("===================")
print("\tpreprocessing training dataset")
fully_processed = preprocess(training_dataset, True)
half_processed = preprocess(training_dataset)
print("\tbuilding training feature map")
print("\t\thandling VADER")
vader_features = convert_to_features(half_processed, 'VADER')
print("\t\thandling TextBlob")
textblob_features = convert_to_features(half_processed, 'TextBlob')
print("\t\thandling SentiWordNet")
swn_features = convert_to_features(fully_processed, 'SentiWordNet')
features_map = {}
for method in methods():
    if method == 'SentiWordNet':
        features_map[method] = swn_features
    elif method == 'TextBlob':
        features_map[method] = textblob_features
    else:
        features_map[method] = vader_features
print("\ttraining models")
models = training(features_map, False, True)
# Validate
#fully_processed = preprocess(validation_dataset, True)
#half_processed = preprocess(validation_dataset)
#labels = get_labels(validation_dataset)
#vader_features = convert_to_features(half_processed, 'VADER')
#textblob_features = convert_to_features(half_processed, 'TextBlob')
#swn_features = convert_to_features(fully_processed, 'SentiWordNet')
#features_map = {}
#for method in methods():
#    if method == 'SentiWordNet':
#        features_map[method] = swn_features
#    elif method == 'TextBlob':
#        features_map[method] = textblob_features
#    else:
#        features_map[method] = vader_features
#best_model_idx = validate(models, features_map, labels)

print("Total execution time: " + str(datetime.now() - startTime))
