from prepare_data import get_training_dataset
from prepare_data import get_validation_dataset
from prepare_data import get_testing_dataset
from prepare_data import preprocess
from feature_conversion import convert_to_features
from experiment_design import get_labels
from datetime import datetime
import numpy as np

startTime = datetime.now()
print("Parsing datasets")
training_dataset = get_training_dataset()
print("Training dataset size: " + str(len(training_dataset)) + " entries")
validation_dataset = get_validation_dataset()
print("Validation dataset size: " + str(len(validation_dataset)) + " entries")
testing_dataset = get_testing_dataset().sample(frac=0.25)
print("Testing dataset size: " + str(len(testing_dataset)) + " entries")

# Train
print("Processing Training Dataset")
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
print("\tsaving training features")
np.save("data/training_vader.npy", vader_features)
np.save("data/training_textblob.npy", textblob_features)
np.save("data/training_swn.npy", swn_features)

# Validate
print("Processing Validation Dataset")
print("===================")
print("\tpreprocessing validation dataset")
fully_processed = preprocess(validation_dataset, True)
half_processed = preprocess(validation_dataset)
print("\tgetting labels")
labels = get_labels(validation_dataset)
np.save("data/validation_labels.npy", labels)
print("\tbuilding validation feature map")
print("\t\thandling VADER")
vader_features = convert_to_features(half_processed, 'VADER')
print("\t\thandling TextBlob")
textblob_features = convert_to_features(half_processed, 'TextBlob')
print("\t\thandling SentiWordNet")
swn_features = convert_to_features(fully_processed, 'SentiWordNet')
print("\tsaving validation features")
np.save("data/validation_vader.npy", vader_features)
np.save("data/validation_textblob.npy", textblob_features)
np.save("data/validation_swn.npy", swn_features)

# Test
print("Processing Test Dataset")
print("===================")
print("\tpreprocessing testing dataset")
fully_processed = preprocess(testing_dataset, True)
half_processed = preprocess(testing_dataset)
print("\tgetting labels")
labels = get_labels(testing_dataset)
np.save("data/testing_labels.npy", labels)
print("\tbuilding testing features")
print("\t\thandling VADER")
vader_features = convert_to_features(half_processed, 'VADER')
print("\t\thandling TextBlob")
textblob_features = convert_to_features(half_processed, 'TextBlob')
print("\t\thandling SentiWordNet")
swn_features = convert_to_features(fully_processed, 'SentiWordNet')
print("\tsaving testing features")
np.save("data/testing_vader.npy", vader_features)
np.save("data/testing_textblob.npy", textblob_features)
np.save("data/testing_swn.npy", swn_features)


print("Total execution time: " + str(datetime.now() - startTime))
