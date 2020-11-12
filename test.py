import numpy as np
from experiment_design import get_labels
from prepare_data import get_training_dataset
from prepare_data import get_validation_dataset
from prepare_data import preprocess
from feature_conversion import convert_to_features

training_dataset = get_training_dataset().sample(frac=0.02)

fully_processed = preprocess(training_dataset, True)
half_processed = preprocess(training_dataset)
test = half_processed
for i in test:
    print(i)
