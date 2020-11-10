import numpy as np

from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

def remove_sentiment_variability(arrays):
    output = []
    for i in range(len(arrays)):
        inner = []
        temp = arrays[i]
        for j in range(len(temp)):
            if j != len(temp) - 1:
                inner.append(temp[j])
        output.append(inner)
    return np.asarray(output)

def remove_outliers(arrays):
    lof = LocalOutlierFactor()
    is_outlier = lof.fit_predict(arrays)
    output = []
    for i in range(len(is_outlier)):
        if is_outlier[i] == 1:
            output.append(arrays[i].tolist())
    return np.asarray(output)

def get_scaler():
    return StandardScaler()

def train_scaler(scaler, arrays):
    scaler.fit(arrays)

def scale(scaler, arrays):
    return scaler.transform(arrays)

def repeated_kmeans(arrays, num_repetitions):
    best_model = None
    best_real_percent = 0.0
    target_real_percent = 70.0
    for i in range(num_repetitions):
        kmeans = KMeans(n_clusters=2, n_init=1).fit(arrays)
        labels = kmeans.labels_
        total = 0
        positive = 0
        negative = 0
        for i in range(len(labels)):
            total += 1
            if labels[i] == 1:
                positive += 1
            else:
                negative += 1
        real_percent = max(positive, negative)/total * 100
        if (
            best_model is None
            or abs(real_percent - target_real_percent) < abs(best_real_percent - target_real_percent)
        ):
            best_real_percent = real_percent
            best_model = kmeans
    return best_model

# Given a model, returns a map indicating whether the labels 1 and 0 correspond to real (True) or fake (False)
def get_label_mapping(model):
    labels = model.labels_
    output = {}
    one = 0
    zero = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            one += 1
        else:
            zero += 1
    if one > zero:
        output[1] = True
        output[0] = False
    else:
        output[1] = False
        output[0] = True
    return output

# example usage
scaler = get_scaler()
a = np.array([[0,0,0,0], [1,1,1,1], [9999,9999,9999,0]])
a = remove_sentiment_variability(a)
print(a)
a = remove_outliers(a)
print(a)
train_scaler(scaler, a)
a = scale(scaler, a)
print(a)
model = repeated_kmeans(a, 10)
labels = get_label_mapping(model)
print(labels)
b = np.array([[10,10,10]])
b = scale(scaler, b)
print(b)
outcome = model.predict(b)
print(outcome)
print(labels[outcome[0]])
print(model.labels_)
