import numpy as np

from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

class Model:
    # Represents an instance of a repeated k-means (RKM) model
    model = None
    scaler = None
    label_map = None
    use_sentiment = False
    
    def __init__(self):
        self.scaler = StandardScaler()

    def __remove_outliers(self, arrays):
        lof = LocalOutlierFactor()
        is_outlier = lof.fit_predict(arrays)
        output = []
        for i in range(len(is_outlier)):
            if is_outlier[i] == 1:
                output.append(arrays[i].tolist())
        return np.asarray(output)

    def __remove_sentiment_variability(self, arrays):
        output = []
        for i in range(len(arrays)):
            inner = []
            temp = arrays[i]
            for j in range(len(temp)):
                if j != len(temp) - 1:
                    inner.append(temp[j])
            output.append(inner)
        return np.asarray(output)

    def __repeated_kmeans(self, arrays, num_repetitions, target_real_percent):
        best_model = None
        best_real_percent = 0.0
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
    def __get_label_mapping(self, model):
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

    def train(self, arrays, num_repetitions, use_sentiment_variability, target_real_percent):
        data = arrays
        self.use_sentiment = use_sentiment_variability
        if not use_sentiment_variability:
            data = self.__remove_sentiment_variability(data)
        cleaned_data = self.__remove_outliers(data)
        scaled_data = self.scaler.fit_transform(cleaned_data)
        self.model = self.__repeated_kmeans(scaled_data, num_repetitions, target_real_percent)
        self.label_map = self.__get_label_mapping(self.model)

    # Classifies a review (1-d array) as real (true) or fake (false)
    def classify(self, review):
        if self.model is None:
            raise RuntimeError("Classification before model was trained")           
        new = np.asarray([review.tolist()])
        if not self.use_sentiment:
            new = self.__remove_sentiment_variability(new)
        scaled = self.scaler.transform(new)
        return self.label_map[self.model.predict(scaled)[0]]

# example usage
if __name__ == "__main__":
    model = Model()
    a = np.array([[0,0,0,0], [1,1,1,1], [9999,9999,9999,0]])
    model.train(a, 10, True)
    b = np.array([10,10,10,10])
    outcome = model.classify(b)
    print(outcome)
