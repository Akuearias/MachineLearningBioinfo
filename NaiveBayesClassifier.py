class NaiveBayes:
    def __init__(self):
        self.class_probs = {}
        self.feature_probs = {}
        self.classes = set()
        self.vocabulary = set()

    def train(self, X, y):
        '''
        X is list of feature lists.
        y is list of labels.
        '''

        samples = len(y)
        class_counts = {}
        feature_counts = {}

        for features, label in zip(X, y):
            self.classes.add(label)
            class_counts[label] = class_counts.get(label, 0) + 1

            for i, feature in enumerate(features):
                k = (label, i, feature)
                feature_counts[k] = feature_counts.get(k, 0) + 1
                self.vocabulary.add((i, feature))

        # Prior probability
        for c in self.classes:
            self.class_probs[c] = class_counts.get(c, 0) / samples

        # Condition probability P = (feature | class) and use Laplace smoothing.
        for c in self.classes:
            for i, value in [(i, v) for (i, v) in self.vocabulary]:
                k = (c, i, value)
                count = feature_counts.get(k, 0)
                total = sum([feature_counts.get((c, i, v), 0) for (ci, i2, v) in feature_counts if ci == c and i2 == i])
                vocab_size = len(set(v for (j, v) in self.vocabulary if j == i))
                self.feature_probs[k] = (count + 1) / (total + vocab_size)


    def predict(self, X):
        predictions = []
        for features in X:
            class_scores = {}
            for c in self.classes:
                p = self.class_probs[c]
                for i, feature in enumerate(features):
                    key = (c, i, feature)
                    p *= self.feature_probs.get(key, 1e-6) # avoid 0

                class_scores[c] = p

            # Choose the class with the highest posterior probability
            C = max(class_scores, key = class_scores.get)
            predictions.append(C)

        return predictions


# Example
if __name__ == "__main__":
    X = [['Sunny', 'Hot', 'High', 'Weak'],
            ['Sunny', 'Hot', 'High', 'Strong'],
            ['Overcast', 'Hot', 'High', 'Weak'],
            ['Rain', 'Mild', 'High', 'Weak'],
            ['Rain', 'Cool', 'Normal', 'Weak'],
            ['Rain', 'Cool', 'Normal', 'Strong'],
            ['Overcast', 'Cool', 'Normal', 'Strong'],
         ['Sunny', 'Mild', 'High', 'Weak'],
         ['Sunny', 'Cool', 'Normal', 'Weak'],
         ['Rain', 'Mild', 'Normal', 'Weak'],
         ['Sunny', 'Mild', 'High', 'Strong'],
         ['Overcast', 'Mild', 'High', 'Strong'],
         ['Overcast', 'Hot', 'Normal', 'Weak'],
         ['Rain', 'Mild', 'High', 'Strong']]

    y = ['N', 'N', 'Y', 'Y', 'Y', 'N', 'Y',
         'N', 'Y', 'Y', 'Y', 'Y', 'Y', 'N']

    nb = NaiveBayes()
    nb.train(X, y)

    print(nb.predict([['Rain', 'Mild', 'Normal', 'Strong']]))