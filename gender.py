import nltk
import random
from nltk.corpus import names

labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
                [(name, 'female') for name in names.words('female.txt')])
random.shuffle(labeled_names)


class Gender:

    def __init__(self, name):
        self.name = name

    def gender_features(self, name):
        return {'suffix1': name[-1:],
                'suffix2': name[-2:],
                'prefix1': name[1],
                'prefix2': name[:2]}

    def get_nltk_data(self):
        labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
                        [(name, 'female') for name in names.words('female.txt')])
        random.shuffle(labeled_names)
        featuresets = [(self.gender_features(n), gender) for (n, gender) in labeled_names]
        self.train_set, self.test_set = featuresets[500:], featuresets[:500]


    def train_classifier(self):
        self.classifier = nltk.NaiveBayesClassifier.train(self.train_set)
        self.accuracy = nltk.classify.accuracy(self.classifier, self.test_set)
        
    def classifier_classify(self, name):
        return self.classifier.classify(self.gender_features(name))

    def run(self):
        self.get_nltk_data()
        self.train_classifier()
        return {'prediction': self.classifier_classify(self.name), 'accuracy': self.accuracy} 


if __name__ == "__main__":
    test = Gender("Kathleen")
    test.run() 
