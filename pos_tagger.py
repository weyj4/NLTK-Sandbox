import nltk
from nltk.corpus import brown

class POS:

    def pos_features(self, sentence, i):
        features = {"suffix(1)": sentence[i][-1:],
                    "suffix(2)": sentence[i][-2:],
                    "suffix(3)": sentence[i][-3:]}
        if i == 0:
            features["prev-word"] = "<START>"
        else:
            features["prev-word"] = sentence[i-1]
        return features

    def get_featuresets(self):
        tagged_sents = brown.tagged_sents(categories='news')
        featuresets = []
        for tagged_sent in tagged_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            for i, (word, tag) in enumerate(tagged_sent):
                featuresets.append( (self.pos_features(untagged_sent, i), tag) )
        self.featuresets = featuresets

    def test_and_train_sets(self):
        size = int(len(self.featuresets) * 0.1)
        self.train_set, self.test_set = self.featuresets[size:], self.featuresets[:size]

    def train_classifier(self):
        self.classifier = nltk.NaiveBayesClassifier.train(self.train_set)

    def classifier_classify(self, sent, index):
        return self.classifier.classify(self.pos_features(sent, index))

    def run(self,sent,index):
        self.get_featuresets()
        self.test_and_train_sets()
        self.train_classifier()
        return {'prediction': self.classifier_classify(sent, index)}

if __name__ == "__main__":
    test = POS()
    print test.run("This toy is mine", 4) 
