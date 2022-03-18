#
#   @author: Mehmet Kaan Erol
#
import pickle
from xgboost import XGBClassifier

class XClassifier():
    def __init__(self, classifier_model_filepath, load=False):
        # Parameters
        self.objective='binary:hinge'
        self.max_depth=11
        self.colsample_bytree=0.6

        self.classifier_model_filepath = classifier_model_filepath

        # model
        if (load):
            with open(self.classifier_model_filepath, 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.model = XGBClassifier(
                objective=self.objective,
                max_depth=self.max_depth,
                colsample_bytree=self.colsample_bytree
                )

    def train(self, data, target, save=False):
        self.model = self.model.fit(data, target)
        if (save):
            with open(self.classifier_model_filepath, 'wb') as f:
                pickle.dump(self.model, f, protocol=4)

    def test(self, data):
        pred = self.model.predict(data)
        return pred
