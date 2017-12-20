import pickle
import numpy as np

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''
    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit
        pass

    def is_good_enough(self,classifier_list,alpha_list,X,y):
        '''Optional'''
        result = np.zeros(y.shape[0])
        for i in range(len(classifier_list)):
            result = result + alpha_list[i] * classifier_list[i].predict(X)
        result[result>0] = 1
        result[result<0] = -1
        count = 0   #count the number of result that equals to y
        for j in range(result.shape[0]):
            if(result[j] != y[j]):
                count+=1
                return False
        return True

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        w = np.zeros(X.shape[0])
        w[:] = 1 / X.shape[0]
        classifier_list = []
        alpha_list = []
        for i in range(self.n_weakers_limit):
            X_train = X
            if len(classifier_list) != 0:
                if self.is_good_enough(classifier_list,alpha_list,X,y) == True:
                    break
            clf = self.weak_classifier(criterion='gini', max_depth=1)
            clf.fit(X_train, y,w)
            answer = clf.predict(X_train)
            err_sum = 0.0
            for j in range(y.shape[0]):
                if answer[j] != y[j]:
                    err_sum += w[j]
            if(err_sum != 0):
                alpha = 0.5 * np.log((1-err_sum)/err_sum)
            z = 0.0   #Normalize parameter
            for m in range(y.shape[0]):
                z = z + w[m] * np.exp((-1)*alpha*y[m]*answer[m])
            #the new distribution of w
            for n in range(y.shape[0]):
                w[n] = (w[n]/z) * np.exp((-1)*alpha*y[n]*answer[n])
            classifier_list.append(clf)
            alpha_list.append(alpha)
        return classifier_list,alpha_list


    def predict_scores(self, X,classifier_list,alpha_list):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            classifier_list:the list of the weak classifiers
            alpha_list: the list of the coefficient of weak classifiers
        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        result = np.zeros(X.shape[0])
        for i in range(len(classifier_list)):
            result = result + alpha_list[i] * classifier_list[i].predict(X)
        return result

    def predict(self, X, classifier_list,alpha_list,threshold=0):
        '''Predict the catagories for given samples.

        Args:
            X: An ndarray indicating the  to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.
            classifier_list:the list of the weak classifiers
            alpha_list: the list of the coefficient of weak classifiers
        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        result = self.predict_scores(X,classifier_list,alpha_list)
        result[result > threshold] = 1
        result[result < threshold] = -1
        return result

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
