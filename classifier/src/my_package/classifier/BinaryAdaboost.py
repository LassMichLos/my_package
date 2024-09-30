import numpy as np

class DecisionStump():
    '''
    Decision stump is used as weak classifier, simply splits data into 2 parts .
    '''
    def __init__(self):
        self.polarity = 1 #Decide which part is classified as positiv or negativ.
        self.feature_idx = None #Choose the feature to split
        self.threshold = None #Find the threshhold to split
        self.alpha = None #Meassure the accuracy of the classifier


    def split(self, X):
        '''
        Function that splits the dataset based on thereshold of chosen feature and classifies them. 
        '''
        n_samples = X.shape[0]
        feature = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            #if polarity is 1, the left side is classified as negativ
            predictions[feature < self.threshold] = -1
        else:
            #if polarity is -1, the right side is classified as negativ
            predictions[feature > self.threshold] = -1

        return predictions


class Adaboost():
    '''
    Adaboost assembles a specific number of weak learners in order to make a stronger, preciser classification.
    Weak learners are sequentially created. The later will focus more on misclassifications of the former to be able to reclassify them.
    '''
    def __init__(self, n_clf=7, weak_learner = DecisionStump()):
        self.n_clf = n_clf #Number of weak learner
        self.weak_learner = weak_learner #Type of weak learner

    def fit(self, X, y):
        '''
        Function finding weak learners sequentially and returning them as a list.
        Default weak learner is DecisionStump.
        '''       
        n_samples, n_features = X.shape    
        w = np.full(n_samples, (1 / n_samples)) #Initialize weights to 1/N
        self.clfs = [] #List to store all splits
        
        #Iterate through classifiers
        for _ in range(self.n_clf):
            clf = self.weak_learner
            min_error = float('inf')
            
            # greedy search to find best threshold and feature
            for feature_i in range(n_features):
                feature = X[:, feature_i]
                thresholds = np.unique(feature)
                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[feature < threshold] = -1  
                    #Calculate error = sum of weights of misclassified samples                    
                    misclassified = w[y != predictions]
                    error = sum(misclassified) 
                    if error > 0.5:
                        error = 1 - error
                        p = -1
                    #Store the best 
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i
                        min_error = error

            #Calculate predictions and update weights
            clf.alpha = 0.5 * np.log((1.0 - min_error) / (min_error + 1e-10))
            predictions = clf.split(X)
            w *= np.exp(-clf.alpha * y * predictions)           
            w /= np.sum(w)  # Normalize to one

            # Save classifier
            self.clfs.append(clf)

    def predict(self, X):
        '''
        Function predicts classification based on split of all weak learner
        '''
        clf_preds = [clf.alpha * clf.split(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)

        return y_pred
