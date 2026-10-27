# Logistic Regression From Scratch

# Libraries
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

# Data
iris = load_iris()

#feature data frame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

#feature matrix
X = df.to_numpy()
#array of true labels
y = iris.target

#logistic regression
class LogisticRegression():

    def __init__(self, X, lr, w=None, b=None):
        #feature array
        self.X = X
        #learning rate
        self.lr = lr
        #setup weights and bias
        self.w = np.random.random(X.shape[0])
        self.b = np.random.random(X.shape[0])


    
        
print(LogisticRegression(X, y).b)