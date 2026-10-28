# Logistic Regression From Scratch

# Libraries
from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd

#logistic regression
class LogisticRegression():

    def __init__(self, lr=0.01, epochs=1000):
        
        #learning rate
        self.lr = lr

        #setup Epochs
        self.epochs = epochs

        #setup weights and bias if not set already
        self.w = None
        self.b = None

        #setup loss array
        self.loss_array = []

        return

    #sigmoid function
    def Sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def Predict_Prob(self, X):

        #calculate linear combination
        z = self.Scale(X) @ self.w + self.b

        # Clip z between -500 and 500 to prevent exponential overflow
        z = np.clip(z, -500, 500)

        #sigmoid activation
        y_probs = self.Sigmoid(z)

        #return predictions
        return y_probs

    #scale
    def Scale(self, X):
        return (X - self.X_mean) / (self.X_std + 1e-8)
    # training engine
    def Fit(self, X, y):

        #setup loss array
        self.loss_array = []
        #feature array mean and std
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)

        #scale
        self.X_scaled = self.Scale(X)

        #setup feature matrix info
        self.m = X.shape[0] # rows
        self.n = X.shape[1] # columns

        #setup random weights
        self.w = np.random.random(self.n) / 10
        self.b = np.random.random()  /10

        #training loop
        for epoch in range(self.epochs):

            y_probs = self.Predict_Prob(X)

            current_loss = self.Loss(y, y_probs)
            self.loss_array.append(float(current_loss))

            self.w, self.b = self.Gradient_Descent(y, y_probs, self.w, self.b)

        return

    def Predict(self, X):
        return np.where(self.Predict_Prob(X) >= 0.5, 1, 0)
    
    def Loss(self, y, y_probs):

        #set an epsilon to cap infinite loss
        epsilon = 1e-15
        #BCEL function
        return np.where(y == 1, -np.log(y_probs + epsilon), -np.log(1 - y_probs + epsilon)).mean()

    def Gradient_Descent(self, y, y_probs, w, b):
        error_vector = y_probs - y

        dw = self.X_scaled.T @ error_vector / self.m
        db = error_vector.sum() / self.m # divide by number of rows

        new_w = w - (self.lr * dw)
        new_b = b - (self.lr * db)

        return new_w, new_b

#   Training
# ------------

#Seed
np.random.seed(42)

# Data
cancer = load_breast_cancer()

#feature data frame
df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)

#feature matrix
X = df.to_numpy()

#array of true labels
y = cancer.target

#Model
model = LogisticRegression(lr=0.01, epochs=1000)

model.Fit(X, y)


#   Validation
# -----------------

#imports
from sklearn.metrics import classification_report, confusion_matrix

#dataset
validation_df = pd.read_csv(r"Datasets\breast_cancer_validation_100k.csv")

val_X = validation_df.drop(columns='target').to_numpy()
val_y = validation_df['target'].to_numpy()

#predict
predictions = model.Predict(val_X)

# Evaluation
print(classification_report(val_y, predictions))
print(confusion_matrix(val_y, predictions))