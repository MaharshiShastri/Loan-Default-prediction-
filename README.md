# Loan-Default-prediction-
# For this basic implementation, we only need these modules from sklearn.datasets import load_breast_cancer from sklearn.model_selection import train_test_split from sklearn.tree import Decision Tree Classifier from sklearn.ensemble import AdaBoostClassifier

# from sklearn.metrics import classification_report # Import classification_report # Rem import

# Load the well-known Breast Cancer dataset

#Split into train and test sets

x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test train_test_split(x, y, test_size=0.25,

random state-23)

#The base learner will be a decision tree with depth = 2

tree - Decision TreeClassifier(max_depth underline - 2 , random_state te = 23 )

#AdaBoost initialization

#It's defined the decision tree as the base learner

#The number of estimators will be 5

#The penalizer for the weights of each estimator is 0.1

adaboost AdaBoost Classifier(estimator tree, n_estimators = 50 , # Changed n_estimators to 50

learning_raterandom_state z = 23 ) = 0 ,1,

#Train!

adaboost.fit(x_train, y_train)

# Evaluation

print(f"Train score: {adaboost.score(x_train, y_train)}")

print(f"Test score: {adaboost.score(x_test, y_test)}")

#Generate additional evaluation metrics # Removed lines below

#y_pred adaboost.predict(x_test)

#print("\nClassification Report:")

#print(classification_report(y_test, y_pred))