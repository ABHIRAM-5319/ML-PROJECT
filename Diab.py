#PROJECT REPORT

#import the all libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

#read csv file
df=pd.read_csv("/content/diabetes_prediction.csv")
df

df.shape

#check null values
df.info()

df.isna().sum()

X=df[["gender","age","hypertension","heart_disease","smoking_history","bmi","HbA1c_level","blood_glucose_level"]]
X # Taking all data except target value

Y=df['diabetes']
Y # Target value

#split the data using train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=40) #test_size means 20% of data is taking for testing

x_train

x_test

y_train

y_test

#completed of splitting the data into training and testing data

"""Before Giving the test data ml algorithm
1. we should convert the all values into numericals(Onehotencoder)
2. there should be no null values(SimpleImputer)
3. use standardscalar to avoid the wrong values"""

#Seperate numerical features and categorical features
numerical_features =["age","hypertension","heart_disease","bmi","HbA1c_level","blood_glucose_level"]
numerical_features

categorical_features=["gender","smoking_history"]
categorical_features

#check the total null values in numerical and categorical features
df[numerical_features].isnull().sum()

df[categorical_features].isnull().sum()

# To fill fill the null values we use the below methods
# numerical_transformer
#categorical_transformer

numerical_transformer=Pipeline(
    steps=[
        ("Imputer",SimpleImputer(strategy="mean")),
        ("StandardScalar",StandardScaler())
    ]
)

numerical_transformer

categorical_transformer = Pipeline(
    steps=[
        ("Imputer", SimpleImputer(strategy="most_frequent")),
        ("OneHotEncoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

categorical_transformer

# ColumnTransformer : fitting the numerical and categorical transformers into ColumnTransformer is named as preprocessor with the data is given from numerical and categorical features

preprocessor=ColumnTransformer(
    transformers=[
        ("numerical",numerical_transformer,numerical_features),
        ("categorical",categorical_transformer,categorical_features)
    ]
)

preprocessor

# After creating a Preprocessor we completed a ML algorithm now we should create a model

lr_model = Pipeline(
    steps=[
        ("Preprocessor", preprocessor),
        ("Classifier", LogisticRegression(solver='liblinear'))
    ]
)

lr_model

lr_model=lr_model.fit(x_train,y_train) # train the model for understanding the patterns from the training data

y_predict=lr_model.predict(x_test) # predicting the target data by usind all the data

y_predict

y_test

lr_acc = accuracy_score(y_test,y_predict) #main goal is improving accuracy using other algorithms

lr_acc

knn_model=Pipeline(
    steps=[
        ("Preprocessor",preprocessor),
        ("Classifier",KNeighborsClassifier())
    ]
)

knn_model.fit(x_train,y_train)

knn_y_pred=knn_model.predict(x_test)

knn_y_pred

y_test

knn_acc = accuracy_score(y_test,knn_y_pred)

knn_acc

dt_model = Pipeline(steps=[
    ("Preprocessor", preprocessor),
    ("Classifier", DecisionTreeClassifier())
])

dt_model.fit(x_train, y_train)

dt_y_predict = dt_model.predict(x_test)
dt_y_predict

dt_accuracy = accuracy_score(y_test, dt_y_predict)

dt_accuracy

NB_model = Pipeline(steps=[
    ("Preprocessor",preprocessor),
    ("Classifier", GaussianNB())
])

NB_model.fit(x_train, y_train)

nb_y_pred=NB_model.predict(x_test)
nb_y_pred

nb_acc = accuracy_score(y_test, nb_y_pred)
nb_acc

svm_model = Pipeline(steps=[
    ("Preprocessor", preprocessor),
    ("Classifier", SVC())
])

svm_model.fit(x_train, y_train)

svm_y_pred = svm_model.predict(x_test)
svm_y_pred

svm_acc=accuracy_score(y_test,svm_y_pred)
svm_acc

rf_model = Pipeline(steps=[
    ("Preprocessor", preprocessor),
    ("Classifier", RandomForestClassifier())
])

rf_model.fit(x_train, y_train)

rf_y_pred=rf_model.predict(x_test)
rf_y_pred

rf_acc=accuracy_score(y_test,rf_y_pred)
rf_acc

# Define the hyperparameter grid for logistic regression
param_grid = {
    'Classifier__C': [0.01, 0.1, 1, 10, 100],
    'Classifier__penalty': ['l1', 'l2']
}

# Initialize GridSearchCV
grid_search = GridSearchCV(lr_model, param_grid, cv=5, scoring='accuracy')

# Fit the model with grid search
grid_search.fit(x_train, y_train)


# Print best parameters and best score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation accuracy: {:.2f}".format(grid_search.best_score_))

#testcase accuracy for Linear Regression
print(" Test accuracy for Linear Regression",lr_acc)

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='diabetes', palette='pastel')
plt.title('Distribution of Diabetes Outcome')
plt.xlabel('Outcome (0 = Non-Diabetic, 1 = Diabetic)')
plt.ylabel('Count')
plt.show()

print("\nAccuracy Scores of All Models:")
print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
print(f"K-Nearest Neighbors Accuracy: {knn_acc:.4f}")
print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")
print(f"Naive Bayes Accuracy: {nb_acc:.4f}")
print(f"Support Vector Machine Accuracy: {svm_acc:.4f}")
print(f"Random Forest Accuracy: {rf_acc:.4f}")