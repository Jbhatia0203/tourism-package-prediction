# work with one-dimensional series objects and two-dimensional data frames
import pandas as pd
import numpy as np

# import random forest classifier library
from sklearn.ensemble import RandomForestClassifier

# import xgboost classifier library
from xgboost import XGBClassifier

# pipeline
from imblearn.pipeline import Pipeline

# For evaluating model performance
from sklearn.metrics import accuracy_score, recall_score, precision_score, \
f1_score, confusion_matrix

# for model serialization
import joblib

# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

print("Model Evaluation Criteria/Metrics")
print("------------------------------")
print("1. True Positive(TP): Both actual and predicted outcomes for target 'ProdTaken' are 1.")
print("2. False Positive(FP): Actual outcome is 0 but Predicted outcome for target 'ProdTaken' is 1.")
print("3. True Negative(TN): Both actual and predicted outcomes for target feature 'ProdTaken' are 0.")
print("4. False Negative(FN): Actual outcome is 1 but Predicted outcome for target 'ProdTaken' is 0.")
print("5. Accuracy metric = (TP + TN) / (TP + TN + FP + FN)")
print("6. Precision metric = TP / (TP + FP)")
print("7. Recall metric = TP / (TP + FN)")
print("------------------------------")
print("\n")
print("From above criteria: False Negative should be minimized so that model does \
not wrongly perceive the interested customers as those not buying the product.")
print("So recall metric is key for this use case and needs to be maximised")
print("\n")

def confusion_matrix_sklearn(model, predictors, target):
  '''
   function computes confusion matrix for a given model and data with percentages
   of TP, TN, FP, and FN.

   'model' parameter: classifier
   'predictors' parameter: Xtrain or Xtest (independent variables)
   'target' parameter: ytrain or ytest (dependent variable)
  '''
  y_pred = model.predict(predictors)
  cm = confusion_matrix(target, y_pred)
  labels = np.asarray(
      [
        ["{0:0.0f}".format(item) + \
         "\n{0:.2%}".format(item / cm.flatten().sum())]\
        for item in cm.flatten()

      ]
  ).reshape(2, 2)

  # draw plot-image of size 6 by 4
  plt.figure(figsize=(6, 4))
  '''
  plot the heatmap using sns library with annotations as:
  TP with TP% data, TN with TN% data, FP with FP% data, FN with FN% data, and
  colormap as different shades representing heat indicators
  '''
  sns.heatmap(cm, annot=labels, fmt="", cmap="Blues")
  # set axis-labels for the plot
  plt.ylabel("Actual Outcome Label")
  plt.xlabel("Predicted Label")
  # set plot title
  plt.title("Confusion Matrix")
  # display plot
  plt.show()

def evaluate_model_performance(model, predictors, target, index_label):
  '''
   function computes different performance metrics for a given model and data.

    'model' parameter: classifier
    'predictors' parameter: Xtrain or Xtest (independent variables)
    'target' parameter: ytrain or ytest (dependent variable)
  '''
  # predict using input features
  pred = model.predict(predictors)
  # compute accuracy
  accuracy = accuracy_score(target, pred)
  # compute recall
  recall = recall_score(target, pred)
  # compute precision
  precision = precision_score(target, pred)
  # compute f1-score
  f1 = f1_score(target, pred)
  # creating a performance metrics dataframe
  df_perf = pd.DataFrame(
      {"Accuracy": accuracy, "Recall": recall, "Precision": precision, "F1": f1},
      index=[index_label],
  )

  return df_perf

print("\n")

Xtrain = pd.read_csv("Xtrain.csv")
ytrain = pd.read_csv("ytrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytest = pd.read_csv("ytest.csv")

print("Random Forest classifier - with default parameters")
print("------------------------------")
rforest = RandomForestClassifier(random_state=42)
print("Random Forest classifier initialized")
rforest.fit(Xtrain, ytrain)
print("Random Forest model with default parameters trained successfully.\n")
print("Displaying confusion matrix for Random Forest classifier...\n")
confusion_matrix_sklearn(rforest, Xtrain, ytrain)
print("\n")
print("Random Forest classifier model performance with training set as below:\n")
df_perf_rforest_train = evaluate_model_performance(rforest, Xtrain, ytrain, 'Random_Forest_with_train_set')
print(df_perf_rforest_train)
print("\n")
print("Random Forest classifier model performance with test set as below:\n")
df_perf_rforest_test = evaluate_model_performance(rforest, Xtest, ytest, 'Random_Forest_with_test_set')
print(df_perf_rforest_test)
print("------------------------------")
print("Observations:")
print("Random Forest model is Overfit because the recall metric is 100% with \
train set but it is approx. 67.4% with test set. So hyper-parameters tuning reqd.")
print("\n")
print("Random Forest classifier: fine-tuning with Hyper-parameters")
print("using RandomSearch optimization...")
print("------------------------------")
hyper_parameters = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10],
    "class_weight": [None, 'balanced']
}
print("hyper parameters values grid as below")
print(hyper_parameters)
# Run the random search
from sklearn.model_selection import RandomizedSearchCV

grid_obj = RandomizedSearchCV(
    estimator=rforest,
    param_distributions=hyper_parameters,
    n_iter=10,              # try 10 random combinations
    scoring="recall",
    cv=3,                   # fewer folds
    n_jobs=2,               # 2 or 4 parallel jobs: avoid excessive parallelism
    random_state=42
)

grid_obj = grid_obj.fit(Xtrain, ytrain)

# get the best estimator for random forest model
rf_tuned = grid_obj.best_estimator_

# train the best algorithm with train set
rf_tuned.fit(Xtrain, ytrain)
print("Random Forest model with tuned hyper-parameters trained successfully.\n")
print("Displaying confusion matrix for fine-tuned Random Forest with test-set...")
confusion_matrix_sklearn(rf_tuned, Xtest, ytest)
print("Fine-tuned Random Forest classifier performance with test-set as below:\n")
df_perf_rforest_tuned = evaluate_model_performance(rf_tuned, Xtest, ytest, 'Random_Forest_fine_tuned')
print("------------------------------")
print(df_perf_rforest_tuned)
print("------------------------------")
print("\n")
print("Observations on fine-tuned Random Forest model performance:\n")
print("1. Accuracy = 87%, Recall = 69%, Precision = 66%, F1 = 67%")
print("2. There is very little improvement in Recall score: by 1%")
print("3. Boosting classifiers: ADABoost or XGBoost can be used")
print("\n")
print("------------------------------")
print("\n")
print("XGBoost classifier - with default parameters")
print("------------------------------")

# Define the model
xgb = XGBClassifier(
    use_label_encoder=False,   # suppress warning
    eval_metric='logloss',     # required for newer versions
    random_state=42
)
print("XGBoost classifier initialized")
xgb.fit(Xtrain, ytrain)
print("XGBoost model with default parameters trained successfully.\n")
print("Displaying confusion matrix for XGBoost classifier...\n")
confusion_matrix_sklearn(xgb, Xtrain, ytrain)
print("\n")
print("XGBoost classifier model performance with training set as below:\n")
df_perf_xgb_train = evaluate_model_performance(xgb, Xtrain, ytrain, 'XGBoost_with_train_set')
print(df_perf_xgb_train)
print("\n")
print("XGBoost classifier model performance with test set as below:\n")
df_perf_xgb_test = evaluate_model_performance(xgb, Xtest, ytest, 'XGBoost_with_test_set')
print(df_perf_xgb_test)
print("------------------------------")
print("Observations:")
print("1. XGBoost model is Overfit because the recall metric is 99.7% with \
train set but it is approx. 78.8% with test set.")
print("2. Precision also reduced by nearly 10%.")
print("So hyper-parameters tuning reqd.")
print("\n")
print("XGBoost classifier: fine-tuning with Hyper-parameters")
print("using RandomSearch optimization...")
print("------------------------------")

# Define parameter distributions
hyperparameters_xgb = {
    'n_estimators': [100, 200], # no. of trees good for given learning rates
    'learning_rate': [0.1, 0.2, 0.3], # 0.1 to 0.3 range avoids overfitting
    'scale_pos_weight': [10, 12]  # useful for class imbalance
}

# 'subsample': [0.6, 0.8, 1.0],
# 'colsample_bytree': [0.6, 0.8, 1.0],

# Randomized search setup
grid_obj_xgb = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=hyperparameters_xgb,
    n_iter=10,          # number of random combinations
    scoring='recall',   # optimize for recall (can change to 'f1' or 'accuracy')
    cv=3,               # 3-fold cross-validation
    verbose=1,
    n_jobs=2,
    random_state=42
)

grid_obj_xgb.fit(Xtrain, ytrain)
print("Best hyperparameters for Random Search with xgb classifier:", grid_obj_xgb.best_params_)
# get the best estimator for xgboost model
xgb_tuned = grid_obj_xgb.best_estimator_

# train the best algorithm with train set
xgb_tuned.fit(Xtrain, ytrain)
print("XGBoost model with tuned hyper-parameters trained successfully.\n")
print("Displaying confusion matrix for fine-tuned XGBoost with test-set...")
confusion_matrix_sklearn(xgb_tuned, Xtest, ytest)
print("Fine-tuned XGBoost classifier performance with test-set as below:\n")
df_perf_xgb_tuned = evaluate_model_performance(xgb_tuned, Xtest, ytest, 'XGBoost_fine_tuned')
print("------------------------------")
print(df_perf_xgb_tuned)
print("------------------------------")
print("\n")
print("Observations on fine-tuned XGBoost model performance as below:\n")
print("1. Accuracy = 90.9%, Recall = 90.6%, Precision = 71%, F1 = 79.6%")
print("2. There is tremendous improvement. Focussed Recall score is 90.6%")
print("3. Accuracy and F1 have dipped a little: by 5%, its a controlled change.")
print("\n")
df_compare_scores = pd.concat([df_perf_rforest_test,
                              df_perf_rforest_tuned, 
                              df_perf_xgb_test, 
                              df_perf_xgb_tuned],
                              axis=0)
'''
index=['Random_Forest_with_test_set',
'Random_Forest_fine_tuned',
'XGBoost_with_test_set',
'XGBoost_fine_tuned']
'''
print("------------------------------")
print("Models Performance Comparison")
print("------------------------------")
print(df_compare_scores)
print("------------------------------")
print("\n")
print("Inference/Recommendation below:")
print("The fine-tuned XGBoost classifier is the best model with highest Recall score as 90.6%")
print("------------------------------")
print("Creating the joblib file for the best model: XGBoost fine-tuned classifier...")
joblib.dump(xgb_tuned, "tourism-package-prediction-classifier.joblib")
print("joblib file created successfully.\n")
print("------------------------------")

# hugging face login profile id
hf_login_id = "JaiBhatia020373"

# set name of the new repository on the Hugging face hub
repo_name = "tourism-package-prediction-model"

# repository type - model repository
repo_type = "model"

repo_id = hf_login_id + "/" + repo_name

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists and if not then create it
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")



# upload the serialized model file (.joblib) to model repository
print("Uploading the serialized model file (.joblib) to model repository...")
try:
    api.upload_file(
        path_or_fileobj="tourism-package-prediction-classifier.joblib",
        path_in_repo="tourism-package-prediction-classifier.joblib",
        repo_id=repo_id,
        repo_type=repo_type
    )
    print("joblib model file uploaded to Hugging Face repo successfully.")
    print("------------------------------")
except HfHubHTTPError as e:
    print(f"Error uploading file: {e}") 
    print("------------------------------")

