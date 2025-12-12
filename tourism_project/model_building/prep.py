# work with one-dimensional series objects and two-dimensional data frames
import pandas as pd
# for model selection and model building
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for scaling of numerical continuous features and significant discrete features
from sklearn.preprocessing import StandardScaler
# for converting categorical data to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi
# ignore all warning-type notifications
import warnings
warnings.filterwarnings('ignore')

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))

hf_login_id = "JaiBhatia020373"

# set name of the new repository on the Hugging face hub
repo_name = "tourism-package-prediction"

# DATASET_PATH = "hf://datasets/praneeth232/machine-failure-prediction/machine-failure-prediction.csv"
DATASET_PATH = "hf://datasets" + "/" + hf_login_id + "/" + repo_name + "/" + "tourism.csv"
df_trans = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# display the columns metadata of the dataset
print("Columns metadata of dataset is as below:\n")
print(df_trans.info())

# check duplicate rows
print("Checking for duplicate rows in the dataset...\n")
duplicate_rows_count = df_trans.duplicated().sum()
print("Number of duplicate rows:", str(duplicate_rows_count))

if duplicate_rows_count > 0:
    df_trans.drop_duplicates(inplace=True)
    print("Duplicate rows removed successfully.\n")
else:
    print("Observation:\n", "No duplicate rows found in the dataset.\n")

# Removing redundant columns from the dataset
print("Removing redundant columns from the dataset...\n")
df_trans.drop(columns=["CustomerID", "Unnamed: 0"], inplace=True)
print("Observation/Action:\n", "Redundant columns 'CustomerID' and 'Unnamed' \
dropped successfully.\n")
print("Quick view of updated columns structure as below (first 5 rows):...\n")
print(df_trans.head())

# check missing/null values
print("Checking for missing/null values in the dataset...\n")
missing_values_count_columnwise = df_trans.isnull().sum()
print(f"Columnwise missing value counts as below\n, {missing_values_count_columnwise}")
print("\n")

# apply standard scaling to all significant numerical columns: normalize continuous
# data like MonthlyIncome, Age, and other significant discrete-count columns
lst_numerical_cols_to_scale = \
 ['Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
  'NumberOfTrips', 'NumberOfChildrenVisiting', 'PitchSatisfactionScore',
  'MonthlyIncome']
print("Scaling of numerical features...\n")
scaler = StandardScaler()
df_trans[lst_numerical_cols_to_scale] = scaler.fit_transform(df_trans[lst_numerical_cols_to_scale])
print("Numerical features scaled successfully.\n")

# convert object-type columns to category types
for col in df_trans.select_dtypes(include="object").columns:
    df_trans[col] = df_trans[col].astype("category")

# Encoding ordinal (rank-based) category columns: ProductPitched, Designation,
# CityTier, and PreferredPropertyStar through label encode
print("Encoding ordinal category columns: ProductPitched, Designation, CityTier \
 and PreferredPropertyStar through label encoding...\n")

label_encoder = LabelEncoder()
df_trans['ProductPitched'] = label_encoder.fit_transform(df_trans['ProductPitched'])
df_trans['Designation'] = label_encoder.fit_transform(df_trans['Designation'])
df_trans['CityTier'] = label_encoder.fit_transform(df_trans['CityTier'])
df_trans['PreferredPropertyStar'] = label_encoder.fit_transform(df_trans['PreferredPropertyStar'])
print("Ordinal category columns encoded successfully.\n")

# Encoding category columns: TypeofContact, Occupation, Gender, MaritalStatus,
# Passport through one-hot encode
print("Encoding nominal category columns: TypeofContact, Occupation, Gender, \
MaritalStatus, Passport and OwnCar through one-hot encoding...\n")
lst_nominal_categories = ['TypeofContact', 'Occupation', 'Gender',
                          'MaritalStatus', 'Passport', 'OwnCar']
df_encoded = pd.get_dummies(df_trans, columns=lst_nominal_categories, drop_first=True)
print(df_encoded)
print("Nominal category columns encoded successfully.\n")

# df_encoded contains both untouched and encoded columns so updating dataset
df_trans = df_encoded.copy(deep=True)
print("Scaling of numerical features and encoding of category columns \
 completed successfully.\n")
print("First 10 rows of cleaned, scaled, and encoded final dataset as below:\n")
print(df_trans.head(10))
print("\n")
print("Splitting dataset into train and test sets in 75%:25% ratio...\n")

# Split into X (features columns) and y (target column)
print("Target column: 'ProdTaken' column\n")
target_col = 'ProdTaken'

X = df_trans.drop(columns=[target_col])
y = df_trans[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.25, random_state=42
    )

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

# https://huggingface.co/datasets/JaiBhatia020373/tourism-package-prediction/blob/main/Xtrain.csv
Xtrain_set = pd.read_csv("Xtrain.csv")
print("Xtrain dataset shape:", Xtrain_set.shape)

ytrain_set = pd.read_csv("ytrain.csv")
print("ytrain dataset shape:", ytrain_set.shape)

Xtest_set = pd.read_csv("Xtest.csv")
print("Xtest dataset shape:", Xtest_set.shape)

ytest_set = pd.read_csv("ytest.csv")
print("ytest dataset shape:", ytest_set.shape)

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

repo_id = hf_login_id + "/" + repo_name
repo_type = "dataset"

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id=repo_id,
        repo_type=repo_type
    )

    print(f"File '{file_path}' uploaded to Hugging Face space successfully.")

print("Splitting dataset completed successfully.\n")
