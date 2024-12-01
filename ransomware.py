# -*- coding: utf-8 -*-
"""Ransomware.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1F7DieUzDP5fNd1InP3xye5nl2Jy1Jik3

# Malware Prediction

## Process for Malware Infestation

![malware attack process.webp](attachment:5a73e1de-9bbc-4818-abbe-d0259c1db756.webp)

## Problem Statement

**Task:** We are developing a script for an antivirus package aimed at identifying malware before executing `.exe` files on Windows. Our approach involves analyzing specific properties of executable files, such as:

- **ImageBase**: The preferred base address where the file's image is loaded into memory.
- **VersionInformationSize**: The size of the version information resource, which may provide details about the file's version, company, and product name.
- **SectionsMaxEntropy**: The maximum entropy value across all sections of the executable, indicating the level of randomness and potential obfuscation techniques used.

My objective is to design a method to assess these properties to identify potential malware. The script should flag any `.exe` files that exhibit unusual or suspicious values for these properties, indicating a higher likelihood of being malicious.

## Installing Libraries
"""

# !pip install lazypredict
# !pip install lime

"""## Importing Libraries and Dataset"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.ensemble import *
from sklearn.metrics import *
import lime.lime_tabular
from imblearn.over_sampling import SMOTE
from collections import Counter

df=pd.read_csv(r"C:\Users\sec\Downloads\Ransomware.csv",sep='|')
df.head()

"""## Dataset Exploration"""

# Total records
print(df.shape[0])

df.describe()

"""## Null Value Check"""

df.isnull().sum()

"""## Distribution of Labelled Data"""

df.legitimate.value_counts() #1 means legitimate, 0 means malware

df.info()

plt.pie(df.legitimate.value_counts().values.tolist(), labels=['Safe','Ransomware'], autopct='%.2f%%')
plt.legend()
plt.title(f"Distribution of Labelled Data, total - {len(df)}")
plt.show()

"""## Unique Files/Objects (MD5)"""

df.head()

df.md5.nunique()

# There are no same files as no 2 files can have same md5 without the same content
df.md5.isnull().sum()

"""## Analyis of Correlated Attributes"""

df.head()

# Correlation b/w independent columns/attributes
sns.heatmap(df.drop(['Name','md5','legitimate'], axis=1).corr())
plt.show()

# Create correlation matrix
corr_matrix = df.drop(['Name','md5','legitimate'], axis=1).corr().abs() # Includes negative correlation as well

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

print(to_drop)

# Drop features
df.drop(to_drop, axis=1, inplace=True)

sns.heatmap(df.drop(['Name','md5','legitimate'], axis=1).corr())

"""## Most Relevant Features for Prediction using IV and WoE"""

def iv_woe(data, target, bins=10, show_woe=False):
    #Empty Dataframe
    newDF,woeDF = pd.DataFrame(), pd.DataFrame()

    #Extract Column Names
    cols = data.columns

    #Run WOE and IV on all the independent variables
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
            binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Events']/d['% of Non-Events'])
        d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        print("Information value of " + ivars + " is " + str(round(d['IV'].sum(),6)))
        temp =pd.DataFrame({"Variable" : [ivars], "IV" : [d['IV'].sum()]}, columns = ["Variable", "IV"])
        newDF=pd.concat([newDF,temp], axis=0)
        woeDF=pd.concat([woeDF,d], axis=0)

        #Show WOE Table
        if show_woe == True:
            print(d)
    return newDF, woeDF

df.legitimate.dtypes

iv, woe = iv_woe(df.drop(['Name','md5'],axis=1), 'legitimate')

iv.sort_values(by = 'IV', ascending=False)

thresh = 1
res = len(iv)-len(iv[iv['IV']>thresh])
print(res) # Total 14 features which are relevant (greater than threshold)

features = iv.sort_values(by = 'IV', ascending=False)['Variable'][:res].values.tolist()

print(features,'\n')
print('Total number of features-\n',len(features))

"""## Training Machine Learning Model"""

X = df[features]
y = df['legitimate']

randomseed = 42 # Used for replicating the experiment

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=randomseed)

print(X_test.shape[0] + X_train.shape[0])
print('Training labels shape:', y_train.shape)
print('Test labels shape:', y_test.shape)
print('Training features shape:', X_train.shape)
print('Test features shape:', X_test.shape)

"""### Lazy Predict - AutoML"""

# highmem_classifiers = ["LabelSpreading", "LabelPropagation", "BernoulliNB", "KNeighborsClassifier",
#                        "ElasticNetClassifier", "GradientBoostingClassifier", "HistGradientBoostingClassifier"]

# # Remove the high memory classifiers from the list
# classifiers = [c for c in lazypredict.Supervised.CLASSIFIERS if c[0] not in highmem_classifiers]

# clf = LazyClassifier(classifiers=classifiers, verbose=0, ignore_warnings=True, custom_metric=None)
# models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# print(models)

rf = RandomForestClassifier(random_state = randomseed)

rf.fit(X_train,y_train)

pred = rf.predict(X_test)
pred_proba = rf.predict_proba(X_test)

# Extract probabilities for the positive class (1)
pred_proba = np.array([prob[1] for prob in pred_proba])

"""## Confusion Matrix"""

cm = confusion_matrix(y_test,pred)
cm

# Classes
classes = ['Safe', 'Malware']

cmd = ConfusionMatrixDisplay(cm, display_labels=classes)
cmd.plot()
plt.show()

# Extract TP, FP, FN, TN
TP = cm[1, 1]
FP = cm[0, 1]
FN = cm[1, 0]
TN = cm[0, 0]

# Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)

# Precision
precision = TP / (TP + FP)

# Recall
recall = TP / (TP + FN)

# F1 Score
f1 = 2 * (precision * recall) / (precision + recall)

# Matthews Correlation Coefficient (MCC)
mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

# False Positive Rate (FPR)
fpr = FP / (FP + TN)

# Calculate AUC score
auc = roc_auc_score(y_test, pred_proba)

# Print all the metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"False Positive Rate: {fpr:.4f}")
print(f"AUC Score: {auc:.4f}")

"""## Explainability for Testing Dataset using LIME"""

explainer_lime = lime.lime_tabular.LimeTabularExplainer(X_train.values,feature_names=X_train.columns,
                                                        verbose=True, mode='classification')

"""## Legitimate File Explanation"""

i = 1 # index from X_test

exp_lime = explainer_lime.explain_instance(X_test.values[i], rf.predict_proba, num_features=5)

exp_lime.show_in_notebook(show_table=True)

"""## Malware File Explanation"""

i = 6 # index from X_test

exp_lime = explainer_lime.explain_instance(X_test.values[i], rf.predict_proba, num_features=5)

exp_lime.show_in_notebook(show_table=True)

"""## Fixing Data Imbalance - SMOTE TOMEK

![smote-tomek.webp](attachment:db128a92-5b41-4d0e-b6ea-ac1b3f5165d1.webp)

Synthetic minority oversampling technique, also termed as SMOTE, is a clever way to perform over-sampling over the minority class to avoid overfitting(unlike random over-sampling that has overfitting problems). In SMOTE, a subset of minority class is taken and new synthetic data points are generated based on it. These synthetic data points are then added to the original training dataset as additional examples of the minority class.

SMOTE technique overcomes the overfitting problem from random over-sampling as there is no replication of the examples. Secondly, as no data points are removed from the dataset, so no loss of useful information.
"""

counter = Counter(y_train)
print('Before', counter)

# oversampling the train dataset using SMOTE
smt = SMOTE()
X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)

counter = Counter(y_train_sm)
print('After', counter)

rf = RandomForestClassifier(random_state = randomseed)
rf.fit(X_train_sm, y_train_sm)

pred = rf.predict(X_test)
pred_proba = rf.predict_proba(X_test)

# Extract probabilities for the positive class (1)
pred_proba = np.array([prob[1] for prob in pred_proba])

cm = confusion_matrix(y_test,pred)
cm


# Classes
classes = ['Safe', 'Malware']

cmd = ConfusionMatrixDisplay(cm, display_labels=classes)
cmd.plot()
plt.show()

# Extract TP, FP, FN, TN
TP = cm[1, 1]
FP = cm[0, 1]
FN = cm[1, 0]
TN = cm[0, 0]

# Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)

# Precision
precision = TP / (TP + FP)

# Recall
recall = TP / (TP + FN)

# F1 Score
f1 = 2 * (precision * recall) / (precision + recall)

# Matthews Correlation Coefficient (MCC)
mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

# False Positive Rate (FPR)
fpr = FP / (FP + TN)

# Calculate AUC score
auc = roc_auc_score(y_test, pred_proba)

# Print all the metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"False Positive Rate: {fpr:.4f}")
print(f"AUC Score: {auc:.4f}")

"""## Explainability using SMOTE TOMEK enhanced data-model"""

explainer_lime = lime.lime_tabular.LimeTabularExplainer(X_train.values,feature_names=X_train.columns,
                                                        verbose=True, mode='classification')

"""## Legitimate File Explanation"""

i = 1 # index from X_test

exp_lime = explainer_lime.explain_instance(X_test.values[i], rf.predict_proba, num_features=5)

exp_lime.show_in_notebook(show_table=True)

"""## Malware File Explanation"""

i = 6 # index from X_test

exp_lime = explainer_lime.explain_instance(X_test.values[i], rf.predict_proba, num_features=5)

exp_lime.show_in_notebook(show_table=True)

"""1. Prediction of a sample (index 6) for malware increased from 92.9% to 97.7%
2. Prediction of a sample (index 1) for legitimate increased from 23% to 47%
"""
import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import pefile

# 모델 로드 함수
def load_model(model_path):
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Please train and save the model first.")
        return None, None

    try:
        with open(model_path, "rb") as file:
            model_data = pickle.load(file)

        if isinstance(model_data, dict) and "model" in model_data and "features" in model_data:
            print("Model and features loaded successfully.")
            return model_data["model"], model_data["features"]
        else:
            print("Invalid model format. Expected a dictionary with 'model' and 'features'.")
            return None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


# 단일 파일에서 Feature 추출
def extract_features(file_path):
    """단일 파일에서 Feature 추출"""
    try:
        if not file_path.endswith(".exe"):
            raise ValueError("Not a .exe file")

        pe = pefile.PE(file_path)
        features = {
            "FileName": os.path.basename(file_path),
            "ImageBase": pe.OPTIONAL_HEADER.ImageBase,
            "SizeOfImage": pe.OPTIONAL_HEADER.SizeOfImage,
            "CheckSum": pe.OPTIONAL_HEADER.CheckSum,
            "Subsystem": pe.OPTIONAL_HEADER.Subsystem,
            "MajorSubsystemVersion": pe.OPTIONAL_HEADER.MajorSubsystemVersion,
            "MinorSubsystemVersion": pe.OPTIONAL_HEADER.MinorSubsystemVersion,
            "SectionsMaxEntropy": max(section.get_entropy() for section in pe.sections),
            "SectionsMinEntropy": min(section.get_entropy() for section in pe.sections),
            "SectionsMeanEntropy": sum(section.get_entropy() for section in pe.sections) / len(pe.sections),
        }
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# 디렉토리 내 모든 .exe 파일에서 Feature 추출
def process_directory(directory_path):
    data = []
    skipped_files = []
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if file_name.endswith(".exe"):
            features = extract_features(file_path)
            if features:
                data.append(features)
            else:
                skipped_files.append(file_name)
        else:
            skipped_files.append(file_name)
    if skipped_files:
        print(f"Skipped files (not .exe or errors occurred): {skipped_files}")
    return data


# 누락된 Feature를 채우는 함수
def fill_missing_features(features_df, feature_columns):
    for col in feature_columns:
        if col not in features_df.columns:
            features_df[col] = 0  # 기본값 설정 (필요에 따라 조정 가능)
    return features_df


# 랜섬웨어 여부 판별 함수
def predict_ransomware(features_df, model, feature_columns, threshold=0.5):
    # 누락된 Feature 채우기
    features_df = fill_missing_features(features_df, feature_columns)

    # 모델 예측 수행
    feature_data = features_df[feature_columns]
    probabilities = model.predict_proba(feature_data)[:, 1]
    predictions = (probabilities < threshold).astype(int)

    # 결과 생성
    features_df['Probability'] = probabilities
    features_df['Prediction'] = predictions
    features_df['Result'] = features_df['Prediction'].apply(lambda x: 'Safe' if x == 1 else 'Ransomware')
    return features_df[['FileName', 'Probability', 'Result']]


# Prediction 수행
if __name__ == "__main__":
    # 모델 파일 경로
    model_path = r"C:\Users\sec\Downloads\ransomware_detection_model.pkl"

    # 모델 로드
    model, feature_columns = load_model(model_path)
    if model is None or feature_columns is None:
        print("Failed to load model. Exiting...")
        exit()

    print("Model loaded successfully.")
    print(f"Model type: {type(model)}")
    print(f"Feature columns: {feature_columns}")

    # Feature 추출
    directory_path = r"C:\Users\sec\Downloads"  # .exe 파일 경로
    features_list = process_directory(directory_path)

    # Feature DataFrame 생성
    if not features_list:
        print("No features extracted. Exiting...")
        exit()

    features_df = pd.DataFrame(features_list)

    # Threshold 설정
    threshold = 0.5
    print(f"Using fixed threshold: {threshold}")

    try:
        prediction_results = predict_ransomware(features_df, model, feature_columns, threshold=threshold)
        print("Prediction Results:")
        print(prediction_results)
        prediction_results.to_csv("prediction_results.csv", index=False)

        # 모든 칼럼 포함 파일 저장
        output_path_with_probabilities = "prediction_results_with_all_features.csv"
        features_df.to_csv(output_path_with_probabilities, index=False)
        print(f"File with probabilities and results saved to: {output_path_with_probabilities}")

    except Exception as e:
        print(f"Prediction failed: {e}")

    
