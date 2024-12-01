import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# 데이터셋 로드
data_path = r"C:\Users\sec\Downloads\Ransomware.csv"  # 데이터셋 경로 (파일 경로를 정확히 지정해주세요)
df = pd.read_csv(data_path, sep='|')

# 데이터를 샘플링하여 모델 학습 속도 향상
df_sample = df.sample(frac=0.1, random_state=42)  # 10%만 샘플링

features = df_sample.drop(columns=['legitimate', 'Name', 'md5'])
target = df_sample['legitimate']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# 데이터 정규화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SVM 모델 학습
svm_model = SVC(kernel='linear', probability=True, max_iter=1000)  # max_iter 설정하여 최대 학습 횟수 지정
svm_model.fit(X_train, y_train)

# 데이터셋 정보 확인
print(df.info())  # 데이터프레임의 열 확인

# 데이터프레임 열 출력 (확인용)
print(df.columns)

# 'label'과 'attack_cat'을 삭제하는 대신, 실제로 삭제해야 할 열들을 확인 후 수정합니다.
# 예시로 'legitimate'을 타겟으로 사용하고 'Name', 'md5'를 제외합니다.

features = df.drop(columns=['legitimate', 'Name', 'md5'])  # 'label'이 아닌 'legitimate'을 타겟으로 설정
target = df['legitimate']  # 'legitimate'을 타겟으로 사용

# 데이터 나누기: 훈련 데이터와 테스트 데이터 분리 (70% 훈련, 30% 테스트)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# 데이터 정규화 (SVM은 데이터의 스케일에 민감하기 때문에 정규화 필요)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_model = SVC(kernel='rbf', C=10, gamma=0.1, probability=True, max_iter=1000)
svm_model.fit(X_train, y_train)

# 모델 예측
y_pred = svm_model.predict(X_test)

# 정확도 출력
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# 분류 성능 평가 (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Confusion Matrix 시각화
fig, ax = plt.subplots(figsize=(6,6))
ax.matshow(cm, cmap="Blues", alpha=0.5)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', fontsize=16)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# 분류 성능 보고서
print(classification_report(y_test, y_pred))

# 예측 확률
y_prob = svm_model.predict_proba(X_test)[:, 1]
print("Prediction probabilities for the positive class (Ransomware):")
print(y_prob)
