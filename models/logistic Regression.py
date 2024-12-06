import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# 데이터 로드
file_path = r"C:\Users\sec\Downloads\Ransomware.csv"  # 데이터 파일 경로
data = pd.read_csv(file_path, delimiter='|')

# 특징(X)과 타겟 변수(y) 설정
X = data.drop(columns=['Name', 'md5', 'legitimate'])
y = data['legitimate']

# 특성 데이터 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 데이터 분리 (훈련 데이터와 테스트 데이터)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 로지스틱 회귀 모델 학습
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train, y_train)

# 테스트 데이터 예측
y_pred = logistic_model.predict(X_test)

# 모델 평가
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
