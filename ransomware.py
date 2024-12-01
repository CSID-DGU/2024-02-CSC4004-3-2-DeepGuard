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

    
