import numpy as np  
import pandas as pd  
import pickle
from sklearn.model_selection import train_test_split

'''
saved_dict = {}

dfs = []

# UNSW_NB 데이터셋 하나의 dataframe으로 병합
for i in range(1,5):
    path = 'AI_part/UNSW_NB15/UNSW-NB15_{}.csv' 
    dfs.append(pd.read_csv(path.format(i), header = None))
all_data = pd.concat(dfs).reset_index(drop=True)

# 칼럼 이름 all data에 적용
df_col = pd.read_csv('AI_part/UNSW_NB15/NUSW-NB15_features.csv', encoding='ISO-8859-1')
df_col['Name'] = df_col['Name'].apply(lambda x: x.strip().replace(' ', '').lower())
all_data.columns = df_col['Name']

saved_dict['columns'] = df_col['Name'][df_col['Name']!='label'].tolist()

# train set과 validation set으로 분할 
train, test = train_test_split(all_data, test_size=0.3, random_state=16)

# attack 유형 결측값을 normal로 변경
train['attack_cat'] = train.attack_cat.fillna(value='normal').apply(lambda x: x.strip().lower())

# ct_flw_http_mthd과 is_ftp_login 칼럼의 결측값 0으로 변경
train['ct_flw_http_mthd'] = train.ct_flw_http_mthd.fillna(value=0)
train['is_ftp_login'] = (train.is_ftp_login.fillna(value=0)).astype(int)

train['ct_ftp_cmd'] = train['ct_ftp_cmd'].replace(to_replace=' ', value=0).astype(int)
saved_dict['binary_col'] = ['is_sm_ips_ports', 'is_ftp_login']

# is_ftp_login 열의 값이 1보다 큰 경우 1로 변환하여 binray하게 정리
train['is_ftp_login'] = np.where(train['is_ftp_login']>1, 1, train['is_ftp_login'])

# service열의 "-" 값은 "None"으로 변경, attack_cat 열의 'backdoors'를 'backdoor'로 변환하여 범주형 데이터를 통일
train['service'] = train['service'].apply(lambda x:"None" if x=="-" else x)
train['attack_cat'] = train['attack_cat'].replace('backdoors','backdoor', regex=True).apply(lambda x: x.strip().lower())


# 정리된 학습 및 테스트 데이터를 CSV 파일로 저장
train.to_csv('AI_part/UNSW_NB15/train_alldata_EDA.csv', index=False)
test.to_csv('AI_part/UNSW_NB15/train_alldata_EDA.csv', index=False)

# dictionary pkl형식으로 저장
pickle.dump(saved_dict, open('AI_part/UNSW_NB15/save_dict.pkl', 'wb'))
'''

train = pd.read_csv('AI_part/UNSW_NB15/train_alldata_EDA.csv')
test = pd.read_csv('AI_part/UNSW_NB15/test_alldata_EDA.csv')

# Utility function
def multi_corr(col1, col2="label", df=train):
    '''
    두 개의 열 간 상관관계를 계산하고, 첫 번째 열에 로그 변환을 적용했을 때의 상관관계도 함께 계산하여 출력
    '''
    corr = df[[col1, col2]].corr().iloc[0,1]
    log_corr = df[col1].apply(np.log1p).corr(df[col2])

    print("Correlation : {}\nlog_Correlation: {}".format(corr, log_corr))

def corr(col1, col2="label", df=train):
    """
    두 열 간의 상관계수를 계산
    """
    return df[[col1, col2]].corr().iloc[0,1]

# train dataset의 covariance matrix 계산
corr_matrix = train.corr().abs()

# 상삼각 행렬 계산
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# 높은 상관관계를 가진 feature 계산
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
#saved_dict['corr_col'] = to_drop

# 상관계수가 높은 feature 제거
train.drop(columns=to_drop, inplace=True)

# 새로운 feature 계산
train['network_bytes'] = train['sbytes'] + train['dbytes']

# 다중 클래스 분류에 불필요한 열 제거
train.drop(['srcip', 'sport', 'dstip', 'dsport', 'attack_cat'], axis=1, inplace=True)
#saved_dict['to_drop'] = ['srcip', 'sport', 'dstip', 'dsport', 'attack_cat']

log1p_col = ['dur', 'sbytes', 'dbytes', 'sload', 'dload', 'spkts', 'stcpb', 
             'dtcpb', 'smeansz', 'dmeansz', 'sjit', 'djit', 'network_bytes']

#saved_dict['log1p_col'] = log1p_col

# train 데이터셋의 각 열에 대한 최빈값 저장
mode_dict = train.mode().iloc[0].to_dict()

# 열에 대해 log1p 함수를 적용하여 데이터의 스케일 조정 및 분포 정규화 작업 수행
def log1p_transform(col, df=train):

    new_col = col+'_log1p'
    df[new_col] = df[col].apply(np.log1p)
    df.drop(col, axis=1, inplace=True)

# 로그 변환 수행
for col in log1p_col:
    log1p_transform(col, df=train)

# x, y set 생성
x_train, y_train = train.drop(columns=['label']), train['label']
x_test, y_test = test.drop(columns=['label']), test['label']

# 나중에 사용하기 위해 저장
pickle.dump((x_train, y_train), open('.final_ipynb/final_train.pkl', 'wb'))
pickle.dump((x_test, y_test), open('.final_ipynb/final_test.pkl', 'wb'))

