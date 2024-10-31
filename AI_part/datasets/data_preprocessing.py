import numpy as np  
import pandas as pd  
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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
test.to_csv('AI_part/UNSW_NB15/test_alldata_EDA.csv', index=False)

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
train_numeric = train.apply(pd.to_numeric, errors='coerce')
corr_matrix = train_numeric.corr().abs()

# 상삼각 행렬 계산
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# 높은 상관관계를 가진 feature 계산
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
saved_dict['corr_col'] = to_drop

# 상관계수가 높은 feature 제거
train.drop(columns=to_drop, inplace=True)

# 새로운 feature 계산
train['network_bytes'] = train['sbytes'] + train['dbytes']

# 다중 클래스 분류에 불필요한 열 제거
train.drop(['srcip', 'sport', 'dstip', 'dsport', 'attack_cat'], axis=1, inplace=True)
saved_dict['to_drop'] = ['srcip', 'sport', 'dstip', 'dsport', 'attack_cat']

col_unique_values = train.nunique()
col = col_unique_values[col_unique_values>200].index

log1p_col = ['dur', 'sbytes', 'dbytes', 'sload', 'dload', 'spkts', 'stcpb', 
             'dtcpb', 'smeansz', 'dmeansz', 'sjit', 'djit', 'network_bytes']

saved_dict['log1p_col'] = log1p_col

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
pickle.dump((x_train, y_train), open('AI_part/UNSW_NB15/final_train.pkl', 'wb'))
pickle.dump((x_test, y_test), open('AI_part/UNSW_NB15/final_test.pkl', 'wb'))

# categorical feature와 numerical 분리
cat_col = ['proto', 'service', 'state']
num_col = list(set(x_train.columns) - set(cat_col))

saved_dict['cat_col'] = cat_col
saved_dict['num_col'] = num_col

# standardizing numerical data
scaler = StandardScaler()
scaler = scaler.fit(x_train[num_col])

x_train[num_col] = scaler.transform(x_train[num_col])

# onehot encoding cagtegorical data
service_ = OneHotEncoder(handle_unknown='ignore')
proto_ = OneHotEncoder(handle_unknown='ignore')
state_ = OneHotEncoder(handle_unknown='ignore')
ohe_service = service_.fit(x_train.service.values.reshape(-1,1))
ohe_proto = proto_.fit(x_train.proto.values.reshape(-1,1))
ohe_state = state_.fit(x_train.state.values.reshape(-1,1))

# replace the original categorical column to onehotencoded columns
for col, ohe in zip(['proto', 'service', 'state'], [ohe_proto, ohe_service, ohe_state]):
    x = ohe.transform(x_train[col].values.reshape(-1,1))
    tmp_df = pd.DataFrame(x.todense(), columns=[str(col) + '_' + str(i) for i in ohe.categories_[0]])
    x_train = pd.concat([x_train.drop(col, axis=1), tmp_df], axis=1)
    
file_path = 'AI_part/UNSW_NB15/'

pickle.dump(scaler, open(file_path+'scaler.pkl', 'wb'))  # Standard scaler
pickle.dump(saved_dict, open(file_path+'saved_dict.pkl', 'wb'))  # Dictionary with important parameters
pickle.dump(mode_dict, open(file_path+'mode_dict.pkl', 'wb')) # 최빈값

# Onehot encoder for categorical columns
pickle.dump(ohe_proto, open(file_path+'ohe_proto.pkl', 'wb'))
pickle.dump(ohe_service, open(file_path+'ohe_service.pkl', 'wb'))
pickle.dump(ohe_state, open(file_path+'ohe_state.pkl', 'wb'))

# Cleaned and processed train data
pickle.dump((x_train, y_train), open(file_path+'final_train.pkl', 'wb'))

# pipeline functions for test data
def clean_data(data):
    '''
    Cleans given raw data. Performs various cleaning, removes Null and wrong values.
    Check for columns datatype and fix them.
    '''
    numerical_col = data.select_dtypes(include=np.number).columns  # All the numerical columns list
    categorical_col = data.select_dtypes(exclude=np.number).columns  # All the categorical columns list
    
    # Cleaning the data
    for col in data.columns:
        val = mode_dict[col]  # Mode value of the column in train data
        data[col] = data[col].fillna(value=val)
        data[col] = data[col].replace(' ', value=val)
        data[col] = data[col].apply(lambda x:"None" if x=="-" else x)

        # Fixing binary columns
        if col in saved_dict['binary_col']:
            data[col] = np.where(data[col]>1, val, data[col])

    # Fixing datatype of columns
    bad_dtypes = list(set(categorical_col) - set(saved_dict['cat_col']))
    for bad_col in bad_dtypes:
        data[bad_col] = data[bad_col].astype(float)
    
    return data


def apply_log1p(data):
    '''
    Performs FE on the data. Apply log1p on the specified columns create new column and remove those original columns.
    '''
    for col in saved_dict['log1p_col']:
        new_col = col + '_log1p'  # New col name
        data[new_col] = data[col].apply(np.log1p)  # Creating new column on transformed data
        data.drop(col, axis=1, inplace=True)  # Removing old columns
    return data

def standardize(data):
    '''
    Stanardize the given data. Performs mean centering and varience scaling.
    Using stanardscaler object trained on train data.
    '''
    data[saved_dict['num_col']] = scaler.transform(data[saved_dict['num_col']])
    
    return data

# Parametrs
saved_dict = pickle.load(open(file_path+'saved_dict.pkl', 'rb'))
# Mode value of all the columns
mode_dict = pickle.load(open(file_path+'mode_dict.pkl', 'rb'))
# Stanardscaler object
scaler = pickle.load(open(file_path+'scaler.pkl', 'rb'))

# One hot encoder objects
ohe_proto = pickle.load(open(file_path+'ohe_proto.pkl', 'rb'))
ohe_service = pickle.load(open(file_path+'ohe_service.pkl', 'rb'))
ohe_state = pickle.load(open(file_path+'ohe_state.pkl', 'rb'))


# Resetting index of test data
x_test.reset_index(drop=True, inplace=True)

# Adding column names
x_test.columns = saved_dict['columns']

# Creating new Feature
x_test['network_bytes'] = x_test['dbytes'] + x_test['sbytes']

# Droping all the unwanted columns
dropable_col = saved_dict['to_drop'] + saved_dict['corr_col']
x_test.drop(columns=dropable_col, inplace=True)

# Cleaning data using clean_data()
x_test = clean_data(x_test)

# FE: applying log1p using apply_log1p()
x_test = apply_log1p(x_test)
print(x_test.shape)

# Standardscaling using stanardize()
x_test = standardize(x_test)
print(x_test.head())

# Onehot encoding categorical columns using ohencoding()
for col, ohe in zip(['proto', 'service', 'state'], [ohe_proto, ohe_service, ohe_state]):
    x = ohe.transform(x_test[col].values.reshape(-1,1))
    tmp_df = pd.DataFrame(x.todense(), columns=[str(col) + '_' + str(i) for i in ohe.categories_[0]])
    x_test = pd.concat([x_test.drop(col, axis=1), tmp_df], axis=1)
    
print(x_test.head())
print(all(x_train.columns == x_test.columns))

# Cleaned and processed test data
pickle.dump((x_test, y_test), open(file_path+'final_test.pkl', 'wb'))
