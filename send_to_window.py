import requests

def upload_file_to_windows(file_path, windows_server_url):

    with open(file_path, 'rb') as file:
        files = {'file': file}
        response = requests.post(windows_server_url, files=files)

    if response.status_code == 200:
        print("File successfully uploaded:", response.json())
    else:
        print("Failed to upload file:", response.json())

# 예제 호출
upload_file_to_windows(
    file_path="./final_dataset.csv",  # 전송할 파일 경로
    windows_server_url="http://192.168.135.128:5000/upload"  # Flask 서버 URL
)
