import requests

# Flask 서버 URL
UPLOAD_URL = "http://192.168.56.101:5000/upload"  # 우분투 서버의 업로드 URL
LOCAL_FILE_PATH = r"C:\Users\loveh\Desktop\pcap\ransomware_traffic.pcap"  # 전송할 .pcap 파일 경로

def upload_file():
    """HTTP 방식으로 파일을 업로드"""
    try:
        with open(LOCAL_FILE_PATH, 'rb') as f:
            files = {'file': f}
            response = requests.post(UPLOAD_URL, files=files)
        if response.status_code == 200:
            print(f"파일 업로드 성공: {response.json()}")
        else:
            print(f"파일 업로드 실패: {response.json()}")
    except Exception as e:
        print(f"파일 업로드 중 오류 발생: {e}")

if __name__ == "__main__":
    upload_file()
