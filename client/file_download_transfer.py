import os
import time
import paramiko
import requests

WATCH_FOLDER = r"C:\Users\공개SW 클라이언트\Downloads"  # 감시할 폴더 (Windows)

# 임시 확장자 목록 (다운로드 중인 파일 필터링)
TEMP_EXTENSIONS = {".tmp", ".crdownload", ".partial"}

# 이전에 감지된 파일 목록을 초기화
previous_files = set(os.listdir(WATCH_FOLDER))

def wait_until_file_is_stable(file_path, timeout=10, interval=0.5):
    """파일이 일정 시간 동안 크기가 변하지 않으면 True 반환"""
    start_time = time.time()
    last_size = -1
    while time.time() - start_time < timeout:
        current_size = os.path.getsize(file_path)
        if current_size == last_size:
            return True  # 파일이 안정화되었음
        last_size = current_size
        time.sleep(interval)
    return False

# Windows 서버 설정 정보
WINDOWS_IP = "192.168.135.131"  # Windows 서버 IP 주소
UPLOAD_URL = f"http://{WINDOWS_IP}:5000/upload"  # 파일 업로드 URL
# ANALYSIS_URL = f"http://{WINDOWS_IP}:5000/start_analysis"  # 분석 시작 URL

def send_file_and_request_analysis(file_path):
    """파일을 Windows 서버로 전송하고 분석 요청을 보냄"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    # 파일 이름만 추출
    file_name = os.path.basename(file_path)

    # HTTP 파일 업로드
    try:
        with open(file_path, 'rb') as f:
            response = requests.post(UPLOAD_URL, files={'file': f})
        if response.status_code == 200:
            print(f"File {file_name} uploaded successfully: {response.json()}")
        else:
            print(f"Failed to upload file: {response.json()}")
            return
    except requests.RequestException as e:
        print(f"Error uploading file: {e}")
        return

# 주기적으로 폴더를 스캔하여 새 파일을 감지
while True:
    # 현재 폴더의 파일 목록을 가져옴
    current_files = set(os.listdir(WATCH_FOLDER))
    # 새로 생성된 파일 탐색
    new_files = current_files - previous_files
    
    for file_name in new_files:
        file_path = os.path.join(WATCH_FOLDER, file_name)
        
        # 파일이 임시 파일인지 확인
        file_extension = os.path.splitext(file_name)[1].lower()
        if file_extension in TEMP_EXTENSIONS:
            print(f"Ignoring temporary file: {file_path}")
            continue
        
        # 파일이 안정화된 상태인지 확인 후 전송
        print(f"New file detected: {file_path}")
        if wait_until_file_is_stable(file_path):
            send_file_and_request_analysis(file_path)
        else:
            print(f"File not stable or download incomplete: {file_path}")

    # 이전 파일 목록 업데이트
    previous_files = current_files
    # 1초마다 폴더를 다시 스캔
    time.sleep(1)
