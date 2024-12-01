import os
import time
import psutil
import requests
import subprocess

# 감시할 폴더 경로
WATCHED_FOLDER = r"C:\Users\공개SW 클라이언트\Desktop"

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

    # # HTTP 분석 요청
    # try:
    #     response = requests.get(ANALYSIS_URL, params={'file': file_name})
    #     if response.status_code == 200:
    #         print(f"Analysis started successfully: {response.json()}")
    #     else:
    #         print(f"Failed to start analysis: {response.json()}")
    # except requests.RequestException as e:
    #     print(f"Error sending analysis request: {e}")

def terminate_process(pid, process_name):
    try:
        # 프로세스 종료
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/F"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        print(f"프로세스 {process_name} (PID {pid})가 성공적으로 종료되었습니다.")
    except subprocess.CalledProcessError as e:
        print(f"프로세스 {process_name} (PID {pid}) 종료 실패: {e.stderr}")

def monitor_processes():
    print(f"'{WATCHED_FOLDER}' 폴더에서 실행되는 파일 감시를 시작합니다.")
    while True:
        try:
            # 실행 중인 모든 프로세스 검사
            for proc in psutil.process_iter(['pid', 'name', 'exe']):
                try:
                    process_path = proc.info['exe']
                    process_name = proc.info['name']
                    process_id = proc.info['pid']

                    # 실행 파일 경로가 None이거나 다른 드라이브에 있는 경우 무시
                    if process_path and os.path.splitdrive(WATCHED_FOLDER)[0] == os.path.splitdrive(process_path)[0]:
                        # 프로세스 경로가 감시 폴더에 속하는지 확인
                        if os.path.commonpath([WATCHED_FOLDER, process_path]) == WATCHED_FOLDER:
                            print(f"감지된 프로세스: {process_name}, 경로: {process_path}, PID: {process_id}")
                            
                            # 프로세스 종료
                            terminate_process(process_id, process_name)
                            return process_path  # 실행 파일 경로 반환
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    continue
            time.sleep(1)  # 1초 간격으로 체크
        except KeyboardInterrupt:
            print("프로세스 감시가 종료되었습니다.")
            return None

if __name__ == "__main__":
    while True:
        # 감지된 파일 경로를 가져옴
        detected_file_path = monitor_processes()
        if detected_file_path:
            send_file_and_request_analysis(detected_file_path)
        else:
            break
