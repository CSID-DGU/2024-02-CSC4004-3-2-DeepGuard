import os
import time
import paramiko
import requests
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 설정 정보
UBUNTU_IP = "192.168.135.130"
UBUNTU_USERNAME = "hyunseok"
UBUNTU_PASSWORD = "0130"
UBUNTU_FOLDER = "/home/hyunseok/Desktop/sw_project/server"
WATCH_FOLDER = "다운로드"

# 파일 전송 및 분석 요청 함수
def send_file_and_request_analysis(file_path):
    file_name = os.path.basename(file_path)

    # 파일 전송 (SFTP 사용)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(UBUNTU_IP, username=UBUNTU_USERNAME, password=UBUNTU_PASSWORD)
    sftp = ssh.open_sftp()
    sftp.put(file_path, os.path.join(UBUNTU_FOLDER, file_name))
    sftp.close()
    ssh.close()
    print(f"Transferred {file_name} to Ubuntu for analysis.")

    # HTTP 요청을 보내서 Ubuntu에서 분석 시작
    try:
        response = requests.get(f"http://{UBUNTU_IP}:5000/start_analysis?file={file_name}")
        if response.status_code == 200:
            print("Analysis started on Ubuntu.")
        else:
            print("Failed to start analysis.")
    except requests.RequestException as e:
        print(f"Error sending analysis request: {e}")

# 새로운 파일이 생성되었을 때 처리
class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            print(f"New file detected: {event.src_path}")
            send_file_and_request_analysis(event.src_path)

# 폴더 모니터링 설정
observer = Observer()
event_handler = NewFileHandler()
observer.schedule(event_handler, WATCH_FOLDER, recursive=False)
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()
