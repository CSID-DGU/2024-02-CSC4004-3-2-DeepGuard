import os
import time
import subprocess
from pyshark import LiveCapture

# 랜섬웨어 샘플 실행 경로
RANSOMWARE_PATH = r"C:\Users\loveh\Desktop\analysis\test_exe.exe"

# pcap 파일 저장 경로
PCAP_SAVE_PATH = r"C:\Users\loveh\Desktop\pcap\ransomware_traffic.pcap"

UPLOAD_URL = "http://192.168.56.101:5000/upload"  # 우분투 서버의 업로드 URL
LOCAL_FILE_PATH = r"C:\Users\loveh\Desktop\pcap\ransomware_traffic.pcap"  # 전송할 .pcap 파일 경로

# 네트워크 인터페이스 이름 (Wireshark에서 확인 가능)
NETWORK_INTERFACE = "Ethernet1"  # 윈도우 환경의 기본 인터페이스 이름 (Wi-Fi, Ethernet 등)

def capture_traffic(duration, save_path, interface):
    """
    네트워크 트래픽을 캡처하여 .pcap 파일로 저장
    :param duration: 캡처 지속 시간 (초)
    :param save_path: 저장할 .pcap 파일 경로
    :param interface: 캡처할 네트워크 인터페이스 이름
    """
    print(f"네트워크 트래픽 캡처 시작: {save_path}")
    try:
        # PyShark 라이브 캡처 설정
        capture = LiveCapture(interface=interface, output_file=save_path)
        capture.sniff(timeout=duration)  # 지정된 시간 동안 캡처
        print(f"트래픽 캡처 완료: {save_path}")
    except Exception as e:
        print(f"네트워크 트래픽 캡처 중 오류 발생: {e}")

def execute_ransomware(file_path):
    """
    랜섬웨어 샘플 실행
    :param file_path: 실행할 랜섬웨어 샘플의 경로
    """
    try:
        print(f"랜섬웨어 실행: {file_path}")
        # 랜섬웨어 실행
        subprocess.Popen([file_path], shell=True)
    except Exception as e:
        print(f"랜섬웨어 실행 중 오류 발생: {e}")

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
    # 캡처 시간 설정 (초 단위)
    CAPTURE_DURATION = 60  # 1분 동안 트래픽 캡처

    # Step 1: 랜섬웨어 실행
    execute_ransomware(RANSOMWARE_PATH)

    # Step 2: 트래픽 캡처 시작
    time.sleep(5)  # 랜섬웨어 실행 후 캡처 시작 지연
    capture_traffic(CAPTURE_DURATION, PCAP_SAVE_PATH, NETWORK_INTERFACE)

    upload_file()