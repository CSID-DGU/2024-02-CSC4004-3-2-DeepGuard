import tkinter as tk
import json
import os
from threading import Thread
from file_execution import monitor_processes, send_file_and_request_analysis  # 파일 실행 모니터링 및 전송 함수

# 스위치 상태를 저장할 파일 경로
SETTINGS_FILE = "switch_settings.json"

class SwitchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("보안 스위치")
        self.root.geometry("300x300")
        self.root.configure(bg="#1e1e2f")  # 배경 색상

        # 스위치 상태 로드
        self.status = self.load_status()

        # 모니터링 상태를 위한 변수 초기화
        self.monitor_thread = None
        self.monitoring_active = False

        # 스위치 캔버스 생성
        self.canvas = tk.Canvas(self.root, width=200, height=200, bg="#1e1e2f", highlightthickness=0)
        self.canvas.pack(pady=50)

        # 원형 스위치 배경
        self.switch_bg = self.canvas.create_oval(25, 25, 175, 175, fill="#2b2b3d", outline="#6c63ff", width=3)

        # 전원 버튼 아이콘 (중앙 선과 곡선)
        self.power_line = self.canvas.create_line(100, 60, 100, 100, fill="#6c63ff", width=3)
        self.power_arc = self.canvas.create_arc(70, 70, 130, 130, start=0, extent=300, style="arc", outline="#6c63ff", width=3)

        # 상태 표시 (ON/OFF 텍스트)
        self.status_text = self.canvas.create_text(100, 140, text=self.get_status_text(), fill="white", font=("Arial", 16))

        # 클릭 이벤트 바인딩
        self.canvas.tag_bind(self.switch_bg, "<Button-1>", self.toggle_switch)
        self.canvas.tag_bind(self.power_line, "<Button-1>", self.toggle_switch)
        self.canvas.tag_bind(self.power_arc, "<Button-1>", self.toggle_switch)
        self.canvas.tag_bind(self.status_text, "<Button-1>", self.toggle_switch)

        # 현재 상태에 따라 UI 업데이트
        self.update_ui()
        self.apply_security()

    def toggle_switch(self, event=None):
        """스위치 상태 변경"""
        self.status = "off" if self.status == "on" else "on"
        self.save_status()
        self.update_ui()
        self.apply_security()

    def update_ui(self):
        """스위치 UI 업데이트"""
        if self.status == "on":
            self.canvas.itemconfig(self.switch_bg, fill="#6c63ff")
        else:
            self.canvas.itemconfig(self.switch_bg, fill="#2b2b3d")
        self.canvas.itemconfig(self.status_text, text=self.get_status_text())

    def get_status_text(self):
        """ON/OFF 상태 텍스트 반환"""
        return "ON" if self.status == "on" else "OFF"

    def apply_security(self):
        """Switch ON/OFF에 따라 모니터링 활성화 또는 비활성화"""
        if self.status == "on":
            self.start_monitoring()
        else:
            self.stop_monitoring()

    def start_monitoring(self):
        """파일 실행 모니터링 시작"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = Thread(target=self.run_file_monitoring, daemon=True)
            self.monitor_thread.start()
            print("파일 실행 모니터링 및 전송이 활성화되었습니다.")

    def stop_monitoring(self):
        """파일 실행 모니터링 중단"""
        self.monitoring_active = False
        print("파일 실행 모니터링이 비활성화되었습니다.")

    def run_file_monitoring(self):
        """file_execution.py의 프로세스 실행 및 전송"""
        try:
            while self.monitoring_active:
                # 파일 모니터링 실행
                detected_file = monitor_processes()  # 감지된 파일 경로 반환
                if detected_file:
                    send_file_and_request_analysis(detected_file)  # 파일 전송 및 요청
        except Exception as e:
            print(f"모니터링 중 오류 발생: {e}")

    def load_status(self):
        """저장된 스위치 상태 로드"""
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as file:
                data = json.load(file)
                return data.get("status", "off")
        return "off"

    def save_status(self):
        """스위치 상태 저장"""
        with open(SETTINGS_FILE, 'w') as file:
            json.dump({"status": self.status}, file)

if __name__ == "__main__":
    root = tk.Tk()
    app = SwitchApp(root)
    root.mainloop()
