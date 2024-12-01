import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import random  # 테스트용 랜덤 상태값 생성

# 실시간 탐지 상태를 위한 전역 변수
is_detection = False

# 상태에 따른 UI 업데이트
def update_ui(status_code=None):
    if not is_detection:
        header_frame.configure(bg="#9e9e9e")  # 회색
        header_label.configure(bg="#9e9e9e", fg="white")
        status_label.configure(text="탐지 대기 중", bg="#9e9e9e", fg="white")
        icon_label.configure(image="", text="🔍", fg="#9e9e9e")  # 진한 회색 이모지
    else:
        # 탐지 ON 상태
        if status_code == 0:
            # 안전한 파일
            header_frame.configure(bg="#4caf50")  # 초록색
            header_label.configure(bg="#4caf50", fg="white")
            status_label.configure(text="안전한 상태입니다", bg="#4caf50", fg="white")
            icon_label.configure(image="", text="✅", fg="#4caf50")  # 초록색 이모지
        elif status_code == 1:
            # 랜섬웨어 탐지
            header_frame.configure(bg="#f44336")  # 빨간색
            header_label.configure(bg="#f44336", fg="white")
            status_label.configure(text="랜섬웨어 파일이 탐지되었습니다!", bg="#f44336", fg="white")
            icon_label.configure(image="", text="⚠️", fg="#f44336")  # 빨간색 이모지
        else:
            # 실시간 탐지 중
            header_frame.configure(bg="#2196f3")  # 파란색
            header_label.configure(bg="#2196f3", fg="white")
            status_label.configure(text="실시간 탐지 중", bg="#2196f3", fg="white")
            icon_label.configure(image="", text="⏳", fg="#2196f3")  # 파란색 이모지

# 검사 버튼 클릭
def toggle_detection():
    global is_detection
    is_detection = not is_detection  # 탐지 상태 전환

    if is_detection:
        check_button.configure(text="실시간 탐지 OFF")
        # 랜덤으로 Status Code 받아오기 (테스트용)
        status_code = random.choice([0, 1, None])  # 실제론 서버 또는 데이터 소스에서 받아옴
        update_ui(status_code)
        # Status Code == 1일 때 추가 팝업 처리
        if status_code == 1:
            user_response = messagebox.askyesno("위험 경고", "랜섬웨어가 탐지되었습니다. 제거하시겠습니까?")
            if user_response:
                messagebox.showinfo("결과", "랜섬웨어가 제거되었습니다.")
            else:
                messagebox.showinfo("결과", "랜섬웨어 제거가 취소되었습니다.")
            update_ui(None)  # 다시 탐지 중 상태로 복귀
        elif status_code == 0:
            user_response = messagebox.askyesno("안전 파일", "안전한 파일입니다. 다운로드를 계속 하시겠습니까?")
            if user_response:
                messagebox.showinfo("결과", "파일 다운로드를 계속합니다.")
            else:
                messagebox.showinfo("결과", "파일 다운로드를 취소합니다.")
            update_ui(None)  # 다시 탐지 중 상태로 복귀
    else:
        check_button.configure(text="실시간 탐지 ON")
        update_ui()  # 탐지 OFF 상태로 전환

# 프로그램 종료
def on_exit():
    if messagebox.askokcancel("종료", "프로그램을 종료하시겠습니까?"):
        window.destroy()

# 메인 창 설정
window = tk.Tk()
window.title("DeepGuard")
window.geometry("800x600")
window.resizable(False, False)
window.configure(bg="#f0f0f0")

# 스타일 설정
style = ttk.Style()
style.configure("TButton", font=("Arial", 12), padding=10)
style.configure("TLabel", font=("Arial", 14), background="#f0f0f0")

# 헤더 섹션
header_frame = tk.Frame(window, bg="#9e9e9e", height=100)  # 기본 회색
header_frame.pack(fill="x")

header_label = tk.Label(header_frame, text="DeepGuard", font=("Arial", 20, "bold"), bg="#9e9e9e", fg="white")
header_label.pack(side="left", padx=20, pady=20)

status_label = tk.Label(header_frame, text="탐지 대기 중", font=("Arial", 16), bg="#9e9e9e", fg="white")
status_label.pack(side="right", padx=20, pady=20)

# 메인 섹션
main_frame = tk.Frame(window, bg="#f0f0f0", pady=20)
main_frame.pack(fill="both", expand=True)

# 초기 아이콘 상태
icon_label = tk.Label(main_frame, text="🔍", font=("Arial", 50), bg="#f0f0f0", fg="white")
icon_label.place(relx=0.5, rely=0.4, anchor="center")  # 중앙 정렬

# 검사 버튼
check_button = ttk.Button(main_frame, text="실시간 탐지 ON", command=toggle_detection)
check_button.place(relx=0.5, rely=0.6, anchor="center")  # 중앙 정렬

# 이벤트 루프 시작
window.protocol("WM_DELETE_WINDOW", on_exit)
window.mainloop()
