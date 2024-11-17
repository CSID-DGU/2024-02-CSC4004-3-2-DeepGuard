import subprocess

# VMware Workstation 설정
VMRUN_PATH = r"C:\Program Files (x86)\VMware\VMware Workstation\vmrun.exe"
VMX_PATH = r"C:\Users\loveh\Documents\Virtual Machines\Server(window)\Server(window).vmx"
SNAPSHOT_NAME = "Clean_state"

# 가상머신 상태 확인
def is_vm_running():
    try:
        result = subprocess.run([VMRUN_PATH, "list"], capture_output=True, text=True, check=True)
        return VMX_PATH in result.stdout
    except Exception as e:
        print(f"[ERROR] Failed to check VM status: {e}")
        return False

# 가상머신 종료
def stop_vm():
    try:
        print(f"[INFO] Stopping VM '{VMX_PATH}'...")
        subprocess.run([VMRUN_PATH, "stop", VMX_PATH], check=True)
        print("[INFO] VM stopped successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to stop VM: {e.stderr}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

# 스냅샷 복구
def restore_snapshot():
    try:
        print(f"[INFO] Restoring snapshot '{SNAPSHOT_NAME}' for VM '{VMX_PATH}'...")
        subprocess.run([VMRUN_PATH, "revertToSnapshot", VMX_PATH, SNAPSHOT_NAME], check=True)
        print("[INFO] Snapshot restored successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to restore snapshot: {e.stderr}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

# 메인 워크플로우
def main():
    if is_vm_running():
        print("[INFO] VM is currently running. Stopping it before restoring snapshot...")
        stop_vm()
    restore_snapshot()

if __name__ == "__main__":
    main()
