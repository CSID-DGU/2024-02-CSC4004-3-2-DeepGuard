import pyshark

def list_available_interfaces():
    print("가능한 네트워크 인터페이스:")
    interfaces = pyshark.tshark.tshark.get_tshark_interfaces()
    for interface in interfaces:
        print(interface)

if __name__ == "__main__":
    list_available_interfaces()
