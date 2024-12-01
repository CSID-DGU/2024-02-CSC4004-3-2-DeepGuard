import subprocess
import pandas as pd
import os
from receive_pcap import *

def run_zeek(pcap_path, zeek_output_dir):
    if not os.path.exists(zeek_output_dir):
        os.makedirs(zeek_output_dir)
    command = f"zeek -r {pcap_path}"
    subprocess.run(command, shell=True, cwd=zeek_output_dir)
    print("Zeek analysis completed.")

def run_argus(pcap_path, argus_output_file):
    command = f"argus -r {pcap_path} -w {argus_output_file}"
    subprocess.run(command, shell=True)
    print("Argus analysis completed.")

    summary_csv = argus_output_file.replace(".argus", "_summary.csv")
    command_summary = f"ra -r {argus_output_file} -n -s stime dur proto sbytes dbytes spkts dpkts > {summary_csv}"
    subprocess.run(command_summary, shell=True)

    # Argus 데이터를 Pandas로 변환 후 필드 수정
    argus_data = pd.read_csv(summary_csv, delim_whitespace=True)  # 공백으로 구분된 데이터 읽기
    if "Proto" in argus_data.columns:
        argus_data.rename(columns={"Proto": "proto"}, inplace=True)  # 열 이름 변경
    argus_data.to_csv(summary_csv, index=False)
    print(f"Processed Argus summary saved to {summary_csv}")
    return summary_csv

def process_zeek_logs(zeek_output_dir, output_csv):
    conn_log = os.path.join(zeek_output_dir, "conn.log")
    if os.path.exists(conn_log):
        # 파일을 열고 메타데이터 제거
        with open(conn_log, 'r') as f:
            lines = f.readlines()
        # 필드 이름 추출
        fields_line = [line for line in lines if line.startswith("#fields")]
        if fields_line:
            fields = fields_line[0].strip().split("\t")[1:]  # 필드 이름
            if "proto" not in fields:
                fields.append("proto")  # 기본값 추가
            # 데이터 필터링 (메타데이터 제거)
            data_lines = [line for line in lines if not line.startswith("#")]
            # Pandas로 변환
            import io
            data = pd.read_csv(io.StringIO("\n".join(data_lines)), delimiter="\t", names=fields)
            data.to_csv(output_csv, index=False)
            print(f"Zeek connection log saved to {output_csv}")
        else:
            print("No #fields line found in conn.log")
    else:
        print("conn.log not found.")

def merge_and_calculate_features(zeek_csv, argus_csv, final_output_csv):
    zeek_data = pd.read_csv(zeek_csv)
    argus_data = pd.read_csv(argus_csv)

    print("Zeek Data Columns:", zeek_data.columns)
    print("Argus Data Columns:", argus_data.columns)

    merged_data = pd.merge(zeek_data, argus_data, on="proto", how="outer")

    merged_data["is_sm_ips_ports"] = (
        (merged_data["id.orig_h"] == merged_data["id.resp_h"]) &
        (merged_data["id.orig_p"] == merged_data["id.resp_p"])
    ).astype(int)

    merged_data.to_csv(final_output_csv, index=False)
    print(f"Final dataset saved to {final_output_csv}")

def main():
    pcap_path = "/home/hyunseok/Desktop/pcap_to_feature/ransomware_traffic.pcap"
    zeek_output_dir = "./zeek"
    argus_output_file = "./argus_output.argus"
    zeek_csv = "./zeek_conn.csv"
    argus_csv = "./argus_output_summary.csv"
    final_output_csv = "./final_dataset.csv"

    run_zeek(pcap_path, zeek_output_dir)
    process_zeek_logs(zeek_output_dir, zeek_csv)
    summary_csv = run_argus(pcap_path, argus_output_file)

    merge_and_calculate_features(zeek_csv, summary_csv, final_output_csv)

if __name__ == "__main__":
    main()
