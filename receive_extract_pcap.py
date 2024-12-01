from flask import Flask, request, jsonify
import subprocess
import pandas as pd
import os

app = Flask(__name__)

UPLOAD_FOLDER = "/home/hyunseok/Desktop/pcap_to_feature"  # Directory to save uploaded PCAP files
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Zeek execution
def run_zeek(pcap_path, zeek_output_dir):
    if not os.path.exists(zeek_output_dir):
        os.makedirs(zeek_output_dir)
    command = f"zeek -r {pcap_path}"
    subprocess.run(command, shell=True, cwd=zeek_output_dir)
    print("Zeek analysis completed.")

# Argus execution
def run_argus(pcap_path, argus_output_file):
    command = f"argus -r {pcap_path} -w {argus_output_file}"
    subprocess.run(command, shell=True)
    print("Argus analysis completed.")

    summary_csv = argus_output_file.replace(".argus", "_summary.csv")
    command_summary = f"ra -r {argus_output_file} -n -s stime dur proto sbytes dbytes spkts dpkts > {summary_csv}"
    subprocess.run(command_summary, shell=True)

    # Modify Argus summary for consistency
    argus_data = pd.read_csv(summary_csv, delim_whitespace=True)  # Parse space-separated data
    if "Proto" in argus_data.columns:
        argus_data.rename(columns={"Proto": "proto"}, inplace=True)  # Rename Proto to proto
    argus_data.to_csv(summary_csv, index=False)
    print(f"Processed Argus summary saved to {summary_csv}")
    return summary_csv

# Zeek log processing
def process_zeek_logs(zeek_output_dir, output_csv):
    conn_log = os.path.join(zeek_output_dir, "conn.log")
    if os.path.exists(conn_log):
        with open(conn_log, 'r') as f:
            lines = f.readlines()
        fields_line = [line for line in lines if line.startswith("#fields")]
        if fields_line:
            fields = fields_line[0].strip().split("\t")[1:]  # Extract field names
            if "proto" not in fields:
                fields.append("proto")  # Add proto if missing
            data_lines = [line for line in lines if not line.startswith("#")]
            import io
            data = pd.read_csv(io.StringIO("\n".join(data_lines)), delimiter="\t", names=fields)
            data.to_csv(output_csv, index=False)
            print(f"Zeek connection log saved to {output_csv}")
        else:
            print("No #fields line found in conn.log")
    else:
        print("conn.log not found.")

# Merge and calculate additional features
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

# Flask route to handle file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"status": "fail", "message": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "fail", "message": "No selected file"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    print(f"File {file.filename} uploaded successfully.")

    # Process the uploaded PCAP file
    zeek_output_dir = "./zeek"
    argus_output_file = "./argus_output.argus"
    zeek_csv = "./zeek_conn.csv"
    argus_csv = "./argus_output_summary.csv"
    final_output_csv = "./final_dataset.csv"

    run_zeek(file_path, zeek_output_dir)
    process_zeek_logs(zeek_output_dir, zeek_csv)
    summary_csv = run_argus(file_path, argus_output_file)
    merge_and_calculate_features(zeek_csv, summary_csv, final_output_csv)

    return jsonify({
        "status": "success",
        "message": f"File {file.filename} processed successfully.",
        "output": final_output_csv
    }), 200

if __name__ == "__main__":
    app.run(host='192.168.56.101', port=5000)
