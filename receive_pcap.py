from flask import Flask, request, jsonify
import os

app = Flask(__name__)

UPLOAD_FOLDER = "/home/hyunseok/Desktop/pcap"  # 저장 디렉토리
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    """파일 업로드 엔드포인트"""
    if 'file' not in request.files:
        return jsonify({"status": "fail", "message": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "fail", "message": "No selected file"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    return jsonify({"status": "success", "message": f"File {file.filename} uploaded successfully"}), 200

if __name__ == "__main__":
    app.run(host='192.168.56.101', port=5000)
