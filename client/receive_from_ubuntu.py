from flask import Flask, request, jsonify
import os

app = Flask(__name__)

UPLOAD_FOLDER = "C:\\Users\\공개SW 클라이언트\\Desktop\\공소"  # 업로드된 파일을 저장할 폴더 경로
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
    # 서버 시작
    app.run(host='0.0.0.0', port=5000)
