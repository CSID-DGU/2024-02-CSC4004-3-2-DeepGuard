from flask import Flask, request, jsonify
import os

app = Flask(__name__)

    # 저장할 폴더 경로
UPLOAD_FOLDER = r"C:\Users\loveh\Desktop\analysis"


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_file():
    """클라이언트에서 파일 업로드를 처리"""
    if 'file' not in request.files:
        return jsonify({"status": "fail", "message": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "fail", "message": "No selected file"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    return jsonify({"status": "success", "message": f"File {file.filename} uploaded successfully"}), 200

# @app.route('/start_analysis', methods=['GET'])
# def start_analysis():
#     """파일 분석 요청을 처리"""
#     file_name = request.args.get('file')
#     if not file_name:
#         return jsonify({"status": "fail", "message": "No file specified"}), 400

#     file_path = os.path.join(UPLOAD_FOLDER, file_name)
#     if not os.path.exists(file_path):
#         return jsonify({"status": "fail", "message": "File not found"}), 404

#     # 분석 로직 (여기에 추가 가능)
#     print(f"Starting analysis for {file_path}...")
#     import time
#     time.sleep(10)  # 분석 시뮬레이션

#     return jsonify({"status": "success", "message": f"Analysis for {file_name} completed"}), 200



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
