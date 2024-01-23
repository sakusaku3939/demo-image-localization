from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
import time
from werkzeug.utils import secure_filename

from models.YoloLSTM import YoloLSTM
from datasets.dataset_utils import load_cropped_image

token = "62BA9128-BC63-4865-9F92-B332BB4D682C"
tile_width = 0.45  # 1タイルの長さ (m)

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return "Hello World!"


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.headers.get("AccessToken") != token:
        print("error 400: Access denied")
        return jsonify({"error": "Access denied"}), 400

    if 'file' not in request.files:
        print("error 400: No file part")
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        print("error 400: No selected file")
        return jsonify({"error": "No selected file"}), 400

    if file:
        file_path = os.path.join('uploads', secure_filename(file.filename))
        file.save(file_path)
        result = predict_from_path(file_path)
        return jsonify(result, 200)


# GPUデバイスの設定
def init_device():
    torch.multiprocessing.freeze_support()
    device = torch.device("cuda")
    print(f"Device type: {device}")
    return device


def predict_from_path(file_path):
    torch.multiprocessing.freeze_support()
    batch_size = 1
    num_workers = 2
    device = init_device()

    start_time = time.time()

    model = YoloLSTM().to(device)
    model = model.eval()
    model.load_state_dict(torch.load("models/YoloLSTM-Bi/best_model.pth"))

    test_loader = load_cropped_image(file_path, batch_size, num_workers)

    with torch.no_grad():
        model = model.eval()
        pred = None

        for i, data in enumerate(test_loader):
            inputs = [d.to(device) for d in data[0]]

            pred = model(inputs)
            print(pred)
            print(f"\nx: {pred.tolist()[0][0]}, y: {pred.tolist()[0][1]}")

    end_time = time.time()
    if pred is not None:
        x, y = pred.tolist()[0][0], pred.tolist()[0][1]
    else:
        x, y = 7, 7

    return {"x": x, "y": y, "time": end_time - start_time}


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)
