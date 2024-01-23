from flask import Flask, request, jsonify
import torch
import time
from torch import nn
from models.YoloLSTM import YoloLSTM
from datasets.dataset_utils import load_cropped_image

tile_width = 0.45  # 1タイルの長さ (m)

app = Flask(__name__)


@app.route("/")
def index():
    return "Hello World!"


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # ここで画像ファイルを変数に格納する処理を行う
        # 例えば、ファイルを保存する場合:
        # file.save(os.path.join('uploads', secure_filename(file.filename)))

        return "Hello World"  # Flutterアプリに文字列を返す


# GPUデバイスの設定
def init_device():
    torch.multiprocessing.freeze_support()
    device = torch.device("cuda")
    print(f"Device type: {device}")
    return device


def predict():
    torch.multiprocessing.freeze_support()
    batch_size = 1
    num_workers = 2
    device = init_device()

    start_time = time.time()

    model = YoloLSTM().to(device)
    model = model.eval()
    model.load_state_dict(torch.load("models/YoloLSTM-Bi/best_model.pth"))

    test_loader = load_cropped_image(batch_size, num_workers)
    mae_function = nn.L1Loss()

    with torch.no_grad():
        model = model.eval()

        for i, data in enumerate(test_loader, 0):
            inputs = [d.to(device) for d in data[0]]
            target = data[1].to(device)

            pred = model(inputs)
            print(f"\n推定した座標: x: {pred.tolist()[0][0]}, y: {pred.tolist()[0][1]}")
            print(f"正解の座標: x: {target.tolist()[0][0]}, y: {target.tolist()[0][1]}")
            print(f"平均誤差: {mae_function(pred, target) * tile_width} m")

    end_time = time.time()
    print("\n")
    print(f"1枚あたりの推定にかかった時間: {(end_time - start_time) / i} 秒")


if __name__ == "__main__":
    # predict()
    app.run(host="0.0.0.0", debug=False)
