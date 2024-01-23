import glob
import os
import re

import torch
from PIL import Image
from torch.utils.data import Dataset
from ultralytics import YOLO


class CropDataset(Dataset):
    def __init__(self, file_path, transform):
        super().__init__()

        # 画像をYOLOでクロップする
        name = os.path.split(file_path)[1].split(".")[0]
        if not os.path.exists(f"cropped_http_data/{name}"):
            yolo(input_path=file_path, output_path=f"cropped_http_data", name=name)

        cropped_paths = sorted(glob.glob(f"cropped_http_data/{name}"))
        self.dataset = []

        # datasetにラベルと画像パスを追加
        for i, c_path in enumerate(cropped_paths):
            dir_names = sorted(os.listdir(c_path))
            for d_name in dir_names:
                file_paths = []

                # ディレクトリ配下にあるクロップ画像のパスを取得
                for current_dir, sub_dirs, files_list in os.walk(f"{c_path}/{d_name}"):
                    # 人のラベルをデータセットから除外
                    # dir_name = os.path.basename(current_dir)
                    # if dir_name == "person":
                    #     continue

                    for f_name in files_list:
                        file_paths.append(os.path.join(current_dir, f_name))

                position = [float(p) for p in re.findall(r'\d+', c_path)]
                self.dataset.append({"xy": position, "paths": file_paths})

        self.transform = transform

    def __getitem__(self, index):
        file_paths = self.dataset[index]["paths"]
        images = []

        for f_path in file_paths:
            img = Image.open(f_path)

            # 画像を前処理
            if self.transform is not None:
                img = self.transform(img)

            images.append(img)

        xy = self.dataset[index]["xy"]
        return images, xy

    def __len__(self):
        return len(self.dataset)


def yolo(input_path, output_path, name):
    model = YOLO("yolov8x.pt")
    model(input_path, project=output_path, name=name, save_crop=True, conf=0.1)


def collate_fn(batch_list):
    # バッチリストから空でない要素だけを選択
    filtered_batch_list = [data for data in batch_list if data[0] and data[1]]

    images_list = [data[0] for data in filtered_batch_list]
    label_list = [data[1] for data in filtered_batch_list]

    # batchリストを1つのTensorにまとめる
    images = [torch.stack(batch) for batch in images_list]
    labels = torch.tensor(label_list)

    return images, labels
