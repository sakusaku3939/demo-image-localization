import os
import random
import shutil


def random_file_rename(base_path):
    for folder_name in os.listdir(base_path):  # 1から65までのフォルダ
        folder_path = os.path.join(base_path, folder_name)

        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue

        # フォルダ内のファイルリストを取得
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        if not files:
            print(f"No files found in {folder_path}")
            continue

        # ランダムにファイルを選択
        selected_file = random.choice(files)
        selected_file_path = os.path.join(folder_path, selected_file)

        # 新しいファイル名を設定
        new_file_name = f"{folder_name}.{selected_file.split('.')[-1]}"  # 拡張子を保持
        new_file_path = os.path.join("data", new_file_name)

        # ファイルを新しい名前でコピー
        shutil.copy2(selected_file_path, new_file_path)
        print(f"Copied {selected_file} to {new_file_name}")


base_path = "train"
random_file_rename(base_path)
