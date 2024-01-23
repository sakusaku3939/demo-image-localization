import torch
import torchvision.transforms as transforms

from datasets.crop_dataset import CropDataset, collate_fn


def load_cropped_image(file_path, batch_size, num_workers):
    # データの前処理
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # データセットの読み込み
    dataset = CropDataset(file_path, transform)

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return test_loader
