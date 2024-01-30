import json
import torch
import numpy as np
import tensorstore as ts
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader

from data.parameters import build_parameter_dict


def main():
    parameter_dict = build_parameter_dict()

    tensorstore_file_template = parameter_dict["tensorstore_file_template"]
    dataset_statistics_file = parameter_dict["dataset_statistics_json"]

    with open(dataset_statistics_file, 'r') as f:
        dataset_statistics = json.load(f)
    
    # https://pytorch.org/vision/main/generated/torchvision.transforms.TrivialAugmentWide.html
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=dataset_statistics["mean"], std=dataset_statistics["std"]),
    ])

    train_dataset = TensorStoreDataset(tensorstore_file_template, dataset_statistics, transform, split="train")

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        pin_memory=False,
        num_workers=0
    )

    for batch in train_loader:
        images = batch["images"]
        tags = batch["tags"]
        fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    
        for i, image in enumerate(images):
            image = image.squeeze(0).numpy()
            image = np.moveaxis(image, 0, 2)
            axes[i].imshow(image)
            axes[i].axis("off")
            axes[i].set_title(f"Image {i+1}")

        plt.savefig("./data/plotted_images.png")
        plt.close()
        break


def open_tensorstore(tensorstore_file):
    return ts.open({
        "driver": "zarr",
        "kvstore": {
            "driver": "file",
            "path": tensorstore_file
        }
    }).result()


class TensorStoreDataset(Dataset):
    def __init__(self, tensorstore_file_template, dataset_statistics, transform, split):
        self.image_tensorstore = open_tensorstore(
            tensorstore_file_template.format(split=split, data_type="image")
        )
        self.tags_tensorstore = open_tensorstore(
            tensorstore_file_template.format(split=split, data_type="tags")
        )
        
        self.total_samples = dataset_statistics["count"][split]
        
        self.transform = transform      

    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        image = self.image_tensorstore[idx, :, :, :].read().result()
        image = self.transform(image)
        
        tags = self.tags_tensorstore[idx, :].read().result()

        return {
            "images": image,
            "tags": tags
        }


if __name__ == "__main__":
    main()
