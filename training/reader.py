import json
import torch
import deeplake
import numpy as np
import lightning as L
from einops import rearrange
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader, BatchSampler, DistributedSampler

from params.parameters import build_parameter_dict


def main():
    parameter_dict = build_parameter_dict()

    file_root = parameter_dict["file_root"]
    deeplake_file_template = parameter_dict["deeplake_file_template"]
    matmul_precision = parameter_dict["training"]["matmul_precision"]
    num_workers = parameter_dict["training"]["num_workers"]
    dataset_statistics_file = parameter_dict["dataset_statistics_json"]

    with open(dataset_statistics_file, 'r') as f:
        dataset_statistics = json.load(f)
    
    if matmul_precision is not None:
        torch.set_float32_matmul_precision(matmul_precision)
    
    # https://pytorch.org/vision/main/generated/torchvision.transforms.TrivialAugmentWide.html
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=dataset_statistics["mean"], std=dataset_statistics["std"]),
    ])

    fabric = L.Fabric(accelerator="cuda", devices=1, strategy="fsdp")
    fabric.launch()
    
    train_dataset = DeepLakeDataset(file_root, deeplake_file_template, dataset_statistics, transform, split="train")

    train_sampler = BatchSampler(
        DistributedSampler(
            train_dataset,
            num_replicas=1,
            rank=0,
            shuffle=True,
            seed=0,
            drop_last=True),
        batch_size=4,
        drop_last=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    train_loader = fabric.setup_dataloaders(train_loader, use_distributed_sampler=False)

    for batch in train_loader:
        images = batch["images"]
        # batch (sampler) and batch (loader)
        images = rearrange(images, "bs bl c h w -> (bs bl) c h w")
        tags = batch["tags"]
        fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    
        for i, image in enumerate(images):
            # add label with list of tags
            image = rearrange(image, "c h w -> h w c")
            image = image.detach().cpu().numpy()
            axes[i].imshow(image)
            axes[i].axis("off")
            axes[i].set_title(f"Image {i+1}")

        plt.savefig("./data/plotted_images.png")
        plt.close()
        break


class DeepLakeDataset(Dataset):    
    def __init__(self, file_root, deeplake_file_template, dataset_statistics, transform, split):
        lake_path = deeplake_file_template.format(root=file_root, split=split)
        self.lake = deeplake.dataset(lake_path, read_only=True)
        
        self.total_samples = dataset_statistics["count"][split]
        self.transform = transform      

    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, indices):
        image_batch = self.lake.images[indices, :, :, :].numpy()
        image_batch = torch.tensor(image_batch, dtype=torch.uint8)
        image_batch = rearrange(image_batch, "b h w c -> b c h w")
        image_batch = self.transform(image_batch)
        
        tags_batch = self.lake.tags[indices, :].numpy()
        tags_batch = torch.tensor(tags_batch.astype(np.float32), dtype=torch.float32)

        return {
            "images": image_batch,
            "tags": tags_batch
        }


if __name__ == "__main__":
    main()
