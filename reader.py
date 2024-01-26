import json
import math
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import dask.dataframe as dd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader

from data.parameters import build_parameter_dict


def main():
    parameter_dict = build_parameter_dict()

    train_parquet_file = parameter_dict["train_parquet_file"]
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

    dataset = ParquetDataset(train_parquet_file, transform)

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        pin_memory=False,
        num_workers=0
    )

    for batch in loader:
        images = batch["images"]
        tags = batch["tags"]
        fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    
        for i, image in enumerate(images):
            image = image.squeeze(0).numpy()
            image = np.moveaxis(image, 0, 2)
            axes[i].imshow(image)
            axes[i].axis("off")
            axes[i].set_title(f"Image {i+1}")

        plt.show()
        break


class ParquetDataset(Dataset):
    def __init__(self, parquet_file, transform, partitions):
        spark = SparkSession.builder.appName("reader").getOrCreate()
        parquet_url = f"file://{parquet_file}"
        df = spark.read.parquet(parquet_url)
        self.num_rows = df.count()
        
        self.partition_size = math.ceil(self.num_rows / partitions)
        divisions = [num * self.partition_size for num in range(partitions + 1)]
        divisions[-1] = self.num_rows

        self.ddf = dd.read_parquet(parquet_file)
        self.ddf = self.ddf.reset_index(drop=True)
        print(self.ddf.head())
        print(self.ddf.npartitions)
        print(self.ddf.divisions)

        self.transform = transform      

    def __len__(self):
        return self.num_rows
    
    def __getitem__(self, idx):
        partition_idx, row_idx = divmod(idx, self.partition_size)
        partition_df = self.ddf.get_partition(partition_idx).compute()
        row = partition_df.iloc[row_idx].to_dict()

        # image
        images = row["image_binary"]
        images = Image.open(BytesIO(images))
        images = self.transform(images)
        # tags
        tags = row["tag_binary"]
        tags = np.load(BytesIO(tags))
        tags = torch.from_numpy(tags.astype(np.float32))

        return {
            "images": images,
            "tags": tags
        }


if __name__ == "__main__":
    main()
