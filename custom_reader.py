import json
import torch
import asyncio
from einops import rearrange
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as transforms

from data.parameters import build_parameter_dict
from tensorstore_reader import CustomTensorStoreDataLoader


async def main():
    parameter_dict = build_parameter_dict()

    tensorstore_file_template = parameter_dict["tensorstore_file_template"]
    dataset_statistics_file = parameter_dict["dataset_statistics_json"]

    with open(dataset_statistics_file, 'r') as f:
        dataset_statistics = json.load(f)
        
    # https://pytorch.org/vision/main/generated/torchvision.transforms.TrivialAugmentWide.html
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=dataset_statistics["mean"], std=dataset_statistics["std"]),
    ])

    root = "data"
    train_image_ts_file = tensorstore_file_template.format(root=root, split="train", data_class="image")
    train_tags_ts_file = tensorstore_file_template.format(root=root, split="train", data_class="tags")
    tensorstore_inputs_dict = {
        "image": {
            "tensorstore_file": train_image_ts_file,
            "cache_limit": 100_000_000
        },
        "tags": {
            "tensorstore_file": train_tags_ts_file,
            "cache_limit": 100_000_000
        }
    }
    
    train_loader = ImageLoader(
        length=dataset_statistics["count"]["train"],
        batch_size=4,
        shuffle=True,
        transform=transform
    )
    await train_loader.init_tensorstores(tensorstore_inputs_dict)

    while train_loader.remaining_indices:
        batch = await train_loader.get_batch()
        images = batch["images"]
        tags = batch["tags"]
        fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    
        for i, image in enumerate(images):
            image = rearrange(image, "c h w -> h w c")
            image = image.squeeze().numpy()
            axes[i].imshow(image)
            axes[i].axis("off")
            axes[i].set_title(f"Image {i+1}")

        plt.savefig("./data/plotted_images.png")
        plt.close()
        break


class ImageLoader(CustomTensorStoreDataLoader):
    def __init__(self, length, batch_size, shuffle,
                 transform):
        super().__init__(length, batch_size, shuffle)
        self.transform = transform
    
    async def batch_logic(self, indices):
        image_batch = await self.tensorstore_dict["image"][indices, :, :, :].read()
        image_batch = torch.tensor(image_batch, dtype=torch.uint8)
        image_batch = rearrange(image_batch, "b h w c -> b c h w")
        image_batch = self.transform(image_batch)
        
        tags_batch = await self.tensorstore_dict["tags"][indices, :].read()
        tags_batch = torch.tensor(tags_batch, dtype=torch.float32)

        return {
            "images": image_batch,
            "tags": tags_batch
        }


if __name__ == "__main__":
    asyncio.run(main())