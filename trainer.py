import json
import math
import torch
import asyncio
import lightning as L
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import ConvNextV2Config
from torchvision.ops import sigmoid_focal_loss
import torchvision.transforms.v2 as transforms

from custom_reader import ImageLoader
from data.parameters import build_parameter_dict
from model import ConvNextV2ForMultiLabelClassification


parameter_dict = build_parameter_dict()

# data stuff
tensorstore_file_template = parameter_dict["tensorstore_file_template"]
matmul_precision = parameter_dict["training"]["matmul_precision"] 
use_fabric = parameter_dict["training"]["use_fabric"]
num_epochs = parameter_dict["training"]["num_epochs"]
learning_rate = parameter_dict["training"]["learning_rate"]
batch_size = parameter_dict["training"]["batch_size"]

dataset_statistics_file = parameter_dict["dataset_statistics_json"]
tag_indices_file = parameter_dict["tag_indices_json"]



with open(dataset_statistics_file, 'r') as f:
    dataset_statistics = json.load(f)

with open(tag_indices_file, 'r') as f:
    tag_indices = json.load(f)

if matmul_precision is not None:
    torch.set_float32_matmul_precision(matmul_precision)

split_transforms = {
    "train":
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=dataset_statistics["mean"], std=dataset_statistics["std"]),
        ]),
    "valid":
        transforms.Compose([
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=dataset_statistics["mean"], std=dataset_statistics["std"]),
        ])
}

shuffles = {
    "train": True,
    "valid": False
}


async def get_loader(split, cache_limit):
    image_ts_file = tensorstore_file_template.format(root="data", split=split, data_class="image")
    tags_ts_file = tensorstore_file_template.format(root="data", split=split, data_class="tags")
    tensorstore_inputs_dict = {
        "image": {
            "tensorstore_file": image_ts_file,
            "cache_limit": cache_limit
        },
        "tags": {
            "tensorstore_file": tags_ts_file,
            "cache_limit": cache_limit
        }
    }

    loader = ImageLoader(
        length=dataset_statistics["count"][split],
        batch_size=batch_size,
        shuffle=shuffles[split],
        transform=split_transforms[split]
    )
    await loader.init_tensorstores(tensorstore_inputs_dict)
    return loader


async def training():
    train_loader = await get_loader("train", 100_000_000)
    valid_loader = await get_loader("valid", 100_000_000)
    
    config = ConvNextV2Config()
    model = ConvNextV2ForMultiLabelClassification(
        config,
        output_size=config.hidden_sizes[-1],
        vocab_size=tag_indices["vocab_size"],
        use_sigmoid=False
    )
    
    if not use_fabric:
        model = model.to("cuda")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # loss_function = torch.nn.BCELoss()

    if use_fabric:
        fabric = L.Fabric(accelerator="cuda", devices=1, strategy="fsdp")
        fabric.launch()

        model, optimizer = fabric.setup(model, optimizer)
        train_loader = fabric.setup_dataloaders(train_loader)

    model.train()
    for epoch in tqdm(range(num_epochs)):
        with tqdm(total=math.ceil(train_loader.length / batch_size)) as pbar:
            while train_loader.remaining_indices:
                optimizer.zero_grad()
                
                batch = await train_loader.get_batch()
                images = batch["images"]
                tags = batch["tags"]
                if not use_fabric:
                    images = images.to("cuda")
                    tags = tags.to("cuda")

                output = model(images)
                loss = sigmoid_focal_loss(output, tags, reduction="sum")
                # precision, recall, F1, confusion matrix, accuracy, positive accuracy, negative accuracy
                
                if use_fabric:
                    fabric.backward(loss)
                else:
                    loss.backward()
                optimizer.step()
                
                pbar.update(1)
            
            train_loader.reset_indices()
        
        print(loss)


if __name__ == "__main__":
    asyncio.run(training())
