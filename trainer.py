import json
import torch
import lightning as L
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import ConvNextV2Config
from torchvision.ops import sigmoid_focal_loss
import torchvision.transforms.v2 as transforms

from reader import ParquetDataset
from parameters import parameter_dict
from model import ConvNextV2ForMultiLabelClassification


NUM_EPOCHS = 10
LEARNING_RATE = 3e-4


# data stuff
large_file_root = parameter_dict["large_file_root"]
train_parquet_file = parameter_dict["train_parquet_file"]
train_parquet = f"{large_file_root}/{train_parquet_file}"

dataset_statistics_file = parameter_dict["dataset_statistics_json"]
tag_indices_file = parameter_dict["tag_indices_json"]

with open(dataset_statistics_file, 'r') as f:
    dataset_statistics = json.load(f)

with open(tag_indices_file, 'r') as f:
    tag_indices_file = json.load(f)

transform = transforms.Compose([
    transforms.ToImage(),
    transforms.RandomHorizontalFlip(),
    transforms.TrivialAugmentWide(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=dataset_statistics["mean"], std=dataset_statistics["std"]),
])

dataset = ParquetDataset(train_parquet, transform, partitions=8)

dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    pin_memory=False,
    num_workers=0
)

config = ConvNextV2Config()
model = ConvNextV2ForMultiLabelClassification(
    config,
    output_size=768,
    vocab_size=tag_indices_file["vocab_size"],
    use_sigmoid=False
)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
# loss_function = torch.nn.BCELoss()

fabric = L.Fabric(accelerator="cuda", devices=1, strategy="fsdp")
fabric.launch()

model, optimizer = fabric.setup(model, optimizer)
dataloader = fabric.setup_dataloaders(dataloader)

model.train()
for epoch in tqdm(range(NUM_EPOCHS)):
    for batch in tqdm(dataloader, leave=False):
        optimizer.zero_grad()

        images = batch["images"]
        tags = batch["tags"]

        output = model(images)
        loss = sigmoid_focal_loss(output, tags, reduction="sum")
        # precision, recall, F1, confusion matrix, accuracy, positive accuracy, negative accuracy

        fabric.backward(loss)
        optimizer.step()
    
    print(loss)
