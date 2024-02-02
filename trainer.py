import json
import torch
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
num_workers = parameter_dict["training"]["num_workers"]
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

train_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.RandomHorizontalFlip(),
    transforms.TrivialAugmentWide(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=dataset_statistics["mean"], std=dataset_statistics["std"]),
])

valid_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=dataset_statistics["mean"], std=dataset_statistics["std"]),
])

if matmul_precision is not None:
    torch.set_float32_matmul_precision(matmul_precision)

train_dataset = TensorStoreDataset(tensorstore_file_template, dataset_statistics, train_transform, "train")
valid_dataset = TensorStoreDataset(tensorstore_file_template, dataset_statistics, valid_transform, "valid")

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=False,
    num_workers=num_workers
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=False,
    num_workers=num_workers
)

config = ConvNextV2Config()
model = ConvNextV2ForMultiLabelClassification(
    config,
    output_size=config.hidden_sizes[-1],
    vocab_size=tag_indices["vocab_size"],
    use_sigmoid=False
)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# loss_function = torch.nn.BCELoss()

if use_fabric:
    fabric = L.Fabric(accelerator="cuda", devices=1, strategy="fsdp")
    fabric.launch()

    model, optimizer = fabric.setup(model, optimizer)
    train_loader = fabric.setup_dataloaders(train_loader)

model.train()
for epoch in tqdm(range(num_epochs)):
    for batch in tqdm(train_loader, leave=False):
        optimizer.zero_grad()

        images = batch["images"]
        tags = batch["tags"]

        output = model(images)
        loss = sigmoid_focal_loss(output, tags, reduction="sum")
        # precision, recall, F1, confusion matrix, accuracy, positive accuracy, negative accuracy
        
        if use_fabric:
            fabric.backward(loss)
        else:
            loss.backward()
        optimizer.step()
    
    print(loss)
