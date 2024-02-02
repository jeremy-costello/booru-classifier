import json
import torch
import lightning as L
from tqdm import tqdm
from torch.utils.data import DataLoader, BatchSampler, DistributedSampler
from transformers import ConvNextV2Config
from torchvision.ops import sigmoid_focal_loss
import torchvision.transforms.v2 as transforms

from reader import DeepLakeDataset
from data.parameters import build_parameter_dict
from model import ConvNextV2ForMultiLabelClassification


parameter_dict = build_parameter_dict()

# data stuff
deeplake_file_template = parameter_dict["deeplake_file_template"]
matmul_precision = parameter_dict["training"]["matmul_precision"]
num_workers = parameter_dict["training"]["num_workers"]
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

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.TrivialAugmentWide(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=dataset_statistics["mean"], std=dataset_statistics["std"]),
])

valid_transform = transforms.Compose([
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=dataset_statistics["mean"], std=dataset_statistics["std"]),
])

fabric = L.Fabric(accelerator="cuda", devices=1, strategy="fsdp")
fabric.launch()

train_dataset = DeepLakeDataset(deeplake_file_template, dataset_statistics, train_transform, "train")
valid_dataset = DeepLakeDataset(deeplake_file_template, dataset_statistics, valid_transform, "valid")

train_sampler = BatchSampler(DistributedSampler(train_dataset, shuffle=True), batch_size=batch_size, drop_last=True)
valid_sampler = BatchSampler(DistributedSampler(valid_dataset, shuffle=False), batch_size=batch_size, drop_last=False)

train_loader = DataLoader(
    train_dataset,
    sampler=train_sampler,
    pin_memory=True,
    num_workers=num_workers
)

valid_loader = DataLoader(
    valid_dataset,
    sampler=valid_sampler,
    pin_memory=True,
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

model, optimizer = fabric.setup(model, optimizer)
train_loader = fabric.setup_dataloaders(train_loader, use_distributed_sampler=False)

model.train()
for epoch in tqdm(range(num_epochs)):
    for batch in tqdm(train_loader, leave=False):
        optimizer.zero_grad(set_to_none=True)
        
        images = batch["images"].squeeze(0)
        tags = batch["tags"].squeeze(0)

        output = model(images)
        loss = sigmoid_focal_loss(output, tags, reduction="sum")
        # precision, recall, F1, confusion matrix, accuracy, positive accuracy, negative accuracy
        
        fabric.backward(loss)
    
    print(loss)
    