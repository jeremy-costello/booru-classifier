# https://github.com/jzhang38/TinyLlama/blob/main/pretrain/tinyllama.py
import json
import time
import torch
import lightning as L
from tqdm import tqdm
from pathlib import Path
from typing import Union
from transformers import ConvNextV2Config
from torchvision.ops import sigmoid_focal_loss
import torchvision.transforms.v2 as transforms
from pytorch_lightning.loggers import WandbLogger
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from torch.utils.data import DataLoader, BatchSampler, DistributedSampler
from transformers.models.convnextv2.modeling_convnextv2 import ConvNextV2Layer

from reader import DeepLakeDataset
from data.parameters import build_parameter_dict
from model import ConvNextV2ForMultiLabelClassification
from utils import step_csv_logger, get_default_supported_precision


# load this from parameters.py too
model_name = None
name = None
out_dir = None

parameter_dict = build_parameter_dict()

matmul_precision = parameter_dict["training"]["matmul_precision"]
num_devices = parameter_dict["training"]["num_devices"]
precision = parameter_dict["training"]["precision"]
tpu = parameter_dict["training"]["tpu"]
num_workers = parameter_dict["training"]["num_workers"]
global_batch_size = parameter_dict["training"]["global_batch_size"]
micro_batch_size = parameter_dict["training"]["micro_batch_size"]
num_epochs = parameter_dict["training"]["num_epochs"]
warmup_steps = parameter_dict["training"]["warmup_steps"]
log_step_interval = parameter_dict["training"]["log_step_interval"]
save_step_interval = parameter_dict["training"]["save_step_interval"]
eval_step_interval = parameter_dict["training"]["eval_step_interval"]

learning_rate = parameter_dict["training"]["learning_rate"]
weight_decay = parameter_dict["training"]["weight_decay"]
beta1 = parameter_dict["training"]["beta1"]
beta2 = parameter_dict["training"]["beta2"]
grad_clip = parameter_dict["training"]["grad_clip"]
decay_lr = parameter_dict["training"]["decay_lr"]
min_lr = parameter_dict["training"]["min_lr"]

batch_size = global_batch_size // num_devices
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
warmup_iters = warmup_steps * gradient_accumulation_steps

with open(parameter_dict["dataset_statistics_json"], 'r') as f:
    dataset_statistics = json.load(f)

mean = dataset_statistics["mean"]
std = dataset_statistics["std"]
count_train = dataset_statistics["count"]["train"]
count_valid = dataset_statistics["count"]["valid"]

with open(parameter_dict["tag_indices_json"], 'r') as f:
    tag_indices = json.load(f)

vocab_size = tag_indices["vocab_size"]

max_step = num_epochs * count_train // global_batch_size
max_iters = max_step * gradient_accumulation_steps
lr_decay_iters = max_iters
log_iter_interval = log_step_interval * gradient_accumulation_steps

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
logger = step_csv_logger("out", name, flush_logs_every_n_steps=log_iter_interval)
wandb_logger = WandbLogger()


def setup(
    resume: Union[bool, Path] = False
) -> None:
    devices = num_devices
    # should this be elsewhere?
    if matmul_precision is not None:
        torch.set_float32_matmul_precision(matmul_precision)
    precision = precision or get_default_supported_precision(training=True, tpu=tpu)
    
    if num_devices > 1:
        if tpu:
            devices = "auto"
            strategy = XLAStrategy(sync_module_states=False)
        else:
            strategy = FSDPStrategy(
                auto_wrap_policy={ConvNextV2Layer},
                activation_checkpointing_policy=None,
                state_dict_type="full",
                limit_all_gathers=True,
                cpu_offload=False
            )
    else:
        strategy = "auto"
    
    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=[logger, wandb_logger])
    fabric.print(hparams)
    main(fabric, resume)


# increment seed by 1 per device rank
def create_dataloaders():
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
    
    train_dataset = DeepLakeDataset(parameter_dict["deeplake_file_template"], dataset_statistics, train_transform, "train")
    valid_dataset = DeepLakeDataset(parameter_dict["deeplake_file_template"], dataset_statistics, valid_transform, "valid")

    # batch sizes in sampler and loader then concatenate along first dimension?
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
    
    return train_loader, valid_loader


def main(fabric, resume):
    #fabric.launch()

    # https://github.com/jzhang38/TinyLlama/blob/main/lit_gpt/speed_monitor.py
    # monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=log_iter_interval)
    
    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    
    # auto configs?
    # config = Config.from_name(model_name)
    config = ConvNextV2Config()
    
    # add seeding to this (and other places?)
    train_loader, valid_loader = create_dataloaders()
    train_loader, valid_loader = fabric.setup_dataloaders(train_loader, valid_loader,
                                                          use_distributed_sampler=False)

    # left out some model loading stuff from tinyllama
    model = ConvNextV2ForMultiLabelClassification(
        config,
        output_size=config.hidden_sizes[-1],
        vocab_size=vocab_size,
        use_sigmoid=False
    )
    model, optimizer = fabric.setup(model)
    
    # foreach / fused ???
    optimizer = torch.optim.AdamW(model.get_optimizer_groups(weight_decay),
                                lr=learning_rate, betas=(beta1, beta2))
    fabric.setup_optimizers(optimizer)
    
    state = {
        "model": model,
        "optimizer": optimizer,
        "hparams": hparams,
        "iter_num": 0,
        "step_count": 0
    }
    
    if resume:
        resume = sorted(out_dir.glob("*.pth"))[-1]
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)
    
    train_time = time.perf_counter()
    train(fabric, state, train_loader, valid_loader, resume)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
    

def train(fabric, state, train_loader, valid_loader, resume):
    model = state["model"]
    optimizer = state["optimizer"]
    
    #validate(fabric, model, valid_loader)
    
    # left out some meta flops stuff from tinyllama
    
    total_t0 = time.perf_counter()
    
    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()
    
    initial_iter = state["iter_num"]
    curr_iter = 0
    
    loss_function = sigmoid_focal_loss

    # more stuff from tinyllama to add here
    model.train()
    for epoch in tqdm(range(num_epochs)):
        for batch in tqdm(train_loader, leave=False):
            optimizer.zero_grad(set_to_none=True)
            
            images = batch["images"].squeeze(0)
            tags = batch["tags"].squeeze(0)

            output = model(images)
            loss = loss_function(output, tags, reduction="sum")
            # precision, recall, F1, confusion matrix, accuracy, positive accuracy, negative accuracy
            
            fabric.backward(loss)
        
        print(loss)
        