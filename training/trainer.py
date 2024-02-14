# adapted from:
# https://github.com/jzhang38/TinyLlama/blob/main/pretrain/tinyllama.py

import json
import math
import time
import torch
import random
from torch import nn
import lightning as L
from pathlib import Path
from einops import rearrange
import torch.nn.functional as F
from typing import Union, Tuple
from transformers import ConvNextV2Config
from torchvision.ops import sigmoid_focal_loss
import torchvision.transforms.v2 as transforms
from pytorch_lightning.loggers import WandbLogger
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from torch.utils.data import DataLoader, BatchSampler, DistributedSampler
from transformers.models.convnextv2.modeling_convnextv2 import ConvNextV2Layer

from reader import DeepLakeDataset
from params.parameters import build_parameter_dict
from model import ConvNextV2ForMultiLabelClassification
from utils import get_default_supported_precision, num_parameters

use_tqdm = True
if use_tqdm:
    from tqdm import tqdm

# https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel-modify-sdp-pt-lightning.html
sagemaker_training = False
if sagemaker_training:
    import os
    import smdistributed.dataparallel.torch.torch_smddp
    from lightning.fabric.plugins.environments.lightning import LightningEnvironment
    
    cluster_environment = LightningEnvironment()
    cluster_environment.world_size = lambda: int(os.environ["WORLD_SIZE"])
    cluster_environment.global_rank = lambda: int(os.environ["RANK"])
    
    # accelerator = gpu ???
    process_group_backend = "smddp"
else:
    cluster_environment = None
    process_group_backend = None


parameter_dict = build_parameter_dict()

seed_maybe = parameter_dict["training"]["seed"]
if seed_maybe is None:
    random.seed()
    seed = random.randint(0, 2 ** 32 - 1)
else:
    seed = seed_maybe

model_name = parameter_dict["training"]["model_name"]
name = parameter_dict["training"]["name"]
out_root = parameter_dict["training"]["out_root"]
out_dir = Path(out_root) / name

matmul_precision = parameter_dict["training"]["matmul_precision"]
num_devices = parameter_dict["training"]["num_devices"]
num_nodes = parameter_dict["training"]["num_nodes"]
precision_maybe = parameter_dict["training"]["precision"]
tpu = parameter_dict["training"]["tpu"]
num_workers = parameter_dict["training"]["num_workers"]
global_batch_size = parameter_dict["training"]["global_batch_size"]
micro_batch_size = parameter_dict["training"]["micro_batch_size"]
nano_batch_size = parameter_dict["training"]["nano_batch_size"]
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
loss_multiplier = parameter_dict["training"]["loss_multiplier"]

batch_size = global_batch_size // num_devices
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
warmup_iters = warmup_steps * gradient_accumulation_steps

assert micro_batch_size % nano_batch_size == 0
outer_batch_size = micro_batch_size // nano_batch_size
inner_batch_size = nano_batch_size

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
wandb_logger = WandbLogger()


def setup(
    resume: Union[bool, Path] = False
) -> None:
    devices = num_devices
    precision = precision_maybe or get_default_supported_precision(training=True, tpu=tpu)
    
    if num_devices > 1:
        if tpu:
            devices = "auto"
            strategy = XLAStrategy(sync_module_states=False)
        else:
            strategy = FSDPStrategy(
                cluster_environment=cluster_environment,
                process_group_backend=process_group_backend,
                auto_wrap_policy={ConvNextV2Layer},
                activation_checkpointing_policy=None,
                state_dict_type="full",
                limit_all_gathers=True,
                cpu_offload=False
            )
    else:
        strategy = "auto"
    
    fabric = L.Fabric(
        strategy=strategy,
        devices=devices,
        num_nodes=num_nodes,
        precision=precision,
        loggers=[wandb_logger]
    )
    fabric.print(hparams)
    main(fabric, resume)


def main(fabric, resume):
    # skipped monitor from tinyllama
    
    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    
    # auto configs?
    # https://github.com/jzhang38/TinyLlama/blob/main/lit_gpt/config.py
    # config = Config.from_name(model_name)
    config = ConvNextV2Config()
    
    train_loader, valid_loader = create_dataloaders(
        outer_batch_size=outer_batch_size,
        inner_batch_size=inner_batch_size,
        fabric=fabric,
        seed=seed
    )
    train_loader, valid_loader = fabric.setup_dataloaders(train_loader, valid_loader,
                                                          use_distributed_sampler=False)
    
    fabric.seed_everything(seed)
    
    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        # all model stuff in config, custom _init_weights?
        model = ConvNextV2ForMultiLabelClassification(
            config,
            output_size=config.hidden_sizes[-1],
            vocab_size=vocab_size,
            use_sigmoid=False
        )
        model.init_weights()
    
    fabric.print(f"Time to instantiate model: {(time.perf_counter() - t0):.2f} seconds")
    fabric.print(f"Total parameters {num_parameters(model):,}")
    
    # https://lightning.ai/docs/fabric/stable/advanced/compile.html
    model = torch.compile(model, mode="reduce-overhead")
    model = fabric.setup(model, _reapply_compile=True)
    
    optimizer = torch.optim.AdamW(model.get_optimizer_groups(weight_decay),
                                  lr=learning_rate, betas=(beta1, beta2),
                                  foreach=False)
    optimizer = fabric.setup_optimizers(optimizer)
    
    state = {
        "model": model,
        "optimizer": optimizer,
        "hparams": hparams,
        "epoch_count": 0,
        "epoch_iter_count": 0,
        "epoch_step_count": 0,
        "total_iter_count": 0,
        "total_step_count": 0
    }
    
    if resume:
        resumed = True
        if isinstance(resume, Path):
            resume_pth = resume
        elif isinstance(resume, bool):
            pth_glob = sorted(out_dir.glob("*.pth"))
            if pth_glob:
                resume_pth = pth_glob[-1]
            else:
                resumed = False
        else:
            raise ValueError("Invalid type for resume. Should be bool or Path")
        
        if resumed:
            fabric.print(f"Resuming training from {resume_pth}")
            fabric.load(resume_pth, state)
        else:
            fabric.print("Could not find file to resume from. Starting new training run.")
    
    if seed != state["hparams"]["seed"]:
        raise ValueError(f"seed in hyperparameters.py ({seed_maybe}) should be the same as"
                         f" seed in fabric state ({state['hparams']['seed']}))")
    
    fabric.print(state["hparams"])
    fabric.print(seed)
    
    train_time = time.perf_counter()
    train(fabric, state, train_loader, valid_loader, resume)
    fabric.print(f"Training time: {(time.perf_counter() - train_time):.2f} seconds")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
    

def train(fabric, state, train_loader, valid_loader, resume):
    model = state["model"]
    optimizer = state["optimizer"]
    
    validate(fabric, model, valid_loader)
    
    # skipped flops stuff from tinyllama
    
    total_t0 = time.perf_counter()
    
    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()
    
    initial_epoch = state["epoch_count"]
    initial_epoch_iter = state["epoch_iter_count"]
    initial_total_iter = state["total_iter_count"]
    curr_epoch_iter = 0
    
    loss_function = sigmoid_focal_loss
    
    if state["total_iter_count"] >= max_iters:
        fabric.print("Total iterations >= maximum iterations")
        return

    average_train_loss = torch.zeros(1).to(fabric.device)
    if fabric.global_rank == 0 and use_tqdm:
        outer_pbar = tqdm(range(num_epochs))
    
    for epoch in range(num_epochs):
        if not resume:
            assert epoch == state["epoch_count"]
        if fabric.global_rank == 0 and use_tqdm:
            inner_pbar = tqdm(range(len(train_loader)), leave=False)
        
        for train_batch in train_loader:
            if resume:
                if epoch < initial_epoch:
                    continue
                elif curr_epoch_iter < initial_epoch_iter:
                    curr_epoch_iter += 1
                    continue
                else:
                    resume = False
                    curr_epoch_iter = -1
                    fabric.barrier()
                    fabric.print(f"Resume finished. Took {time.perf_counter() - total_t0} seconds")

            if state["total_iter_count"] >= max_iters:
                break            
                        
            images = train_batch["images"]
            images = rearrange(images, "bs bl c h w -> (bs bl) c h w")
            tags = train_batch["tags"]
            tags = rearrange(tags, "bs bl t -> (bs bl) t")
            
            is_accumulating = (state["total_iter_count"] + 1) % gradient_accumulation_steps != 0
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                outputs = model(images)
                loss = loss_multiplier * loss_function(outputs, tags, reduction="sum")
                fabric.backward(loss / gradient_accumulation_steps)
                average_train_loss += loss / gradient_accumulation_steps
            
            if not is_accumulating:
                lr = get_lr(state["total_iter_count"]) if decay_lr else learning_rate
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                
                fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                state["epoch_step_count"] += 1
                state["total_step_count"] += 1
                
                with torch.no_grad():
                    average_train_loss = fabric.all_reduce(average_train_loss, reduce_op="sum")
                if fabric.global_rank == 0:
                    fabric.log_dict(
                        {
                            "metric/train/loss": average_train_loss
                        },
                        state["total_step_count"]
                    )
                average_train_loss = 0
            # idk how tpu training works
            elif fabric.device.type == "xla":
                xm.mark_step()
            
            state["epoch_iter_count"] += 1
            state["total_iter_count"] += 1
            
            # monitor stuff
            if not is_accumulating and state["total_step_count"] % eval_step_interval == 0:
                val_dict = validate(fabric, model, valid_loader)
                # monitor stuff
                with torch.no_grad():
                    val_dict = fabric.all_reduce(val_dict, reduce_op="mean")
                if fabric.global_rank == 0:
                    fabric.log_dict(
                        {
                            "metric/val/loss": val_dict["loss"],
                            "metric/val/precision": val_dict["precision"],
                            "metric/val/recall": val_dict["recall"],
                            "metric/val/f1": val_dict["f1"],
                            "total_images": (state["total_iter_count"] + 1) * micro_batch_size * fabric.world_size
                        },
                        state["total_step_count"]
                    )
                fabric.barrier()
            if not is_accumulating and state["total_step_count"] % save_step_interval == 0:
                checkpoint_path = out_dir / f"iter-{state['total_iter_count']:06d}-ckpt.pth"
                fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
                fabric.save(checkpoint_path, state)
            
            if fabric.global_rank == 0:
                inner_pbar.update(1)
        
        if fabric.global_rank == 0:
            inner_pbar.close()
        
        state["epoch_count"] += 1
        if fabric.global_rank == 0:
            outer_pbar.update(1)
    
    if fabric.global_rank == 0:
        outer_pbar.close()


@torch.no_grad()
def validate(fabric: L.Fabric, model: nn.Module, valid_loader: DataLoader) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    # does this require fabric.world_size? or is that done automatically
    # https://lightning.ai/docs/fabric/stable/advanced/distributed_communication.html
    valid_batches = count_valid // (fabric.world_size * micro_batch_size)
        
    losses = torch.zeros(valid_batches, device=fabric.device)
    tp = torch.zeros(vocab_size, device=fabric.device)
    fp = torch.zeros(vocab_size, device=fabric.device)
    fn = torch.zeros(vocab_size, device=fabric.device)
    
    if fabric.global_rank == 0 and use_tqdm:
        inner_pbar = tqdm(range(len(valid_loader)), leave=False)
    
    for k, valid_batch in enumerate(valid_loader):
        images = valid_batch["images"]
        images = rearrange(images, "bs bl c h w -> (bs bl) c h w")
        tags = valid_batch["tags"]
        tags = rearrange(tags, "bs bl t -> (bs bl) t")
        
        outputs = model(images)
        loss = F.binary_cross_entropy_with_logits(outputs, tags, reduction="sum")
        losses[k] = loss.item()
        
        # convert for cuda bitwise and
        tags = tags.to(torch.int64)
        
        threshold = 0.5
        predicted = outputs >= threshold
        tp += (predicted & tags).sum(dim=0)
        fp += (predicted & ~tags).sum(dim=0)
        fn += (~predicted & tags).sum(dim=0)
        
        if fabric.global_rank == 0:
            inner_pbar.update(1)
    
    if fabric.global_rank == 0:
        inner_pbar.close()

    epsilon = 1e-12
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)

    model.train()
    # probably something better than mean for precision, recall, f1?
    return {
        "loss": losses.mean(),
        "precision": precision.mean(),
        "recall": recall.mean(),
        "f1": f1.mean()
    }


def create_dataloaders(outer_batch_size: int, inner_batch_size: int, fabric: L.Fabric, seed: int) -> Tuple[DataLoader, DataLoader]:
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
    
    train_dataset = DeepLakeDataset(parameter_dict["file_root"], parameter_dict["deeplake_file_template"],
                                    dataset_statistics, train_transform, "train")
    valid_dataset = DeepLakeDataset(parameter_dict["file_root"], parameter_dict["deeplake_file_template"],
                                    dataset_statistics, valid_transform, "valid")

    train_sampler = BatchSampler(
        DistributedSampler(
            train_dataset,
            num_replicas=fabric.world_size,
            rank=fabric.global_rank,
            shuffle=True,
            seed=seed,
            drop_last=True),
        batch_size=inner_batch_size,
        drop_last=True)
    
    valid_sampler = BatchSampler(
        DistributedSampler(
            valid_dataset,
            num_replicas=fabric.world_size,
            rank=fabric.global_rank,
            shuffle=False,
            drop_last=True),
        batch_size=inner_batch_size,
        drop_last=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=outer_batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=outer_batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return train_loader, valid_loader


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    from jsonargparse import CLI
    
    if matmul_precision is not None:
        torch.set_float32_matmul_precision(matmul_precision)
    
    CLI(setup)
