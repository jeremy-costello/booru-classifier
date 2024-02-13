import os


PARAMETER_DICT = {
    "debug": True,
    "booru_url": "https://safebooru.org",
    "file_root": "./mount/data",
    "deeplake_file_template": "{root}/{split}_lake",
    "files": {
        "database_file": "booru.db",
        "image_save_root": "images",
        "skeleton_parquet_file": "skeleton.parquet",
        "tag_counts_json": "tag_counts.json",
        "tag_indices_json": "tag_indices.json",
        "dataset_statistics_json": "dataset_statistics.json",
    },
    "scraping": {
        "max_retries": 10,
        "post_min_batch_size": 8192,
        "image_batch_size": 8192,
        "tag_batch_size": 512,
        "download_type": "posts"
    },
    "dataset": {
        "skeleton_partitions": 32,
        "min_tag_count": 50,
        "tag_count_limit": 5000,
        "validation_fraction": 0.05,
        "image_size": 224,
        "channel_size": 3,
        "final_chunks": 256,
        "stats_rounding": 4,
    },
    "training": {
        "out_root": "./mount/out",
        "seed": None,
        "model_name": None,
        "name": None,
        "matmul_precision": "high",
        "num_nodes": 1,
        "num_devices": 1,
        "precision": "bf16-mixed",
        "num_workers": 0,
        "global_batch_size": 64,
        "micro_batch_size": 16,
        "num_epochs": 50,
        "warmup_steps": 2000,
        "log_step_interval": 10,
        "save_step_interval": 10,
        "eval_step_interval": 10,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "beta1": 0.9,
        "beta2": 0.999,
        "grad_clip": 1.0,
        "decay_lr": True,
        "min_lr": 3e-5,
        "loss_multiplier": 256
    }
}


def build_parameter_dict():
    parameter_dict = {
        "debug": PARAMETER_DICT["debug"],
        "booru_url": PARAMETER_DICT["booru_url"],
        "file_root": PARAMETER_DICT["file_root"],
        "deeplake_file_template": PARAMETER_DICT["deeplake_file_template"],
        "scraping": PARAMETER_DICT["scraping"],
        "dataset": PARAMETER_DICT["dataset"],
        "training": PARAMETER_DICT["training"]
    }

    file_root = PARAMETER_DICT["file_root"]
    for key, value in PARAMETER_DICT["files"].items():
        parameter_dict[key] = f"{file_root}/{value}"
    
    parameter_dict["training"]["tpu"] = \
        True if os.environ["TPU_TRAINING"] == "true" else False
        
    return parameter_dict
