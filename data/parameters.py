PARAMETER_DICT = {
    "booru_url": "https://safebooru.org",
    "file_root": "./data",
    "tensorstore_file_template": "{root}/{data_class}_store_{split}.ts",
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
        "debug": True,
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
        "load_tensorstores": True
    },
    "training": {
        "matmul_precision": None,
        "num_workers": 0,
        "num_epochs": 10,
        "learning_rate": 3e-4,
        "batch_size": 16
    }
}


def build_parameter_dict():
    parameter_dict = {
        "booru_url": PARAMETER_DICT["booru_url"],
        "tensorstore_file_template": PARAMETER_DICT["tensorstore_file_template"],
        "scraping": PARAMETER_DICT["scraping"],
        "dataset": PARAMETER_DICT["dataset"],
        "training": PARAMETER_DICT["training"]
    }

    file_root = PARAMETER_DICT["file_root"]
    for key, value in PARAMETER_DICT["files"].items():
        parameter_dict[key] = f"{file_root}/{value}"
        
    return parameter_dict
