PARAMETER_DICT = {
    "booru_url": "https://safebooru.org",
    "file_root": "./data",
    "files": {
        "database_file": "booru.db",
        "image_save_root": "images",
        "tensorstore_file_template": "{data_type}_store_{split}.ts",
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
    },
    "dataset": {
        "skeleton_partitions": 32,
        "min_tag_count": 50,
        "tag_count_limit": 5000,
        "validation_fraction": 0.05,
        "image_size": 224,
        "channel_size": 3,
        "final_chunks": 10,
        "stats_rounding": 4,
        "load_tensorstores": True
    },
    "training": {
        "num_epochs": 10,
        "learning_rate": 3e-4,
        "batch_size": 16
    }
}


def build_parameter_dict():
    parameter_dict = {
        "booru_url": PARAMETER_DICT["booru_url"],
        "scraping": PARAMETER_DICT["scraping"],
        "dataset": PARAMETER_DICT["dataset"],
        "training": PARAMETER_DICT["training"]
    }

    file_root = PARAMETER_DICT["file_root"]
    for key, value in PARAMETER_DICT["files"].items():
        parameter_dict[key] = f"{file_root}/{value}"
        
    parameter_dict["tensorstore_file_template"] = \
        parameter_dict["tensorstore_file_template"].lstrip(".").strip("/")
    
    return parameter_dict
