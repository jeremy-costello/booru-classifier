PARAMETER_DICT = {
    "booru_url": "https://safebooru.org",
    "large_file_root": "./data",
    "large_files": {
        "database_file": "safebooru.db",
        "image_save_root": "images",
        "train_parquet_file": "train.parquet",
        "valid_parquet_file": "valid.parquet",
        "image_tensorstore_file": "images_store.ts",
        "tags_tensorstore_file": "tags_store.ts"
    },
    "small_file_root": "./data",
    "small_files": {
        "skeleton_parquet_file": "skeleton.parquet",
        "tag_counts_json": "tag_counts.json",
        "tag_indices_json": "tag_indices.json",
        "dataset_statistics_json": "dataset_statistics.json",
    }
}


def build_parameter_dict():
    parameter_dict = {
        "booru_url": PARAMETER_DICT["booru_url"]
    }

    for file_type in ["large_file", "small_file"]:
        root = PARAMETER_DICT[f"{file_type}_root"]

        for key, value in PARAMETER_DICT[f"{file_type}s"].items():
            parameter_dict[key] = f"{root}/{value}"
    
    return parameter_dict
