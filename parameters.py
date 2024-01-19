parameter_dict = {
    "booru_url": "https://safebooru.org",
    "database_file": "./data/safebooru.db",
    "skeleton_parquet_file": "./data/skeleton.parquet",
    "tag_counts_json": "./data/tag_counts.json",
    "tag_indices_json": "./data/tag_indices.json",
    "dataset_statistics_json": "./data/dataset_statistics.json",
    "large_file_root": None,  # absolute file path
    # large files ↓↓↓
    "train_parquet_file": "train.parquet",
    "valid_parquet_file": "valid.parquet",
    "image_tensorstore_file": "images_store.ts",
    "tags_tensorstore_file": "tags_store.ts"
}
