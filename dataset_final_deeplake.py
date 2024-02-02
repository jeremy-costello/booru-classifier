import os
import json
import shutil
import sqlite3
import deeplake
import numpy as np
from tqdm import tqdm
import dask.dataframe as dd
from PIL import Image, UnidentifiedImageError

from data.parameters import build_parameter_dict


parameter_dict = build_parameter_dict()

database_file = parameter_dict["database_file"]
image_save_root = parameter_dict["image_save_root"]
tag_indices_file = parameter_dict["tag_indices_json"]
skeleton_parquet = parameter_dict["skeleton_parquet_file"]
deeplake_file_template = parameter_dict["deeplake_file_template"]
dataset_statistics_json = parameter_dict["dataset_statistics_json"]

validation_fraction = parameter_dict["dataset"]["validation_fraction"]
image_size = parameter_dict["dataset"]["image_size"]
channel_size = parameter_dict["dataset"]["channel_size"]
chunks = parameter_dict["dataset"]["final_chunks"]
stats_rounding = parameter_dict["dataset"]["stats_rounding"]

conn = sqlite3.connect(database_file)
cursor = conn.cursor()

cursor.execute("SELECT tag_splitter FROM meta_info")
tag_splitter_tuple = cursor.fetchall()[0]
assert len(tag_splitter_tuple) == 1
tag_splitter = tag_splitter_tuple[0]

conn.close()

with open(tag_indices_file, 'r') as f:
    tag_indices = json.load(f)

vocab_size = tag_indices["vocab_size"]


def image_and_tags(id_, tags, file_extension):   
    image_path = f"{image_save_root}/{id_}.{file_extension}"
    try:
        try:
            image = Image.open(image_path)
        except Image.DecompressionBombError:
            return None
        
        try:
            image = image.convert("RGBA")
            image = image.convert("RGB")
            image = image.resize((image_size, image_size))
        except OSError:
            return None
    except UnidentifiedImageError:
        return None
    
    image_array = np.array(image, dtype=np.uint8)
    image_float = image_array.astype(np.float32) / 255.0
    image_mean = np.mean(image_float, axis=(0, 1))
    image_std = np.std(image_float, axis=(0, 1))
    
    tags = tags.split(tag_splitter)
    indices = [tag_indices["tag2idx"][tag] for tag in tags]
    tags_array = np.zeros(vocab_size, dtype=np.int64)
    tags_array[indices] = 1
    
    output = (id_, image_array, image_mean, image_std, tags_array)
    return output


# https://stackoverflow.com/questions/41728527/how-to-apply-a-function-to-a-dask-dataframe-and-return-multiple-values
def iat_dask_wrapper(row):
    return image_and_tags(
        id_=row["id"],
        tags=row["tags"],
        file_extension=row["file_extension"]
    )


iat_schema = {
    0: int,
    1: np.zeros((image_size, image_size, channel_size), dtype=np.uint8),
    2: np.zeros(channel_size, dtype=np.float32),
    3: np.zeros(channel_size, dtype=np.float32),
    4: np.zeros(vocab_size, dtype=np.int64)
}

skeleton_url = skeleton_parquet.lstrip(".").strip("/")
full_skeleton_url = f"file://{os.getcwd()}/{skeleton_url}"
ddf = dd.read_parquet(full_skeleton_url)

total_rows = len(ddf)
validation_set = set(ddf["id"].sample(frac=validation_fraction).compute())

ddf["chunk"] = ddf["id"] % chunks

dataset_stats = {
    "count": {
        "train": 0,
        "valid": 0
    },
    "mean": 0,
    "std": 0
}

root = "data"
lakes = dict()
for split in ["train", "valid"]:
    lake_path = deeplake_file_template.format(root=root, split=split)
    lakes[split] = deeplake.empty(lake_path, overwrite=True)

with lakes["train"], lakes["valid"]:
    for split in ["train", "valid"]:
        lakes[split].create_tensor("images", htype="image.rgb", sample_compression="jpg")
        lakes[split].create_tensor("tags", htype="class_label")
    
    for chunk in tqdm(range(chunks)):
        chunked_ddf = ddf[ddf["chunk"] == chunk]
        chunked_ddf = chunked_ddf.drop(columns=["chunk"])
        chunked_ddf = chunked_ddf.apply(iat_dask_wrapper, axis=1, result_type="expand", meta=iat_schema)
        chunked_ddf = chunked_ddf.dropna()
        chunked_ddf = chunked_ddf.reset_index(drop=True)
        chunked_ddf.columns = ["id", "image_array", "image_mean", "image_std", "tags_array"]
        
        approximate_total = total_rows // chunks
        for index, row in tqdm(chunked_ddf.iterrows(), total=approximate_total, leave=False):
            if row.id not in validation_set:
                split = "train"
                dataset_stats["mean"] += row.image_mean
                dataset_stats["std"] += row.image_std
            else:
                split = "valid"

            lakes[split].append(row.image_array)
            lakes[split].append(row.tags_array)
                            
            dataset_stats["count"][split] += 1

print(dataset_stats["count"])

for stat in ["mean", "std"]:
    dataset_stats[stat] = [round(stat, stats_rounding) for stat in
                           (dataset_stats[stat] / dataset_stats["count"]["train"]).tolist()]

with open(dataset_statistics_json, "w") as f:
    json.dump(dataset_stats, f, indent=4)
