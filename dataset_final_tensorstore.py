import os
import json
import sqlite3
import numpy as np
import pandas as pd
from tqdm import tqdm
from io import BytesIO
import tensorstore as ts
import dask.dataframe as dd
from PIL import Image, UnidentifiedImageError

from parameters import build_parameter_dict


# still testing this. lock-file error with tensorstore
IMAGE_SIZE = 224
CHANNEL_SIZE = 3
CHUNKS = 10


parameter_dict = build_parameter_dict()

database_file = parameter_dict["database_file"]
image_save_root = parameter_dict["image_save_root"]
tag_indices_file = parameter_dict["tag_indices_json"]
skeleton_parquet = parameter_dict["skeleton_parquet_file"]
image_tensorstore_file = parameter_dict["image_tensorstore_file"]
tags_tensorstore_file = parameter_dict["tags_tensorstore_file"]

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
        ### REMOVE!!!
        except FileNotFoundError:
            return None
        
        try:
            image = image.convert("RGB")
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        except OSError:
            return None
    except UnidentifiedImageError:
        return None
    
    image_array = np.array(image, dtype=np.uint8)
    image_float = image_array.astype(np.float32) / 255.0
    image_mean = np.mean(image_float, axis=(0, 1)).tolist()
    image_std = np.std(image_float, axis=(0, 1)).tolist()
    
    ### FIX!!!!!
    tags = tags[1:-1].split(" ")
    tags = [tag.strip("\n").strip("'") for tag in tags]
    # tags = tags.split(tag_splitter)
    indices = [tag_indices["tag2idx"][tag] for tag in tags]
    tag_array = np.zeros(vocab_size, dtype=np.int64)
    tag_array[indices] = 1

    with BytesIO() as memfile:
        np.save(memfile, image_array)
        image_binary = bytearray(memfile.getvalue())
    
    with BytesIO() as memfile:
        np.save(memfile, tag_array)
        tag_binary = bytearray(memfile.getvalue())
    
    output = pd.Series({
        "id": id_,
        "image_binary": image_binary,
        "image_mean": image_mean,
        "image_std": image_std,
        "tag_binary": tag_binary
    })
    return output


iat_schema = {
    "id": int,
    "image_binary": object,
    "image_mean": object,
    "image_std": object,
    "tag_binary": object
}

skeleton_url = skeleton_parquet.lstrip(".").strip("/")
full_skeleton_url = f"file://{os.getcwd()}/{skeleton_url}"
ddf = dd.read_parquet(full_skeleton_url)

ddf = ddf[ddf["id"] < 100]

total_rows = len(ddf)

"""
# https://google.github.io/tensorstore/python/tutorial.html
image_dataset = ts.open({
    "driver": "zarr",
    "kvstore": {
        "driver": "file",
        "path": image_tensorstore_file
    },
    "metadata": {
        "compressor": {
            "id": "gzip"
        },
        "dtype": "|u1",
        "shape": [total_rows, IMAGE_SIZE, IMAGE_SIZE, CHANNEL_SIZE],
        "chunks": [1, IMAGE_SIZE, IMAGE_SIZE, CHANNEL_SIZE]
    },
    "create": True,
    "delete_existing": True
}).result()

tags_dataset = ts.open({
    "driver": "zarr",
    "kvstore": {
        "driver": "file",
        "path": tags_tensorstore_file
    },
    "metadata": {
        "compression": {
            "type": "gzip"
        },
        "dtype": "|i8",
        "shape": [total_rows, vocab_size],
        "chunks": [1, vocab_size]
    },
    "create": True,
    "delete_existing": True
}).result()
"""

ddf["chunk"] = ddf["id"] % np.ceil(total_rows / CHUNKS)

current_index = 0
for chunk in range(CHUNKS):
    chunked_ddf = ddf[ddf["chunk"] == chunk]
    chunked_ddf = chunked_ddf.drop(columns=["chunk"])
    chunked_ddf = chunked_ddf.apply(lambda row: image_and_tags(row["id"], row["tags"], row["file_extension"]), axis=1, meta=iat_schema)
    print(chunked_ddf.compute())
    chunked_ddf = chunked_ddf.dropna()
    
    for index, row in chunked_ddf.iterrows():
        print(index, row)
        
    

"""
df = df.orderBy(F.rand())
df = df.withColumn("index", F.monotonically_increasing_id())

df = df.withColumn("chunk", F.col("index") % math.ceil(df.count() / CHUNKS))
df = df.drop(F.col("index"))

current_index = 0
for chunk in range(CHUNKS):
    chunked_df = df.filter(df["chunk"] == chunk)
    chunked_df = chunked_df.withColumn("result", image_and_tags_spark(
        F.col("id"), F.col("tags"), F.col("file_extension")))
    chunked_df = chunked_df.selectExpr("id", "result.*")
    chunked_df = chunked_df.na.drop()

    row_list = chunked_df.collect()

    for row in row_list:
        image = np.load(BytesIO(row.image_binary))
        tags = np.load(BytesIO(row.tags_binary))

        image_dataset[current_index, :, :, :].write(image).result()
        tags_dataset[current_index, :].write(tags).result()

        current_index += 1

    break
"""