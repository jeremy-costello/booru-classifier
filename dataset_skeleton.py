import os
import json
import random
import sqlite3
from tqdm import tqdm
import dask.dataframe as dd

from parameters import build_parameter_dict


PARTITIONS = 32
MIN_COUNT = 50
COUNT_LIMIT = 5000
LIMITED_TYPES = ["artist", "copyright", "model"]
IGNORE_TAGS = [
    # alphabetical
    "",
    "&",
    "-",
    # nonetype
    # metadata
    "animated",
    "animated_gif",
    "copyright_request",
    "duplicate",
    "gif",
    "me",
    "model_name",
    "model_reqeust",
    "music",
    "name_request",
    "sound",
    "source_request",
    "sourced",
    "tagme",
    "translation_request",
    "webm"
]


parameter_dict = build_parameter_dict()

database_file = parameter_dict["database_file"]
tag_indices_file = parameter_dict["tag_indices_json"]
tag_counts_file = parameter_dict["tag_counts_json"]
skeleton_parquet = parameter_dict["skeleton_parquet_file"]

conn = sqlite3.connect(database_file)
cursor = conn.cursor()

cursor.execute("SELECT tag_splitter FROM meta_info")
tag_splitter_tuple = cursor.fetchall()[0]
assert len(tag_splitter_tuple) == 1
tag_splitter = tag_splitter_tuple[0]

cursor.execute("SELECT * FROM downloaded_tags")

tag_list = cursor.fetchall()
name_list = [row[0] for row in tag_list]
count_list = [row[1] for row in tag_list]
type_list = [row[2] for row in tag_list]

tag_set = set()
limit_dict = dict()
for name, count, type_ in zip(name_list, count_list, type_list):
    if name not in IGNORE_TAGS and count >= MIN_COUNT:
        tag_set.add(name)
        if type_ in LIMITED_TYPES and count > COUNT_LIMIT:
            limit_dict[name] = {
                "ids": [],
                "seen": 0
            }

cursor.execute("SELECT id FROM images WHERE downloaded = 1")
id_list = [row[0] for row in cursor.fetchall()]

dataset_dict = dict()
ids_to_remove_set = set()
for id_ in tqdm(id_list):
    cursor.execute("SELECT tags, file_extension from posts WHERE id = ?", (id_,))
    tags_string, file_extension = cursor.fetchall()[0]
    full_tag_list = tags_string.split(tag_splitter)

    to_add = True
    final_tag_list = []
    for tag in full_tag_list:
        if tag in tag_set:
            final_tag_list.append(tag)
        if tag in limit_dict.keys():
            # reservoir sampling
            if limit_dict[tag]["seen"] < COUNT_LIMIT:
                limit_dict[tag]["ids"].append(id_)
                limit_dict[tag]["seen"] += 1
            else:
                idx = random.randint(0, limit_dict[tag]["seen"])
                limit_dict[tag]["seen"] += 1
                if idx < COUNT_LIMIT:
                    ids_to_remove_set.add(limit_dict[tag]["ids"][idx])
                    limit_dict[tag]["ids"][idx] = id_
                else:
                    to_add = False
                    break
    if to_add:
        dataset_dict[id_] = {
            "tags": tag_splitter.join(final_tag_list),
            "file_extension": file_extension
        }

conn.close()

for key in (set(dataset_dict.keys()) & ids_to_remove_set):
    del dataset_dict[key]

tag_counts = dict()
for info_dict in dataset_dict.values():
    tags, _ = info_dict.values()
    for tag in tags:
        if tag in tag_counts.keys():
            tag_counts[tag] += 1
        else:
            tag_counts[tag] = 1

tag_indices = {
    "vocab_size": 0,
    "tag2idx": dict(),
    "idx2tag": dict()
}
for tag, count in tag_counts.items():
    if count >= MIN_COUNT:
        idx = tag_indices["vocab_size"]
        tag_indices["tag2idx"][tag] = idx
        tag_indices["idx2tag"][idx] = tag
        tag_indices["vocab_size"] += 1

ids_to_remove_set = set()
for id_, info_dict in dataset_dict.items():
    tags = info_dict["tags"]
    pruned_tags = [tag for tag in tags if tag in tag_indices["tag2idx"].keys()]
    if pruned_tags:
        dataset_dict[id_]["tags"] = pruned_tags
    else:
        ids_to_remove_set.add(id_)

for key in ids_to_remove_set:
    del dataset_dict[key]

with open(tag_indices_file, "w") as f:
    json.dump(tag_indices, f, indent=4)

tag_counts = {key: value for key, value in
              sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
              if value >= MIN_COUNT}
with open(tag_counts_file, "w") as f:
    json.dump(tag_counts, f, indent=4)

# gotta fix this
ddf = dd.from_dict(dataset_dict, npartitions=PARTITIONS)

skeleton_url = skeleton_parquet.lstrip(".").strip("/")
full_skeleton_url = f"file://{os.getcwd()}/{skeleton_url}"
ddf.to_parquet(full_skeleton_url, write_options={'compression': 'snappy'}).compute()

# input: id -> image array || output: tag strings -> n-hot tags
print(len(dataset_dict))
print(len(ddf))

# tag -> index || index -> tag
print(tag_indices["vocab_size"])
