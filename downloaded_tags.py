import math
import sqlite3
from tqdm import tqdm

from data.parameters import build_parameter_dict


parameter_dict = build_parameter_dict()

database_file = parameter_dict["database_file"]
batch_size = parameter_dict["scraping"]["tag_batch_size"]

conn = sqlite3.connect(database_file)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS downloaded_tags (
        name TEXT PRIMARY KEY,
        count INTEGER,
        type TEXT
    )
""")
conn.commit()

cursor.execute("SELECT tag_splitter FROM meta_info")
tag_splitter_tuple = cursor.fetchall()[0]
assert len(tag_splitter_tuple) == 1
tag_splitter = tag_splitter_tuple[0]

cursor.execute("SELECT id FROM images WHERE downloaded = 1")
id_list = [row[0] for row in cursor.fetchall()]

tags_list = []
cursor.execute("SELECT COUNT(*) FROM posts")
num_batches = math.ceil(cursor.fetchone()[0] / batch_size)
for batch in tqdm(range(num_batches)):
    id_batch = id_list[batch * batch_size : (batch + 1) * batch_size]
    question_marks = ", ".join("?" for _ in id_batch)
    query = f"SELECT tags FROM posts WHERE id in ({question_marks})"
    cursor.execute(query, tuple(id_batch))
    tags_list.extend([row[0] for row in cursor.fetchall()])

tag_dict = dict()
for tags in tqdm(tags_list):
    for tag in tags.split(tag_splitter):
        if tag in tag_dict.keys():
            tag_dict[tag]["count"] += 1
        else:
            tag_dict[tag] = {
                "count": 1
            }
            cursor.execute("SELECT type FROM tags WHERE name = ?", (tag,))
            tag_type_list = cursor.fetchone()
            if tag_type_list is None:
                tag_type = None
            else:
                tag_type = tag_type_list[0]
            tag_dict[tag]["type"] = tag_type

for key, value in tqdm(tag_dict.items()):
    query = f"INSERT INTO downloaded_tags (name, count, type) VALUES (?, ?, ?)"
    cursor.execute(query, (key, value["count"], value["type"]))
conn.commit()
