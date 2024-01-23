import math
import asyncio
import aiohttp
import sqlite3
import itertools
import xml.etree.ElementTree as ET
from datetime import datetime
from tqdm.asyncio import tqdm, tqdm_asyncio

from parameters import build_parameter_dict


YEAR_CUTOFF = 2024
TAG_SPLITTER = "||"
MIN_BATCH_SIZE = 8192
MAX_RETRIES = 10
IGNORE_KEYS = []
INT_KEYS = ["height", "score", "sample_width", "sample_height", "id",
            "width", "change", "creator_id", "preview_width", "preview_height"]
MAP_DICT = {
    "": None,
    "true": True,
    "false": False
}

parameter_dict = build_parameter_dict()

booru_url = parameter_dict["booru_url"]
database_file = parameter_dict["database_file"]


def create_tables(cursor):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS meta_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            year_cutoff INTEGER,
            tag_splitter TEXT
        )
    """)
    cursor.execute("SELECT COUNT(*) FROM meta_info")
    count = cursor.fetchone()[0]

    if count == 0:
        cursor.execute("INSERT INTO meta_info (year_cutoff, tag_splitter) VALUES (?, ?)", (YEAR_CUTOFF, TAG_SPLITTER))
    elif count == 1:
        cursor.execute("UPDATE meta_info SET year_cutoff = ?, tag_splitter = ? WHERE id = 1", (YEAR_CUTOFF, TAG_SPLITTER))
    else:
        raise ValueError(f"Table 'meta_info' should only have 1 entry, but it has {count}!")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS file_extensions (
            file_extension TEXT PRIMARY KEY
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS completed_tags (
            tag_name TEXT PRIMARY KEY
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY,
            width INTEGER,
            height INTEGER,
            score INTEGER,
            file_url TEXT,
            parent_id INTEGER,
            sample_url TEXT,
            sample_width INTEGER,
            sample_height INTEGER,
            preview_url TEXT,
            rating TEXT,
            tags TEXT,
            change INTEGER,
            md5 TEXT,
            creator_id INTEGER,
            has_children BOOLEAN,
            created_at TEXT,
            status TEXT,
            source TEXT,
            has_notes BOOLEAN,
            has_comments BOOLEAN,
            preview_width INTEGER,
            preview_height INTEGER,
            file_extension TEXT
        )
    """)


async def main():
    await get_all_posts()


async def get_xml_tree(tag, page, session):
    url = f"{booru_url}/index.php?page=dapi&s=post&q=index&tags={tag}&pid={page}&limit=100"

    for retry in itertools.count():
        try:
            async with session.get(url) as r:
                if r.status == 200:
                    xml_tree = ET.fromstring(await r.text())
                else:
                    continue
            if len(xml_tree) > 0:
                return xml_tree
            else:
                return None
        except ET.ParseError:
            pass
        except aiohttp.client_exceptions.ClientConnectorError:
            pass
        except aiohttp.client_exceptions.ClientOSError:
            pass
        except aiohttp.client_exceptions.ClientPayloadError:
            pass
        except asyncio.TimeoutError:
            pass
        if retry >= MAX_RETRIES:
            raise TimeoutError("Max retries exceeded.")


async def get_posts_from_tag_page(tag, page, post_id_set, session):
    xml_tree = await get_xml_tree(tag, page, session)
    if xml_tree is None:
        return None
    else:
        post_dicts = []
        for element in xml_tree:
            attrib = element.attrib
            post_id = attrib["id"]
            if post_id not in post_id_set:
                post_dict = {
                    "id": post_id
                }
                input_date = attrib["created_at"]
                dt_object = datetime.strptime(input_date, '%a %b %d %H:%M:%S %z %Y')
                gmt_time = dt_object - dt_object.utcoffset()
                if gmt_time.year < YEAR_CUTOFF:
                    for key, value in attrib.items():
                        if key in IGNORE_KEYS:
                            continue
                        elif key in INT_KEYS and value != "":
                            post_dict[key] = int(value)
                        elif key == "tags":
                            post_dict[key] = value.strip().replace(" ", TAG_SPLITTER)
                        elif value in MAP_DICT.keys():
                            post_dict[key] = MAP_DICT[value]
                        else:
                            post_dict[key] = value

                    created_at = gmt_time.strftime('%Y-%m-%d %H:%M:%S')
                    post_dict["created_at"] = created_at

                    file_extension = post_dict["file_url"].split(".")[-1]
                    post_dict["file_extension"] = file_extension
                    
                    post_dicts.append(post_dict)
        return post_dicts


def update_database(results, cursor):
    for post_dict_list in tqdm(results, leave=False):
        if post_dict_list is not None:
            for post_dict in post_dict_list:
                post_id = post_dict["id"]
                cursor.execute("SELECT EXISTS (SELECT 1 FROM posts WHERE id = ?)", (post_id,))
                post_id_exists = cursor.fetchone()[0]
                if not post_id_exists:
                    file_extension = post_dict["file_extension"]
                    cursor.execute("SELECT EXISTS (SELECT 1 FROM file_extensions WHERE file_extension = ?)", (file_extension,))
                    file_extension_exists = cursor.fetchone()[0]
                    if not file_extension_exists:
                        cursor.execute("INSERT INTO file_extensions (file_extension) VALUES (?)", (file_extension,))
                    
                    columns = ", ".join(post_dict.keys())
                    question_marks = ", ".join("?" for _ in post_dict.values())

                    query = f"INSERT INTO posts ({columns}) VALUES ({question_marks})"
                    cursor.execute(query, tuple(post_dict.values()))


async def get_posts_from_tag(tag_list, page_list, post_id_set):
    async with aiohttp.ClientSession() as session:
        tasks = [get_posts_from_tag_page(tag, page, post_id_set, session) for tag, page in zip(tag_list, page_list)]
        results = await tqdm_asyncio.gather(*tasks, leave=False)
    return results

    
async def get_all_posts():
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    create_tables(cursor)
    conn.commit()

    cursor.execute("SELECT name FROM tags")
    full_tag_list = [row[0] for row in cursor.fetchall()]

    tag_list = []
    page_list = []
    for tag in tqdm(full_tag_list):
        cursor.execute("SELECT EXISTS (SELECT 1 FROM completed_tags WHERE tag_name = ?)", (tag,))
        tag_completed = cursor.fetchone()[0]
        if not tag_completed:
            cursor.execute("SELECT count FROM tags WHERE name = ?", (tag,))
            count = cursor.fetchone()[0]

            total_pages = min(math.ceil(count / 100), 2001)
            tag_list.extend([tag] * total_pages)
            page_list.extend(list(range(total_pages)))

            assert len(tag_list) == len(page_list)
            if len(tag_list) >= MIN_BATCH_SIZE:
                cursor.execute("SELECT id FROM posts")
                post_id_set = set([row[0] for row in cursor.fetchall()])
                
                results = await get_posts_from_tag(tag_list, page_list, post_id_set)
                update_database(results, cursor)

                unique_tag_list = list(set(tag_list))
                for tag in unique_tag_list:
                    cursor.execute(f"INSERT INTO completed_tags (tag_name) VALUES (?)", (tag,))
                conn.commit()

                tag_list = []
                page_list = []
    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
