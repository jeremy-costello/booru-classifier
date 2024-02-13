import os
import math
import asyncio
import aiohttp
import sqlite3
import aiofiles
import itertools
from pathlib import Path
from tqdm.asyncio import tqdm, tqdm_asyncio

from params.parameters import build_parameter_dict


GOOD_EXTENSIONS = ["jpeg", "jpg", "png"]


parameter_dict = build_parameter_dict()

debug = parameter_dict["debug"]
database_file = parameter_dict["database_file"]
image_save_root = parameter_dict["image_save_root"]
batch_size = parameter_dict["scraping"]["image_batch_size"]
max_retries = parameter_dict["scraping"]["max_retries"]
download_type = parameter_dict["scraping"]["download_type"]


def create_tables(cursor):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY,
            file_url TEXT,
            downloaded BOOLEAN,
            extension TEXT,
            status_code INTEGER,
            timeout BOOLEAN,
            size_bytes INTEGER
        )
    """)


async def main():
    await download_all_images()


async def download_image(post_id, file_url, file_extension, save_root, session):   
    status_code = None
    timeout = False
    if file_extension not in GOOD_EXTENSIONS:
        downloaded = False
    else:
        save_path = save_root.joinpath(f"{post_id}.{file_extension}")

        for retry in itertools.count():
            try:
                async with session.get(file_url) as r:
                    if r.status == 200:
                        async with aiofiles.open(save_path, "wb") as f:
                            await f.write(await r.read())
                            downloaded = True
                            break
                    else:
                        downloaded = False
                        status_code = r.status
                        # not sure whether to keep this break or not?
                        # elif for other status codes?
                        break
            except RuntimeError:
                downloaded = False
            except asyncio.TimeoutError:
                downloaded = False
            except aiohttp.client_exceptions.ClientOSError:
                downloaded = False
            except aiohttp.client_exceptions.ClientConnectorError:
                downloaded = False
            except aiohttp.client_exceptions.ClientPayloadError:
                downloaded = False
            except aiohttp.client_exceptions.ServerDisconnectedError:
                downloaded = False
            except OSError:
                downloaded = False
            if retry >= max_retries:
                timeout = True
                # break or raise TimeoutError?
                break
    
    image_dict = {
        "id": int(post_id),
        "file_url": file_url,
        "downloaded": downloaded,
        "extension": file_extension,
        "status_code": status_code,
        "timeout": timeout,
        "size_bytes": os.stat(save_path).st_size if downloaded else None
    }
    return image_dict


def download_type_else():
    raise ValueError("Invalid value: to_download must be 'posts' or 'timeouts'.")


def update_database(results, cursor, download_type):
    for image_dict in tqdm(results, leave=False):
        image_id = image_dict["id"]

        if download_type == "posts":
            cursor.execute("SELECT EXISTS (SELECT 1 FROM images WHERE id = ?)", (image_id,))
            image_id_exists = cursor.fetchone()[0]
        elif download_type == "timeouts":
            image_id_exists = False
        else:
            download_type_else()
        
        if not image_id_exists:
            if download_type == "posts":
                columns = ", ".join(image_dict.keys())
                question_marks = ", ".join("?" for _ in image_dict.values())
                query = f"INSERT INTO images ({columns}) VALUES ({question_marks})"
                cursor.execute(query, tuple(image_dict.values()))
            elif download_type == "timeouts":
                set_values = ", ".join(f"{column} = ?" for column in image_dict.keys())
                query = f"UPDATE images SET {set_values} WHERE id = ?"
                cursor.execute(query, tuple(list(image_dict.values()) + [image_id]))
            else:
                download_type_else()


async def download_image_batch(post_id_batch, file_url_batch, file_extension_batch, save_root):
    async with aiohttp.ClientSession() as session:
        tasks = [download_image(post_id, file_url, file_extension, save_root, session)
                 for post_id, file_url, file_extension in zip(post_id_batch, file_url_batch, file_extension_batch)]
        results = await tqdm_asyncio.gather(*tasks, leave=False)
    return results


async def download_all_images():
    save_root = Path(image_save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    create_tables(cursor)
    conn.commit()

    cursor.execute("SELECT id FROM images")
    image_set = set([row[0] for row in cursor.fetchall()])

    if debug:
        num_batches = 1
    else:
        if download_type == "posts":
            cursor.execute("SELECT COUNT(*) FROM posts")
        elif download_type == "timeouts":
            cursor.execute(f"SELECT COUNT(*) FROM images WHERE timeout = 1")
        else:
            download_type_else()

        num_batches = math.ceil(cursor.fetchone()[0] / batch_size)
    
    for batch in tqdm(range(num_batches)):
        if download_type == "posts":
            query = "SELECT id, file_url, file_extension FROM posts LIMIT ? OFFSET ?"
        elif download_type == "timeouts":
            query = "SELECT id, file_url, extension FROM images WHERE timeout = 1 LIMIT ? OFFSET ?"
        else:
            download_type_else()
        
        cursor.execute(query, (batch_size, batch * batch_size))
        data_list = cursor.fetchall()
        post_id_batch = [post[0] for post in data_list]

        skip_batch = False
        if download_type == "posts":
            skip_batch = set(post_id_batch).issubset(image_set)
        elif download_type == "timeouts":
            pass
        else:
            download_type_else()

        if not skip_batch:
            file_url_batch = [post[1] for post in data_list]
            file_extension_batch = [post[2] for post in data_list]

            # pass image_set and download_type?
            results = await download_image_batch(post_id_batch, file_url_batch, file_extension_batch, save_root)
            update_database(results, cursor, download_type)
            conn.commit()


if __name__ == "__main__":
    asyncio.run(main())
