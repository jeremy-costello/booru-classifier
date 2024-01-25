import string
import aiohttp
import asyncio
import sqlite3
import itertools
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm, tqdm_asyncio

from parameters import build_parameter_dict


MAX_RETRIES = 10
DEBUG = True


parameter_dict = build_parameter_dict()

booru_url = parameter_dict["booru_url"]
database_file = parameter_dict["database_file"]
search_dict = dict()


async def main():
    await get_all_tags()


def create_tables(cursor):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tag_types (
            type TEXT PRIMARY KEY
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tags (
            name TEXT PRIMARY KEY,
            count INTEGER,
            type TEXT
        )
    """)


async def get_soup(tag_start, session):
    url = f"{booru_url}/index.php?page=tags&s=list&tags={tag_start}*&sort=asc&order_by=tag"
    for retry in itertools.count():
        try:
            async with session.get(url) as r:
                if r.status == 200:
                    soup = BeautifulSoup(await r.text(), "html.parser")
                    return soup
                else:
                    continue
        except aiohttp.client_exceptions.ClientConnectorError:
            pass
        except asyncio.TimeoutError:
            pass
        if retry >= MAX_RETRIES:
            raise TimeoutError("Max retries exceeded.")


async def get_tags_from_tag_start(tag_start, depth, session):
    soup = await get_soup(tag_start, session)
    table = soup.find("table", {"class": "highlightable"})
    tag_tuples = []
    # skip header row
    for tr in table.find_all("tr")[1:]:
        tds = tr.find_all("td")
        count = tds[0].text
        if count:
            count = int(count)
            if count > 0:
                name = tds[1].span.a.text
                type_ = tds[2].text.split(" ")[0].strip(",")
                tag_tuples.append((name, count, type_))
            else:
                return None
        else:
            search_dict[depth].append(tag_start)
            return None
    return tag_tuples


async def get_tags_from_tag_start_list(search_tags, depth):
    async with aiohttp.ClientSession() as session:
        tasks = [get_tags_from_tag_start(tag_start, depth, session) for tag_start in search_tags]
        results = await tqdm_asyncio.gather(*tasks)
    return results


def update_database(results, cursor):
    for tag_tuples in tqdm(results, leave=False):
        if tag_tuples is not None:
            for name, count, type_ in tag_tuples:
                cursor.execute("SELECT EXISTS (SELECT 1 FROM tags WHERE name = ?)", (name,))
                name_exists = cursor.fetchone()[0]
                if not name_exists:
                    cursor.execute("INSERT INTO tags (name, count, type) VALUES (?, ?, ?)", (name, count, type_))
                    cursor.execute("SELECT EXISTS (SELECT 1 FROM tag_types WHERE type = ?)", (type_,))
                    file_extension_exists = cursor.fetchone()[0]
                    if not file_extension_exists:
                        cursor.execute(f"INSERT INTO tag_types (type) VALUES (?)", (type_,))


async def get_all_tags():
    remove_characters = ["*", "\n", "\r", "\t"]

    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()
    
    create_tables(cursor)
    conn.commit()

    alphabet = string.printable
    for character in remove_characters:
        alphabet = alphabet.replace(character, "")
    alphabet = list(set(alphabet.lower()))

    depth = 0
    search_dict[depth] = [""]
    
    search_tags = alphabet.copy()
    while search_dict[depth]:
        depth += 1
        search_dict[depth] = []

        search_tags = []
        for tag_root in search_dict[depth - 1]:
            search_tags.extend([tag_root + letter for letter in alphabet])
        results = await get_tags_from_tag_start_list(search_tags, depth)
        
        update_database(results, cursor)
        conn.commit()
        
        if DEBUG and depth == 2:
            break
        
    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
