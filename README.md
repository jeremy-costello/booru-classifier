# booru-classifier
Download tagged images from any booru with similar structure to [safebooru](https://safebooru.org/).

# howto
- run the docker compose with ```docker-compose up``` (may need sudo)
- attach to the docker compose with ```docker-compose exec booru-classifier /bin/bash``` (may need ```sudo``` and/or ```-u root```)
- fill in common parameters in ```./data/parameters.py``` and move this file into the ```./data``` folder in the docker image
- run ```scrape_tags_async.py``` to scrape all tags from the booru.
- run ```scrape_posts_async.py``` to scrape the first ~2000 pages of posts for each tag (if applicable).
- run ```scrape_images_async.py``` to scrape images from posts.
- run ```downloaded_tags.py``` to get tags from downloaded images.
- run ```dataset_skeleton.py``` to get tag data jsons and create Dask DF of image data.
- run ```dataset_final_tensorstore.py``` to store image arrays in a [tensorstore](https://google.github.io/tensorstore/).
- run ```reader.py``` to test the pytorch dataset is working (saves a sample in ```./data/plotted_images.py```).
- run ```trainer.py``` for a basic training loop.

# todo
- tensorstore async
- finish training code
- train a model
