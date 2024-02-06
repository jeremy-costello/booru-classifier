# booru-classifier
Download tagged images from any booru with similar structure to [safebooru](https://safebooru.org/).

# howto
- rename ```.env.example``` to ```.env```
- input ```HOST_DATA_PATH``` into ```.env```. this is where persistent data will be saved.
- run the docker compose with ```docker-compose up``` (may need sudo)
- attach to the docker compose with ```docker-compose exec booru-classifier /bin/bash``` (may need ```sudo``` and/or ```-u root```)
- run ```mv parameters.py ./data``` to move the parameters file onto the persistent volume. can edit this file if required.
- run ```scrape_tags_async.py``` to scrape all tags from the booru.
- run ```scrape_posts_async.py``` to scrape the first ~2000 pages of posts for each tag (if applicable).
- run ```scrape_images_async.py``` to scrape images from posts.
- run ```downloaded_tags.py``` to get tags from downloaded images.
- run ```dataset_skeleton.py``` to get tag data jsons and create Dask DF of image data.
- run ```dataset_final_deeplake.py``` to store image and tag arrays in a [deeplake](https://github.com/activeloopai/deeplake) dataframe.
- run ```reader.py``` to test the pytorch dataset is working (saves a sample on the persistent volume as ```plotted_images.png```).
- run ```trainer.py```to train a model.

# todo
- test training code
- add docker volume for training outputs
- train a model
