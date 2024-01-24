# booru-classifier
Download tagged images from any booru with similar structure to [safebooru](https://safebooru.org/).

# howto
- fill in common parameters in ```parameters.py``` (some ```*.py``` files also have inputs at the top)
- build the docker image with ```docker build -t booru-classifier .```
- run the docker container with ```docker run -it booru-classifier```
- run ```scrape_tags_async.py``` to scrape all tags from the booru.
- run ```scrape_posts_async.py``` to scrape the first ~2000 pages of posts for each tag (if applicable).
- run ```scrape_images_async.py``` to scrape images from posts.
- run ```downloaded_tags.py``` to get tags from downloaded images.
- run ```dataset_skeleton.py``` to get tag data jsons and create Spark DF of image data.
- run ```dataset_final_tensorstore.py``` to store image arrays in a [tensorstore](https://google.github.io/tensorstore/). (WIP)

# todo
- tensorstore doesn't work on WSL (lock file issue). add docker or test on actual Ubuntu
- update pytorch dataset
- finish training code
- train a model
