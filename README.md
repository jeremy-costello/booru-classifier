# booru-classifier
Download tagged images from any booru with similar structure to [safebooru](https://safebooru.org/).

# howto
## docker
### scraping
- rename ```.env.example``` to ```.env```
  - input ```HOST_DATA_PATH```. this is where persistent data will be saved.
- run the docker compose with ```docker-compose -f dc-scraping.yml up``` (may need ```sudo```)
- attach to the docker compose with ```docker-compose -f dc-scraping.yml exec booru-scraping /bin/bash``` (may need ```sudo```)
- run ```mv parameters.py ./params``` to move the parameters file onto the persistent volume. can edit this file if required.
- run ```scrape_tags_async.py``` to scrape all tags from the booru
- run ```scrape_posts_async.py``` to scrape the first ~2000 pages of posts for each tag (if applicable)
- run ```scrape_images_async.py``` to scrape images from posts
- run ```downloaded_tags.py``` to get tags from downloaded images
- run ```dataset_skeleton.py``` to get tag data jsons and create Dask DF of image data
- run ```dataset_final_deeplake.py``` to store image and tag arrays in a [deeplake](https://github.com/activeloopai/deeplake) dataframe

### training
- rename ```.env.example``` to ```.env```
  - input ```HOST_DATA_PATH```. this is where persistent data will be saved.
  - input ```WANDB_API_KEY```. this is your [Weights & Biases](https://wandb.ai/site) api key.
  - input ```TPU_TRAINING```. set to "true" if training with TPU. hasn't been tested.
- run the docker compose with ```docker-compose -f dc-training.yml up``` (may need ```sudo```)
- attach to the docker compose with ```docker-compose -f dc-training.yml exec booru-training /bin/bash``` (may need ```sudo``` and/or ```-u root```)
- run mv parameters.py ./params to move the parameters file onto the persistent volume. can edit this file if required
- run ```reader.py``` to test the pytorch dataset is working (saves a sample on the persistent volume as ```plotted_images.png```)
- run ```trainer.py```to train a model

## aws ec2
### scraping
- same as docker
- upload ```train_lake```, ```valid_lake```, ```dataset_statistics.json```, and ```tag_indices.json``` into a S3 Bucket

### training
- attach to an EC2 instance
  - make sure it has an IAM role with the ```AmazonS3FullAccess``` permission.
- run ```wget https://raw.githubusercontent.com/jeremy-costello/booru-classifier/main/aws_setup.sh```
- run ```chmod +x aws_setup.sh```
- run ```wget https://raw.githubusercontent.com/jeremy-costello/booru-classifier/main/.env.example```
- run ```mv .env.example .env```
  - input ```WANDB_API_KEY```. this is your [Weights & Biases](https://wandb.ai/site) api key.
  - input ```TPU_TRAINING```. set to "true" if training with TPU. hasn't been tested.
  - input ```BUCKET_NAME```. this is the name of the S3 bucket your scraping data is in.
- run ```./aws_setup.sh```
- edit parameters in ```params/parameters.py``` (e.g. using vim)
- run ```python trainer.py```

# todo
- train a model
