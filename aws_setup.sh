source ".env"
cd ~
mkdir training
sudo yum remove awscli -y
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
wget https://s3.amazonaws.com/mountpoint-s3-release/latest/x86_64/mount-s3.rpm
sudo yum install ./mount-s3.rpm -y
rm awscliv2.zip
rm -rf aws
rm mount-s3.rpm
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
eval "$(./bin/micromamba shell hook -s posix)"
./bin/micromamba shell init -s bash -p ~/micromamba
micromamba create -n training
sudo yum install git -y
git clone https://github.com/jeremy-costello/booru-classifier.git
cd booru-classifier/training
micromamba activate training
micromamba install -f conda-lock.yml -y
python -m pip install deeplake==3.8.*
cp *.py ~/training
cp -r ../params ~/training
cd ~/training
mkdir mount
mount-s3 $BUCKET_NAME mount
export WANDB_API_KEY=$WANDB_API_KEY
export TPU_TRAINING=$TPU_TRAINING
