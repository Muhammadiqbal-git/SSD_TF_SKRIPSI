## Configure the environtment

1.install wsl
>install wsl

2.get miniconda
>wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

3.in the same yml file directory, do
>conda env create --name envname -f environment.yml

[opsional]if you want to use GPU(nvidia) to train.
4.check is nvidia driver is already installed
>nvidia-smi

5.if yes, jump to step 6. below is nvidia driver for cuda-toolkit 11.8(which is already installed in the env)
https://developer.nvidia.com/cuda-11-8-0-download-archive
> sudo apt install nvidia-cuda-toolkit

6.check to confirm installation
> nvcc -V

6.install cudnn 
>pip install nvidia-cudnn-cu11==8.6.0.163

7.check your installation
>python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU')); print(tf.test.is_built_with_cuda())"
[<list of gpu device>]
True

your done!




--------
iqbal

