# Keras GAN
Shishir Tandale

## Description
Implementation of a Generative Adversarial Network (GAN) architecture using
Keras. The design was bootstrapped off of [this excellent Medium article]
and was redesigned to work with higher resolution, full color images in order to
work with [this Pokémon dataset]. This dataset might still need additional
processing in order for it to work for my purposes, but it's an excellent start.
Currently, the dataset is completely unlabeled with our Critic aiming to determine
simply whether or not an image belongs to a given domain. The intention is to stage
this algorithm on test datasets and be able to scale it up to work with arbitrary
image domains.

In order to support this massive increase in data complexity
compared to MNIST, the design was updated to reflect this [compilation of best practices].
I also changed the Critic to use [Wasserstein distance using this article].
Using Wasserstein distance is useful since it allows our Critic to provide more
useful feedback to our Generator. WGAN also allows us to process discrete
input domains such as text in addition to continuous domains like image data.
[WGAN-GP] improves vanilla WGAN by using a Gradient Penalty instead of simply
clipping gradients. This has an effect of reducing mode collapse. The specific
implementation I went for was inspired by [this keras-contrib example].

For future work, [PacGAN] stands out as an excellent idea to improve algorithm
stability over increasingly sparse datasets.

[this excellent Medium article]: https://medium.com/datadriveninvestor/generative-adversarial-network-gan-using-keras-ce1c05cfdfd3
[this Pokémon dataset]: https://www.kaggle.com/kvpratama/pokemon-images-dataset
[compilation of best practices]: https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/
[Wasserstein distance using this article]: https://medium.com/@jonathan_hui/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490
[WGAN-GP]: https://arxiv.org/abs/1704.00028
[this keras-contrib example]: https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
[PacGAN]: http://swoh.web.engr.illinois.edu/pacgan.html

## Results

This project is still a work in progress and currently does not work consistently
on its target dataset. It works well on much simpler data like MNIST, but
experiences a huge degree of mode collapse and divergence of the Generator
and Critic within 60 to 100 epochs.

## Usage

To run the current implementation, make sure you have a recent version of
Python 3.x (I recommend grabbing the latest [Anaconda3]) as well as [Keras]
and [TensorFlow] as its backing library. Make sure your datasets are configured
properly. Currently, this project expects a dataset located at `data/pokemon`.
It also assumes the existence of an `output` in which it will occasionally write
output images.

`python main.py --batchsize 32 --epochs 60`

There are two optional arguments:
* `batchsize`: Defaults to 64
* `epochs`: Defaults to 100

[Anaconda3]: https://www.anaconda.com/distribution/
[Keras]: https://keras.io
[TensorFlow]: https://www.tensorflow.org/
