# Shishir Tandale
# Bootstrapped from:
# https://medium.com/datadriveninvestor/generative-adversarial-network-gan-using-keras-ce1c05cfdfd3

import numpy as np
# easily draw examples
import pandas as pd
import matplotlib.pyplot as plt
# progress bar
from tqdm import tqdm
# keras will be using tensorflow backend
import keras
import keras.backend as K
from keras.layers import Dense, Dropout, Input, Activation, BatchNormalization, Lambda, Conv2D, Conv2DTranspose, Reshape, Flatten
from keras.models import Model, Sequential
from keras.initializers import RandomNormal
from keras.applications.vgg16 import VGG16
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import adam
from keras.preprocessing.image import ImageDataGenerator

dataset = "pokemon"
generator_input_shape = (160,)
image_shape = (64, 64, 3)

def plot_generated_images(epoch, generated_images, dim=(5,5), figsize=(10,10)):
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'output/{dataset}_gan_e{epoch}.png')

def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)

def gradient_penalty_loss(averaged_samples):
    def _gpl(y_true, y_pred):
        gradients = K.gradients(y_pred, averaged_samples)[0]
        gradients_sqr = K.square(gradients)
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        gradient_penalty = 10*K.square(gradient_l2_norm - 1)
        return K.mean(gradient_penalty)
    return _gpl

def create_generator():
    generator = Sequential()
    generator.add(Dense(4*4*10, input_shape=generator_input_shape))
    generator.add(Reshape((4,4,10)))
    generator.add(Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu', kernel_initializer=RandomNormal(stddev=0.02)))
    #generator.add(BatchNormalization())
    generator.add(Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu', kernel_initializer=RandomNormal(stddev=0.02)))
    #generator.add(BatchNormalization())
    generator.add(Conv2DTranspose(32, (5, 5), strides=2, padding='same', activation='relu', kernel_initializer=RandomNormal(stddev=0.02)))
    #generator.add(BatchNormalization())
    generator.add(Conv2DTranspose(3, (5, 5), strides=2, padding='same', activation='tanh', kernel_initializer=RandomNormal(stddev=0.02)))
    return generator

def create_critic():
    critic = Sequential()
    critic.add(Conv2D(32, (5, 5), strides=2, kernel_initializer=RandomNormal(stddev=0.02)))
    critic.add(LeakyReLU(0.2))
    critic.add(Dropout(0.3))
    critic.add(Conv2D(32, (5, 5), strides=2, kernel_initializer=RandomNormal(stddev=0.02)))
    critic.add(LeakyReLU(0.2))
    critic.add(Dropout(0.3))
    critic.add(Conv2D(32, (3, 3), strides=2, kernel_initializer=RandomNormal(stddev=0.02)))
    critic.add(LeakyReLU(0.2))
    critic.add(Dropout(0.3))
    critic.add(Conv2D(3, (3, 3), strides=2, kernel_initializer=RandomNormal(stddev=0.02)))
    critic.add(Flatten())
    critic.add(Dense(units=1, kernel_initializer=RandomNormal(stddev=0.02)))
    return critic

def training(epochs=1, batch_size=128):
    image_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
    )
    image_generator = image_datagen.flow_from_directory(
        f'./data/{dataset}',
        class_mode=None,
        target_size=(image_shape[0], image_shape[1]),
        batch_size=batch_size)
    num_critic_iter = 1
    # build model
    generator = create_generator()
    critic = create_critic()
    critic.trainable = False

    generator_input = Input(shape=generator_input_shape)
    generator_layers = generator(generator_input)
    generator_output = critic(generator_layers)

    generator_model = Model(
        inputs=[generator_input],
        outputs=[generator_output]
    )
    generator_model.compile(
        loss=wasserstein_loss,
        optimizer=adam(lr=0.0001, beta_1=0.5, beta_2=0.9)
    )

    critic.trainable = True
    generator_predictive_model = Model(
        inputs=[generator_input],
        outputs=[generator_layers]
    )
    generator_predictive_model.compile(
        loss=perceptual_loss,
        optimizer=adam(lr=0.0001, beta_1=0.5, beta_2=0.9)
    )

    real_samples_input = Input(shape=image_shape)
    generator_input_for_critic = Input(shape=generator_input_shape)
    generated_samples_for_critic = generator(generator_input_for_critic)
    random_weights = K.random_uniform((batch_size, 1, 1, 1))
    averaged_samples = Lambda(
        lambda t: random_weights*t[0] + (1-random_weights)*t[1]
    )(inputs=[real_samples_input, generated_samples_for_critic])
    critic_output_avg_samples = critic(averaged_samples)
    critic_output_from_generator = critic(generated_samples_for_critic)
    critic_output_from_real = critic(real_samples_input)

    critic_model = Model(
        inputs=[real_samples_input, generator_input_for_critic],
        outputs=[critic_output_from_real, critic_output_from_generator, critic_output_avg_samples]
    )
    critic_model.compile(loss=[
        wasserstein_loss,
        wasserstein_loss,
        gradient_penalty_loss(averaged_samples)
    ], optimizer=adam(lr=0.0001, beta_1=0.5, beta_2=0.9))

    # loop epochs
    for e in range(1, epochs+1):
        print(f'Epoch {e}')
        image_generator = image_datagen.flow_from_directory(
            f'./data/{dataset}',
            class_mode=None,
            target_size=(image_shape[0], image_shape[1]),
            batch_size=batch_size)
        num_batches = int(817/batch_size)

        acc_loss, wr, wf, wa = 0.0, 0.0, 0.0, 0.0
        # use tqdm as a progress meter
        for _ in tqdm(range(int(num_batches/num_critic_iter))):
            # allow training discriminator multiple times before engaging gan
            for _ in range(num_critic_iter):
                # generate a random normal vector with 100 dimensions
                noise = np.random.normal(-1,1, [batch_size, generator_input_shape[0]])
                # fake_images = generator.predict(noise)
                real_images = (next(image_generator)*2)-1
                # group our images together so we can run it at once
                loss, a, b, c = critic_model.train_on_batch(
                    [real_images, noise],
                    [np.ones(batch_size), -np.ones(batch_size), np.zeros(batch_size)]
                )
                acc_loss += loss
                (wr, wf, wa) = (wr+a, wf+b, wa+c)
            # train the generator using the gan
            # generate a random normal vector with 100 dimensions
            noise = np.random.normal(-1,1, [batch_size, generator_input_shape[0]])
            sample = generator_model.predict(noise)
            generator_model.train_on_batch(noise, -np.ones(batch_size))
        # regularize losses
        acc_loss /= num_critic_iter*int(num_batches/num_critic_iter)
        wr /= num_critic_iter*int(num_batches/num_critic_iter)
        wf /= num_critic_iter*int(num_batches/num_critic_iter)
        wa /= num_critic_iter*int(num_batches/num_critic_iter)
        # occasionally output examples
        if e % 10 == 0:
            print(f'loss:{acc_loss} real:{wr} fake:{wf} avg:{wa}')
            # generate images
            test_batch_size = 25
            noise = np.random.normal(-1,1, [test_batch_size, generator_input_shape[0]])
            generated_images = (generator_predictive_model.predict(noise)+1)/2
            plot_generated_images(e, generated_images)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=64)
    args = parser.parse_args()
    training(
        args.epochs,
        args.batchsize
    )
