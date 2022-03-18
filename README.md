# Generative Adversarial Nets
![Image](https://www.tensorflow.org/tutorials/generative/images/gan1.png)

GANs are used for teaching a deep learning model to generate new data from that same distribution of training data. Invented by Ian Goodfellow in 2014 in the paper Generative Adversarial Nets. They are made up of two different models, a generator and a discriminator. The generator produces synthetic or fake images which look like training images. The discriminator looks at an image and the output and checks if the image is real or fake. While training, the generator generates better fake images and fools the discriminator to believe that the generated image is a real image and the discriminator tries to become better at detection and classifying whether the image is real or fake.

![Image](https://www.tensorflow.org/tutorials/generative/images/gan2.png)

# DCGAN
DCGAN is one of the popular and successful network design for GAN. It mainly composes of convolution layers without max pooling or fully connected layers. It uses convolutional stride and transposed convolution for the downsampling and the upsampling. The figure below is the network design for the generator.

![Image](https://editor.analyticsvidhya.com/uploads/2665314.png)

Here is the summary of DCGAN:
- Replace all max pooling with convolutional stride
- Use transposed convolution for upsampling.
- Eliminate fully connected layers.
- Use Batch normalization except the output layer for the generator and the input layer of the discriminator.
- Use ReLU in the generator except for the output which uses tanh.

![Image](https://www.tensorflow.org/images/gan/dcgan.gif)

