![image](https://github.com/AbhinavJangra29/GANS/assets/107471490/2a3b7d4b-8def-4fa9-a86c-33e25aec21e6)

<h1>Generative Adversarial Networks (GANs)</h1>

Welcome to the world of Generative Adversarial Networks (GANs)! This repository explores the fascinating field of GANs, a type of machine learning model that's great at generating new content, whether it's images, text, or more. 

<h2>What are GANs?</h2>

Let us break it down..

**Generative**: Something that can generate, create, or produce.

**Adversarial**: Involving adversaries or opponents in a conflict or competition.
So, in simple terms, GANs are a type of model or algorithm that consists of two parts – one that creates or generates content, and another that evaluates or discriminates the generated content. They are like two opponents in a game, where one is trying to create something (like images, text, etc.), and the other is trying to figure out whether what's created is real or not. This dynamic competition helps the model get better at generating content that is increasingly realistic.

So they are kind of playing a game against each other where both are trying to get better against each other.


**Types of GANs Covered**

1. **Simple GAN**
   
The basic GAN architecture involves a straightforward Generator-Discriminator setup. The Generator creates data, and the Discriminator tries to distinguish between real and generated samples.(might use linear layer)

![image](https://github.com/AbhinavJangra29/GANS/assets/107471490/5fb80de9-441c-4218-b45c-3f544d4d279a)


3. **DCGAN (Deep Convolutional GAN)**
   
DCGAN enhances the basic GAN by incorporating deep convolutional networks. This is especially powerful for image generation tasks, allowing the model to learn hierarchical features effectively.
A dcgan unlike simple gan uses convulation as an operation.

5. WGAN (Wasserstein GAN)
   
WGAN introduces a new way to train GANs by using Wasserstein distance, providing more stable training and better gradient flow. This helps overcome some of the training challenges faced by traditional GANs.
just think of wasserstien distance as the distance between two probablistic distributions and our generator is trying to minimise the difference between the actual and generated image's probablistic distribution.

![image](https://github.com/AbhinavJangra29/GANS/assets/107471490/1856c611-3943-43e4-b3fd-6b18365078ef)


6. **Pix2Pix**
   
Pix2Pix is a type of GAN used for image-to-image translation. It learns to map input images to output images in a supervised manner. For example, turning satellite images into maps or black-and-white photos into color.

![image](https://github.com/AbhinavJangra29/GANS/assets/107471490/a4f5cae2-6d02-4697-8892-4882605d08ae)


6. **CycleGAN**
CycleGAN extends GANs for unpaired image translation. It can learn to transform images from one domain to another without requiring paired examples during training. Think turning horses into zebras without needing images of the same scene in both styles.

![image](https://github.com/AbhinavJangra29/GANS/assets/107471490/35fe3334-4f78-4bf1-929b-e7a46de08613)


<h1>But how do GANs learn??</h1>

![image](https://github.com/AbhinavJangra29/GANS/assets/107471490/46001360-0ba2-4e8f-8313-595574925e5b)


In the GAN training process, the generator takes input from the latent space to create an image. This generated image is then presented to the discriminator for evaluation. The discriminator's role is to classify the generated image as either fake or real based on its quality. If the discriminator correctly identifies the image as fake, the generator adjusts its parameters accordingly during training. The discriminator strives to accurately classify every image, while the generator aims to produce images that deceive the discriminator, leading it to classify them as real. This iterative adversarial dynamic continues until the generator consistently generates images that closely resemble real data, achieving the desired outcome.

<h3>so just sit back relax and grab a pen and a page and get started</h3>










