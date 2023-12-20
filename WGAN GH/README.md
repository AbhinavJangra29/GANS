<h1>IDEA BEHIND WGAN </h1>
<p>
The W in wgans stand for wasserstein distance. what is this wasserstien distance? it can be basically described as distance between 2 probablity distributions. distance between 2 distributions? Yes when we say distance between 2 probablity distribution we mean the distance between their means. So in WGAN what we do is instead of keeping a discriminator like usual which tells us real or fake , wee use a critic that tells us how far is the generated sample distribution from the original image distriubtion(critic generates a score). Thats it, the generator tries to minimise the distance between distribution of orginal sample and real sample . And the discriminator tries to keep that distance as large as possible. In that way both train against each other.

Pg is the generated sample probablity distribution and Pr is for the real sample
![image](https://github.com/AbhinavJangra29/GANS/assets/107471490/7fee49f3-b13b-4809-ab5e-422743afa083)


in the paper they did a whole lot of math but if we just take an overview it can be summarise as:

![image](https://github.com/AbhinavJangra29/GANS/assets/107471490/db7f32dd-fa9d-43b5-b3e8-79488b7deaea)

discriminator(critic) wants to maximise this score, to maintain a distinguishibility.

generator wants to minimise this score and make the gen image distribution as close as the real one.

to enforce the constraint a weight clipping is used here(it basically clips the weight between 0.01 and -0.01)


read this amazing article:   [link](https://jonathan-hui.medium.com/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490)
</p>
<h2>why are they preffered?</h2>

  -Wasserstein distance measures the "work" needed to transform one data distribution to another.

  
  -This provides a clear and continuous measure of how realistic the generated data is.

  
  -WGANS leverage this distance metric to avoid vanishing gradients and train more effectively.


read this amazing article: 

[link](https://jonathan-hui.medium.com/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490](https://towardsdatascience.com/deep-convolutional-vs-wasserstein-generative-adversarial-network-183fbcfdce1f)


<h1>ARCHITECTURE</h1>
<p>
The architecture is simple invloving some basic conv nets, comments are added in the code for more understanding. ;)
</p>

<h1>CONS</h1>
weight clipping is a clearly terrible way to enforce the constraint in the diagram(lipschitz contraint) . if the clipping parameter is large then it can take a long time for any weight to reach their limit, thereby making it harder to train the critic till optimality. if clipping is small then it leads to vanishing gradient. So we introduce a new concept called gradient penalty , which is more effective in this case.



