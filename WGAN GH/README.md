<h1>IDEA BEHIND WGAN </h1>
<p>
The W in wgans stand for wasserstein distance. what is this wasserstien distance? it can be basically described as distance between 2 probablity distributions. distance between 2 distributions? Yes when we say distance between 2 probablity distribution we mean the distance between their means. So in WGAN what we do is instead of keeping a discriminator like usual which tells us real or fake , wee use a critic that tells us how far is the generated sample distribution from the original image distriubtion(critic generates a score). Thats it, the generator tries to minimise the distance between distribution of orginal sample and real sample . And the discriminator tries to keep that distance as large as possible. In that way both train against each other.

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



