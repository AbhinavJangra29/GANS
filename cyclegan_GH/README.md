<h1> <b>WHAT IS CYCLE GAN?</b> </h1>
<p>

![image](https://github.com/AbhinavJangra29/GANS/assets/107471490/e86eff36-f13a-4cfe-b618-1f1166d76983)


## Overview

CycleGAN is a type of Generative Adversarial Network (GAN) designed for image-to-image translation. Imagine having two artists, each specializing in drawing specific things, like horses or zebras . it works between any 2 domains.

## Key Components

1. **Two Artists (Forgers):**
   - Artists specialize in drawing realistic horses and zebras.

2. **Art Critics:**
   - Critics evaluate realism, one for horses and one for zebras.

3. **Training Process:**
   - Forger 1 turns a horse photo into a zebra painting, evaluated by the zebra critic.
   - Forger 2 turns a zebra photo into a horse painting, evaluated by the horse critic.
   - Adjustments are made based on critic feedback.

4. **Cycle Consistency:**
   - After turning a horse into a zebra, the second forger tries to turn it back into a horse.
   - The same process happens in reverse.
   - Goal: Consistency in both translations.

5. **Iteration and Learning:**
   - This process repeats iteratively to improve the forgers and critics.

6. **Result:**
   - Skilled forgers can translate between horses and zebras, ensuring consistency in both directions.

## Outcome

CycleGAN learns to transform images between two different styles while maintaining a cycle consistency. It's like training artists to create realistic translations that are reversible.

</p>

<h1>ARCHITECTURE</h1>

<h2>generator</h2>

![image](https://github.com/AbhinavJangra29/GANS/assets/107471490/0b4d5ec1-942f-4d8f-8952-e8b6a99730d7)

<p>
  this is the structure of generator consists of conv(downsampling),residual blocks and conv(upsampling) .Residual block act as skip connections facilitating smooth flow of gradients.
Transposed conv layers are used as upsampling layers. Important point is the genrator outputs a feature map of same size as the inputted image, because it just the same image transferred to a different domain.
</p>

<h2>discriminator</h2>

![image](https://github.com/AbhinavJangra29/GANS/assets/107471490/ea08d7f9-0c90-4de5-94ca-f90a9778f93a)


<p>
Discriminator is a simple conv net , consisting of some convulation layers. Important part is that the discrimintor instead of returning a single output as real or fake ,returns a matrix/grid of NxN where each matrice grid has a value between 0-1(fake or real) and each grid is responsible for looking on a larger patch of image in the inputted image. This is a reason this type of gan network falls in the category of PatchGan because the disrciminator outputs a grid where each element is responsible for looking on a larger patch.
</p>

<h1>TRAINING AND LOSSES</h1>

<p>
  training is simple. we have 2 generators:horse_gen and zebra_gen ,2 discriminators:horse_disc and zebra_disc (in case of horse/zebra cyclegan)
  let us take a look at training:
  
  
    - take a zebra image put it into horse_gen -> now you have a fake zebra
      
    
    -capture what is the response of horse_disc on real horses, lets call this first response
  
    
    -capture what is the response of horse_disc on fake_horses(which were originally zebra),lets     call this second response
    
    
    -the first reponse should be close to 1 cus they are real horses.
    
    
    -the second response should be close to 0 cus they are fake horses.
    
    
    -combine both losses for horse_disc training
    

similarly for zebra_disc..

The twist actually lies in the generator training, where we also use 2 other losses as well which make the cycle gan ,cycle gan. Cycle loss and identity loss.

<b>let us understand cycle loss:</b>

we take a horse and put it into zebra_gen , it will output a genarated zebra image. If we put the generated zebra image back in the horse generator it should ideally output the reconstructed horse image which was initially used to make generated zebra. This is the whole concept of cycle loss.

<b>let us understand identity loss:</b>

if we put a original horse image into a horse generator , it should ideally not manipulate the image , because it is already a horse and need not to be transformed. This way we train the generator into manage all these losses at a time inluding the adverserial loss.
  
  
  
</p>

<h1>DATASET</h1>
<p>
this is the link to horse/zebra dataset : 
  
  [link](https://www.kaggle.com/suyashdamle/cyclegan)

  keep the heirarchy as : project folder->data->train(train folder has horses and zebra folders inside) and test(test folder has horses and zebra folders inside). or modify it as per your needs.

here are the pretrained weights: 

[link](https://github.com/aladdinpersson/Machine-Learning-Collection/releases/download/1.0/CycleGAN_weights.zip)

run the model on gpu , it is extremely slow.
</p>

