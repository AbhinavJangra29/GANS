<h1> <b>WHAT IS PIX 2 PIX GAN?</b> </h1>
<p>It is one of the image to image translation model. The model in this case learns a mapping of real image to transformed domain images, for example: satellite images to google maps image etc (or cartoon outline sketch to colored illustration) the usecases are numerous.
  
![image](https://github.com/AbhinavJangra29/GANS/assets/107471490/045148bd-71f7-4915-b1e0-0d61fb5eba3b)



</p>

<h1>ARCHITECTURE</h1>
<h2>generator</h2>
<p> Generator is very much inspired (not exactly) by U-net i.e (conv->downsample x 4 times)+(conv->upsample x 4 times) aslo added skip connections for good gradient flow.

  
![image](https://github.com/AbhinavJangra29/GANS/assets/107471490/11f74b41-0df3-445c-aa38-510b9238cbda)

# Machine Learning Masterpiece

## Sketch Understanding (Downsampling):
The machine looks at the sketch (input image) and uses `nn.Conv2d` to understand the big shapes and forms. This is like the initial downsampling where it simplifies things.

## Adding Details (Blocks with More Features):
It zooms in and adds more details by increasing the features in each block. This is similar to focusing on smaller things and adding details progressively.

## Squeezing Info (Bottleneck):
It squishes all the important info into a small space. The `bottleneck` layer compresses the information.

## Bringing Back Details (Upsampling):
It uses the squeezed info to bring back details. `nn.ConvTranspose2d` is like adding details back during upsampling, and this happens in steps with different blocks.

## Putting It Together (Concatenation):
Finally, it puts everything together. Concatenation is like combining different elements to get a colorful painting.

## A Bit Random (Dropout):
Sometimes, it adds a sprinkle of randomness to keep things interesting. `nn.Dropout` is like randomly removing some details.

So, the code is like a little artist transforming a simple sketch into a detailed and colorful masterpiece step by step! 🎨✨


</p>
<h2>discriminator</h2>
<p>Discriminator is basically just a couple of CNN layers, output of the discriminator architecture is unlike other GANs which just output a value between 0 or 1 (representing fake or real) ,to have a better understanding think of the output as a nXn matrix where each pixel has the value 0 or 1 showing if that particular patch is fake or not. That particular patch is responsible for looking larger patches in the original image. comments are added in the code for better understanding.</p>


<h1>DATASET</h1>
<p>
  download the dataset from here 
  
  [link](https://www.kaggle.com/vikramtiwari/pix2pix-dataset)

  data looks like:![1](https://github.com/AbhinavJangra29/GANS/assets/107471490/093e81b6-ecfa-4684-947d-726f3811376a)

  this is a single image from the train dataset, we call the left half as x and rest y where y is the target.To make the code work  data heirarchy should be :
-project folder->data folder->maps folder-> train and test folder.
## MapDataset: Custom Dataset for Image Processing

### Overview:

- **Dataset Handling Code Overview:**
  - Custom dataset class: `MapDataset`.
  - Loads image data from a specified directory.

### Initialization:

- **Dataset Initialization:**
  - Sets up the root directory and retrieves a list of image files.

### Dataset Length:

- **Dataset Length:**
  - Implements the `__len__` method to return the total number of images in the dataset.

### Dataset Item Retrieval:

- **Dataset Item Retrieval:**
  - Implements the `__getitem__` method to get an item at a specified index.
  - Reads and splits each image into input and target parts.

### Data Augmentations:

- **Data Augmentations:**
  - Applies augmentations using transformations from the `config` module.

### Transformation Pipelines:

- **Transformation Pipelines:**
  - Defines separate transformation pipelines for input and target images.

### Usage Example:

- **Dataset Usage Example:**
  - Creates an instance of `MapDataset` for a specific training directory.
  - Uses PyTorch's `DataLoader` to iterate through batches of data.
  - Prints the shape of input images in each batch.
  - Saves sample images ("x.png" and "y.png") for visualization.



</p>

<h1>TRAINING AND LOSSES</h1>
<p>
 The training goes as : we have a training input x and corresponding transformation label y ,
we feed x to the generator and it generates a response as y_fake . Also we feed the whole image to the discriminator to capture the discriminator response on the whole image (x+y).
The discriminator should identify x+y as 1(real sample). Then we feed the combination of (x+y_fake) to the discriminator and it shoudl identify (x+y_fake) as 0(fake sample). we minimise both losses and in this way we train the discriminator.

To train the generator we keep a checkpoint on the discriminator response on fake sample , and the generator tries to makes that response as 1(to fool the discriminator). And this goes on and on.
below is an easy diagram to understand the key concept involved

![WhatsApp Image 2023-12-17 at 3 35 47 PM](https://github.com/AbhinavJangra29/GANS/assets/107471490/fd272e2a-cced-4554-8d55-820f421fd898)


Apart from these 2 losses we also try to minimise another loss:L1 loss which is the pixelwise difference between y and y_fake.
  
</p>

<p>To use the model download all files and put them in a folder including the dataset and use any ide to train the model, also use cuda because the model is very heavy and slow</p>

<h1>TRAINED MODEL WEIGHTS:
  
  [link](https://github.com/aladdinpersson/Machine-Learning-Collection/releases/download/1.0/Pix2Pix_Weights_Satellite_to_Map.zip)

</h1>




