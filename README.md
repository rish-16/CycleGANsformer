# CycleGANsformer
Unpaired Image-to-Image Translation using Transformer-based GANs.

### About
This is an independent research project to build a Convolution-free GAN using Transformers for unpaired image-to-image translation between two domains (eg: horse and zebra, painting and photograph, seasons, etc.). It's fully implemented with `pytorch` and `torchvision`, and was inspired by the GANsformer, TransGAN, and CycleGAN papers.

### Usage [WIP]
I've prepared a `CycleGANsformer` wrapper over the entire model. You can install it via `pip` like so:

```bash
$ pip install pytorch-cyclegansformer
```

You can use the wrapper like so:

```python
import torch
from cyclegansformer import CycleGANsformer

x = torch.rand((1, 256, 256, 3)) # your input image
cgf = CycleGANsformer()

output_img = cgf(x) # can be viewed using matplotlib
```

### Training [WIP]
You can even train your own CycleGANsformer from scratch using the provided `ImageDatasetLoader`. Here, `path_to_x` and `path_to_y` represent the canonical filepaths to your training dataset comprising of two disjoint sets of images from two domains (eg: horses and zebras). Ensure you have the following directory structure:

```
my_image_dataset/
    |- train/
        |- HORSES
            |- horse_1.jpg
            |- horse_2.jpg
            |- ...
            |- horse_n.jpg
        |- ZEBRAS
            |- zebra_1.jpg
            |- zebra_2.jpg
            |- ...
            |- zebra_m.jpg
    |- test/
        |- HORSES
            |- horse_1.jpg
            |- horse_2.jpg
            |- ...
            |- horse_n.jpg
        |- ZEBRAS
            |- zebra_1.jpg
            |- zebra_2.jpg
            |- ...
            |- zebra_m.jpg
```

> Here, `n` is the number of horse images (X) and `m` is the number of zebra images (Y). 

Once ready, you can start the training process (ideally on some acceleration hardware) like so:

```python
import torch
from cyclegansformer import CycleGANsformer, ImageDatasetLoader

img_ds = ImageDatasetLoader(path_to_x, path_to_y)
cgf = CycleGANsformer()

cgf.fit(img_ds, epochs=200, alpha_decay=True) # proceeds to train – ideally use GPU, not CPU
```

### Credits
Credits to Aladdin Persson for the CycleGAN tutorial found [here](https://www.youtube.com/watch?v=4LktBHGCNfw), to Phil Wang for his implementation of the [Vision Transformer](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py) by Dosovitskiy et al., and [TransGAN](https://arxiv.org/abs/2102.07074) by Jiang et al.

### License
[MIT](https://github.com/rish-16/CycleGANsformer/blob/rish-dev/LICENSE)
