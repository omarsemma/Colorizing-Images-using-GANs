
import numpy as np
from PIL import Image, ImageCms

# Open image and discard alpha channel which makes wheel round rather than square
im = Image.open('image.jpg').convert('RGB')

# Convert to Lab colourspace
srgb_p = ImageCms.createProfile("sRGB")
lab_p  = ImageCms.createProfile("LAB")

rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
Lab = ImageCms.applyTransform(im, rgb2lab)
L,a,b = Lab.split()
L.save('./checkpoints/test/lab.jpg')