import numpy as np
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt

image = Image.open("../lenna.png")

plt.imshow(image)
plt.show()

image2=image.resize((32,32))

w,h=image.size
resize_w= (int)(w/2)
resize_h= (int)(h/2)
image_2=image.resize((resize_w, resize_h))

plt.imshow(image2)
plt.show()

print(image2)
print(type(image2))
print(image2.size)
print(image2.mode)
print(image2.getpixel((0,0)))

image_rotate=image.transpose(Image.ROTATE_180)
image_rotate_LR=image.transpose(Image.FLIP_LEFT_RIGHT)
image_rotate_180=image.transpose(Image.ROTATE_180)

plt.imshow(image_rotate)
image_rotate_LR.show()
image_rotate_180.show()
image2.show()
image_2.show()
plt.show()

blurred_image=image.filter(ImageFilter.BLUR)

plt.imshow(blurred_image)
plt.show()

blurred_image.save("./blurred_Lenna.png")