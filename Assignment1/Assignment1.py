from PIL import Image

image = Image.open("../lenna.png")

w,h=image.size
resize_w= (int)(w/2)
resize_h= (int)(h/2)
image_2=image.resize((resize_w, resize_h))

image_rotate_LR=image.transpose(Image.FLIP_LEFT_RIGHT)
image_rotate_180=image.transpose(Image.ROTATE_180)

image.show()
image_rotate_LR.show()
image_rotate_180.show()
image_2.show()
