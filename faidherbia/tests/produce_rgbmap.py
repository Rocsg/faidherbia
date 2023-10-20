from PIL import Image
import numpy as np

# Open the RGB image and convert it to grayscale
r_image = np.array(Image.open('/home/rfernandez/rgbgroundr.png').convert('L'))
g_image = np.array(Image.open('/home/rfernandez/rgbgroundg.png').convert('L'))
b_image = np.array(Image.open('/home/rfernandez/rgbgroundb.png').convert('L'))

r_image   = (r_image   / r_image.max()   * 255).astype(np.uint8)
g_image   = (g_image   / g_image.max()   * 255).astype(np.uint8)
b_image   = (b_image   / b_image.max()   * 255).astype(np.uint8)


# Create a new RGB image with the 8-bit brightness channel
rgb_8bit = np.dstack((r_image, g_image, b_image))

#save this as a rgb image
im = Image.fromarray(rgb_8bit)
im.save("/home/rfernandez/rgbground_assembled.png")

