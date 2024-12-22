import cv2
import numpy as np
from skimage.exposure import rescale_intensity

def convolute(Image, Kernel):
    '''
    :param np.array Image: grayscale image to apply kernel to
    :param np.aray|list Kernel: Kernel to apply images
    '''
    assert len(Image.shape) == 2, "Image must be 2d"

    # image shape
    (iH, iW) = Image.shape[:2]
    (kH, kW) = Kernel.shape[:2]
    
    # copute and apply the length of padding on each 
    # sides of the images to preserve original image dimension
    padding_size = (kW - 1) // 2
    padded_image = cv2.copyMakeBorder(Image, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_REPLICATE)

    # create a dark frame for the oiutput image
    # each pixel will be changed till the entire
    # frame finanlly becomes the output images
    output_image = np.zeros((iH, iW), dtype="float")

    # loop over each rows and columns as we slide the
    # the kernel from left to right and top to bottom
    for y in np.arange(padding_size, iH + padding_size):
        for x in np.arange(padding_size, iW + padding_size):

            # select a region of image that has same size with
            # the kernel
            roi = padded_image[y - padding_size:y + padding_size + 1, x - padding_size:x + padding_size + 1]

            # apply convolution(element wise multiplication) bwtween the
            # the kerel and ROI and then compute sum of elemtent in the
            # resulting amtrix
            conv = (roi * Kernel).sum()

            # replace the current pixel value with result of convolution
            output_image[y-padding_size:, x-padding_size] = conv
    
    # reescale output image to 0, 255
    output_image = rescale_intensity(output_image, in_range=(0, 255))
    output_image = (output_image * 255).astype('uint8')

    # retunr output image
    return output_image

# load image
image = "yuji.jpeg"
image = cv2.imread(image)

# defines differnet kernels
K = np.array([
    [-1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
])
sharpen = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])
blurr = np.ones((7, 7), dtype="float") * (1/(7 * 7))

dic = {
    "blurr": blurr,
    "sharpen": sharpen,
    "hand kernel": K
}

# display original image
cv2.imshow("original", image)

# loop over kernel and apply each of them
# on the image
for key in dic.keys():
    # apply filters on each channels of the image
    channel_conv = [convolute(i, dic[key]) for i in cv2.split(image)]
    conv_image = cv2.merge(channel_conv)

    # display image
    cv2.imshow(key, conv_image)
    
cv2.waitKey(0) & 0xff
cv2.destroyAllWindows()