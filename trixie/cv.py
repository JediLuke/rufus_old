import cv2
import numpy as np

class ImgGreyscale():

    def run(self, img_arr):
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        return img_arr



class BirdsEyePerspectiveTxfrm():
    #http://www.coldvision.io/2017/03/02/advanced-lane-finding-using-opencv/
    def compute_perspective_transform(self, binary_image):
        # Define 4 source and 4 destination points = np.float32([[,],[,],[,],[,]])
        shape = binary_image.shape[::-1] # (width,height)
        w = shape[0]
        h = shape[1]
        transform_src = np.float32([ [580,450], [160,h], [1150,h], [740,450]])
        transform_dst = np.float32([ [0,0], [0,h], [w,h], [w,0]])
        M = cv2.getPerspectiveTransform(transform_src, transform_dst)
        return M

    def run(self, img_arr):
        M = self.compute_perspective_transform(img_arr)
        return cv2.warpPerspective(img_arr, M, (img_arr.shape[1], img_arr.shape[0]), flags=cv2.INTER_NEAREST)  # keep same size as input image



class AdaptiveThreshold():

    def __init__(self, high_threshold=255):
        self.high_threshold = high_threshold
        

    def run(self, img_arr):
        return cv2.adaptiveThreshold(img_arr, self.high_threshold,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 115, 1)



class ImgCanny():

    def __init__(self, low_threshold=60, high_threshold=110):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        
        
    def run(self, img_arr):
        return cv2.Canny(img_arr, 
                         self.low_threshold, 
                         self.high_threshold)

    

class ImgGaussianBlur():

    def __init__(self, kernal_size=5):
        self.kernel_size = kernal_size
        
    def run(self, img_arr):
        return cv2.GaussianBlur(img_arr, 
                                (self.kernel_size, self.kernel_size), 0)



class DrawLine():
    
    def __init__(self, line_start, line_finish):
        self.line_start = line_start
        self.line_finish = line_finish
    
    def run(self, img_arr):
        return cv2.line(img_arr, self.line_start, self.line_finish, (255,0,0), 5)




class ImgCrop:
    """
    Crop an image to an area of interest. Works by cropping num pixels in from side of each border.
    """
    def __init__(self, top=0, bottom=0, left=0, right=0):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        
    def run(self, img_arr):
        width, height, _ = img_arr.shape
        img_arr = img_arr[self.top:height-self.bottom, 
                          self.left: width-self.right]
        return img_arr
        


class ImgStack:
    """
    Stack N previous images into a single N channel image, after converting each to grayscale.
    The most recent image is the last channel, and pushes previous images towards the front.
    """
    def __init__(self, num_channels=3):
        self.img_arr = None
        self.num_channels = num_channels

    def rgb2gray(self, rgb):
        '''
        take a numpy rgb image return a new single channel image converted to greyscale
        '''
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
        
    def run(self, img_arr):
        width, height, _ = img_arr.shape        
        gray = self.rgb2gray(img_arr)
        
        if self.img_arr is None:
            self.img_arr = np.zeros([width, height, self.num_channels], dtype=np.dtype('B'))

        for ch in range(self.num_channels - 1):
            self.img_arr[...,ch] = self.img_arr[...,ch+1]

        self.img_arr[...,self.num_channels - 1:] = np.reshape(gray, (width, height, 1))

        return self.img_arr

        
        
class Pipeline():
    def __init__(self, steps):
        self.steps = steps
    
    def run(self, val):
        for step in self.steps:
            f = step['f']
            args = step['args']
            kwargs = step['kwargs']
            
            val = f(val, *args, **kwargs)
        return val
    