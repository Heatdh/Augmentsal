import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
#import albumentations as A
import random

def augment(method, image_dir, export_dir):
    for i in os.listdir(image_dir):
        img = cv2.imread(os.path.join(image_dir, i))
        if method == "MotionBlur1":
            # fspecial('motion', 15, 0) cv2 imfilter
            transformed = cv2.filter2D(img, -1, cv2.getGaussianKernel(15, 0))
        elif method == "MotionBlur2":
            # fspecial('motion', 35, 90) cv2 imfilter
            transformed = cv2.filter2D(img, -1, cv2.getGaussianKernel(35, 90))
        elif method == "Noise1":
            # imnoise('gaussian', 0, 0.1) cv2
            transformed = cv2.addWeighted(
                img, 0.9, np.zeros(img.shape, img.dtype), 0, 10)
        elif method == "Noise2":
            # imnoise('gaussian', 0, 0.2) cv2
            transformed = cv2.addWeighted(
                img, 0.8, np.zeros(img.shape, img.dtype), 0, 20)
        elif method == "Contrast1":
            # imadjust('stretch', [0.3 0.7], []) cv2
            transformed = cv2.addWeighted(
                img, 1.5, np.zeros(img.shape, img.dtype), 0, -50)
        elif method == "Contrast2":
            # imadjust [], [0.4 0.6] cv2
            transformed = cv2.addWeighted(
                img, 0.5, np.zeros(img.shape, img.dtype), 0, 50)
        elif method == "Mirroring":
            # flipud
            transformed = cv2.flip(img, 0)
        elif method == "Inversion":
            # imcomplement
            transformed = cv2.bitwise_not(img)

        elif method == "Shearing1":
            # imwar[0.5 0.5 0; 0 1 0; 0 0 1] cv2
            rows, cols, ch = img.shape
            pts1 = np.float32([[0, 0], [cols, 0], [0, rows]])
            pts2 = np.float32([[0, 0], [cols, 0], [cols/2, rows]])
            M = cv2.getAffineTransform(pts1, pts2)
            transformed = cv2.warpAffine(img, M, (cols, rows))
        elif method == "Shearing2":
            # imwar[1 0.5 0; 0 1 0; 0 0 1] cv2
            rows, cols, ch = img.shape
            pts1 = np.float32([[0, 0], [cols, 0], [0, rows]])
            pts2 = np.float32([[cols/2, 0], [cols, 0], [0, rows]])
            M = cv2.getAffineTransform(pts1, pts2)
            transformed = cv2.warpAffine(img, M, (cols, rows))
        elif method == "Shearing3":
            # imwar[1 0 0; 0.5 1 0; 0 0 1] cv2
            rows, cols, ch = img.shape
            pts1 = np.float32([[0, 0], [cols, 0], [0, rows]])
            pts2 = np.float32([[0, 0], [cols, rows/2], [0, rows]])
            M = cv2.getAffineTransform(pts1, pts2)
            transformed = cv2.warpAffine(img, M, (cols, rows))

        elif method == "Rotate-45":
            # rotate -45 degree cv2
            rows, cols, ch = img.shape
            M = cv2.getRotationMatrix2D((cols/2, rows/2), -45, 1)
            transformed = cv2.warpAffine(img, M, (cols, rows))


        elif method == "Rotate135":
            # rotate -135 degree cv2
            rows, cols, ch = img.shape
            M = cv2.getRotationMatrix2D((cols/2, rows/2), -135, 1)
            transformed = cv2.warpAffine(img, M, (cols, rows))

        elif method == "boundaries":
            # canny 0.3 sqrt(0.2)
            transformed = cv2.Canny(img, 0.3, np.sqrt(2))

        elif method == "Cropping1":
            # cut 1080x200 left side imcrop
            transformed = img[0:1080, 0:200]           
        elif method == "Cropping2":
            # cut 200x1920 top side cv2 imcrop
            transformed = img[0:200, 0:1920]

        elif method == "JpegCompression1":
            # quality 5  cv2 imwrite else 5
            cv2.imwrite(os.path.join(export_dir, i), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 5])
        elif method == "JpegCompression2":
            # quality 0
            cv2.imwrite(os.path.join(export_dir, i), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 0])
        else:
            print("No method found")
            break
        # save same image name in export_dir
        if method != "JpegCompression1" and method != "JpegCompression2":
            cv2.imwrite(export_dir+'/'+i, transformed)

def show_1sample_each(methods, path):
    for method in methods:
        # get a random image from the path
        img = random.choice(os.listdir(path))
        # read the image
        img = cv2.imread(os.path.join
                            (path, img))
        # convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # show the image
        plt.imshow(img)
        plt.title(method)
        plt.show()






if __name__ == "__main__":
    print("methods imported")
    #  make a list of methods
    methods = ["MotionBlur1", "MotionBlur2", "Noise1", "Noise2", "Contrast1", "Contrast2", "Mirroring", "Inversion", "Shearing1", "Shearing2", "Shearing3", "Rotate-45", "Rotate135", "boundaries", "Cropping1", "Cropping2", "JpegCompression1", "JpegCompression2"]
    methods_c = ["Rotate-45", "Rotate135", "boundaries", "Cropping1", "Cropping2", "JpegCompression1", "JpegCompression2"]
    # create a directory for each method in side the augmented directory skipping it if it already exists

    for method in methods_c:  
        if not os.path.exists("augmented/images/"+method):
            os.makedirs("augmented/images/"+method)

        # augment each method
    for method in methods_c:
        print("augmenting:",method)
        augment(method, "dataset/images/train/","augmented/images/"+method)
    


    """fig = plt.figure(figsize=(20, 20))
    # sample same image from each method and show it in a 6x3 grid
    img_name = random.choice(os.listdir("saliency/augmented/"+methods[0]))
    img_name_only = img_name.split("/")[-1]
    print(img_name_only)
    for i in range(1, 19):
        img = random.choice(os.listdir("saliency/augmented/"+methods[i-1]))
        img = cv2.imread(os.path.join("saliency/augmented/"+methods[i-1], img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig.add_subplot(6, 3, i)
        plt.imshow(img)
        plt.title(methods[i-1]"""


    # execute test script on each method
    #for method in methods:
    #    print(method)
    #    os.system("python test.py --dataroot saliency/augmented/"+method+" --name "+method+" --model test --no_dropout --gpu_ids -1")
    #    print("Done")

        
    

