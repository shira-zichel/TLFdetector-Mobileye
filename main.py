try:
    import os
    import json
    import glob
    import argparse

    import numpy as np
    from scipy import signal as sg
    from scipy.ndimage.filters import maximum_filter

    from PIL import Image

    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise

import matplotlib.pyplot as plt
import numpy
from PIL import Image, ImageEnhance
from scipy import ndimage
def plot(data, title):
    plot.i += 1
    plt.subplot(2,2,plot.i)

    plt.imshow(data)
    plt.gray()
    """"for r in data:
        for s in r:
            #print(type(s))
            if s is not int and s[0]>=100 and s[1]>=100 and s[2]>=100:
                s[0] = 250
                s[1] = 0
                s[2] = 0
    img = Image.fromarray(data)
    img.save('my.png')
    img.show()"""
    plt.title(title)
plot.i = 0
def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    plot(c_image, 'Original')
    c_image2 = c_image[:, :, 0]

    # A very simple and very narrow highpass filter
    kernel = np.array([[-1/25, -1/25, -1/25, -1/25, -1/25],
                       [-1/25, 1/25, 1/25,  1/25, -1/25],
                       [-1/25, 1/25, 15/25, 1/25, -1/25],
                       [-1/25, 1/25, 1/25, 1/25, -1/25],
                       [-1/25, -1/25, -1/25, -1/25, -1/25]])
    highpass_3x3 = ndimage.convolve(c_image2, kernel)
    plot(highpass_3x3, 'Simple 3x3 Highpass')
    y = highpass_3x3.shape[0]
    x = highpass_3x3.shape[1]

    #max_3x3 = maximum_filter(highpass_3x3, size=5)

    list_x=[]
    list_y=[]
    for i in range(x):
        for j in range(y):
            if highpass_3x3[j,i]==1:
                #print([j,i])
                list_x.append(i)
                list_y.append(j)
    #plt.show()
    return list_x,list_y, [], []
find_tfl_lights(np.array(Image.open(r"C:\Users\User\Desktop\Studies\Excellenteam\Mobileye\New folder\aachen_000008_000019_leftImg8bit.png")))

### GIVEN CODE TO TEST YOUR IMPLENTATION AND PLOT THE PICTURES
def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    show_image_and_gt(image, objects, fig_num)

    red_x, red_y, green_x, green_y = find_tfl_lights(image)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=4)



def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""

    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = r"C:\Users\User\Desktop\Studies\Excellenteam\Mobileye\New folder"

    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))

    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')

        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)

    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


#if __name__ == '__main__':
    #main()
