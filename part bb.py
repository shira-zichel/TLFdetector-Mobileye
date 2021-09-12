import glob, os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import random
def create_pic():
    #os.chdir(r"C:\Users\מירי\Desktop\test")
    #json_path=r"C:\Users\מירי\Desktop\test"
    #path=r"C:\Users\מירי\Desktop\test"
    all_labels=[]
    all_images=[]
    for root, dirs, files in os.walk(r"C:\Users\מירי\Desktop\files\gtFine\val"):
        for file in files:
            if file.endswith(".json"):
                json_name = root+"/"+ file
                f = open(json_name, )
                data = json.load(f)
                pixel_list = []
                not_traffic_list = []
                for i in data["objects"]:
                    if i["label"] == "traffic light":
                        pixel_list.append(i["polygon"][0])
                base_name = file[0:-21]
                name = base_name + "_leftImg8bit.png"
                r=root
                r=r.split("\\")
                r[5]="leftImg8bit"
                r="\\".join(r)
                image_path = r + "/" + name
                image = Image.open(image_path)
                width, height = image.size
                for i in range(len(pixel_list)):
                    x = random.randint(0, width)
                    y = random.randint(0, height)
                    not_traffic_list.append([x, y])
                f.close()
                images_tfl = []
                labels_tfl = []
                for i in pixel_list:
                    left = i[0] - 40
                    top = i[1] - 40
                    right = i[0] + 41
                    bottom = i[1] + 41
                    if left < 0:
                        right += abs(left)
                        left = 0
                    if right > width:
                        left -= (right - width)
                        right = width
                    if top < 0:
                        bottom += abs(top)
                        top = 0
                    if bottom > height:
                        top -= (bottom - height)
                        bottom = height
                    im1 = image.crop((left, top, right, bottom))
                    traffic_u = np.array(im1, dtype=np.int16)
                    traffic_u = np.uint8(traffic_u)
                    images_tfl.append(traffic_u)
                    traffic_u = np.array(1, dtype=np.int16)
                    traffic_u = np.uint8(traffic_u)
                    labels_tfl.append(traffic_u)

                images_not_tfl = []
                labels_not_tfl = []
                for i in not_traffic_list:
                    left = i[0] - 40
                    top = i[1] - 40
                    right = i[0] + 41
                    bottom = i[1] + 41
                    if left < 0:
                        right += abs(left)
                        left = 0
                    if right > width:
                        left -= (right - width)
                        right = width
                    if top < 0:
                        bottom += abs(top)
                        top = 0
                    if bottom > height:
                        top -= (bottom - height)
                        bottom = height
                    im1 = image.crop((left, top, right, bottom))
                    # im1.show()
                    traffic_u = np.array(im1, dtype=np.int16)
                    traffic_u = np.uint8(traffic_u)
                    images_not_tfl.append(traffic_u)
                    traffic_u = np.array(0, dtype=np.int16)
                    traffic_u = np.uint8(traffic_u)
                    labels_not_tfl.append(traffic_u)
                all_labels.extend(labels_tfl)
                all_labels.extend(labels_not_tfl)
                all_images.extend(images_tfl)
                all_images.extend(images_not_tfl)
    with open(r"C:\Users\מירי\Desktop\data_dir1\val\labels.bin",
              "wb") as f:  # or choose 'w+' mode - read "open()" documentation
        np.array(all_labels).tofile(f)
    with open(r"C:\Users\מירי\Desktop\data_dir1\val\data.bin",
              "wb") as f:  # or choose 'w+' mode - read "open()" documentation
        np.array(all_images).tofile(f)

create_pic()