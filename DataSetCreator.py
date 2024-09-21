import os
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as con 
import shutil
import random
import numpy as np
import cv2
from tqdm import tqdm
import threading
import time
valpercent = 0.1
folder_path = r'' #input folder with labels and instances 
ToDelete = []# insert folders to clear
folder_contents = os.listdir(folder_path)
Do_Preprocessing = True
AugProbabilities = [1,0.4,0.15,0.3,0.45,0.3,0.4,0.6,0.6]
num_workers = 4 # threads


s = set()
X = []
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize to the model's input shape
])
def UnsharpMasking_PIL(image):
    image = np.array(image)
    gaussian_3 = cv2.GaussianBlur(image, (0, 0), 2.0)
    unsharp_image = cv2.addWeighted(image, 2.0, gaussian_3, -1.0, 0)
    return Image.fromarray(unsharp_image)
def UnsharpMasking_cv2(image):
    gaussian_3 = cv2.GaussianBlur(image, (0, 0), 2.0)
    unsharp_image = cv2.addWeighted(image, 2.0, gaussian_3, -1.0, 0)
    return unsharp_image
def Preprocess(img):
    if(Do_Preprocessing):
        color_image = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        ycrcb_img = cv2.cvtColor(color_image, cv2.COLOR_RGB2YCrCb)
        ycrcb_img[:, :, 0] = clahe.apply(ycrcb_img[:, :, 0])#cv2.equalizeHist(ycrcb_img[:, :, 0])
        equalized_color_image = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2RGB)
        return transform(Image.fromarray(UnsharpMasking_cv2(equalized_color_image)))
    else:
        return transform(img)
    
def delete():
    for path in ToDelete:
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
lock1 = threading.Lock()
lock2 = threading.Lock()

def getID():
    global current_size
    with lock2:
        id = current_size
        current_size += 1
    return id

def CreateLoop(val:bool,ImageSet:set,index:int):
    pmod = "train"
    if(val):
        pmod = "val"
    
    for i in tqdm(ImageSet,desc=f"{pmod} {index}"):
        count_per_i = 0
        path = os.path.join(folder_path,i)
        img = Preprocess(Image.open(path).convert('RGB'))
    
        id = getID()
        img.save(r"PATH\{}\images\{}.jpg".format(pmod,str(id).zfill(8))) # replace PATH with output path
        shutil.copy(os.path.splitext(path)[0] + ".txt", r"PATH\{}\labels".format(pmod)) # replace PATH with output path
        directory, filename = os.path.split(os.path.splitext(path)[0] + ".txt") # replace PATH with output path
        os.rename(r"PATH\{}\labels\{}".format(pmod,filename), r"PATH\{}\labels\{}.txt".format(pmod,str(id).zfill(8))) # replace PATH with output path
        
        count_per_i += 1
    Done[index] = True
delete()
print("Deleted Old Dataset")
for item in folder_contents:
    file_name_without_extension, file_extension = os.path.splitext(item)
    if(file_extension == ".jpeg" or file_extension == ".png" or file_extension == ".jpg"):
        s.add(item)

s1 = set()
s2 = set()
for i in s:
    rand = random.random()
    if(rand < valpercent):
        s2.add(i)
    else:
        s1.add(i)
current_size = 0
for i in range(2):
    prevsize = current_size
    Done = [False for _ in range(num_workers)]
    Threads = []
    Sets = [set() for _ in range(num_workers)]
    baseset = s1 if i == 0 else s2
    isval = (i == 1)
    
    startcount = 0 if i == 0 else 100000
    to = 0
    for path in baseset:
        Sets[to].add(path)
        to = (to + 1) % num_workers
    
    for thread_idx in range(num_workers):
        t = threading.Thread(target=CreateLoop, args=(isval, Sets[thread_idx], thread_idx,))
        t.start()
        Threads.append(t)
    for t in Threads:
        t.join()
    s = "val" if isval else "train"
    print(f"{s} size is {current_size-prevsize}\n")



