# Fish classifier segmentation web application

Data was downloaded from [Kaggle](https://www.kaggle.com/crowww/a-large-scale-fish-dataset)

# Software
- App developed Ubuntu 20.04 
- Python 3.8 

**Python libraries I uesd**
- Tensorflow
- Pyqt5
- cv2
- numpy
- pickle
- glob
- matplotlib


## 1. Buildig U-net model ##
 - **Part 1**. For building U-net model first I need make a inputs for my model if you check dataset you will see there is 2 type of images for each class one of them RGB one of
them is Masked first I need make RGB images as a input and Masked images as output (The file for making inputs [here](https://github.com/tural327/Fish_classifier_desk_app/blob/master/make_inputs.py))
first approach was deciding image shapes I made it "128,128,3" and I collected all class RGB images and make it single inputs masked files also same by using **glob** input
then created X and y inputs :


```python

for rgb_img in loc_of_rgb1:
    index = loc_of_rgb1.index(rgb_img)
    img_read = cv2.imread(rgb_img)
    img_size = cv2.resize(img_read,(img_h,img_w))
    img_array = np.array(img_size)
    X[index] = img_array
    percentage = (index * 100)/len(loc_of_rgb1)
    ## just chkening current status of loop 
    print("{}% of process done for RGB".format(round(percentage),1))
    
    
for mask in loc_of_masked1:
    index_mask = loc_of_masked1.index(mask)
    mask_img = cv2.imread(mask)
    mask_gray = cv2.cvtColor(mask_img,cv2.COLOR_BGR2GRAY)
    mask_gray_size = cv2.resize(mask_gray,(img_h,img_w))
    mask_array = np.array(mask_gray_size).reshape(img_h,img_w,1)
    y[index_mask] = mask_array
    percentage = (index_mask * 100)/len(loc_of_masked1)
    ## just chkening current status of loop 
    print("{}% of process done for mask".format(round(percentage), 1))
    
 ```
