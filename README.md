# Fish classifier segmentation desk application

## How does it works ?
Application was built base on kaggle dataset and making classification fish types ("Black Sea Sprat","Gilt-Head Bream","Hourse Mackerel","Red Mullet","Red Sea Bream","Sea Bass","Shrimp","Striped Red Mullet","Trout")

When you run .py file or app you will see first main display 

![](https://github.com/tural327/Fish_classifier_desk_app/blob/master/some_other_files/main.png)

For generate your image you need first drag and drop your image file **Drag and Drop Here** area for see your image you need just push **Show** button , if your image okay then you can push Generator button for see your segmented and dedtected images left side of app 

![](https://github.com/tural327/Fish_classifier_desk_app/blob/master/some_other_files/result.png)

Buttom of main dispaly class of fish type going to appear if you wondering to change displays like if you want to see segmented or dedtected images image at the main display you need just select which of the display you wandering display your image then click change display button .so you done =)

![](https://github.com/tural327/Fish_classifier_desk_app/blob/master/some_other_files/end_res.gif)


Data was downloaded from [Kaggle](https://www.kaggle.com/crowww/a-large-scale-fish-dataset)
## Application Details
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

X = np.zeros((len(loc_of_rgb1),img_h,img_w,img_c), dtype=np.uint8)

y = np.zeros((len(loc_of_masked1),img_h,img_w,1),dtype=np.bool)


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

X and y files saved by using pickle 

- **Part 2.** [U-net model](https://github.com/tural327/Fish_classifier_desk_app/blob/master/U_net.py)
Model was :


```python

inputs = tf.keras.layers.Input((img_h,img_w,img_c))
s = tf.keras.layers.Lambda(lambda x: x/255)(inputs)

#U-nets
c1 = tf.keras.layers.Conv2D(16,(3,3),activation='relu', kernel_initializer='he_normal',padding='same')(inputs)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16,(3,3),activation='relu', kernel_initializer='he_normal',padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

c2 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer="he_normal",padding="same")(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c2)
p2 = tf.keras.layers.MaxPool2D((2,2))(c2)

c3 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c3)
p3 = tf.keras.layers.MaxPool2D((2,2))(c3)

c4 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c4)
p4 = tf.keras.layers.MaxPool2D((2,2))(c4)

c5 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c5)


u6 = tf.keras.layers.Conv2DTranspose(32,(2,2),strides=(2,2),padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6,c4])
c6 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.3)(c6)
c6 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64,(2,2),strides=(2,2),padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7,c3])
c7 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32,(2,2),strides=(2,2),padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8,c2])
c8 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c8)


u9 = tf.keras.layers.Conv2DTranspose(16,(2,2),strides=(2,2),padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9,c1], axis=3)
c9 = tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c9)



outputs = tf.keras.layers.Conv2D(1,(1,1),activation="sigmoid")(c9)

```

For checking condition of model loss and accuracy scores was ploted

Accurancy of U-net was :


![](https://github.com/tural327/Fish_classifier_desk_app/blob/master/some_other_files/2.png)

Loss of U-net was :


![](https://github.com/tural327/Fish_classifier_desk_app/blob/master/some_other_files/1.png)

so its okay for saving model because network did not overfitted

## 1. Buildig image classification model ## 

I used tensorflow image classification guide for build my model
so for makeing inputs I used:

```python

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  file_loc,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_h, img_w),
  batch_size=32)


```

For validation was same things....

Model was look like 

```python


model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(9)
])


```

As I did for U-net same I want to see my loss and accuracy how worked so :

For accuracy :

![](https://github.com/tural327/Fish_classifier_desk_app/blob/master/some_other_files/class_acc.png)

For loss :

![](https://github.com/tural327/Fish_classifier_desk_app/blob/master/some_other_files/class.png)



## Combining all results ## 

Display U-net result as image was little problem because image dimension was 1 but we need 3 dimensions images so for handiling it i uesed simply convertor such as:

```python

display_unet = np.dstack([y_pred_a[0], y_pred_a[0], y_pred_a[0]])


```

and for printin fish class name i write simple python script [test_class.py](https://github.com/tural327/Fish_classifier_desk_app/blob/master/desk_app_for_fish/test_class.py) here just i loaded my model for testing image need to convert as a tensor for that i used :

```python

    tensor = tf.keras.preprocessing.image.load_img(
        picture, grayscale=False, color_mode='rgb', target_size=(128, 128),
        interpolation='nearest'
        )
    input_arr = tf.keras.preprocessing.image.img_to_array(tensor)

    tesnor_for_pred = tf.reshape(input_arr, [1, 128, 128, 3])
    

```
after that mission was done our model was ready for web app part 
