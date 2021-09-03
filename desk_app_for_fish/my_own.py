import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
import cv2
from keras.models import load_model
import numpy as np
from PyQt5.QtGui import QFont,QPixmap
import tensorflow as tf
import os
import imageio
from test_class import find_class
from PyQt5 import QtGui
model = load_model('u_net.h5')
model_c = load_model("img_class.h5")
img_h = 128
img_w = 128
img_c = 3

class Button(QPushButton):
    def __init__(self, title, parent):
        super().__init__(title, parent)
        self.setAcceptDrops(True)


    def dragEnterEvent(self, e):
        m = e.mimeData()
        if m.hasUrls():
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        m = e.mimeData()
        if m.hasUrls():
            print(m.urls()[0].toLocalFile())
            self.parent().main_display.setText("Image Loaded \n Please Push <Show> \n For View")
            self.parent().second_display.setText("Image Loaded \n Please Push <Generator> \n For View")
            try:
                try:
                    os.remove("img.jpg")
                    img = cv2.imread(m.urls()[0].toLocalFile())
                    cv2.imwrite("img.jpg",img)
                except:
                    img = cv2.imread(m.urls()[0].toLocalFile())
                    cv2.imwrite("img.jpg", img)
            except:
                self.parent().main_display.setText("Don't load more images")





class Main_space(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Fish')
        self.setWindowIcon(QtGui.QIcon('index.png'))
        self.setFixedWidth(800)
        self.setFixedHeight(600)

        ## Main Img display
        self.main_display_text = QtWidgets.QLabel(self)
        self.main_display_text.setText("Main Display")
        self.main_display_text.setFont(QFont("Times",13))
        self.main_display_text.setGeometry(0,50,200,25)
        self.main_display_text.move(210,10)

        self.main_display = QtWidgets.QLabel(self)
        self.pixmap = QPixmap('img.jpg')
        self.main_display.setPixmap(self.pixmap)
        self.main_display.setScaledContents(True)
        self.main_display.setGeometry(20, 45, 460, 340)

        #Second Display
        self.second_display = QtWidgets.QLabel(self)
        self.pixmap_second = QPixmap('img.jpg')
        self.second_display.setPixmap(self.pixmap_second)
        self.second_display.setScaledContents(True)
        self.second_display.setGeometry(520, 45, 260, 160)

        self.second_display_text = QtWidgets.QLabel(self)
        self.second_display_text.setText("Second Display")
        self.second_display_text.setFont(QFont("Times",13))
        self.second_display_text.setGeometry(0,50,200,25)
        self.second_display_text.move(600,10)


        self.third_display_text = QtWidgets.QLabel(self)
        self.third_display_text.setText("Third Display")
        self.third_display_text.setFont(QFont("Times",13))
        self.third_display_text.setGeometry(0,50,200,25)
        self.third_display_text.move(600,220)

        self.third_display = QtWidgets.QLabel(self)
        self.pixmap_third = QPixmap('img.jpg')
        self.third_display.setPixmap(self.pixmap_second)
        self.third_display.setScaledContents(True)
        self.third_display.setGeometry(520, 260, 260, 160)

        self.fish_class_names = QtWidgets.QLabel(self)
        self.fish_class_names.setText("")
        self.fish_class_names.setFont(QFont("Times",13))
        self.fish_class_names.setGeometry(0,50,500,25)
        self.fish_class_names.move(20,400)
#############  Combobox display #############################
        self.original_image = QtWidgets.QLabel(self)
        self.original_image.setText("Show Original Image at :")
        self.original_image.setFont(QFont("Times",13))
        self.original_image.setGeometry(0,50,200,25)
        self.original_image.move(20,440)

        self.orginal_display = QComboBox(self)
        self.orginal_display.setGeometry(200,443,120,25)
        self.orginal_display .addItem("Main Display")
        self.orginal_display.addItem("Second Display")
        self.orginal_display.addItem("Third Display")

        self.segmented_image = QtWidgets.QLabel(self)
        self.segmented_image.setText("Show Segmented Image at :")
        self.segmented_image.setFont(QFont("Times",13))
        self.segmented_image.setGeometry(0,50,200,25)
        self.segmented_image.move(20,480)

        self.segmented_display = QComboBox(self)
        self.segmented_display.setGeometry(220,483,120,25)
        self.segmented_display .addItem("Main Display")
        self.segmented_display.addItem("Second Display")
        self.segmented_display.addItem("Third Display")

        self.object_d_image = QtWidgets.QLabel(self)
        self.object_d_image.setText("Show Detected Image at :")
        self.object_d_image.setFont(QFont("Times",13))
        self.object_d_image.setGeometry(0,50,200,25)
        self.object_d_image.move(20,520)

        self.object_d_display = QComboBox(self)
        self.object_d_display.setGeometry(220,523,120,25)
        self.object_d_display .addItem("Main Display")
        self.object_d_display.addItem("Second Display")
        self.object_d_display.addItem("Third Display")

        self.dis_change =QPushButton("Change Display",self)
        self.dis_change.pressed.connect(self.img)
        self.dis_change.setGeometry(100,200,120,30)
        self.dis_change.move(340, 440)

        self.dis_change2 = QPushButton("Change Display", self)
        self.dis_change2.pressed.connect(self.img)
        self.dis_change2.setGeometry(100, 200, 120, 30)
        self.dis_change2.move(360, 480)

        self.dis_change3 = QPushButton("Change Display", self)
        self.dis_change3.pressed.connect(self.img)
        self.dis_change3.setGeometry(100, 200, 120, 30)
        self.dis_change3.move(360, 520)

        self.dis_change.pressed.connect(self.display_set_orgi)
        self.dis_change2.pressed.connect(self.display_set_seg)
        self.dis_change3.pressed.connect(self.display_set_tr)

        ## Show my image button

        self.show_but =QPushButton("Show",self)
        self.show_but.pressed.connect(self.img)
        self.show_but.setGeometry(100,200,100,50)
        self.show_but.move(665, 540)

        ## U-net button

        self.u_net_but =QPushButton("Generator",self)
        self.u_net_but.move(500,560)
        self.u_net_but.pressed.connect(self.img_u_net)
        self.u_net_but.setGeometry(100, 200, 100, 50)
        self.u_net_but.move(540, 540)



        self.initUI()
    

    def initUI(self):
        #Drag and drop section settings
        button = Button("Drag and Drop \n Here",self)
        button.resize(130,90)
        button.move(650, 440)

    def display_set_orgi(self):
        orginal = str(self.orginal_display.currentText())

        if orginal == "Main Display":
            self.pixmap = QPixmap('img.jpg')
            self.main_display.setPixmap(self.pixmap)
            self.main_display.setScaledContents(True)
        elif orginal == "Second Display":
            self.pixmap = QPixmap('img.jpg')
            self.second_display.setPixmap(self.pixmap)
            self.second_display.setScaledContents(True)
        elif orginal == "Third Display":
            self.pixmap = QPixmap('img.jpg')
            self.third_display.setPixmap(self.pixmap)
            self.third_display.setScaledContents(True)
    def display_set_seg(self):

        secment = str(self.segmented_display.currentText())

        if secment == "Main Display":
            self.pixmap_second = QPixmap('u_net_img.jpeg')
            self.main_display.setPixmap(self.pixmap_second)
            self.main_display.setScaledContents(True)
        elif secment == "Second Display":
            self.pixmap_second = QPixmap("u_net_img.jpeg")
            self.second_display.setPixmap(self.pixmap_second)
            self.second_display.setScaledContents(True)
        elif secment == "Third Display":
            self.pixmap_second = QPixmap('u_net_img.jpeg')
            self.third_display.setPixmap(self.pixmap_second)
            self.third_display.setScaledContents(True)

    def display_set_tr(self):
        object_tr = str(self.object_d_display.currentText())

        if object_tr == "Main Display":
            self.pixmap_third = QPixmap('track_img.jpeg')
            self.main_display.setPixmap(self.pixmap_third)
            self.main_display.setScaledContents(True)
        elif object_tr == "Second Display":
            self.pixmap_third = QPixmap('track_img.jpeg')
            self.second_display.setPixmap(self.pixmap_third)
            self.second_display.setScaledContents(True)
        elif object_tr == "Third Display":
            self.pixmap_third = QPixmap('track_img.jpeg')
            self.third_display.setPixmap(self.pixmap_third)
            self.third_display.setScaledContents(True)


    def img(self):

        self.pixmap = QPixmap('img.jpg')
        self.main_display.setPixmap(self.pixmap)
        self.main_display.setScaledContents(True)


    def img_u_net(self):
        self.second_display.setText("Please wait...")

        X = np.zeros((1,img_h,img_w,img_c), dtype=np.uint8)
        img_read = cv2.imread("img.jpg")
        img_size = cv2.resize(img_read,(img_h,img_w))
        img_array = np.array(img_size)
        X[0] = img_array
        y_pred_a = model.predict(X)
        display_unet = np.dstack([y_pred_a[0], y_pred_a[0], y_pred_a[0]])
        try:
            os.remove("u_net_img.jpeg")
            imageio.imwrite('u_net_img.jpeg', display_unet)
            self.pixmap_second = QPixmap('u_net_img.jpeg')
            self.second_display.setPixmap(self.pixmap_second)
            self.second_display.setScaledContents(True)
        except:
            imageio.imwrite('u_net_img.jpeg', display_unet)
            self.pixmap_second = QPixmap('u_net_img.jpeg')
            self.second_display.setPixmap(self.pixmap_second)
            self.second_display.setScaledContents(True)


        ### Image Class

        your_fish_name = find_class("img.jpg")
        fish_name_print = "Your fish type is " + str(your_fish_name)
        self.fish_class_names.setText(fish_name_print)
        ### object select
        my_img = cv2.imread("img.jpg")
        my_img_sized = cv2.resize(my_img,(img_h,img_w))
        img = cv2.imread("u_net_img.jpeg")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([0, 0, 48])
        upper_blue = np.array([180, 180, 180])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        contours, hierachy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(my_img_sized, contours, -1, (0, 255, 0), 1)
        try:
            os.remove("track_img.jpeg")
            cv2.imwrite("track_img.jpeg", my_img_sized)
        except:
            cv2.imwrite("track_img.jpeg", my_img_sized)

        self.pixmap_third = QPixmap("track_img.jpeg")
        self.third_display.setPixmap(self.pixmap_third)
        self.third_display.setScaledContents(True)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Main_space()
    ex.show()
    app.exec_()