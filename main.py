#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from keras.preprocessing import image
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk,Image
import numpy as np
import cv2,os,glob

model=tf.keras.models.load_model("my_model_80per.h5")


# In[ ]:


import time

def activity():
    
    global my_img_logo
    global my_img_plate
    global path_logo
    global path_plate
    global button_exit
    global start_time
    global detection_time

    start_time = time.time()
    net = cv2.dnn.readNet('Detection//yolov3_custom_last.weights', 'Detection//yolov3_custom.cfg')

    classes = []
    with open('Detection//obj.names') as f:
        classes=f.read().splitlines()

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(100, 3))

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    img=cv2.imread(path)

    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    #output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])

            if label=="Number plate":
                crop=img[y:y+h,x:x+w]
                path_plate="Numberplate//Numberplate.jpeg"
                cv2.imwrite(path_plate,crop)
                #print(path_plate)
                confidence = str(round(confidences[i],2))
                color = colors[i]

            elif label=="Logo":
                crop=img[y-3:y+h,x:x+w]
                path_logo="Logo//logo.jpeg"
                cv2.imwrite(path_logo,crop)
                #print(path_logo)
                confidence = str(round(confidences[i],2))
                color = colors[i]
        detection_time=str(int(time.time()-start_time))+" sec"
        print(detection_time)
        start_time=time.time()
    
    if os.path.isfile(path_logo):
        my_img_logo=ImageTk.PhotoImage(Image.open(path_logo).resize((100,100)))
    else:
        my_img_logo=ImageTk.PhotoImage(Image.open("sorry.png").resize((100,100)))
    if os.path.isfile(path_plate):
        my_img_plate=ImageTk.PhotoImage(Image.open(path_plate).resize((200,100)))
    else:
        my_img_plate=ImageTk.PhotoImage(Image.open("sorry.png").resize((100,100)))
    
    showresult()
    
    #button_exit=Button(root,text="For pics",command=showresult)
    #button_exit.grid(row=1,column=1)

def showresult():
    
    global label_plate
    global label_logo
    global my_label_logo
    global my_label_plate
    global button_exit
    global path_logo
    global path_plate
    global logo_name
    global plate_name
    global time_info
    
    label_1.grid_forget()
    
    label_logo=Label(text="logo",font = "Calibri 11",fg = "blue",underline=True)
    label_logo.grid(row=2,column=0)
    
    label_plate=Label(text="Numberplate",font = "Calibri 11",fg = "blue",underline=True)
    label_plate.grid(row=2,column=2)
    
    logo_name=Label(text=logo_recog(),font = "Calibri 11",fg = "red")
    logo_name.grid(row=4,column=0)
    
    plate_name=Label(text="None",font = "Calibri 11",fg = "red")
    plate_name.grid(row=4,column=2)
    
    time_info=Label(text="Time info:\n\nDetection[Logo & Plate]: "+detection_time+"\n Logo-recognition : "+recognition,font = "Calibri 11",fg = "black")
    time_info.grid(row=3,column=1)
    
    my_label_logo=Label(image=my_img_logo)
    my_label_logo.grid(row=3,column=0)

    my_label_plate=Label(image=my_img_plate)
    my_label_plate.grid(row=3,column=2)
    
    button_exit.grid_forget()
    button_exit=Button(root,text="clear result",command=clearresult)
    button_exit.grid(row=1,column=1)

def logo_recog():
    global path_logo
    global path_plate
    global label
    global recognition
    if os.path.isfile("Logo//logo.jpeg"):
        classes=["hyundai","Kia","benz","bmw","lexus","nissan","toyota"]
        img_org=image.load_img(path_logo,target_size=(224,224))
        img=np.expand_dims(image.img_to_array(img_org)/255,axis=0)
        test=cv2.imread(path)

        if model.predict(img)[0,np.argmax(model.predict(img))]>0.70:
            Class_name=classes[np.argmax(model.predict(img))]
            Class_Confidence=model.predict(img)[0,np.argmax(model.predict(img))]
            print(Class_name+" "+str(Class_Confidence*100)[0:5]+"%")
            current_time=time.time()
            recognition=str(int(time.time() - start_time))+" sec "
            label=Class_name+" "+str(Class_Confidence*100)[0:5]+"% ["+str(int(time.time() - start_time))+" sec ]"
        else:
            recognition=" --- "
            label= "Cant recognise"
    else:
        recognition=" --- "
        label= "Cant recognise"
    return label
        

    
def clearresult():
    my_label_logo.grid_forget()
    my_label_plate.grid_forget()
    button_exit.grid_forget()
    logo_name.grid_forget()
    plate_name.grid_forget()
    time_info.grid_forget()
    
def clearcontent():
    files = glob.glob('Logo//*.jpeg', recursive=True)
    files += glob.glob('Logo//*.jpg', recursive=True)
    files += glob.glob('Numberplate//*.jpeg', recursive=True)
    files += glob.glob('Numberplate//*.jpg', recursive=True)

    for f in files:
        try:
            os.remove(f)
            print(f+" deleted")
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))
            
def browseFiles():
    global my_label
    global my_img
    global button_exit
    global path
    
    
    clearcontent()
    
    path = filedialog.askopenfilename(initialdir = "C:/Users/valiy/JupytorFiles/Neural Network Practice/RNN/logo_plate_det", 
                                          title = "Select a File",filetypes = (("all files","*.*"),("Text files","*.txt*")))
    my_label.grid_forget()
    my_img=ImageTk.PhotoImage(Image.open(path).resize((600,400)))
    my_label=Label(image=my_img)
    my_label.grid(row=0,column=0,columnspan=3)
    
    button_exit=Button(root,text="Process",command=activity)
    button_exit.grid(row=1,column=1)

root=Tk()
root.geometry("600x600")

my_img=ImageTk.PhotoImage(Image.open("wallpaper.jpg").resize((600,400)))
my_label=Label(image=my_img)
my_label.grid(row=0,column=0,columnspan=3)

path=""
path_plate=""
path_logo=""

button_back=Button(root,text="Select Image",command=browseFiles)
button_forwd=Button(root,text="Exit",command=root.destroy)

button_back.grid(row=1,column=0)
button_forwd.grid(row=1,column=2)

label_1=Label(text="Hit the Proces Button to extract Numberplate and logo")
label_1.grid(row=2,column=1)
 
root.mainloop()

