import tkinter as tk
from tkinter import *

import cv2
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential

emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('./emotion_model.h5')

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}


emoji_dist={0:"./emojis/angry.png",2:"./emojis/disgusted.png",2:"./emojis/fearful.png",3:"./emojis/happy.png",4:"./emojis/neutral.png",5:"./emojis/sad.png",6:"./emojis/surpriced.png"}

global last_frame1                                    
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text=[0]
# --------------------------------------
def show_vid():
    cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap1.isOpened():
        print("cant open the camera1")
    flag1, frame1 = cap1.read()

    frame1 = cv2.resize(frame1,(500,400),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)

    bounding_box = cv2.CascadeClassifier('c:\\users\\islam\\appdata\\local\\programs\\python\\python38\\lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = emotion_model.predict(cropped_img)

        maxindex = int(np.argmax(prediction))
        cv2.putText(frame1, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        show_text[0]=maxindex
    if flag1 is None:
        print ("Major error!")
    elif flag1:
        global last_frame1
        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_frame.imgtk = imgtk
        camera_frame.configure(image=imgtk)
        camera_frame.after(10, show_vid2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

# -------------------------------------------
def show_vid2():
    frame2=cv2.imread(emoji_dist[show_text[0]])
    pic2=cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
    img2=Image.fromarray(pic2)
    imgtk2=ImageTk.PhotoImage(image=img2)
    emoji_frame.imgtk2=imgtk2
    emoji_title.configure(text=emotion_dict[show_text[0]], font=('arial', 30, 'bold'))
    
    emoji_frame.configure(image=imgtk2)
    emoji_frame.after(10, show_vid)

if __name__ == '__main__':
    root=tk.Tk()
    start_y=100
    heading2=Label(root,text="Photo to Emoji",pady=20, font=('arial',45,'bold'),bg='black',fg='#CDCDCD')
    heading2.pack()

    camera_frame = tk.Label(master=root, padx=50, bd=10)
    emoji_frame = tk.Label(master=root, bd=10)

    emoji_title=tk.Label(master=root, bd=10, fg="#CDCDCD", bg='black')
    camera_frame.pack(side=LEFT)
    camera_frame.place(x=10, y=start_y+70)

    emoji_title.pack()
    emoji_title.place(x=700, y=start_y)

    emoji_frame.pack(side=RIGHT)
    emoji_frame.place(x=600, y=start_y+70)
    


    root.title("Photo To Emoji")
    root.geometry("1080x680+100+10")
    root.resizable(0, 0)
    root['bg']='black'
    exitbutton = Button(root, text='Quit',fg="red",command=root.destroy,font=('arial',25,'bold')).pack(side = BOTTOM)
    show_vid()
    show_vid2()
    root.mainloop()