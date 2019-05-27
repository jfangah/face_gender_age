from tkinter import filedialog
from tkinter import *
from keras.models import load_model
import cv2 as cv
import numpy as np
from keras.preprocessing.image import img_to_array
from PIL import Image, ImageTk
import os



def parse_age(age_tensor):
	for i in range(len(age_tensor)):
		if age_tensor[i] < 0.5:
			return i + 1
	return 80

def ROI(path):
	face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
	img = cv.imread(path)
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 2)
	if len(faces) > 0:
		flag = 0
		for x, y, w, h in faces:
			if w > 0.2 * img.shape[1] and h >= 0.2*img.shape[0]:
				if w >= 0.6*img.shape[1] and h >= 0.6*img.shape[0]:
					new_img = gray
					flag = 1
					break
				bound = 15
				if y-30<=0:
					new_img = gray[y:y+h+2*bound, x-10:x+w+10]
					flag = 1
				else:
					new_img = gray[y-2*bound:y+h+20, x-10:x+w+10]
					flag = 1
				break
		if flag == 0:
			new_img = gray
	else:
		new_img = gray
	if new_img.shape[0]<64 and new_img.shape[1]<64:
		new_img = gray
	return new_img
	
def parse_img(path):
	img = ROI(path)
	new_img = cv.resize(img, (64,64))
	x = new_img.reshape(1,64,64,1)
	x = x / 255.0
	return x

def read():
	f = filedialog.askopenfilename(
        parent=root, initialdir='C:/Tutorial',
        title='Choose file',
        filetypes=(("jpeg files","*.jpg"),("all files","*.*"))
        )
	if f == '':
		return
	new_window = Toplevel(root)
	new_window.title(f)
	image = ImageTk.PhotoImage(Image.open(f))
	l1 = Label(new_window, image=image)
	l1.image = image
	l1.pack()
	
	data = parse_img(f)
	is_face = bool(face_model.predict(data)[0][0]>0.5)
	age = parse_age(age_model.predict(data)[0])
	gt = gender_model.predict(data)[0][0]
	#print(gt)
	if is_face == False:
		face_text.configure(text='is face? False')
		age_text.configure(text='age: UnKnown')
		gender_text.configure(text='gender: UnKnown')
		return
	else:
		if gt >= 0.5:
			gender = 'male'
		else:
			gender = 'female'
		face_text.configure(text='is face? True')
		age_text.configure(text='age: '+str(age))
		gender_text.configure(text='gender: '+gender)
		
		return
	

if __name__ == "__main__":
	root = Tk()
	face_text = Label(root, text='is face? ', anchor='w', font=('Times',20))
	age_text = Label(root, text='age: ', anchor='w', font=('Times',20))
	gender_text = Label(root, text='gender:', anchor='w', font=('Times',20))
	face_model = load_model('face_model.h5')
	age_model = load_model('age_model.h5')
	gender_model = load_model('gender_model.h5')

	b = Button(root, text="please select an image", command=read, font=('Times',15, 'bold'))
	b.pack()
	face_text.pack(fill='both')
	age_text.pack(fill='both')
	gender_text.pack(fill='both')

	root.mainloop()