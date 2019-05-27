precondition: face images are resized to 64 by 64
1. run create_gender_csv.py to build a gender csv file
2. run face.py, age.py and gender.py to get 3 models in hdf5 format separately
3. run human_face_gui.py to interactively select a image for demo (haarcascade_frontalface_default.xml from opencv required)