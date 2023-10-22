import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk

#Càrrega del model entrenat
model = load_model('aquí va el model.h5')


#Carregar el classificador de cares de Haar
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Persones
alumnes = ['persona1,persona2,persona3']  #PERSONES (en ordre)

#Finestra
finestra = tk.Tk()
finestra.title("LLISTA D'ASSISTÈNCIA PER RECONEIXEMENT FACIAL")
finestra.geometry('400x300')

#Llista de presència amb StringVar(wideget que s'actualitza sol)
ll_pres = []
for class_label in alumnes:
    ll_pres.append(tk.StringVar())
    ll_pres[-1].set('Absent')

#S'afeigexen els noms a la llista
label_list = []
for i, class_label in enumerate(alumnes):
    label = tk.Label(finestra, text=class_label)
    label.pack()
    label_list.append(label)

#Actualitzar l'assitència
def act_assistencia(index, v):
    ll_pres[index].set(v)

# Actualitzar les etiquetes 
def act_labels():
    for i, label in enumerate(label_list):
        label.config(text=f"{alumnes[i]}: {ll_pres[i].get()}") #Combina el nom amb l'assistència
    finestra.after(1000, act_labels) #Actualitzar cada segon



act_labels()

#Iniciar video
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def process_frame():
    #Capturar el fotograma actual
    ret, frame = cap.read()

    if not ret:
        return

    #Convertir la imatge a escala de grisos
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detectar cares a la imatge
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    #Processar cada cara detectada
    for (x, y, w, h) in faces:
        #Extraure la regió de la cara
        cara = frame[y:y + h, x:x + w].copy()

        #Preprocessament de la imatge de la cara
        cara = cv2.resize(cara, (224, 224))
        cara = cara / 255.0  # Normalitzar els píxels
        cara = np.expand_dims(cara, axis=0)

        #Predir l'etiqueta de classe de la cara utilitzant el model
        prediccions = model.predict(cara)
        predicted_class_index = np.argmax(prediccions)
        predicted_class_label = alumnes[predicted_class_index]

        #Actualitzar assistència
        ll_pres[predicted_class_index].set('Present')

    #Mostrar el fotograma amb les deteccions
    cv2.imshow('Reconeixement Facial', frame)

    #Sortir del bucle amb q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return

    #Continuar processant els fotogrames
    finestra.after(1, process_frame)

#Iniciar el processament dels fotogrames
process_frame()

finestra.mainloop()

cap.release()
cv2.destroyAllWindows()
