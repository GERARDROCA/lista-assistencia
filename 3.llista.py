import cv2
import os
import tkinter as tk

datapath = ' aquí va el dataset' ##############
imagepaths = os.listdir(datapath)
print('imagepath=', imagepaths)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('ModeloFaceFrontalData1.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

faceclassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Llista d'estudiants
estudiants = ['aquí va la llista dels estudiants, en el ordre que estan al dataset']

# Diccionari per a l'estat de llista
estat_llista = {estudiant: 'Desconegut' for estudiant in estudiants}

def actualitzar_llista():
    llista_presentats = []
    llista_absents = []

    for estudiant in estudiants:
        if estat_llista[estudiant] == 'Present':
            llista_presentats.append(estudiant)
        elif estat_llista[estudiant] == 'Absent':
            llista_absents.append(estudiant)

    print("Estudiants presents:", llista_presentats)
    print("Estudiants absents:", llista_absents)

    # Actualitzar la llista a la finestra de l'aplicació
    listbox.delete(0, tk.END)
    for estudiant in estudiants:
        listbox.insert(tk.END, f"{estudiant}: {estat_llista[estudiant]}")

def capturar_frame():
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxframe = gray.copy()
        faces = faceclassif.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            rostro = auxframe[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (150, 105), interpolation=cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)
            cv2.putText(frame, '{}'.format(result), (x, y-5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

            if result[1] < 110:
                nom_estudiant = imagepaths[result[0]].split('.')[0]
                cv2.putText(frame, '{}'.format(nom_estudiant), (x, y-25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Actualitzar l'estat de llista
                if nom_estudiant in estat_llista:
                    estat_llista[nom_estudiant] = 'Present'
                    actualitzar_llista()  # Actualitzar la llista aquí

            else:
                cv2.putText(frame, 'Desconegut', (x, y-20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2.imshow('frame', frame)

        # Programar una nova captura de frame després de 1 mil·lisegon
        window.after(1, capturar_frame)
    else:
        cap.release()
        cv2.destroyAllWindows()

def tancar_finestra():
    cap.release()
    cv2.destroyAllWindows()
    window.destroy()

# Crear la finestra de l'aplicació
window = tk.Tk()
window.title("Aplicació de llista d'estudiants")
window.geometry("400x300")

# Crear un Listbox per mostrar la llista d'estudiants
listbox = tk.Listbox(window)
listbox.pack()

# Botó per actualitzar la llista
btn_actualitzar = tk.Button(window, text="Actualitzar llista", command=actualitzar_llista)
btn_actualitzar.pack()

# Botó per tancar la finestra
btn_tancar = tk.Button(window, text="Tancar finestra", command=tancar_finestra)
btn_tancar.pack()

# Capturar el primer frame
capturar_frame()

# Iniciar el bucle principal de l'aplicació
window.mainloop()