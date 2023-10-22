import cv2
import os
import imutils

# Creació de la carpeta de cada usuari (si no existeix)
nom = 'aquí va el nom de la persona' 
datap = 'aquí va la direcció don vols que vagi la carpeta'
alumnep = os.path.join(datap, nom)

if not os.path.exists(alumnep):
    print('Carpeta Creada:', alumnep)
    os.makedirs(alumnep)

# Creació de les carpetes d'entrenament i validació
train_dir = os.path.join(alumnep, 'train')
val_dir = os.path.join(alumnep, 'validation')

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(val_dir):
    os.makedirs(val_dir)

# Inicialització del classificador de cares
faceClasif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicialització de comptadors
count = 0

# Inicialització de les llistes d'imatges d'entrenament i validació
train_images = []
val_images = []

# Inicialització de la càmera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Obrir càmera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=320)  # Redimensionar la imatge
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertir els colors en una escala de grisos
    auxFrame = frame.copy()  # Copiar cada imatge

    # Detectar cares
    cares = faceClasif.detectMultiScale(gray, 1.3, 5)

    # Crear el rectangle de la cara
    for (x, y, w, h) in cares:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cara = auxFrame[y:y + h, x:x + w]
        cara = cv2.resize(cara, (720, 720), interpolation=cv2.INTER_CUBIC)

        # Afegir les imatges a les llistes corresponents
        if count % 5 == 0:  # Utilitza una cada 5 imatges per validació
            val_images.append(cara)
        else:
            train_images.append(cara)

        count += 1

    cv2.imshow('Reconeixement Facial', frame)

    k = cv2.waitKey(1)
    if k == 27 or count >= 300:
        break

cap.release()
cv2.destroyAllWindows()

# Guarda les imatges a les carpetes d'entrenament i validació
for idx, img in enumerate(train_images):
    cv2.imwrite(os.path.join(train_dir, f'cara_{idx}.jpg'), img)

for idx, img in enumerate(val_images):
    cv2.imwrite(os.path.join(val_dir, f'cara_{idx}.jpg'), img)
