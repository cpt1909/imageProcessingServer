import cv2
import numpy as np

def face_detect(frame):
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    face = face_classifier.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return frame

def grayscale(frame):
    return cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

def resize(frame,dim):
    resized_imag = cv2.resize(frame, (int(dim['width']), int(dim['height'])),interpolation=cv2.INTER_LANCZOS4)
    return resized_imag

def apply_sepia(image):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_image = cv2.transform(image, sepia_filter)
    sepia_image = np.clip(sepia_image, 0, 255)
    return sepia_image.astype(np.uint8)

def apply_pencil_sketch(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted_image = cv2.bitwise_not(gray_image)
    blurred_image = cv2.GaussianBlur(inverted_image, (21, 21), sigmaX=0, sigmaY=0)
    sketch_image = cv2.divide(gray_image, 255 - blurred_image, scale=256)
    return sketch_image

def apply_cartoon_filter(image):
    smooth_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, blockSize=9, C=2)
    cartoon_image = cv2.bitwise_and(smooth_image, smooth_image, mask=edges)
    return cartoon_image

def reduce_noise(image, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21):
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a valid NumPy array representing an image.")

    denoised_image = cv2.fastNlMeansDenoisingColored(
        image, 
        None, 
        h=h, 
        hColor=hColor, 
        templateWindowSize=templateWindowSize, 
        searchWindowSize=searchWindowSize
    )
    return denoised_image
