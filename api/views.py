from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import base64
import cv2
import numpy as np
import json
from imageProcessing.objectDetection import objectDetection
from imageProcessing import imageTransformation as imgt

@csrf_exempt
def imageProcess(request):
    if request.method == "POST":
        try:
            image = request.FILES.get("image")
            choice = int(request.POST["choice"])
            dim = json.loads(request.POST["dim"])
            
            file_bytes = np.frombuffer(image.read(), np.uint8)
            input_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if choice == 1:
                output_img = imgt.face_detect(input_image)
            elif choice == 2:
                output_img = imgt.grayscale(input_image)
            elif choice == 3:
                output_img = imgt.resize(input_image, dim)
            elif choice == 4:
                output_img = imgt.apply_pencil_sketch(input_image)
            elif choice == 5:
                output_img = imgt.apply_cartoon_filter(input_image)
            elif choice == 6:
                output_img = imgt.apply_sepia(input_image)
            elif choice == 7:
                output_img = imgt.reduce_noise(input_image)
            elif choice == 8:
                output_img = objectDetection(input_image)
            
            success, encoded_image = cv2.imencode('.jpg', output_img)
            image_bytes = encoded_image.tobytes()
            return HttpResponse(image_bytes, content_type='image/jpeg', status=200)
        
        except:
            return HttpResponse("Invalid Request", status=400)
    return HttpResponse("Server Status : Online", status = 200)