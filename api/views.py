from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import base64
import cv2
import numpy as np
import json
from imageProcessing.objectDetection import objectDetection
from imageProcessing import imageTransformation as imgt

def base64_to_image(base64_str):
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

def image_to_base64(img):
    _, buffer = cv2.imencode('.PNG', img)
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return base64_str

@csrf_exempt
def imageProcess(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            base64String = data['image']
            choice = int(data['choice'])
            dimensions = data['dim']
            
            if base64String:
                img = base64_to_image(base64String)
                if choice == 1:
                    processed_img = imgt.face_detect(img)
                elif choice == 2:
                    processed_img = imgt.grayscale(img)
                elif choice == 3:
                    processed_img = imgt.resize(img,dimensions)
                elif choice == 4:
                    processed_img = imgt.apply_pencil_sketch(img)
                elif choice == 5:
                    processed_img = imgt.apply_cartoon_filter(img)
                elif choice == 6:
                    processed_img = imgt.apply_sepia(img)
                elif choice == 7:
                    processed_img = imgt.reduce_noise(img)
                elif choice == 8:
                    processed_img = objectDetection(img)


                processed_base64 = image_to_base64(processed_img)
                
                return JsonResponse(
                    {'message':'OK',
                        'output':processed_base64,
                        }, status=200)
        except:
            return JsonResponse({'message':'Invalid Image Parameters / File Size Limit Exceeded !!'}, status=400)
    return HttpResponse("Server Status : ACTIVE")