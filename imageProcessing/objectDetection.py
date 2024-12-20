import os
import cv2
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
namesPath = os.path.join(script_dir,'coco.names')
cfgPath = os.path.join(script_dir,'yolov3.cfg')
weightsPath = os.path.join(script_dir,'yolov3.weights')


def load_yolo_model(config_path, weights_path,  names_path):
    # Load YOLO network
    net = cv2.dnn.readNet(weights_path, config_path)
    with open(names_path, 'r') as f:
        class_names = f.read().strip().split('\n')
    
    # Get YOLO output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    return net, output_layers, class_names

# Perform object detection
def detect_objects(image, net, output_layers, class_names, conf_threshold=0.5, nms_threshold=0.4):
    # Prepare the image for YOLO model
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)
    
    height, width, channels = image.shape
    boxes = []
    confidences = []
    class_ids = []
    
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply Non-Maxima Suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    objects = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = class_names[class_ids[i]]
            confidence = confidences[i]
            objects.append((label, confidence, (x, y, w, h)))
    
    return objects

# Draw bounding boxes on the image
def draw_boxes(image, objects):
    for obj in objects:
        label, confidence, (x, y, w, h) = obj
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"{label} ({confidence*100:.2f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

# Example Usage
def objectDetection(image):
    # Load YOLO model
    net, output_layers, class_names = load_yolo_model(
        config_path = cfgPath,
        weights_path = weightsPath,
        names_path = namesPath
        )

    # image = cv2.imread('image.jpg')
    
    objects = detect_objects(image, net, output_layers, class_names)
    
    result_image = draw_boxes(image, objects)
    
    # cv2.imwrite("result_image.jpg", result_image)
    return result_image