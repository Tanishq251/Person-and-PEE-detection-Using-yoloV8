from ultralytics import YOLO
import cv2
import os
import numpy as np
import argparse

# Non-maximum suppression function
def non_max_suppression(boxes, scores, iou_threshold):
    sorted_indices = np.argsort(scores)[::-1]
    keep_indices = []

    while sorted_indices.size > 0:
        keep_index = sorted_indices[0]
        keep_indices.append(keep_index)
        iou_values = [calculate_iou(boxes[keep_index], boxes[idx]) for idx in sorted_indices[1:]]
        filtered_indices = np.where(np.array(iou_values) < iou_threshold)[0] + 1
        sorted_indices = sorted_indices[filtered_indices]

    return np.array(keep_indices)

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    x_left = max(x1, x3)
    y_top = max(y1, y3)
    x_right = min(x2, x4)
    y_bottom = min(y2, y4)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)

    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou

def perform_inference(image_dir, output_dir, person_model_path, ppe_model_path):
    person_model = YOLO(person_model_path)
    ppe_model = YOLO(ppe_model_path)

    class_colors = {
        'person': (0, 255, 0),
        'ppe': (0, 0, 255)
    }

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)

            # Person detection
            person_results = person_model(image)[0]
            person_boxes = person_results.boxes.xyxy.cpu().numpy()
            person_scores = person_results.boxes.conf.cpu().numpy()
            person_keep_indices = non_max_suppression(person_boxes, person_scores, iou_threshold=0.5)

            # PPE detection
            ppe_results = ppe_model(image)[0]
            ppe_boxes = ppe_results.boxes.xyxy.cpu().numpy()
            ppe_scores = ppe_results.boxes.conf.cpu().numpy()
            ppe_keep_indices = non_max_suppression(ppe_boxes, ppe_scores, iou_threshold=0.5)

            for idx in person_keep_indices:
                x1, y1, x2, y2 = person_boxes[idx]
                confidence = person_scores[idx]
                color = class_colors.get('person', (0, 255, 0))
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(image, f"Person: {confidence:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            for idx in ppe_keep_indices:
                x1, y1, x2, y2 = ppe_boxes[idx]
                confidence = ppe_scores[idx]
                color = class_colors.get('ppe', (0, 0, 255))
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(image, f"PPE: {confidence:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform object detection using YOLOv8 models.')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the directory containing input images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the directory for saving output images.')
    parser.add_argument('--person_model', type=str, required=True, help='Path to the person detection model.')
    parser.add_argument('--ppe_model', type=str, required=True, help='Path to the PPE detection model.')
    args = parser.parse_args()

    perform_inference(args.image_dir, args.output_dir, args.person_model, args.ppe_model)