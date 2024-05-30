import os
import cv2
import shutil

def read_annotations(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    annotations = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        bbox = list(map(float, parts[1:]))
        annotations.append((class_id, bbox))
    return annotations

def save_annotations(file_path, annotations):
    with open(file_path, 'w') as file:
        for class_id, bbox in annotations:
            bbox_str = ' '.join(map(str, bbox))
            file.write(f"{class_id} {bbox_str}\n")

def crop_person(image, bbox):
    h, w, _ = image.shape
    x_center, y_center, width, height = bbox
    x1 = int((x_center - width / 2) * w)
    y1 = int((y_center - height / 2) * h)
    x2 = int((x_center + width / 2) * w)
    y2 = int((y_center + height / 2) * h)
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image, (x1, y1, x2 - x1, y2 - y1)

def adjust_annotations(ppe_annotations, person_bbox, image_shape, cropped_shape):
    adjusted_annotations = []
    orig_w, orig_h = image_shape[1], image_shape[0]
    crop_x, crop_y, crop_w, crop_h = person_bbox

    for class_id, bbox in ppe_annotations:
        x_center, y_center, width, height = bbox
        x_center = x_center * orig_w
        y_center = y_center * orig_h
        width = width * orig_w
        height = height * orig_h

        x_center = x_center - crop_x
        y_center = y_center - crop_y

        x_center = x_center / crop_w
        y_center = y_center / crop_h
        width = width / crop_w
        height = height / crop_h

        if x_center > 0 and x_center < 1 and y_center > 0 and y_center < 1:
            adjusted_annotations.append((class_id, [x_center, y_center, width, height]))

    return adjusted_annotations

def process_images(images_dir, annotations_dir, output_images_dir, output_annotations_dir):
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_annotations_dir, exist_ok=True)

    for image_name in os.listdir(images_dir):
        if not image_name.endswith('.jpg') and not image_name.endswith('.png'):
            continue

        image_path = os.path.join(images_dir, image_name)
        annotation_path = os.path.join(annotations_dir, image_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        image = cv2.imread(image_path)
        annotations = read_annotations(annotation_path)

        person_annotations = [ann for ann in annotations if ann[0] == 0]  # Assuming '0' is the person class
        ppe_annotations = [ann for ann in annotations if ann[0] != 0]

        for i, (class_id, person_bbox) in enumerate(person_annotations):
            cropped_image, crop_bbox = crop_person(image, person_bbox)
            adjusted_annotations = adjust_annotations(ppe_annotations, crop_bbox, image.shape, cropped_image.shape)
            cropped_image_name = image_name.replace('.jpg', f'_{i}.jpg').replace('.png', f'_{i}.png')
            cropped_image_path = os.path.join(output_images_dir, cropped_image_name)
            cropped_annotation_path = os.path.join(output_annotations_dir, cropped_image_name.replace('.jpg', '.txt').replace('.png', '.txt'))

            cv2.imwrite(cropped_image_path, cropped_image)
            save_annotations(cropped_annotation_path, adjusted_annotations)

# Example usage:
images_dir = './images'
annotations_dir = './annotations'
output_images_dir = './cropped_images'
output_annotations_dir = './cropped_annotations'
process_images(images_dir, annotations_dir, output_images_dir, output_annotations_dir)
