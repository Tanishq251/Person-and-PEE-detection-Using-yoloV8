import os
import xml.etree.ElementTree as ET
import argparse

def convert_voc_to_yolo(voc_dir, yolo_dir):
    # Check if input directory exists
    if not os.path.exists(voc_dir):
        print(f"Input directory {voc_dir} does not exist.")
        return

    # Create output directory if it does not exist
    if not os.path.exists(yolo_dir):
        os.makedirs(yolo_dir)
        print(f"Created output directory {yolo_dir}")

    # Read classes
    classes_file = os.path.join(voc_dir, "../classes.txt")
    if not os.path.exists(classes_file):
        print(f"Classes file {classes_file} does not exist.")
        return

    with open(classes_file, "r") as f:
        classes = f.read().strip().split()
        print(f"Classes: {classes}")
    
    annotations_dir = voc_dir
    
    # Check if annotations directory exists and contains XML files
    if not os.path.exists(annotations_dir):
        print(f"Annotations directory {annotations_dir} does not exist.")
        return

    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
    if not xml_files:
        print(f"No XML files found in {annotations_dir}.")
        return

    for ann_file in xml_files:
        print(f"Processing file: {ann_file}")
        tree = ET.parse(os.path.join(annotations_dir, ann_file))
        root = tree.getroot()
        
        image_id = root.find("filename").text.replace(".jpg", "")
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        
        with open(os.path.join(yolo_dir, f"{image_id}.txt"), "w") as yolo_f:
            for obj in root.findall("object"):
                cls = obj.find("name").text
                if cls not in classes:
                    continue
                cls_id = classes.index(cls)
                
                xmlbox = obj.find("bndbox")
                xmin = int(xmlbox.find("xmin").text)
                xmax = int(xmlbox.find("xmax").text)
                ymin = int(xmlbox.find("ymin").text)
                ymax = int(xmlbox.find("ymax").text)
                
                x_center = (xmin + xmax) / 2.0 / width
                y_center = (ymin + ymax) / 2.0 / height
                w = (xmax - xmin) / float(width)
                h = (ymax - ymin) / float(height)
                
                yolo_f.write(f"{cls_id} {x_center} {y_center} {w} {h}\n")
                print(f"Written annotation for {image_id} in YOLO format.")

    print(f"Conversion completed. YOLO annotations are saved in {yolo_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Path to input PascalVOC annotations directory")
    parser.add_argument("output_dir", help="Path to output YOLOv8 directory")
    args = parser.parse_args()
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    
    convert_voc_to_yolo(args.input_dir, args.output_dir)
