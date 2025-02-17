import os
import numpy as np
import cv2
from doclayout_yolo import YOLOv10
from .py_portada_utility_for_layout import (
    fill_gaps_in_boxes,
    remove_edge_boxes,
    adjust_box_widths_and_center,
    adjust_box_heights,
    remove_overlapping_segments,
)

def get_model(fpath=None):
    """
    Retorna una instancia de YOLOv10 utilizando la ruta especificada.
    Si no se proporciona una ruta, se utiliza una ruta predeterminada.
    """
    if fpath is None:
        p = os.path.abspath(os.path.dirname(__file__))
        fpath = f"{p}/modelo/doclayout_yolo_docstructbench_imgsz1024.pt"
    return YOLOv10(fpath)

def raw_predictions(image, model: YOLOv10 = None, imgsz=1024, conf=0.2, device="cpu",
                    iou_threshold=0.5, area_ratio_threshold=0.8, margin=5):
    if model is None:
        model = get_model()
    det_res = model.predict(image, imgsz=imgsz, conf=conf, device=device)
    all_detections = det_res[0].boxes.data.cpu().numpy()
    kept_detections = remove_overlapping_segments(all_detections, iou_threshold, area_ratio_threshold)
    ret_boxes = []
    ret_properties = []
    for detection in kept_detections:
        x1, y1, x2, y2, conf_score, class_id = detection
        ret_boxes.append([int(x1), int(y1), int(x2), int(y2)])
        ret_properties.append({'box':[int(x1), int(y1), int(x2), int(y2)], 'conf_score':conf_score ,'class_id':class_id, 'class_name':det_res[0].names[class_id],'detection':detection})

    return ret_boxes, ret_properties


def extract_fragments_from_image(image, model: YOLOv10 = None, imgsz=1024, conf=0.1, device="cpu",
                                 iou_threshold=0.5, area_ratio_threshold=0.8, margin=5):
    if model is None:
        model = get_model()

    det_res = model.predict(image, imgsz=imgsz, conf=conf, device=device)
    detections = det_res[0].boxes.data.cpu().numpy()

    kept_detections = remove_overlapping_segments(detections, iou_threshold, area_ratio_threshold)
    boxes = np.array([[int(coord) for coord in det[:4]] for det in kept_detections])

    if len(boxes)>0:
        boxes = fill_gaps_in_boxes(boxes)
        boxes = remove_edge_boxes(boxes)
        boxes = adjust_box_widths_and_center(boxes)
        boxes = adjust_box_heights(boxes)

    return boxes

def extract_fragments_and_get_json(image, model: YOLOv10 = None, imgsz=1024, conf=0.05, device="cpu",
                                   iou_threshold=0.5, area_ratio_threshold=0.8, margin=5):
    if model is None:
        model = get_model()

    boxes = extract_fragments_from_image(image, model, imgsz, conf, device, iou_threshold, area_ratio_threshold, margin)
    resp = []
    if len(boxes)==0:
        message = f"Paragraphs not found. Original fragment is added"
        resp.append({"id": 0, "image": image, "message": message})
    else:
        for i, (x1, y1, x2, y2)  in enumerate(boxes, start=1):
            x1_adj = max(0, x1 - margin)
            y1_adj = max(0, y1 - margin)
            x2_adj = min(image.shape[1], x2 + margin)
            y2_adj = min(image.shape[0], y2 + margin)
            segment = image[y1_adj:y2_adj, x1_adj:x2_adj]
            message = f"Found fragment {i}"
            resp.append({"id": i, "image": segment, "message": message})
    return resp


def extract_and_save_segments(image_path, output_dir, model: YOLOv10 = None, imgsz=1024, conf=0.05, device="cpu",
                              iou_threshold=0.5, area_ratio_threshold=0.8, margin=5):
    """
    Extrae y guarda segmentos de una imagen utilizando un modelo YOLOv10, incluyendo la detección de espacios faltantes.
    """
    if model is None:
        model = get_model()

    image = cv2.imread(image_path)
    if image is None:
        print(f"No se pudo cargar la imagen: {image_path}")
        return
    boxes = extract_fragments_from_image(image, model, imgsz, conf, device, iou_threshold, area_ratio_threshold, margin)

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (x1, y1, x2, y2)  in enumerate(boxes, start=1):
        x1_adj = max(0, x1 - margin)
        y1_adj = max(0, y1 - margin)
        x2_adj = min(image.shape[1], x2 + margin)
        y2_adj = min(image.shape[0], y2 + margin)
        segment = image[y1_adj:y2_adj, x1_adj:x2_adj]
        segment_path = os.path.join(output_dir, f"{image_name}_{i:03d}.jpg")
        cv2.imwrite(segment_path, segment)


def process_directory(input_dir, output_dir):
    """
    Procesa todas las imágenes en el directorio de entrada de forma secuencial.
    """
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted(
        [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))
        ]
    )

    model = get_model()

    for image_path in image_files:
        extract_and_save_segments(image_path, model, output_dir)


if __name__ == '__main__':
    
    input_dir = "./demo/bloques/input"
    output_dir = "./demo/bloques/segmentos"

    process_directory(input_dir, output_dir)
