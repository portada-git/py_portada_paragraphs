import os
import numpy as np
from PIL import Image
from .py_portada_utility_for_layout import (
    fill_gaps_in_boxes,
    remove_edge_boxes,
    adjust_box_widths_and_center,
    adjust_box_heights,
    remove_overlapping_segments,
    get_model
)

def process_image(image_path, model, output_dir, imgsz=1024, conf=0.05, device="cpu",
                  iou_threshold=0.5, area_ratio_threshold=0.8, margin=5):
    """
    Procesa una imagen, detecta segmentos y guarda los recortes en el directorio de salida.
    """
    try:
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        det_res = model.predict(image_path, imgsz=imgsz, conf=conf, device=device)
        detections = det_res[0].boxes.data.cpu().numpy()

        kept_detections = remove_overlapping_segments(detections, iou_threshold, area_ratio_threshold)
        boxes = np.array([[int(coord) for coord in det[:4]] for det in kept_detections])

        boxes = fill_gaps_in_boxes(boxes)
        boxes = remove_edge_boxes(boxes)
        boxes = adjust_box_widths_and_center(boxes)
        boxes = adjust_box_heights(boxes)

        with Image.open(image_path) as image:
            img_w, img_h = image.size
            for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
                x1_adj = max(0, x1 - margin)
                y1_adj = max(0, y1 - margin)
                x2_adj = min(img_w, x2 + margin)
                y2_adj = min(img_h, y2 + margin)

                segment = image.crop((x1_adj, y1_adj, x2_adj, y2_adj))
                segment_path = os.path.join(output_dir, f"{base_name}_{i:03d}.jpg")
                segment.save(segment_path)

    except Exception as e:
        print(f"Error procesando {image_path}: {e}")


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
        process_image(image_path, model, output_dir)


if __name__ == '__main__':
    
    input_dir = "./demo/bloques/input"
    output_dir = "./demo/bloques/segmentos"

    process_directory(input_dir, output_dir)
