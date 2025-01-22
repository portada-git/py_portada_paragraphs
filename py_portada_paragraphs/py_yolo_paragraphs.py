from typing import final

import cv2
import numpy as np
from doclayout_yolo import YOLOv10
import os


def create_output_directory(directory):
    """
    Crea un directorio si no existe.
    Args:
        directory (str): Ruta del directorio a crear.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def calculate_iou(box1, box2):
    """
    Calcula la Intersección sobre Unión (IoU) entre dos cajas delimitadoras.
    Args:
        container, box (list): Coordenadas de las cajas en formato [x1, y1, x2, y2].
    Returns:
        tuple: (IoU, área de intersección, área de container, área de box)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    iou = intersection / union if union > 0 else 0

    return iou, intersection, area1, area2


def remove_overlapping_segments(detections, iou_threshold=0.5, area_ratio_threshold=0.8):
    """
    Elimina segmentos superpuestos basándose en IoU y relación de áreas.
    Args:
        detections (list): Lista de detecciones, cada una con formato [x1, y1, x2, y2, conf, class_id].
        iou_threshold (float): Umbral de IoU para considerar superposición.
        area_ratio_threshold (float): Umbral de relación de áreas para considerar superposición.
    Returns:
        list: Lista de detecciones después de eliminar superposiciones.
    """
    sorted_detections = sorted(detections, key=lambda x: x[4], reverse=True)
    kept_detections = []

    for i, detection in enumerate(sorted_detections):
        should_keep = True
        for kept in kept_detections:
            iou, intersection, area1, area2 = calculate_iou(detection[:4], kept[:4])
            if iou > iou_threshold:
                should_keep = False
                break
            if (area1 > area2 and intersection / area2 > area_ratio_threshold) or \
                    (area2 > area1 and intersection / area1 > area_ratio_threshold):
                should_keep = False
                break
        if should_keep:
            kept_detections.append(detection)

    return kept_detections


def add_margin(image, x1, y1, x2, y2, margin):
    """
    Añade un margen alrededor de un segmento, asegurándose de no exceder los límites de la imagen.
    Args:
        image (numpy.ndarray): Imagen original.
        x1, y1, x2, y2 (int): Coordenadas del segmento.
        margin (int): Tamaño del margen a añadir.
    Returns:
        tuple: Nuevas coordenadas del segmento con margen.
    """
    height, width = image.shape[:2]
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(width, x2 + margin)
    y2 = min(height, y2 + margin)
    return x1, y1, x2, y2


def get_model(fpath=None):
    if fpath is None:
        p = os.path.abspath(os.path.dirname(__file__))
        fpath = f"{p}/modelo/doclayout_yolo_docstructbench_imgsz1024.pt"
    return YOLOv10(fpath)


def reprocess_large_segment(model, segment, original_coords, imgsz=1024, conf=0.2, device="cpu"):
    """
    Reprocesa un segmento grande para detectar sub-segmentos.
    Args:
        model: Modelo YOLOv10 para detección.
        segment (numpy.ndarray): Imagen del segmento a reprocesar.
        original_coords (tuple): Coordenadas originales del segmento (x, y).
        imgsz (int): Tamaño de la imagen para el modelo.
        conf (float): Umbral de confianza para las detecciones.
        device (str): Dispositivo para inferencia ("cpu" o "cuda").
    Returns:
        numpy.ndarray: Sub-detecciones ajustadas a las coordenadas originales.
    """
    height, width = segment.shape[:2]
    det_res = model.predict(segment, imgsz=imgsz, conf=conf, device=device)
    sub_detections = det_res[0].boxes.data.cpu().numpy()
    for i in range(len(sub_detections)):
        sub_detections[i][0] += original_coords[0]
        sub_detections[i][1] += original_coords[1]
        sub_detections[i][2] += original_coords[0]
        sub_detections[i][3] += original_coords[1]
    return sub_detections


def is_inside(box1, box2):
    """
    Comprueba si container está completamente dentro de box.
    Args:
        container, box (list): Coordenadas de las cajas en formato [x1, y1, x2, y2].
    Returns:
        bool: True si container está completamente dentro de box.
    """
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    return x1 >= a1 and y1 >= b1 and x2 <= a2 and y2 <= b2


def remove_completely_overlapped_detections(detections):
    """
    Elimina detecciones que están completamente contenidas dentro de otras detecciones.
    Args:
        detections (list): Lista de detecciones.
    Returns:
        list: Lista de detecciones sin las que están completamente dentro de otras.
    """
    kept_detections = []
    num_detections = len(detections)

    for i in range(num_detections):
        detection_i = detections[i]
        box_i = detection_i[:4]
        is_inside_another = False

        for j in range(num_detections):
            if i == j:
                continue
            detection_j = detections[j]
            box_j = detection_j[:4]

            if is_inside(box_i, box_j):
                is_inside_another = True
                break

        if not is_inside_another:
            kept_detections.append(detection_i)

    return kept_detections


def cajas_faltantes(imagen, cajas, umbral=98, margen=0):
    """
    Detecta y crea segmentos para los espacios faltantes entre las cajas detectadas.
    Args:
        imagen (numpy.ndarray): Imagen original.
        cajas (list o numpy.ndarray): Lista de cajas detectadas, cada una con formato [x1, y1, x2, y2].
        umbral (int): Umbral de espacio vertical para considerar un espacio como significativo.
    Returns:
        list: Lista de tuplas (índice, segmento), incluyendo los segmentos originales y los nuevos espacios.
    """
    # Convertir cajas a lista de listas si es un array de numpy
    cajas = cajas.tolist() if isinstance(cajas, np.ndarray) else cajas

    # Ordenar cajas por su coordenada y
    cajas_ordenadas = sorted(cajas, key=lambda x: x[1])
    segmentos = []

    for i, caja in enumerate(cajas_ordenadas):
        x1, y1, x2, y2 = map(int, caja)
        x1 = max(0, x1-margen)
        x2 = min(imagen.shape[1], x2+margen)
        y1 = max(0, y1-margen)
        y2 = min(imagen.shape[0], y2+margen)
        segmento = imagen[y1:y2, x1:x2]
        segmentos.append((i, segmento))
        # Si no es la última caja, verificar el espacio hasta la siguiente
        if i < len(cajas_ordenadas) - 1:
            caja_siguiente = cajas_ordenadas[i + 1]
            espacio = int(caja_siguiente[1]) - y2
            if espacio > umbral:
                # Si hay un espacio significativo, crear un nuevo segmento
                segmento_espacio = imagen[y2:int(caja_siguiente[1]), x1:x2]
                segmentos.append((f"{i}_espacio", segmento_espacio))
        elif i == len(cajas_ordenadas) -1:
            y2 = imagen.shape[0]
            espacio = y2 - y1
            if espacio > umbral:
                # Si hay un espacio significativo, crear un nuevo segmento
                segmento_espacio = imagen[y1:y2, x1:x2]
                segmentos.append((f"{i}_espacio", segmento_espacio))

    return segmentos


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
    all_detections = det_res[0].boxes.data.cpu().numpy()
    kept_detections = remove_overlapping_segments(all_detections, iou_threshold, area_ratio_threshold)
    final_detections = []
    image_height = image.shape[0]

    for detection in kept_detections:
        x1, y1, x2, y2, conf_score, class_id = detection
        segment_height = y2 - y1
        if segment_height > 0.5 * image_height:
            # print(f"Reprocesando segmento grande: {x1}, {y1}, {x2}, {y2}")
            segment = image[int(y1):int(y2), int(x1):int(x2)]
            sub_detections = reprocess_large_segment(model, segment, (x1, y1), imgsz=imgsz, conf=conf, device=device)

            # Agregar sub-detecciones a la lista
            final_detections.extend(sub_detections)
        else:
            final_detections.append(detection)

    # **Nuevo código para eliminar detecciones originales de segmentos grandes**
    # Filtramos las detecciones que no corresponden a segmentos grandes
    filtered_detections = []
    for detection in final_detections:
        x1, y1, x2, y2 = detection[:4]
        segment_height = y2 - y1
        if segment_height <= 0.5 * image_height:
            filtered_detections.append(detection)
    final_detections = filtered_detections

    # **Nuevo código para eliminar detecciones completamente contenidas dentro de otras**
    final_detections = remove_completely_overlapped_detections(final_detections)

    # Ordenar las detecciones finales por coordenada y
    final_detections.sort(key=lambda x: x[1])

    # Aplicar la función cajas_faltantes
    cajas = np.array([detection[:4] for detection in final_detections])
    segmentos_con_espacios = cajas_faltantes(image, cajas, margen=5)
    return segmentos_con_espacios, final_detections, det_res


def extract_fragments_and_get_json(image, model: YOLOv10 = None, imgsz=1024, conf=0.1, device="cpu",
                                   iou_threshold=0.5, area_ratio_threshold=0.8, margin=5):
    if model is None:
        model = get_model()

    segmentos_con_espacios, final_detections, det_res = extract_fragments_from_image(image, model, imgsz, conf, device,
                                                                                     iou_threshold,
                                                                                     area_ratio_threshold,
                                                                                     margin)

    resp = []
    for i, (idx, segmento) in enumerate(segmentos_con_espacios, start=1):
        if isinstance(idx, int):
            # Segmento original
            detection = final_detections[idx]
            x1, y1, x2, y2, conf_score, class_id = detection
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            class_name = det_res[0].names[int(class_id)]
        else:
            # Segmento de espacio
            y1 = y2
            y2 = y1 + segmento.shape[0]
            x1, x2 = 0, image.shape[1]
            class_name = "espacio"
            x1, y1, x2, y2 = add_margin(image, x1, y1, x2, y2, margin)

        message = f"Found fragment {i} of type {class_name}"
        resp.append({"id": i, "class": class_name, "image": segmento, "block": [x1, y1, x2, y2], "message": message})

    return resp


def extract_and_save_segments(image_path, output_dir, model: YOLOv10 = None, imgsz=1024, conf=0.2, device="cpu",
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

    # det_res = model.predict(image_path, imgsz=imgsz, conf=conf, device=device)
    # all_detections = det_res[0].boxes.data.cpu().numpy()
    # kept_detections = remove_overlapping_segments(all_detections, iou_threshold, area_ratio_threshold)
    # final_detections = []
    # image_height = image.shape[0]
    #
    # for detection in kept_detections:
    #     x1, y1, x2, y2, conf_score, class_id = detection
    #     segment_height = y2 - y1
    #     if segment_height > 0.5 * image_height:
    #         print(f"Reprocesando segmento grande: {x1}, {y1}, {x2}, {y2}")
    #         segment = image[int(y1):int(y2), int(x1):int(x2)]
    #         sub_detections = reprocess_large_segment(model, segment, (x1, y1), imgsz=imgsz, conf=conf, device=device)
    #
    #         # Agregar sub-detecciones a la lista
    #         final_detections.extend(sub_detections)
    #     else:
    #         final_detections.append(detection)
    #
    # # **Nuevo código para eliminar detecciones originales de segmentos grandes**
    # # Filtramos las detecciones que no corresponden a segmentos grandes
    # filtered_detections = []
    # for detection in final_detections:
    #     x1, y1, x2, y2 = detection[:4]
    #     segment_height = y2 - y1
    #     if segment_height <= 0.5 * image_height:
    #         filtered_detections.append(detection)
    # final_detections = filtered_detections
    #
    # # **Nuevo código para eliminar detecciones completamente contenidas dentro de otras**
    # final_detections = remove_completely_overlapped_detections(final_detections)
    #
    # # Ordenar las detecciones finales por coordenada y
    # final_detections.sort(key=lambda x: x[1])
    #
    # # Aplicar la función cajas_faltantes
    # cajas = np.array([detection[:4] for detection in final_detections])
    # segmentos_con_espacios = cajas_faltantes(image, cajas)

    segmentos_con_espacios, final_detections, det_res = extract_fragments_from_image(image, model, imgsz, conf, device,
                                                                                     iou_threshold,
                                                                                     area_ratio_threshold,
                                                                                     margin)

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    create_output_directory(output_dir)

    for i, (idx, segmento) in enumerate(segmentos_con_espacios, start=1):
        if isinstance(idx, int):
            # Segmento original
            detection = final_detections[idx]
            x1, y1, x2, y2, conf_score, class_id = detection
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            class_name = det_res[0].names[int(class_id)]
        else:
            # Segmento de espacio
            y1, y2 = segmento.shape[:2]
            x1, x2 = 0, image.shape[1]
            class_name = "espacio"
            x1, y1, x2, y2 = add_margin(image, x1, y1, x2, y2, margin)

        output_filename = f"{output_dir}/{image_name}_{i:03d}_{class_name}.jpg"
        cv2.imwrite(output_filename, segmento)
        print(f"Segmento guardado: {output_filename}")


def process_directory(input_dir, output_dir, model: YOLOv10 = None, **kwargs):
    """
    Procesa todas las imágenes en un directorio.
    """
    if model is None:
        model = get_model()

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_path = os.path.join(input_dir, filename)
            print(f"Procesando imagen: {image_path}")
            extract_and_save_segments(image_path, output_dir, model, **kwargs)


def main():
    """
    Función principal que configura y ejecuta el proceso de segmentación de documentos.
    """
    filepath = "modelo/doclayout_yolo_docstructbench_imgsz1024.pt"
    input_dir = "demo"
    output_dir = "segmentos_yolo"
    iou_threshold = 0.3
    area_ratio_threshold = 0.7
    margin = 5  # Margen en píxeles

    model = YOLOv10(filepath)
    process_directory(input_dir, output_dir, model,
                      iou_threshold=iou_threshold,
                      area_ratio_threshold=area_ratio_threshold,
                      margin=margin)


if __name__ == "__main__":
    main()
