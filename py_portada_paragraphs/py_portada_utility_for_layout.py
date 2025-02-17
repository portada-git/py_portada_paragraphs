from typing import Union
import numpy as np

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


def calculate_coordinates(p1: Union[int, list] = 0, p2: Union[int, list] = 0, x2=0, y2=0):
    """
    Calculate coordinate values with different type of parameters
    :param p1: Can be an int,  a list, or zero. If its type is int, represents the value of x1 (left border). If its
    type is a list, represents a set of box. If the list has 2 positions represents the left/top vertex. if the
    list has 4 positions represents the vertex of the diagonal of a rectangular area as x1, y1, x2, y2.
    :param p2: Can be an int,  a list, or zero. If its type is int, represents the value of y1 (left border). If its
    type is a list, represents the box of the right/bottom vertex of the rectangular area.
    :param x2: represents x2 coordinate of the rectangular area (right border)
    :param y2: represents y2 coordinate of the rectangular area (bottom border)
    :return: 4 integer values representing x1, y1, x2, y2
    """
    if (type(p1) is int or type(p1) is np.int64) and (type(p2) is int or type(p2) is np.int64):
        x1 = p1
        y1 = p2

    if type(p2) is list or type(p2) is tuple or type(p2) is np.ndarray:
        x2, y2 = p2
        x2, y2, _, _ = calculate_coordinates(x2, y2)

    if type(p1) is list or type(p1) is tuple or type(p1) is np.ndarray:
        if 4 == len(p1):
            x1, y1, x2, y2 = p1
            x1, y1, x2, y2 = calculate_coordinates(int(x1), int(y1), int(x2), int(y2))
        else:
            x1, y1 = p1
            x1, y1, _, _ = calculate_coordinates(x1, y1)

    return x1, y1, x2, y2


def horizontal_overlapping_ratio(box1: list, box2: list, threshold: int):
    margin = threshold + threshold
    if len(box1) == 4:
        b1x = [box1[0], box1[2]]
    else:
        b1x = box1

    if len(box2) == 4:
        b2x = [box2[0], box2[2]]
    else:
        b2x = box2
    iwidth = min(b1x[1], b2x[1]) - max(b1x[0], b2x[0])
    return iwidth / (b1x[1] - b1x[0] - margin)


def overlap_horizontally(box1: list, box2: list, threshold: int):
    """
    Check if 2 rectangular areas intersect horizontally or not
    :param box1: represents the box of a rectangular area as a list of integers. If the list has only 2 positions
    the values are exclusively of vertical box (y1, y2). If has 4 positions represents teh 4 box (x1, y1, x2, y2)
    :param box2: represents the box of a rectangular area as a list of integers. If the list has only 2 positions
    the values are exclusively of vertical box (y1, y2). If has 4 positions represents teh 4 box (x1, y1, x2, y2)
    :param threshold: represents the margin of error in absolute value to check the overlapping.
    :return: true if the boxes intersect, false otherwise.
    """
    if len(box1) == 4:
        b1y = [box1[0], box1[2]]
    else:
        b1y = box1

    if len(box2) == 4:
        b2y = [box2[0], box2[2]]
    else:
        b2y = box2

    dif = min(b1y[1], b2y[1]) - max(b1y[0], b2y[0]) - min(threshold, b1y[1] - b1y[0], b2y[1] - b2y[0])
    return dif >= 0


def overlap_vertically(box1: list, box2: list, threshold: int):
    """
    Check if 2 rectangular areas intersect vertically or not
    :param box1: represents the box of a rectangular area as a list of integers. If the list has only 2 positions
    the values are exclusively of vertical box (y1, y2). If has 4 positions represents teh 4 box (x1, y1, x2, y2)
    :param box2: represents the box of a rectangular area as a list of integers. If the list has only 2 positions
    the values are exclusively of vertical box (y1, y2). If has 4 positions represents teh 4 box (x1, y1, x2, y2)
    :param threshold: represents the margin of error in absolute value to check the overlapping.
    :return: true if the boxes intersect, false otherwise.
    """
    if len(box1) == 4:
        b1y = [box1[1], box1[3]]
    else:
        b1y = box1

    if len(box2) == 4:
        b2y = [box2[1], box2[3]]
    else:
        b2y = box2

    ret = min(b1y[1], b2y[1]) - max(b1y[0], b2y[0]) - min(threshold, b1y[1] - b1y[0], b2y[1] - b2y[0]) >= 0
    if not ret and min(b1y[1] - b1y[0], b2y[1] - b2y[0]) < (threshold+threshold):
        try:
            ret = ((min(b1y[1], b2y[1]) - max(b1y[0], b2y[0])) / min(b1y[1] - b1y[0] , b2y[1] - b2y[0])) >= 0.4
        except ZeroDivisionError:
            ret = False
    return ret

def is_similar_distance(x11, x12, x21, x22, threshold):
    margin = threshold
    return abs(x12 - x11 - x22 +x21) - margin < 0

def contains(edges_in_account: list, container: list, box: list, threshold, limit_values=None):
    """
    Check if a rectangular area contains completely other one. The parameter edges_in_account allows to dismiss the check
    of some coordinates. In any case, Even if a coordinate is dismissed, the box to be checked should never exceed
    the maximum limit received through the parameter limit_values
    :param edges_in_account: Coordinates to take int account in the checking. Is a list that can take the values 0, 1,
    2 or 3 or any combination of them.
    :param container: Is a list of the coordinates of the container area
    :param box:Is a list with the coordinates of the box to check
    :param threshold: represents the margin of error in absolute value to check.
    :param limit_values: represents the limite values of the container in case that some edge isn't checked
    :return: True if the container contains the box o false otherwise.
    """
    ret = False
    margin = threshold + threshold
    for i in range(4):
        dif = 0
        if i < 2:
            if i in edges_in_account:
                dif = box[i] - container[i] + (margin if i==0 else min(margin, 0.8*(box[3]-box[1])))
            elif limit_values is not None:
                dif = box[i] - limit_values[i] + (margin if i==0 else min(margin, 0.8*(box[3]-box[1])))
        else:
            if i in edges_in_account:
                dif = container[i] - box[i] + (margin if i==2 else min(margin, 0.8*(box[3]-box[1])))
            elif limit_values is not None:
                dif = limit_values[i] - box[i] + (margin if i==2 else min(margin, 0.8*(box[3]-box[1])))
        ret = dif >= 0
        if not ret:
            break
    return ret


VERTICAL_POSITIONING = 1
HORIZONTAL_POSITIONING = 0

def get_boxes_non_overlapping_positioning(ori, rel_box, fix_box, threshold):
    v = 0
    r = 0
    if ori == VERTICAL_POSITIONING:
        vok = not overlap_vertically(rel_box, fix_box, threshold)
        edges = [1,3]
    else:
        vok = not overlap_horizontally(rel_box, fix_box, threshold)
        edges = [0,2]
    if vok:
        r = _get_relative_loc_in_boxes(edges, rel_box, fix_box, True)
        if r < 0:
            v = fix_box[edges[0]] - rel_box[edges[1]]
        else:
            v = rel_box[edges[0]] - fix_box[edges[1]]
    return vok, r, v


def _get_relative_loc_in_boxes(edges, rel_box, fix_box, min=True):
    hr = rel_box[edges[1]] - rel_box[edges[0]]
    hf = fix_box[edges[1]] - fix_box[edges[0]]
    hf_plus_hr = hf + hr
    if min:
        dif = rel_box[edges[1]] - fix_box[edges[0]]
    else:
        dif = fix_box[edges[1]] - rel_box[edges[0]]
    if dif <= 0:
        status = -1
    elif 0 < dif < hr:
        status = dif / hr - 1
    elif hr <= dif <= hf:
        status = 0
    elif hf < dif < hf_plus_hr:
        status = (dif - hf) / hr
    else:
        status = 1
    return status

def get_relative_top_loc_in_boxes(rel_box, fix_box):
    return _get_relative_loc_in_boxes([1,3], rel_box, fix_box, True)

def get_relative_bottom_loc_in_boxes(rel_box, fix_box):
    return _get_relative_loc_in_boxes([1,3], rel_box, fix_box, False)

def get_relative_left_loc_in_boxes(rel_box, fix_box):
    return _get_relative_loc_in_boxes([0,2], rel_box, fix_box, True)

def get_relative_right_loc_in_boxes(rel_box, fix_box):
    return _get_relative_loc_in_boxes([0,2], rel_box, fix_box, False)

def fill_gaps_in_boxes(boxes):
    """
    Inserta nuevas cajas para llenar huecos verticales significativos entre cajas.
    Se utiliza la caja más ancha para determinar el ancho de las cajas insertadas.
    Cada caja se representa como [x1, y1, x2, y2].  
    Retorna una lista de cajas que incluye las originales y las insertadas.
    """
    boxes = np.array(boxes)
    boxes = boxes[boxes[:, 1].argsort()]
    widest_box = max(boxes, key=lambda box: box[2] - box[0])
    widest_width = widest_box[2] - widest_box[0]
    new_boxes = []
    for i in range(len(boxes) - 1):
        current_box = boxes[i]
        next_box = boxes[i + 1]
        gap = next_box[1] - current_box[3]
        if gap > 20:
            new_box_y1 = current_box[3]
            new_box_y2 = next_box[1]
            new_box_x1 = int((current_box[0] + current_box[2] - widest_width) // 2)
            new_box_x2 = new_box_x1 + widest_width
            new_boxes.append([new_box_x1, new_box_y1, new_box_x2, new_box_y2])
    if new_boxes:
        all_boxes = np.vstack([boxes, new_boxes])
        all_boxes = all_boxes[all_boxes[:, 1].argsort()]
    else:
        all_boxes = boxes
    return all_boxes.tolist()

def remove_edge_boxes(boxes, width_threshold=0.15, edge_threshold=0.2):
    """
    Filtra las cajas consideradas como de borde basándose en la posición de su centro
    y en su ancho. Retorna una lista de cajas que cumplen con los criterios definidos.
    """
    boxes = np.array(boxes)
    box_widths = boxes[:, 2] - boxes[:, 0]
    box_centers = (boxes[:, 0] + boxes[:, 2]) / 2
    median_center = np.median(box_centers)
    max_width = np.max(box_widths)
    center_range = np.max(box_centers) - np.min(box_centers)
    mask = ((np.abs(box_centers - median_center) <= center_range * edge_threshold) |
            (box_widths >= max_width * width_threshold))
    filtered_boxes = boxes[mask]
    return filtered_boxes.tolist()

def adjust_box_widths_and_center(boxes):
    """
    Ajusta todas las cajas para que tengan el ancho máximo encontrado y las centra
    en torno al centro mediano. Retorna la lista de cajas ajustadas.
    """
    boxes = np.array(boxes)
    box_widths = boxes[:, 2] - boxes[:, 0]
    max_width = np.max(box_widths)
    box_centers = (boxes[:, 0] + boxes[:, 2]) / 2
    median_center = int(np.median(box_centers))
    adjusted_boxes = np.zeros_like(boxes)
    adjusted_boxes[:, 0] = median_center - max_width // 2
    adjusted_boxes[:, 2] = median_center + max_width // 2
    adjusted_boxes[:, 1] = boxes[:, 1]
    adjusted_boxes[:, 3] = boxes[:, 3]
    return adjusted_boxes.astype(int).tolist()

def adjust_box_heights(boxes):
    """
    Ajusta las alturas de las cajas distribuyendo equitativamente los espacios verticales
    entre cajas consecutivas. Retorna una lista de cajas con las coordenadas y actualizadas.
    """
    boxes = np.array(boxes)
    sorted_indices = np.argsort(boxes[:, 1])
    sorted_boxes = boxes[sorted_indices]
    adjusted_boxes = []
    for i in range(len(sorted_boxes)):
        current_box = sorted_boxes[i].tolist()
        if i < len(sorted_boxes) - 1:
            next_box = sorted_boxes[i + 1]
            gap = next_box[1] - current_box[3]
            if gap > 0:
                current_box[3] += gap // 2
                sorted_boxes[i + 1][1] -= gap // 2
        adjusted_boxes.append(current_box)
    return adjusted_boxes

def remove_overlapping_segments(detections, iou_threshold=0.5, area_ratio_threshold=0.8):
    """
    Filtra las detecciones superpuestas utilizando umbrales de IoU y razón de área.
    Las detecciones se ordenan por confianza y se retienen aquellas que no se solapan
    excesivamente con las ya guardadas. Retorna la lista de detecciones filtradas.
    """
    sorted_detections = sorted(detections, key=lambda x: x[4], reverse=True)
    kept_detections = []
    for detection in sorted_detections:
        if all(
            not (
                calculate_iou(detection[:4], kept[:4])[0] > iou_threshold or
                ((area1 := (detection[2] - detection[0]) * (detection[3] - detection[1])) >
                 (area2 := (kept[2] - kept[0]) * (kept[3] - kept[1])) and 
                 calculate_iou(detection[:4], kept[:4])[1] / area2 > area_ratio_threshold) or
                (area2 > area1 and 
                 calculate_iou(detection[:4], kept[:4])[1] / area1 > area_ratio_threshold)
            )
            for kept in kept_detections
        ):
            kept_detections.append(detection)
    return kept_detections
