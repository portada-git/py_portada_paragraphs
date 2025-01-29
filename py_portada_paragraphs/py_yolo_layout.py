import os
from ultralytics import YOLO
import numpy as np
from .py_portada_utility_for_layout import contains

def calculate_overlap_vectorized(box, boxes):
    """
    Calcula el porcentaje de superposición entre una caja y un conjunto de cajas de manera vectorizada.

    Args:
    box (np.array): Coordenadas [x1, y1, x2, y2] de la caja de referencia.
    boxes (np.array): Array de coordenadas de las cajas a comparar.

    Returns:
    np.array: Array de porcentajes de superposición.
    """
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    overlap_percentage = intersection_area / np.minimum(box_area, boxes_area)
    return overlap_percentage

def _get_non_overlapping_indexes(boxes):
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    overlaps = np.zeros((len(boxes), len(boxes)))
    for i in range(len(boxes)):
        overlaps[i] = calculate_overlap_vectorized(boxes[i], boxes)

    boxes_to_remove = set()
    for i in range(len(boxes)):
        if i in boxes_to_remove:
            continue

        overlap_indices = np.where(overlaps[i] >= 0.85)[0]
        overlap_indices = overlap_indices[overlap_indices > i]

        if len(overlap_indices) > 0:
            heights = boxes[overlap_indices, 3] - boxes[overlap_indices, 1]
            if heights.max() > (boxes[i, 3] - boxes[i, 1]):
                boxes_to_remove.add(i)
            else:
                boxes_to_remove.update(overlap_indices)

    keep_indices = list(set(range(len(boxes))) - boxes_to_remove)
    return keep_indices

def remove_overlapping_boxes(p_boxes):
    boxes = np.array(p_boxes)
    keep_indices = _get_non_overlapping_indexes(boxes)
    return list(boxes[keep_indices])

def get_tagged_boxes(boxes, classes, confidences, names):
    tagged_boxes = []
    for box, cls, conf in zip(boxes, classes, confidences):
        class_name = names[int(cls)]
        x1, y1, x2, y2 = map(int, box.tolist())
        tagged_boxes.append({'box':[x1, y1, x2, y2], 'class_name':class_name, 'conf':conf})
    return tagged_boxes

def get_boundaries_for_class(boxes, classes, names, class_name_list_to_include=None, class_name_list_to_exclude=None):
    if class_name_list_to_exclude is None:
        class_name_list_to_exclude = list(names.values())
    if class_name_list_to_include is None:
        class_name_list_to_include = []
    boundaries = []
    class_names = []
    for box, cls in zip(boxes, classes):
        class_name = names[int(cls)]
        if class_name in class_name_list_to_include:
            x1, y1, x2, y2 = map(int, box.tolist())
            boundaries.append([x1, y1, x2, y2])
            class_names.append(class_name)
        if class_name not in class_name_list_to_exclude:
            x1, y1, x2, y2 = map(int, box.tolist())
            boundaries.append([x1, y1, x2, y2])
            class_names.append(class_name)
    if len(boundaries)>0:
        boundaries = np.array(boundaries)
        class_names = np.array(class_names)
        index_to_keep = _get_non_overlapping_indexes(boundaries)
        boundaries = list(boundaries[index_to_keep])
        class_names = list(class_names[index_to_keep])
    return boundaries, class_names

def get_page_boundaries(boxes, classes, names, guess_page):
    max_page_area = 0
    max_page = None
    for box, cls in zip(boxes, classes):
        class_name = names[int(cls)]
        if class_name == 'pagina':
            x1, y1, x2, y2 = map(int, box.tolist())
            parea = (x2 - x1) * (y2 - y1)
            if parea > max_page_area:
                max_page_area = parea
                max_page = [x1, y1, x2, y2]
    if max_page is None:
        max_page = guess_page
    return max_page

# def get_tagged_boxes_by_class(tagged_boxes, names, class_name_list_to_include=None, class_name_list_to_exclude=None):
#     if class_name_list_to_exclude is None:
#         class_name_list_to_exclude = list(names.values())
#     if class_name_list_to_include is None:
#         class_name_list_to_include = []
#     ret = []
#     boundaries = []
#     for tagged_box in tagged_boxes:
#         class_name =tagged_box['class_name']
#         if class_name in class_name_list_to_include:
#             x1, y1, x2, y2 = tagged_box['box']
#             boundaries.append([x1, y1, x2, y2])
#             ret.append(tagged_box)
#         if class_name not in class_name_list_to_exclude:
#             ret.append(tagged_box)
#     if len(ret)>0:
#         index_to_keep = _get_non_overlapping_indexes(np.array(boundaries))
#         ret = list(np.array(ret)[index_to_keep])
#     return ret


# def classify_tagged_boxes_by_container(tagged_boxes, guess_page, names):
#     tagged_sections = get_tagged_boxes_by_class(tagged_boxes, names, ['seccion'])
#     if not tagged_sections:
#          tagged_sections = [{'box':[int(guess_page[0]), int(guess_page[1]), int(guess_page[2]), int(guess_page[3])], 'class_name':'seccion', 'conf':-1}]
#     for tagged_section in tagged_sections:
#         tagged_columns = []
#
#
#     return



def classify_boxes_by_container(boxes, containers, max_container=None, container_order_key=1, box_order_key=0):
    if max_container is None:
        max_container = [0, 0, 100000, 100000]
    if not containers:
        containers = [[int(max_container[0]), int(max_container[1]), int(max_container[2]), int(max_container[3])]]
    sorted_containers = sorted(containers, key=lambda s: s[container_order_key])
    result = []
    to_remove = set()
    for container in sorted_containers:
        container_boxes = []
        if len(boxes) > 0:
            for i, box in enumerate(boxes):
                if contains([0,1,2,3], container, box, 30):
                    container_boxes.append(box)
                    to_remove.add(i)
        result.append({
            'container': container,
            'boxes': sorted(container_boxes, key=lambda c: c[box_order_key])
        })
    for index in sorted(to_remove, reverse=True):
        boxes.pop(index)
    return result

def get_model(fpath=None):
    if fpath is None:
        p = os.path.abspath(os.path.dirname(__file__))
        fpath = f"{p}/modelo/yolo11x-layout-882-rtx-6000-ada-48gb.pt"
    return YOLO(fpath)

def get_annotated_prediction(image: np.array, model):
    if model is None:
        model = get_model()
    elif type(model) is str:
        model = get_model(model)
    results = model.predict(image)

    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    names = results[0].names

    ret = []
    for i in range(len(boxes)):
        ret.append({'box': boxes[i], 'class_name': names[classes[i]]})
    return ret

def redefine_sections(sections):
    to_remove = set()
    result = []
    for i, section in enumerate(sections):
        if i in to_remove:
            continue

        s_center = (section['container'][0] + section['container'][2]) / 2
        for j in range(i + 1, len(sections)):
            sj_center = (sections[j]['container'][0] + sections[j]['container'][2]) / 2
            if abs(s_center - sj_center) < 60 and len(section['boxes']) == len(sections[j]['boxes']) and \
                    sections[j]['container'][1] - section['container'][3] < 60:  # threshold == 60
                to_remove.add(j)
                section['container'][3] = max(section['container'][3], sections[j]['container'][3])
                for c in range(len(section['boxes'])):
                    section['boxes'][c][3] = max(section['boxes'][c][3], sections[j]['boxes'][c][3])
        result.append(section)

    return result



def get_sections_and_page(image: np.array, model=None):
    """
    Procesa una imagen individual, detectando y ajustando las columnas.

    Args:
    image (np.array): Ruta de la imagen a procesar.
    model (YOLO): Modelo YOLO cargado para la detección.

    Returns:
    tuple: (imagen_procesada, secciones_ordenadas)
    """
    if model is None:
        model = get_model()
    elif type(model) is str:
        model = get_model(model)
    results = model.predict(image)

    boxes = results[0].boxes.xyxy.cpu().numpy()
    conf = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    names = results[0].names

    guess_page = [np.min(boxes[:, 0]), np.min(boxes[:, 1]), np.max(boxes[:, 2]), np.max(boxes[:, 3])]

    # tagged_boxes = get_tagged_boxes(boxes, classes, conf, names)
    # tagged_sorted_sections = classify_tagged_boxes_by_container(tagged_boxes, guess_page, names)

    header_boxes, _ = get_boundaries_for_class(boxes, classes, names, ['encabezado'])
    section_boxes, _ = get_boundaries_for_class(boxes, classes, names, ['seccion'])
    column_boxes, _ = get_boundaries_for_class(boxes, classes, names, ['columna'])
    block_boxes, _ = get_boundaries_for_class(boxes, classes, names, ['bloque'])
    other_boxes, other_boxes_class_names = get_boundaries_for_class(boxes, classes, names, class_name_list_to_exclude=['columna', 'seccion', 'encabezado',  'pagina', 'bloque'])
    page_box = get_page_boundaries(boxes, classes, names, guess_page)

    sorted_sections = classify_boxes_by_container(column_boxes, section_boxes, guess_page, 1, 0)
    # for section in sorted_sections:
    #     section['boxes'] = classify_boxes_by_container(block_boxes, section['boxes'])

    # for block in block_boxes:
    #     other_boxes.append(block)

    kept_others = []
    to_delete = []
    for other in other_boxes:
        keep = True
        reduce_columns = []
        area_other = (other[2] - other[0]) * (other[3] - other[1])
        need_columns = False
        for s, section in enumerate(sorted_sections):
            if min(section['container'][3], other[3]) - max(section['container'][1], other[1]) > 0:
                need_columns = True
            for c, col in enumerate(section['boxes']):
                p_intersection = max(0, min(other[2], col[2]) - max(other[0], col[0])) * max(0, min(other[3],
                                                                                                    col[3]) - max(
                    other[1], col[1])) / area_other
                if p_intersection > 0.65:
                    keep = False
                    break
                p_lx = (min(col[2], other[2]) - max(col[0], other[0])) / (col[2] - col[0])
                m_y1 = max(col[1], other[1])
                m_y2 = min(col[3], other[3])
                if (m_y2 - m_y1 > 0) and (p_lx > 0.75 or len(reduce_columns) > 1 and p_lx > 0.25):
                    reduce_columns.append([s, c])
            if not keep:
                break

        if keep:
            if need_columns:
                kept_others.append({'container': other, 'boxes': [other]})
            else:
                kept_others.append({'container': other, 'boxes': []})
            for index in reduce_columns:
                cy1 = sorted_sections[index[0]]['boxes'][index[1]][1]
                cy2 = sorted_sections[index[0]]['boxes'][index[1]][3]
                oy1 = other[1]
                oy2 = other[3]
                o_d = oy2 - oy1
                if oy2 - cy1 - o_d > 25 >= cy2 - oy1 - o_d:
                    sorted_sections[index[0]]['boxes'][index[1]][3] = oy1
                    if sorted_sections[index[0]]['boxes'][index[1]][3] - \
                            sorted_sections[index[0]]['boxes'][index[1]][1] < 25:
                        to_delete.append(index)
                elif cy2 - oy1 - o_d > 25 >= oy2 - cy1 - o_d:
                    sorted_sections[index[0]]['boxes'][index[1]][1] = oy2
                    if sorted_sections[index[0]]['boxes'][index[1]][3] - \
                            sorted_sections[index[0]]['boxes'][index[1]][1] < 25:
                        to_delete.append(index)
                else:
                    x1 = sorted_sections[index[0]]['boxes'][index[1]][0]
                    y1 = sorted_sections[index[0]]['boxes'][index[1]][1]
                    x2 = sorted_sections[index[0]]['boxes'][index[1]][2]
                    y2 = sorted_sections[index[0]]['boxes'][index[1]][3]
                    sorted_sections[index[0]]['boxes'][index[1]][3] = oy1
                    if sorted_sections[index[0]]['boxes'][index[1]][3] - \
                            sorted_sections[index[0]]['boxes'][index[1]][1] < 25:
                        to_delete.append(index)
                    if y2 - oy2 >= 25:
                        sorted_sections[index[0]]['boxes'].append([x1, oy2, x2, y2])
    for index in to_delete:
        sorted_sections[index[0]]['boxes'].pop(index[1])
    for section in sorted_sections:
        section['boxes'].sort(key=lambda X: X[0] * 10000 + X[1])

    # kept_others.sort(key=lambda X:X['container'][1] * 10000 + X['container'][0])
    sorted_sections.extend(kept_others)
    sorted_sections.sort(key=lambda X: X['container'][1] * 10000 + X['container'][0])
    sorted_sections = redefine_sections(sorted_sections)
    for sorted_section in sorted_sections:
        if len(sorted_section['boxes']) > 0:
            min_top = sorted_section['container'][3]
            max_bottom = sorted_section['container'][1]
            min_left = sorted_section['container'][2]
            max_right = sorted_section['container'][0]
            for col in sorted_section['boxes']:
                if min_top > col[1]:
                    min_top = col[1]
                if max_bottom < col[3]:
                    max_bottom = col[3]
                if min_left > col[0]:
                    min_left = col[0]
                if max_right < col[2]:
                    max_right = col[2]
            sorted_section['container'][0] = min_left
            sorted_section['container'][1] = min_top
            sorted_section['container'][2] = max_right
            sorted_section['container'][3] = max_bottom
    return sorted_sections, page_box