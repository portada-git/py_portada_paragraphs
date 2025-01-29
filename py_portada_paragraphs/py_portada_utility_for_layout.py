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
        ret = ((min(b1y[1], b2y[1]) - max(b1y[0], b2y[0])) / min(b1y[1] - b1y[0], b2y[1] - b2y[0])) >= 0.4
    return ret


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
                dif = box[i] - container[i] + margin
            elif limit_values is not None:
                dif = box[i] - limit_values[i] + margin
        else:
            if i in edges_in_account:
                dif = container[i] - box[i] + min(margin, 0.8*(box[3]-box[1]))
            elif limit_values is not None:
                dif = limit_values[i] - box[i] + min(margin, 0.8*(box[3]-box[1]))
        ret = dif >= 0
        if not ret:
            break
    return ret
