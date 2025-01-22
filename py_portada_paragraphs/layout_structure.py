from builtins import staticmethod, enumerate
from typing import Union
import numpy

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
    if (type(p1) is int or type(p1) is numpy.int64) and (type(p2) is int or type(p2) is numpy.int64):
        x1 = p1
        y1 = p2

    if type(p2) is list or type(p2) is tuple or type(p2) is numpy.ndarray:
        x2, y2 = p2
        x2, y2, _, _ = calculate_coordinates(x2, y2)

    if type(p1) is list or type(p1) is tuple or type(p1) is numpy.ndarray:
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
    :param margin: represents the margin of error in absolute value to check.
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
                dif = container[i] - box[i] + margin
            elif limit_values is not None:
                dif = limit_values[i] - box[i] + margin
        ret = dif >= 0
        if not ret:
            break
    return ret

def get_writing_area_properties(writing_area_list, i, min_left, max_right, threshold):
    guess_left = min_left
    guess_right = max_right
    is_single = True
    offset = -1

    writing_area = writing_area_list[i]
    while (i + offset) >= 0:
        if overlap_vertically(writing_area_list[i + offset], writing_area,
                              threshold) and not overlap_horizontally(writing_area_list[i + offset],
                                                                                  writing_area, threshold):
            if writing_area_list[i + offset][0] < writing_area[0] and guess_left < writing_area_list[i + offset][2]:
                guess_left = writing_area_list[i + offset][2]
            elif writing_area[0] <= writing_area_list[i + offset][0] < guess_right:
                guess_right = writing_area_list[i + offset][0]
            is_single = False
        offset -= 1
    offset = 1
    # while (i + offset) < len(writing_area_list) and overlap_vertically(writing_area_list[i + offset],
    #                                                                    writing_area, main_layout.threshold):
    while (i + offset) < len(writing_area_list):
        if overlap_vertically(writing_area_list[i + offset], writing_area,
                              threshold) and not overlap_horizontally(writing_area_list[i + offset],
                                                                                  writing_area, threshold):
            if writing_area_list[i + offset][0] < writing_area[0] and guess_left < writing_area_list[i + offset][2]:
                guess_left = writing_area_list[i + offset][2]
            elif writing_area[0] <= writing_area_list[i + offset][0] < guess_right:
                guess_right = writing_area_list[i + offset][0]
            is_single = False
        offset += 1
    guess_width = guess_right - guess_left
    wa = {"area": writing_area, "guess_left": guess_left, "guess_right": guess_right,
          "guess_width": guess_width, "is_single": is_single}
    return wa



class ThresholdAttribute:
    default_threshold = 20

    def __init__(self):
        self._threshold = ThresholdAttribute.default_threshold

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, v):
        self._threshold = v


class AbstractSection:

    def __init__(self, boundaries_or_proposed_section: Union[dict, list, numpy.ndarray] = None,
                 threshold: Union[int, ThresholdAttribute] = None, container=None, proposed_section=None,
                 boundaries=None, is_bottom_expandable=None, page_boundary=None):
        if proposed_section is None and type(boundaries_or_proposed_section) is dict:
            proposed_section = boundaries_or_proposed_section
        if boundaries is None and boundaries_or_proposed_section is not None \
                and type(boundaries_or_proposed_section) is not dict:
            boundaries = boundaries_or_proposed_section
        elif boundaries is None:
            if proposed_section is not None:
                boundaries = proposed_section['box']
            else:
                boundaries = [0, 0, 0, 0]
        if page_boundary is None:
            self.page_boundary = container.page_boundary
        else:
            self.page_boundary = page_boundary
        self.proposed_section = proposed_section
        self.diagonal_points = boundaries
        self.section_container = container
        if threshold is None:
            if self.section_container is None:
                threshold = 40
            else:
                threshold = self.section_container._threshold
        if type(threshold) is ThresholdAttribute:
            self._threshold = threshold
        else:
            self._threshold = ThresholdAttribute()
            if type(threshold) is int:
                self.threshold = threshold
        self._is_bottom_expandable = is_bottom_expandable
        self.upper_fill_level = self.down_fill_level = self.top

    @property
    def threshold(self):
        return self._threshold.threshold

    @threshold.setter
    def threshold(self, v):
        self._threshold.threshold = v

    @property
    def coordinates(self):
        return [self.left, self.top, self.right, self.bottom]

    @property
    def diagonal_points(self):
        return self._diagonal_points

    @property
    def lt_coord(self):
        return self.diagonal_points[0]

    @property
    def rb_coord(self):
        return self.diagonal_points[1]

    @property
    def left(self):
        return self.diagonal_points[0][0]

    @property
    def top(self):
        return self.diagonal_points[0][1]

    @property
    def right(self):
        return self.diagonal_points[1][0]

    @property
    def bottom(self):
        return self.diagonal_points[1][1]

    @diagonal_points.setter
    def diagonal_points(self, p1: Union[int, list] = 0, p2: Union[int, list] = 0, x2=0, y2=0):
        x1, y1, x2, y2 = calculate_coordinates(p1, p2, x2, y2)
        self._diagonal_points = [[x1, y1], [x2, y2]]

    @lt_coord.setter
    def lt_coord(self, p1: Union[int, list] = 0, y=0):
        if type(p1) is list:
            x = p1[0]
            y = p1[1]
        elif type(p1) is int:
            x = p1

        self._diagonal_points[0] = [x, y]

    @rb_coord.setter
    def rb_coord(self, p1: Union[int, list] = 100000, y=100000):
        if type(p1) is list:
            x = p1[0]
            y = p1[1]
        elif type(p1) is int:
            x = p1

        self._diagonal_points[1] = [x, y]

    @left.setter
    def left(self, x=0):
        self._diagonal_points[0][0] = x

    @top.setter
    def top(self, y=0):
        self._diagonal_points[0][1] = y

    @right.setter
    def right(self, x=100000):
        self._diagonal_points[1][0] = x

    @bottom.setter
    def bottom(self, y=100000):
        self._diagonal_points[1][1] = y

    @property
    def width(self):
        return self.right - self.left

    def is_empty(self):
        dif = self.bottom - self.top
        return dif < self.threshold

    @property
    def is_bottom_expandable(self):
        result = self._is_bottom_expandable
        if result is None:
            result = self.section_container.is_bottom_expandable
        return result

    @is_bottom_expandable.setter
    def is_bottom_expandable(self, val):
        self._is_bottom_expandable = val

    def can_be_added(self, writing_area_properties):
        pass

    def add_writing_area(self, writing_area_properties):
        pass


class StructuredSection(AbstractSection):
    """
    A StructuredSection class implements a structured section which can contain 2 kinds of structures: section stack or
    sibling sections

    Attributtes
    -----------


    Methods
    -------

    """

    def __init__(self, boundaries_or_proposed_section: Union[dict, list] = None,
                 threshold: Union[int, ThresholdAttribute] = None, proposed_section=None, boundaries=None,
                 container=None, page_boundary=None):
        super().__init__(boundaries_or_proposed_section, threshold, container, proposed_section, boundaries,
                         is_bottom_expandable=False, page_boundary=page_boundary)
        self._sections = []
        self.is_right_expandable = False
#        self.is_bottom_expandable = True

    @property
    def sections(self):
        return self._sections

    def get_single_sections_as_boxes(self, boxes=None):
        if boxes is None:
            boxes = []
        for section in self.sections:
            section.get_single_sections_as_boxes(boxes)
        return boxes

    def add_new_section(self, section):
        self._sections.append(section)

    # def is_area_inside(self, p1: Union[int, list] = 0, p2: Union[int, list] = 0, x2=0, y2=0):
    #     x1, y1, x2, y2 = calculate_coordinates(p1, p2, x2, y2)
    #     edges = [0, 1]
    #     if not self.is_rigth_expandable:
    #         edges.append(2)
    #     if not self.is_bottom_expandable:
    #         edges.append(3)
    #     return contains(edges, self.coordinates, (x1, y1, x2, y2), self.threshold)


# def _compare(item1, item2):
#     if item1["is_single"] or item2["is_single"]:
#         ret = item1["area"][1] - item2["area"][1]
#     elif _areas_belong_to_the_same_columns(item1["area"], item2["area"], item1, item2, 60):
#         ret = item1["area"][1] - item2["area"][1]
#     else:
#         ret = item1["guess_left"] - item2["guess_left"]
#     return ret

SET_AREA_IN_EXISTING_SECTION = 0
CREATE_SECTION_AND_SET_AREA = -1
REFORM_EXISTING_SECTION_AND_SET_AREA= 1


class MainLayout(StructuredSection):
    def __init__(self, page_boundary, w: int, h: int = 0, threshold=40, proposed_sections=None):
        super().__init__([0, 0, w, h], threshold, page_boundary=page_boundary)
        self.proposed_sections = proposed_sections
        self.unlocated_boxes = []

    def get_single_sections_as_boxes(self):
        boxes = []
        return super().get_single_sections_as_boxes(boxes)

    def get_unlocated_boxes(self):
        boxes = []
        for wa in self.unlocated_boxes:
            boxes.append(wa['area'])
        return boxes

    def search_section(self, writing_area_properties):
        pos = -1
        for i, section in enumerate(self.sections):
            if section.can_be_added(writing_area_properties):
                pos = i
                break
        return pos

    def _has_area_similar_width_to_proposed_columns(self, writing_area_properties, proposed_section):
        if len(proposed_section['columns']) > 0:
            margin = self.threshold + self.threshold + self.threshold
            width_sibling = proposed_section['columns'][0][2] - proposed_section['columns'][0][0]
            if width_sibling == -1:
                dif = 0
            else:
                x1, y1, x2, y2 = writing_area_properties['area']
                dif = abs(x2 - x1 - width_sibling)
                if dif > margin:
                    dif = abs(writing_area_properties['guess_width'] - width_sibling)
            ret = dif <= margin
        else:
            ret = False
        return ret

    def _has_area_similar_center_to_proposed_columns(self, writing_area_properties, proposed_section):
        if len(proposed_section['columns']) > 0:
            margin = self.threshold + self.threshold
            x1, y1, x2, y2 = writing_area_properties['area']
            width_sibling = proposed_section['columns'][0][2] - proposed_section['columns'][0][0]
            dif = abs((((x1 + x2) / 2) % width_sibling) - (width_sibling / 2)) - self.page_boundary[0]
            ret = dif <= margin
        else:
            ret = False
        return ret

    def _has_area_similar_start_to_proposed_columns(self, writing_area_properties, proposed_section):
        if len(proposed_section['columns']) > 0:
            margin = self.threshold + self.threshold
            x1, y1, x2, y2 = writing_area_properties['area']
            width_sibling = proposed_section['columns'][0][2] - proposed_section['columns'][0][0]
            left = proposed_section['columns'][0][0]
            dif = abs(left - x1) % width_sibling
            if dif <= margin:
                dif = abs(writing_area_properties['guess_width'] - width_sibling)
            ret = dif <= margin
        else:
            ret = False
        return ret

    def _exist_a_proposed_column_to_contain_area(self, writing_area_properties, proposed_section):
        result = False
        for column in proposed_section['columns']:
            if contains([0,1,2,3], column, writing_area_properties['area'], self.threshold):
                result = True
                break
        return result

    def is_proposed_section_compatible(self, writing_area_properties, proposed_section):
        result = writing_area_properties['is_single'] == (len(proposed_section['columns']) == 0)
        result = result and contains([0, 1, 2, 3], proposed_section['box'], writing_area_properties['area'],
                                     self.threshold)
        if result and not writing_area_properties['is_single']:
            sw = self._has_area_similar_width_to_proposed_columns(writing_area_properties, proposed_section)
            if not sw:
                sw = self._has_area_similar_center_to_proposed_columns(writing_area_properties, proposed_section)
            if sw:
                sw = self._exist_a_proposed_column_to_contain_area(writing_area_properties, proposed_section)

            result = result and sw
        return result

    def search_proposed_section(self, writing_area_properties):
        pos = -1
        for i, proposed_section in enumerate(self.proposed_sections):
            if self.is_proposed_section_compatible(writing_area_properties, proposed_section):
                pos = i
                break
        return pos

    def add_section_from_proposed_section(self, pos_proposed_section):
        proposed_section = self.proposed_sections[pos_proposed_section]
        if len(proposed_section['columns']) == 0:
            section = SingleSection(self, proposed_section=proposed_section, is_bottom_expandable=False)
        else:
            section = BigSectionOfSibling(self, proposed_section=proposed_section)
        for prev_section in self.sections:
            if prev_section.is_bottom_expandable and horizontal_overlapping_ratio(prev_section.coordinates,
                                                                                  section.coordinates,
                                                                                  self.threshold) >= 0.9:
                prev_section.is_bottom_expandable = False
        self.add_new_section(section)
        self.proposed_sections.pop(pos_proposed_section)

    def add_writing_area(self, writing_area_properties):
        ppos = self.search_proposed_section(writing_area_properties)
        if ppos >= 0:
            pos = len(self.sections)
            self.add_section_from_proposed_section(ppos)
            self.sections[pos].top = writing_area_properties['area'][1]
            self.sections[pos].add_writing_area(writing_area_properties)
        else:
            pos = self.search_section(writing_area_properties)
            if pos >= 0:
                self.sections[pos].add_writing_area(writing_area_properties)
            elif contains([0,1,2,3], self.page_boundary, writing_area_properties['area'], self.threshold):
                self.unlocated_boxes.append(writing_area_properties)
                # if writing_area_properties['is_single']:
                #     # create one extra single section
                #     section = SingleSection(self, [self.left,
                #                                    writing_area_properties['area'][1],
                #                                    self.right,
                #                                    writing_area_properties['area'][1]],is_bottom_expandable=True)
                #     section.add_writing_area(writing_area_properties)
                #     self.sections.append(section)
                #     self.sections.sort(key=lambda x:x.top*10000+x.left)
                # else:
                #     is_overlapping = False
                #     for sec in self.sections:
                #         if overlap_vertically(sec.coordinates, writing_area_properties['area'], self.threshold):
                #             is_overlapping = True
                #             break
                #     # si colinda con otras secciones, limitar la anchura
                #     if is_overlapping:
                #         section = SingleSection(self, [writing_area_properties['area'][0],
                #                                        writing_area_properties['area'][1],
                #                                        writing_area_properties['area'][2],
                #                                        writing_area_properties['area'][1]])
                #     # si no colinda con otras secciones, anchura máxima
                #     else:
                #         section = BigSectionOfSibling(self, [self.left,
                #                                    writing_area_properties['area'][1],
                #                                    self.right,
                #                                    writing_area_properties['area'][1]])
                #         column = SingleSection(self, [writing_area_properties['area'][0],
                #                                       writing_area_properties['area'][1],
                #                                       writing_area_properties['area'][2],
                #                                       writing_area_properties['area'][1]])
                #         section.add_new_section(column)
                #     section.add_writing_area(writing_area_properties)
                #     self.add_new_section(section)
                #     self.sections.sort(key=lambda x:x.top*10000+x.left)




    # def adjust_size_of_sections(self):
    #     #adjust vertically
    #     #order bigsection vertically
    #     sorted(self.sections, key=lambda x: x.top*10000+x.left)
    #     for pos in range(len(self.sections)):
    #         offset = 1
    #         while pos+offset<len(self.sections) and overlap_vertically(self.sections[pos].coodinates,
    #                                                                    self.sections[pos+offset].coodinates):
    #             offset += 1
    #         if offset<len(self.sections):
    #             #need adjust
    #             self.sections[pos].bottom = (self.sections[pos].bottom + self.sections[pos+offset].top)//2
    #

    # def search_action_and_position_for_unlocated_area(self, i, min_left, max_right, threshold):
    #     pos = -1
    #     action = -1
    #     if self.unlocated_boxes[i]['is_single']:
    #
    #
    #     return pos, action

    @staticmethod
    def build_lauoud_from_sections(page_boundary, sections, writing_area_list, w: int, h: int, threshold=None):
        main_layout = MainLayout(page_boundary, w, h, proposed_sections=sections)
        if threshold is not None:
            main_layout.threshold = threshold

        min_left = 100000
        max_right = 0
        for writing_area in writing_area_list:
            a_left = max(writing_area[0], page_boundary[0])
            a_right = min(writing_area[2], page_boundary[2])
            if min_left > a_left:
                min_left = a_left
            if max_right < a_right:
                max_right = a_right

        writing_area_list.sort(key=lambda x: x[1] * 10000 + x[0])
        for i in range(len(writing_area_list)):
            wa = get_writing_area_properties(writing_area_list, i, min_left, max_right, main_layout.threshold)
            main_layout.add_writing_area(wa)

        # for section in main_layout.sections:
        #     if type(section) is BigSectionOfSibling:
        #         for column in section.siblings:
        #             column.top = column.upper_fill_level
        #             column.bottom = column.down_fill_level
        #     section.top = section.upper_fill_level
        #     section.bottom = section.down_fill_level


        # for i in range(len(main_layout.unlocated_boxes)):
        #     pos, action = main_layout.search_action_and_position_for_unlocated_area(i, min_left, max_right, main_layout.threshold)
        #     if action == SET_AREA_IN_EXISTING_SECTION:
        #         # afegir a la secció de la posició pos
        #         main_layout.sections[pos].add_writing_area(main_layout.unlocated_boxes[i])
        #     elif action == REFORM_EXISTING_SECTION_AND_SET_AREA:
        #         # reformar la secció pos
        #         pass
        #     else:
        #         # inserir entre seccions
        #         if main_layout.unlocated_boxes[i]['is_single']:
        #             section = SingleSection(main_layout, [min_left, page_boundary[1], max_right, main_layout.sections[0].top], main_layout.threshold, is_bottom_expandable=False)
        #             section.add_writing_area(main_layout.unlocated_boxes[i])
        #         else:
        #             # ???
        #             pass
        #         main_layout.sections.append(section)
        #         main_layout.sections.sort(key=lambda X: X.top*10000 +  X.left)

        #for i, section in enumerate(main_layout.sections):
        #     if i == 0:
        #         section.top = main_layout.top
        #     elif i == len(main_layout.sections) - 1:
        #         section.bottom = main_layout.bottom
        #     if i < len(main_layout.sections) - 1:
        #         section.bottom = main_layout.sections[i + 1].top
        #     section.left = main_layout.left
        #     section.right = main_layout.right
        #     if type(section) is BigSectionOfSibling:
        #         for j, col in enumerate(section.siblings):
        #             if j == 0:
        #                 col.left = main_layout.left
        #             if j == len(section.siblings) - 1:
        #                 col.right = main_layout.right
        #             if j < len(section.siblings) - 1:
        #                 col.right = section.siblings[j + 1].left = (col.right + section.siblings[j + 1].left) // 2

        return main_layout



class BigSectionOfSibling(StructuredSection):
    def __init__(self, container: MainLayout, boundaries_or_proposed_section: Union[dict, list] = None,
                 threshold: Union[int, ThresholdAttribute] = None, proposed_section=None, boundaries=None):
        super().__init__(boundaries_or_proposed_section, threshold, proposed_section, boundaries, container)
        if self.proposed_section is not None and len(self.proposed_section['columns']) > 0:
            self._width_sibling = self.proposed_section['columns'][0][2] - self.proposed_section['columns'][0][0]
            for col in self.proposed_section['columns']:
                col_sec = SingleSection(self, col)
                self.add_new_section(col_sec)
        else:
            self._width_sibling = -1

    @AbstractSection.top.setter
    def top(self, value):
        for sib in self.siblings:
            if sib.top == self._diagonal_points[0][1]:
                sib.top = value
        self._diagonal_points[0][1] = value

    @AbstractSection.bottom.setter
    def bottom(self, value):
        for sib in self.siblings:
            if sib.bottom == self._diagonal_points[1][1]:
                sib.bottom = value
        self._diagonal_points[1][1] = value

    @property
    def width_sibling(self):
        return self._width_sibling

    @property
    def siblings(self):
        return self.sections

    def _has_area_similar_center(self, writing_area_properties):
        margin = self.threshold + self.threshold + self.threshold
        x1, y1, x2, y2 = writing_area_properties['area']
        dif = abs((((x1 + x2) / 2) % self.width_sibling) - (self.width_sibling / 2)) - self.page_boundary[0]
        return dif <= margin

    def _has_area_similar_start(self, writing_area_properties):
        margin = self.threshold + self.threshold + self.threshold
        x1, y1, x2, y2 = writing_area_properties['area']
        dif = abs(self.width_sibling - x1) % self.width_sibling
        return dif <= margin

    def _has_area_similar_width(self, writing_area_properties):
        margin = self.threshold + self.threshold + self.threshold
        if self.width_sibling == -1:
            dif = 0
        else:
            x1, y1, x2, y2 = writing_area_properties['area']
            dif = abs(x2 - x1 - self.width_sibling)
            if dif > margin:
                dif = abs(writing_area_properties['guess_width'] - self.width_sibling)
        return dif <= margin

    def is_area_compatible(self, writing_area_properties):
        ret = self._has_area_similar_width(writing_area_properties)
        if not ret:
            ret =  self._has_area_similar_center(writing_area_properties)
        return ret

    def search_sibling(self, writing_area_properties):
        pos = -1
        for i, sibling in enumerate(self.siblings):
            if sibling.can_be_added(writing_area_properties):
                pos = i
                break
        return pos

    def can_be_added(self, writing_area_properties):
        result = not writing_area_properties['is_single']
        edges = [0, 1, 2]
        if not self.is_bottom_expandable:
            edges.append(3)
        result = result and contains(edges, self.coordinates, writing_area_properties['area'], self.threshold)
        result = result and self.is_area_compatible(writing_area_properties)
        pos = self.search_sibling(writing_area_properties)
        return result and pos >= 0

    def where_insert_a_column_for(self, writing_area_properties):
        ret = None
        space = None
        # buscar espacio libre entre columnas
        for i, col in enumerate(self.siblings):
            if i == 0:
                free_with = col.left - self.page_boundary[0]
                if free_with - self.width_sibling >= self.threshold and contains([0,2],
                                                                                 [self.page_boundary[0], 0,
                                                                                  col.left, 0],
                                                                                  writing_area_properties['area'],
                                                                                  self.threshold):
                    space = [max(col.left - free_with -self.threshold, 0), col.left]
                    break
            elif i > 0:
                free_with = col.left - self.siblings[i-1].right
                if free_with - self.width_sibling >=  self.threshold and contains([0,2],
                                                                                 [self.siblings[i-1].right, 0,
                                                                                  col.left, 0],
                                                                                  writing_area_properties['area'],
                                                                                  self.threshold):
                    space = [self.siblings[i-1].right, col.left]
                    break
        if space is None:
            free_with = self.page_boundary[2] - self.siblings[-1].right
            if free_with - self.width_sibling >= self.threshold and contains([0,2],
                                                                                 [self.siblings[-1].right, 0,
                                                                                  self.page_boundary[2], 0],
                                                                                  writing_area_properties['area'],
                                                                                  self.threshold):
                space = [self.siblings[-1].right, self.page_boundary[2]]
        if space is not None:
            found = False
            while not found and space[0]<=space[1]:
                ret = [space[0], writing_area_properties['area'][1], space[0]+self.width_sibling, writing_area_properties['area'][1]]
                found = contains([0,1,2], ret, writing_area_properties['area'], self.threshold)
                space[0] += self.width_sibling
        return ret

    def add_writing_area(self, writing_area_properties):
        pos = self.search_sibling(writing_area_properties)
        if pos == -1:
            # boundaries = self.where_insert_a_column_for(writing_area_properties)
            # if boundaries is not None:
            #     # insert
            #     column = SingleSection(self, boundaries)
            #     column.add_writing_area(writing_area_properties)
            #     self.siblings.append(column)
            #     self.siblings.sort(key=lambda x: x[0])
            #     if column.down_fill_level > self.down_fill_level:
            #         self.down_fill_level = column.down_fill_level
            #     if column.upper_fill_level < self.upper_fill_level:
            #         self.upper_fill_level = column.upper_fill_level
            # else:
            #     raise "Error trying add writing area to wrong column"
            raise "Error trying add writing area to wrong column"
        else:
            self.siblings[pos].add_writing_area(writing_area_properties)
            if self.siblings[pos].down_fill_level > self.down_fill_level:
                self.down_fill_level = self.siblings[pos].down_fill_level
            if self.siblings[pos].upper_fill_level < self.upper_fill_level:
                self.upper_fill_level = self.siblings[pos].upper_fill_level


class SingleSection(AbstractSection):
    """
    Class used to represent a single section. In real world, a single section has not layout structure and can only
    contain text.
    This implementation maintains only the rectangle information.

    Attributtes
    -----------


    Methods
    -------

    """

    def __init__(self, container: StructuredSection, boundaries_or_proposed_section: Union[dict, list] = None,
                 threshold: Union[int, ThresholdAttribute] = None, proposed_section=None, boundaries=None,
                 is_bottom_expandable=None):
        super().__init__(boundaries_or_proposed_section, threshold=threshold, container=container,
                         proposed_section=proposed_section, boundaries=boundaries,
                         is_bottom_expandable=is_bottom_expandable)
        self._len = 0
        self._suma_center = 0
        self._is_column = type(container) is BigSectionOfSibling

    @property
    def center(self):
        return self._suma_center / self._len

    def can_be_added(self, writing_area_properties):
        result = writing_area_properties['is_single'] == (not self._is_column)
        edges = [0, 1, 2]
        if not self.is_bottom_expandable:
            edges.append(3)
        result = result and contains(edges, self.coordinates, writing_area_properties['area'], self.threshold)
        return result

    def add_writing_area(self, writing_area_properties):
        x1, y1, x2, y2 = writing_area_properties['area']
        if y1 < self.upper_fill_level:
            self.upper_fill_level = y1
        if y2 > self.down_fill_level:
            self.down_fill_level = y2
        # if y2 > self.bottom:
        #     self.bottom = y2
        self._len += 1
        self._suma_center += (x1 + x2) / 2
        if x1 < self.left:
            self.left = x1
        if x2 > self.right:
            self.right = x2

    def get_single_sections_as_boxes(self, boxes=[]):
        boxes.append(self.coordinates)

    def get_compatible_status(self, p1: Union[int, list] = 0, p2: Union[int, list] = 0, x2=0, y2=0):
        ret = super().get_compatible_status(p1, p2, x2, y2)
        if ret != 0:
            x1, y1, x2, y2 = calculate_coordinates(p1, p2, x2, y2)
            ret = self.get_compatible_status_as_guess(x1, y1, x2, y2)
            if ret != 0:
                c = (x1 + x2) / 2
                ldif = c - self.center
                if abs(ldif) <= self.threshold:
                    ret = 0
                elif ldif < self.threshold:
                    ret = -1
                else:
                    ret = 1
        return ret

    def get_compatible_status_as_guess(self, p1: Union[int, list] = 0, p2: Union[int, list] = 0, x2=0, y2=0):
        ret = 0
        x1, y1, x2, y2 = calculate_coordinates(p1, p2, x2, y2)
        ldif = x1 - self.guess_left + self.threshold
        rdif = self.guess_right - x2 + self.threshold
        if ldif >= 0 and rdif >= 0:
            ret = 0
        elif ldif < 0:
            ret = -1
        else:
            ret = 1
        return ret


def test_build_layout():
    m = MainLayout.build_layout([[10, 10, 90, 50], [10, 50, 100, 60], [10, 60, 90, 70], [10, 72, 90, 85],
                                 [10, 85, 90, 90], [10, 91, 45, 98], [55, 90, 90, 105], [10, 110, 45, 120],
                                 [55, 104, 90, 117], [10, 119, 90, 125], [10, 125, 90, 135]], 105, 2)
    boxes = m.get_single_sections_as_boxes()
    print(boxes)


def test():
    # single structure

    main_layout = MainLayout(100, 2)
    main_layout.add_new_section(SingleSection())
    main_layout.try_to_add_writing_area(10, 10, 90, 50)
    main_layout.try_to_add_writing_area(10, 50, 90, 60)
    try:
        main_layout.try_to_add_writing_area(10, 50, 90, 60)
    except ValueError as e:
        print(e)

    main_layout.try_to_add_writing_area(10, 60, 90, 70)
    main_layout.add_new_section(SingleSection())
    main_layout.try_to_add_writing_area(10, 72, 90, 85)
    main_layout.try_to_add_writing_area(10, 85, 90, 90)
    main_layout.add_new_section(BigSectionOfSibling())
    main_layout.try_to_add_writing_area(10, 91, 40, 98)
    main_layout.try_to_add_writing_area(55, 94, 90, 105)
    main_layout.add_new_section(SingleSection())
    main_layout.try_to_add_writing_area(10, 104, 90, 120)

    print(main_layout.diagonal_points)

    boxes = main_layout.get_single_sections_as_boxes()

    print(boxes)

    m = MainLayout.build_layout([[10, 10, 90, 50], [10, 50, 100, 60], [10, 60, 90, 70], [10, 72, 90, 85],
                                 [10, 85, 90, 90], [10, 91, 40, 98], [55, 94, 90, 105], [10, 110, 40, 120],
                                 [55, 104, 90, 117], [10, 119, 90, 125], [10, 125, 90, 135]], 105)
    boxes = m.get_single_sections_as_boxes()
    print(boxes)


if __name__ == "__main__":
    test()
    test_build_layout()
