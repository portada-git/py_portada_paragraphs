from builtins import staticmethod, enumerate
from typing import Union
import numpy

from .py_portada_utility_for_layout import overlap_vertically, overlap_horizontally, calculate_coordinates, contains, \
    horizontal_overlapping_ratio, is_similar_distance
from .py_yolo_layout import get_annotated_prediction


def get_writing_area_properties(writing_area_list, i, min_left, max_right, threshold):
    guess_left = min_left
    guess_right = max_right
    is_single = True
    offset = -1

    writing_area = writing_area_list[i]
    while (i + offset) >= 0:
        if contains([0, 2], [min_left, 0, max_right, 0], writing_area_list[i + offset],
                    threshold) and overlap_vertically(writing_area_list[i + offset], writing_area,
                                                      threshold) and not overlap_horizontally(
            writing_area_list[i + offset],
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
        if contains([0, 2], [min_left, 0, max_right, 0], writing_area_list[i + offset],
                    threshold) and overlap_vertically(writing_area_list[i + offset], writing_area,
                                                      threshold) and not overlap_horizontally(
            writing_area_list[i + offset],
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


class OutsideUnlocatedBoxesForSection:
    def __init__(self, section):
        self.section = section
        self.boxes = []


class InsideUnlocatedBoxesToColumn:
    def __init__(self, column):
        self.column = column
        self.boxes = []


class ColUnlocatedBoxes:
    def __init__(self):
        self.map_sections_for_prev = {}
        self.map_sections_for_post = {}
        self.map_columns_for_in = {}
        self.prev_outside_unlocated_boxes = []
        self.post_outside_unlocated_boxes = []
        self.inside_unlocated_boxes = []
        self.index_maps_by_type = [self.map_columns_for_in, self.map_sections_for_post, self.map_sections_for_prev]
        self.unlocated_boxes_by_type = [self.inside_unlocated_boxes, self.post_outside_unlocated_boxes,
                                        self.prev_outside_unlocated_boxes]

    def set_unlocated_box(self, unlocated_box_properties, type, section_or_column):
        if type == 0:
            self.set_unlocated_for_inside_type(unlocated_box_properties, section_or_column)
        else:
            self.set_unlocated_for_outside_type(unlocated_box_properties, type, section_or_column)

    def set_unlocated_for_inside_type(self, unlocated_box_properties, column):
        index_map = self.index_maps_by_type[0]
        unlocated_boxes = self.unlocated_boxes_by_type[0]
        if not column in index_map:
            index_map[column] = len(unlocated_boxes)
            unlocated_boxes.append(InsideUnlocatedBoxesToColumn(column))
        unlocated_boxes[index_map[column]].boxes.append(unlocated_box_properties)

    def set_unlocated_for_outside_type(self, unlocated_box_properties, type, section):
        index_map = self.index_maps_by_type[type]
        unlocated_boxes = self.unlocated_boxes_by_type[type]
        if not section in index_map:
            index_map[section] = len(unlocated_boxes)
            unlocated_boxes.append(OutsideUnlocatedBoxesForSection(section))
        unlocated_boxes[index_map[section]].boxes.append(unlocated_box_properties)

    def get_unlocated_boxes_by_type(self, type):
        unlocated_boxes = self.unlocated_boxes_by_type[type]
        return unlocated_boxes


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
                boundaries = proposed_section['container']
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

    def sort_content(self):
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
#
# SET_AREA_IN_EXISTING_SECTION = 0
# CREATE_SECTION_AND_SET_AREA = -1
# REFORM_EXISTING_SECTION_AND_SET_AREA= 1

class MainLayout(StructuredSection):
    def __init__(self, page_boundary, w: int, h: int = 0, threshold=40, proposed_sections=None):
        super().__init__([0, 0, w, h], threshold, page_boundary=page_boundary)
        self.proposed_sections = proposed_sections
        self.unlocated_boxes = []
        self.col_unlocated_boxes = None
        self.__can_use_existing_single_section = True

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
        if len(proposed_section['boxes']) > 0:
            margin = self.threshold + self.threshold + self.threshold
            x1, y1, x2, y2 = writing_area_properties['area']
            area_width = x2 - x1
            dif = margin + 1
            for col in proposed_section['boxes']:
                if not contains([0, 1, 2, 3], col, writing_area_properties['area'], self.threshold):
                    continue
                d = abs(area_width - (col[2] - col[0]))
                if d > margin:
                    d = abs(writing_area_properties['guess_width'] - (col[2] - col[0]))
                if d < dif:
                    dif = d
            ret = dif <= margin
            #
            # width_sibling = proposed_section['boxes'][0][2] - proposed_section['boxes'][0][0]
            # if width_sibling == -1:
            #     dif = 0
            # else:
            #     x1, y1, x2, y2 = writing_area_properties['area']
            #     dif = abs(x2 - x1 - width_sibling)
            #     if dif > margin:
            #         dif = abs(writing_area_properties['guess_width'] - width_sibling)
            # ret = dif <= margin
        else:
            ret = False
        return ret

    def _has_area_similar_center_to_proposed_columns(self, writing_area_properties, proposed_section):
        if len(proposed_section['boxes']) > 0:
            margin = 2 * self.threshold
            x1, y1, x2, y2 = writing_area_properties['area']
            dif = margin + 1
            cent = (x1 + x2) / 2
            for col in proposed_section['boxes']:
                if not contains([0, 1, 2, 3], col, writing_area_properties['area'], self.threshold):
                    continue
                d = abs(cent - (col[0] + col[2]) / 2)
                if d < dif:
                    dif = d
            ret = dif <= margin
        else:
            ret = False
        return ret

    def _exist_a_proposed_column_to_contain_area(self, writing_area_properties, proposed_section):
        result = False
        for column in proposed_section['boxes']:
            if contains([0, 1, 2, 3], column, writing_area_properties['area'], self.threshold):
                result = True
                break
        return result

    def is_proposed_section_compatible(self, writing_area_properties, proposed_section):
        result = (len(proposed_section['boxes']) <= 1) if writing_area_properties['is_single'] else (
                    len(proposed_section['boxes']) > 0)
        result = result and contains([0, 1, 2, 3], proposed_section['container'], writing_area_properties['area'],
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

    def add_section_from_proposed_section(self, pos_proposed_section, is_single_area):
        proposed_section = self.proposed_sections[pos_proposed_section]
        if is_single_area:
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
            self.add_section_from_proposed_section(ppos, writing_area_properties['is_single'])
            self.sections[pos].top = writing_area_properties['area'][1]
            self.sections[pos].add_writing_area(writing_area_properties)
        else:
            pos = self.search_section(writing_area_properties)
            if pos >= 0:
                self.sections[pos].add_writing_area(writing_area_properties)
            elif contains([0, 1, 2, 3], self.page_boundary, writing_area_properties['area'], self.threshold):
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

    def sort_content(self, all=False):
        self.sections.sort(key=lambda x: x.top * 10000 + x.left)
        if all:
            for section in self.sections:
                section.sort_content(all)

    def process_unlocated_area(self, unlocated_pos):
        ret = False
        pos = -1
        status = -10
        found = False
        while not found and pos + 1 < len(self.sections):
            found = True
            for i in range(pos + 1, len(self.sections)):
                status = self.get_status_in_section(self.unlocated_boxes[unlocated_pos]['area'], i)
                if status < 1 and contains([0, 2], self.sections[i].coordinates,
                                           self.unlocated_boxes[unlocated_pos]['area'], self.threshold):
                    pos = i
                    break
            if -1 <= status < -0.5 and pos > -1:
                if self.unlocated_boxes[unlocated_pos]['is_single']:
                    if self.__can_use_existing_single_section and type(self.sections[pos]) is SingleSection:
                        # afegir a la secció pos
                        self.sections[pos].top = self.unlocated_boxes[unlocated_pos]['area'][1]
                        self.sections[pos].add_writing_area(self.unlocated_boxes[unlocated_pos])
                    elif self.__can_use_existing_single_section and pos > 0 and type(
                            self.sections[pos - 1]) is SingleSection:
                        # afegir a la secció pos -1
                        self.sections[pos - 1].bottom = self.unlocated_boxes[unlocated_pos]['area'][3]
                        self.sections[pos - 1].add_writing_area(self.unlocated_boxes[unlocated_pos])
                    else:
                        # inserir una nova secció
                        b = [self.page_boundary[0], self.unlocated_boxes[unlocated_pos]['area'][1],
                             self.page_boundary[2], self.unlocated_boxes[unlocated_pos]['area'][3]]
                        section = SingleSection(self, b)
                        section.add_writing_area(self.unlocated_boxes[unlocated_pos])
                        self.sections.insert(pos, section)
                    self.sort_content()
                    ret = self.__can_use_existing_single_section = True
                else:
                    self.col_unlocated_boxes.set_unlocated_box(self.unlocated_boxes[unlocated_pos], -1,
                                                               self.sections[pos])
                    ret = True
                    self.__can_use_existing_single_section = False

                    # # Afegim nova secció i si cal, ja l'ajuntarem
                    # b = [self.page_boundary[0], self.unlocated_boxes[unlocated_pos]['area'][1], self.page_boundary[2], self.unlocated_boxes[unlocated_pos]['area'][3]]
                    # section = BigSectionOfSibling(self, b)
                    # col = SingleSection(section, self.unlocated_boxes[unlocated_pos]['area'], is_bottom_expandable=True)
                    # section.add_new_section(col)
                    # section.add_writing_area(self.unlocated_boxes[unlocated_pos])
                    # self.sections.insert(pos, section)
            elif -0.5 <= status < 0.5 and pos > -1:
                if self.unlocated_boxes[unlocated_pos]['is_single']:
                    if (type(self.sections[pos]) is SingleSection or len(self.sections[pos].siblings) == 1):
                        section = self.sections[pos] if type(self.sections[pos]) is SingleSection else \
                        self.sections[pos].siblings[0]
                        if section.can_be_added(self.unlocated_boxes[unlocated_pos], True):
                            section.add_writing_area(self.unlocated_boxes[unlocated_pos])
                            section.writing_areas.sort(key=lambda x: x[1])
                            ret = True
                    else:
                        # Contradicció. (if search_sibling > -1 => pertany a una columna else cal veure com es resol)
                        cpos = self.sections[pos].search_sibling(self.unlocated_boxes[unlocated_pos])
                        if cpos > -1:
                            self.col_unlocated_boxes.set_unlocated_box(self.unlocated_boxes[unlocated_pos], 0,
                                                                       self.sections[pos].siblings[cpos])
                            ret = True
                        elif self.sections[pos]._has_area_similar_width(self.unlocated_boxes[unlocated_pos]):
                            # possible error fa la mateixa mida de la columna. Verifiquem si hi cap
                            cp = -1
                            for i, col in enumerate(self.sections[pos].siblings):
                                if col.fits_vertically(self.unlocated_boxes[unlocated_pos]['area']):
                                    cp = i
                                    break
                            if cp > -1:
                                # afegim a la columna
                                self.unlocated_boxes[unlocated_pos]['is_single'] = False
                                self.sections[pos].add_writing_area(self.unlocated_boxes[unlocated_pos])
                                ret = True
                            else:
                                # TODO: Contradicció. Cal veure con es resol
                                pass

                        else:
                            # trencar la secció existent e inserir la nova
                            f, cp1, cp2 = self.sections[pos].search_siblings_to_cut_for_bigger_section(
                                self.unlocated_boxes[unlocated_pos])
                            if f:
                                if cp1 == 0 and cp2 + 1 == len(self.sections[pos].siblings):
                                    # convertir la secció pos en dues i inserir entre elles una nova secció
                                    self.sections[pos].cut(self.unlocated_boxes[unlocated_pos]['area'])
                                    if pos + 1 < len(self.sections) and (
                                            type(self.sections[pos + 1]) is SingleSection or len(
                                            self.sections[pos + 1].siblings) == 1):
                                        new_section = self.sections[pos + 1]
                                        new_section.top = self.unlocated_boxes[unlocated_pos]['area'][1]
                                    else:
                                        new_section = SingleSection(self, self.unlocated_boxes[unlocated_pos]['area'])
                                        self.add_new_section(new_section)
                                    new_section.add_writing_area(self.unlocated_boxes[unlocated_pos])
                                    new_section.sort_content()
                                    self.sort_content()
                                else:
                                    # crear una nova secció i bloquejar la posició de la caixa a col·locar
                                    for i in range(cp1, cp2):
                                        (self.sections[pos].siblings[i].append([
                                            self.sections[pos].siblings[i].left,
                                            self.unlocated_boxes[unlocated_pos]['area'][1],
                                            self.sections[pos].siblings[i].right,
                                            self.unlocated_boxes[unlocated_pos]['area'][3]]))
                                        self.sections[pos].siblings[i].sort_content()
                                    new_section = BigSectionOfSibling(self, self.unlocated_boxes[unlocated_pos]['area'])
                                    new_column = SingleSection(new_section, new_section.coordinates)
                                    new_column.add_writing_area(self.unlocated_boxes[unlocated_pos])
                                    new_section.add_new_section(new_column)
                                    self.add_new_section(new_section)
                                    self.sort_content()
                                ret = True
                            else:
                                # TODO: Contradicció.es crea una nova secció
                                pass

                else:
                    if type(self.sections[pos]) is SingleSection:
                        # TODO: Contradicció. Cal veure con es resol
                        pass
                    elif type(self.sections[pos]) is BigSectionOfSibling:
                        # TODO: agrupem per columnes
                        cpos = self.sections[pos].search_sibling(self.unlocated_boxes[unlocated_pos])
                        if cpos > -1:
                            self.col_unlocated_boxes.set_unlocated_box(self.unlocated_boxes[unlocated_pos], 0,
                                                                       self.sections[pos].siblings[cpos])
                            ret = True
                        else:
                            found = False
                        # for sp in self.sections[pos].siblings:
                        #     if (contains([0,2], sp.coordinates, self.unlocated_boxes[unlocated_pos]['area'],
                        #                  self.threshold*2)
                        #             and is_similar_distance(sp.left, sp.right, self.unlocated_boxes[unlocated_pos]['area'][0],
                        #                                     self.unlocated_boxes[unlocated_pos]['area'][2], (sp.right-sp.left)*0.45)
                        #             and sp.fits_vertically(self.unlocated_boxes[unlocated_pos]['area'])):
                        #         sp.writing_areas.append(self.unlocated_boxes[unlocated_pos]['area'])
                        #         sp.writing_areas.sort(key=lambda x: x[1])
                        #         ret = True
            elif 0.5 <= status <= 1 and pos > -1:
                # TODO: Es troba al final. Cal veure con es resol
                pass
            else:
                # TODO: something was wrong. WHat?
                pass
        return ret

    def get_status_in_section(self, unlocated, spos):
        hu = unlocated[3] - unlocated[1]
        hs = self.sections[spos].bottom - self.sections[spos].top
        hs_plus_hu = hs + hu
        dif = unlocated[3] - self.sections[spos].top
        if dif <= 0:
            status = -1
        elif 0 < dif < hu:
            status = dif / hu - 1
        elif hu <= dif <= hs:
            status = 0
        elif hs < dif < hs_plus_hu:
            status = (dif - hs) / hu
        else:
            status = 1
        return status

    @staticmethod
    def build_lauoud_from_sections(page_boundary, sections, writing_area_list, w: int, h: int, threshold=None,
                                   image=None, lmodel=None):
        main_layout = MainLayout(page_boundary, w, h, proposed_sections=sections)
        if threshold is not None:
            main_layout.threshold = threshold

        min_left = 100000
        max_right = 0

        # margin = threshold + threshold + threshold + threshold + threshold

        # if page_boundary[0] > min(margin, w * 0.05):
        if page_boundary[0] > w * 0.05:
            page_boundary[0] = threshold
        # if page_boundary[1] > min(margin, h * 0.05):
        if page_boundary[1] > h * 0.05:
            page_boundary[1] = threshold
        # if page_boundary[2] < max( w - margin, w * 0.95):
        if page_boundary[2] < w * 0.95:
            page_boundary[2] = w - threshold
        # if page_boundary[3] < max( h - margin, h * 0.95):
        if page_boundary[3] < h * 0.95:
            page_boundary[3] = h - threshold

        to_remove = []
        for i, writing_area in enumerate(writing_area_list):
            if not contains([0, 1, 2, 3], page_boundary, writing_area, main_layout.threshold):
                to_remove.append(i)
                continue
            a_left = writing_area[0]
            a_right = writing_area[2]
            if min_left > a_left:
                min_left = a_left
            if max_right < a_right:
                max_right = a_right

        for i in reversed(to_remove):
            writing_area_list.pop(i)

        writing_area_list.sort(key=lambda x: x[1] * 10000 + x[0])
        for i in range(len(writing_area_list)):
            wa = get_writing_area_properties(writing_area_list, i, min_left, max_right, main_layout.threshold)
            main_layout.add_writing_area(wa)

        main_layout.col_unlocated_boxes = ColUnlocatedBoxes()
        for i in range(len(main_layout.unlocated_boxes) - 1, -1, -1):
            # buscar la secció prèvia a l'àrea
            if main_layout.process_unlocated_area(i):
                main_layout.unlocated_boxes.pop(i)

        # recol·locar inconherències en columnes
        prev_col_unlocated_boxes = main_layout.col_unlocated_boxes.get_unlocated_boxes_by_type(-1)
        for col_unlocated_box in prev_col_unlocated_boxes:
            s = main_layout.sections.index(col_unlocated_box.section)
            to_locate = []
            to_locate_properties = []
            for unloc_wa in col_unlocated_box.boxes:
                to_locate.append(unloc_wa['area'])
                to_locate_properties.append(unloc_wa)
            min_left = main_layout.sections[s].left
            max_right = main_layout.sections[s].right
            for i, wa in enumerate(to_locate):
                new_section = False
                wap = get_writing_area_properties(to_locate, i, min_left, max_right, main_layout.threshold)
                if wap['is_single'] and (
                        type(main_layout.sections[s]) is SingleSection or len(main_layout.sections[s].siblings) == 1):
                    if main_layout.sections[s].can_be_added(to_locate_properties[i]):
                        main_layout.sections[s].add_writing_area(to_locate_properties[i])
                elif not wap['is_single'] and type(main_layout.sections[s]) is BigSectionOfSibling:
                    if main_layout.sections[s].can_be_added(to_locate_properties[i]):
                        main_layout.sections[s].add_writing_area(to_locate_properties[i])
                    elif main_layout.sections[s].can_be_inserted_a_column_for(to_locate_properties[i]):
                        # insert
                        section = SingleSection(main_layout.sections[s], to_locate[i])
                        section.add_writing_area(to_locate_properties[i])
                        main_layout.sections[s].add_new_section(section)
                        main_layout.sections[s].sort_content()
                    else:
                        # nova secció
                        new_section = True
                else:
                    # nova secció
                    new_section = True
                if new_section:
                    if wap['is_single']:
                        # crear una nova SingleSection i afegir la wa en cas contrari
                        if s == 0:
                            min_top = page_boundary[1]
                        else:
                            min_top = min(main_layout.sections[s - 1].bottom, wa[1])
                        max_bottom = main_layout.sections[s].top
                        section = SingleSection(main_layout, [page_boundary[0], min_top, page_boundary[2], max_bottom])
                        section.add_writing_area(to_locate_properties[i])
                    else:
                        # crear una nova BigSectionOfSibling i una SingleSection com a sibling i afegir-hi la wa en cas contrari
                        if s == 0:
                            min_top = page_boundary[1]
                        else:
                            min_top = min(main_layout.sections[s - 1].bottom, wa[1])
                        max_bottom = main_layout.sections[s].top
                        section = BigSectionOfSibling(main_layout,
                                                      [page_boundary[0], min_top, page_boundary[2], max_bottom])
                        col = SingleSection(section, [to_locate[i][0], min_top, to_locate[i][2], max_bottom])
                        col.add_writing_area(to_locate_properties[i])
                        section.add_new_section(col)
                    main_layout.add_new_section(section)
                    main_layout.sort_content()

        overlapped = []
        inside_col_unlocated_boxes = main_layout.col_unlocated_boxes.get_unlocated_boxes_by_type(0)
        for col_unlocated_box in inside_col_unlocated_boxes:
            s = main_layout.sections.index(col_unlocated_box.column.section_container)
            # c = main_layout.sections[s].siblings.index(col_unlocated_box.column)
            to_locate = []
            to_locate_properties = []
            for unloc_wa in col_unlocated_box.boxes:
                to_locate.append(unloc_wa['area'])
                to_locate_properties.append(unloc_wa)

            if type(main_layout.sections[s]) is SingleSection:
                min_left = main_layout.sections[s].left
                max_right = main_layout.sections[s].right
            else:
                min_left = col_unlocated_box.column.left
                max_right = col_unlocated_box.column.right
            for i, wa in enumerate(to_locate):
                wap = get_writing_area_properties(to_locate, i, min_left, max_right, main_layout.threshold)
                if wap['is_single']:
                    # force insert into columns
                    col_unlocated_box.column.writing_areas.append(wa)
                    col_unlocated_box.column.sort_content()
                else:
                    if image is not None and lmodel is not None:
                        p_over = -1
                        for i in range(len(overlapped)):
                            for over_box in overlapped[i]['overlapped_boxes']:
                                dif_center = abs((wa[0] + wa[2])/2 - (over_box[0]+over_box[2])/2)
                                dif_start = abs(wa[0]-over_box[0])
                                min_max_h = max(wa[3], over_box[3]) - min(wa[1], over_box[1])
                                sum_h = (wa[3] - wa[1]) + (over_box[3] - over_box[1])
                                if (overlap_vertically(over_box, wa, main_layout.threshold)
                                        or (dif_start <= main_layout.threshold or dif_center <= main_layout.threshold)
                                        and abs(min_max_h - sum_h) <= threshold):
                                    p_over = i
                                    break
                        if p_over == -1:
                            overlapped.append({'column': col_unlocated_box.column, 'overlapped_boxes':[wa]})
                        else:
                            overlapped[p_over]['overlapped_boxes'].append(wa)
                    else:
                        main_layout.unlocated_boxes.append(to_locate_properties[i])
        #resolve overlapped
        for overlapped_in_column in overlapped:
            fragment_top = overlapped_in_column['overlapped_boxes'][0][1]
            fragment_bottom = overlapped_in_column['overlapped_boxes'][0][3]
            for wa_overlapped in overlapped_in_column['overlapped_boxes']:
                if wa_overlapped[1] < fragment_top:
                    fragment_top = wa_overlapped[1]
                if wa_overlapped[3] > fragment_bottom:
                    fragment_bottom = wa_overlapped[3]
            fragment_image = image[fragment_top + 5:fragment_bottom + 5,
                         overlapped_in_column['column'].left:overlapped_in_column['column'].right]
            fragment_predict = get_annotated_prediction(fragment_image, lmodel)
            new_columns = []
            for pred in fragment_predict:
                if pred['class_name'] == 'columna':
                    new_columns.append(pred['box'])
            if len(new_columns) > 1:
                # cortar e insertar varias singlesection (una por columna nueva) en la seccion existente
                overlapped_in_column['column'].cut([0, fragment_top, 0, fragment_bottom])
                for new_col_box in new_columns:
                    x1 = overlapped_in_column['column'].left + new_col_box[0]
                    y1 = fragment_top
                    x2 = x1 + new_col_box[2] - new_col_box[0]
                    y2 = fragment_bottom
                    new_column = SingleSection(overlapped_in_column['column'].section_container,[x1, y1, x2, y2])
                    overlapped_in_column['column'].section_container.add_new_section(new_column)
                overlapped_in_column['column'].section_container.sort_content()
            else:
                overlapped_in_column['column'].writing_areas.append(wa)
                overlapped_in_column['column'].sort_content()

        prev_col_unlocated_boxes = main_layout.col_unlocated_boxes.get_unlocated_boxes_by_type(1)
        for col_unlocated_box in prev_col_unlocated_boxes:
            s = main_layout.sections.index(col_unlocated_box.section)
            for unloc_wa in col_unlocated_box.boxes:
                main_layout.unlocated_boxes.append(unloc_wa)

        main_layout.sort_content(True)
        return main_layout


class BigSectionOfSibling(StructuredSection):
    def __init__(self, container: MainLayout, boundaries_or_proposed_section: Union[dict, list] = None,
                 threshold: Union[int, ThresholdAttribute] = None, proposed_section=None, boundaries=None):
        super().__init__(boundaries_or_proposed_section, threshold, proposed_section, boundaries, container)
        if self.proposed_section is not None and len(self.proposed_section['boxes']) > 0:
            self._width_sibling = self.proposed_section['boxes'][0][2] - self.proposed_section['boxes'][0][0]
            for col in self.proposed_section['boxes']:
                col_sec = SingleSection(self, col)
                self.add_new_section(col_sec)
        else:
            self._width_sibling = -1

    @AbstractSection.top.setter
    def top(self, value):
        for sib in self.siblings:
            if (self._diagonal_points[0][1] - self.threshold) <= sib.top < value:
                sib.top = value
        self._diagonal_points[0][1] = value

    @AbstractSection.bottom.setter
    def bottom(self, value):
        for sib in self.siblings:
            if (self._diagonal_points[0][1] - self.threshold) >= sib.bottom > value:
                sib.bottom = value
        self._diagonal_points[1][1] = value

    @property
    def width_sibling(self):
        return self._width_sibling

    @property
    def siblings(self):
        return self.sections

    def cut(self, boundaries):
        partial_section_2 = BigSectionOfSibling(self.section_container,
                                                [self.left, boundaries[3], self.right, self.bottom])
        keep = False
        for col in self.siblings:
            partial_column_2 = SingleSection(partial_section_2, [col.left, boundaries[3], col.right, col.bottom])
            for i in range(len(col.writing_areas) - 1, -1, -1):
                if col.writing_areas[i][1] >= boundaries[3]:
                    partial_column_2.writing_areas.append(col.writing_areas.pop(i))
                    keep = True
            if len(col.writing_areas) == 0:
                col._writing_areas = partial_column_2._writing_areas
                col.top = partial_column_2.top
                col.bottom = partial_column_2.bottom
                self.remove(col)
                col.section_container = partial_section_2
                partial_section_2.add_new_section(col)
            else:
                col.bottom = boundaries[1]
                partial_column_2.sort_content()
                partial_section_2.add_new_section(partial_column_2)
        if len(self.siblings)==0:
            self._sections = partial_section_2._sections
            self.top = partial_section_2.top
            self.bottom = partial_section_2.bottom
        else:
            self.bottom = boundaries[1]
            if keep:
                self.section_container.add_new_section(partial_section_2)
                self.section_container.sort_content()

    def sort_content(self, all=False):
        self.siblings.sort(key=lambda x: x.left * 10000 + x.top)
        if all:
            for sibling in self.siblings:
                sibling.sort_content()

    def _has_area_similar_center(self, writing_area_properties, only_width=False):
        margin = self.threshold
        x1, y1, x2, y2 = writing_area_properties['area']
        dif = margin + 1
        cent = (x1 + x2) / 2
        for sibling in self.siblings:
            if only_width:
                edges = [0, 2]
            else:
                edges = [0, 1, 2, 3]
            if not contains(edges, sibling.coordinates, writing_area_properties['area'], self.threshold):
                continue
            d = abs(cent - sibling.center)
            if d < dif:
                dif = d
        return dif <= margin

    def _has_area_similar_width(self, writing_area_properties, only_width=False):
        margin = self.threshold + self.threshold
        x1, y1, x2, y2 = writing_area_properties['area']
        area_width = x2 - x1
        dif = margin + 1
        for col in self.siblings:
            if only_width:
                edges = [0, 2]
            else:
                edges = [0, 1, 2, 3]
            if not contains(edges, col.coordinates, writing_area_properties['area'], self.threshold):
                continue
            d = abs(area_width - (col.right - col.left))
            if d > margin:
                d = abs(writing_area_properties['guess_width'] - (col.left - col.right))
            if d < dif:
                dif = d
        # ret = dif <= margin
        #
        # if self.width_sibling == -1:
        #     dif = 0
        # else:
        #     x1, y1, x2, y2 = writing_area_properties['area']
        #     dif = abs(x2 - x1 - self.width_sibling)
        #     if dif > margin:
        #         dif = abs(writing_area_properties['guess_width'] - self.width_sibling)
        return dif <= margin

    def is_area_compatible(self, writing_area_properties, only_width=False):
        ret = self._has_area_similar_width(writing_area_properties, only_width)
        if not ret:
            ret = self._has_area_similar_center(writing_area_properties)
        return ret

    def search_sibling(self, writing_area_properties, only_width=False):
        pos = -1
        for i, sibling in enumerate(self.siblings):
            if sibling.can_be_added(writing_area_properties):
                pos = i
                break
        return pos

    def search_siblings_to_cut_for_bigger_section(self, writing_area_properties):
        box = writing_area_properties['area']
        width = box[2] - box[0]
        cx1 = self.right
        cx2 = self.right
        cp1 = 0
        cp2 = 0
        found = False
        exit_loop = False
        while not exit_loop:
            cp1 = len(self.siblings)
            for i in range(cp2, len(self.siblings)):
                col = self.siblings[i]
                if col.fits_vertically(box):
                    cp1 = i
                    break
            cp2 = len(self.siblings)
            for i in range(cp1, len(self.siblings)):
                col = self.siblings[i]
                if col.fits_vertically(box):
                    cp2 = i
                elif box[1] - col.bottom + self.threshold >= 0:
                    cp2 = i
                else:
                    break
            cx1 = self.siblings[cp1].left if cp1 < len(self.siblings) else self.right
            cx2 = self.siblings[cp2].right if cp2 < len(self.siblings) else self.right
            if contains([0, 2], [cx1, 0, cx2, 0], box, self.threshold):
                exit_loop = found = True
            else:
                cp2 += 1
                exit_loop = cp2 == len(self.siblings)
        return found, cp1, cp2

    def can_be_added(self, writing_area_properties):
        result = not writing_area_properties['is_single']
        edges = [0, 1, 2]
        if not self.is_bottom_expandable:
            edges.append(3)
        result = result and contains(edges, self.coordinates, writing_area_properties['area'], self.threshold)
        result = result and self.is_area_compatible(writing_area_properties)
        pos = self.search_sibling(writing_area_properties)
        return result and pos >= 0

    def can_be_inserted_a_column_for(self, writing_area_properties):
        ret = False
        # buscar espacio libre entre columnas
        for i, col in enumerate(self.siblings):
            if i == 0:
                free_with = col.left - self.page_boundary[0]
                if free_with - self.width_sibling >= self.threshold and contains([0, 2],
                                                                                 [self.page_boundary[0], 0,
                                                                                  col.left, 0],
                                                                                 writing_area_properties['area'],
                                                                                 self.threshold):
                    ret = True
                    break
            elif i > 0:
                free_with = col.left - self.siblings[i - 1].right
                if free_with - self.width_sibling >= self.threshold and contains([0, 2],
                                                                                 [self.siblings[i - 1].right, 0,
                                                                                  col.left, 0],
                                                                                 writing_area_properties['area'],
                                                                                 self.threshold):
                    ret = True
                    break

        if not ret:
            free_with = self.page_boundary[2] - self.siblings[-1].right
            if free_with - self.width_sibling >= self.threshold and contains([0, 2],
                                                                             [self.siblings[-1].right, 0,
                                                                              self.page_boundary[2], 0],
                                                                             writing_area_properties['area'],
                                                                             self.threshold):
                ret = True
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
        self._writing_areas = []

    @property
    def center(self):
        if self._len == 0:
            ret = (self.left + self.right) / 2
        else:
            ret = self._suma_center / self._len
        return ret

    @property
    def writing_areas(self):
        return self._writing_areas

    def can_be_added(self, writing_area_properties, verify_space=False):
        result = writing_area_properties['is_single'] == (not self._is_column)
        edges = [0, 1, 2]
        if not self.is_bottom_expandable:
            edges.append(3)
        result = result and contains(edges, self.coordinates, writing_area_properties['area'], self.threshold)
        if result and verify_space:
            v = self.fits_vertically(writing_area_properties['area'])
            result = result and v
        return result

    def sort_content(self, all=False):
        self.writing_areas.sort(key=lambda x: x[1] * 10000 + x[0])

    def fits_vertically(self, writing_area):
        v = False
        thr = self.threshold
        for i in range(len(self.writing_areas)):
            wa = self.writing_areas[i]
            if i == 0:
                if contains([1, 3], [0, self.top, 0, wa[1]], writing_area, self.threshold):
                    v = True
                    break
            else:
                if contains([1, 3], [0, self.writing_areas[i - 1][3], 0, wa[1]], writing_area, self.threshold):
                    v = True
                    break
        if not v and contains([1, 3], [0, wa[3], 0, self.bottom], writing_area, self.threshold):
            v = True
        return v

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
        self._writing_areas.append(writing_area_properties['area'])

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

    def cut(self, boundaries):
        partial_section_2 = SingleSection(self.section_container, [self.left, boundaries[3], self.right, self.bottom])
        keep = False
        for i in range(len(self.writing_areas) - 1, -1, -1):
            if self.writing_areas[i][1] >= boundaries[3]:
                partial_section_2.writing_areas.append(self.writing_areas.pop(i))
                keep = True
        if len(self.writing_areas)==0:
            self._writing_areas = partial_section_2._writing_areas
            self.top = partial_section_2.top
            self.bottom = partial_section_2.bottom
        else:
            self.bottom = boundaries[1]
            if keep:
                self.section_container.add_new_section(partial_section_2)
                self.section_container.sort_content()


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
