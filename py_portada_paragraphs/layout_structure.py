from builtins import staticmethod, enumerate
from typing import Union
import numpy

from .py_portada_utility_for_layout import (overlap_vertically, overlap_horizontally, calculate_coordinates, contains,
                                            horizontal_overlapping_ratio, is_similar_distance,
                                            get_relative_right_loc_in_boxes, get_relative_bottom_loc_in_boxes,
                                            get_relative_top_loc_in_boxes, get_relative_left_loc_in_boxes,
                                            get_boxes_non_overlapping_positioning, VERTICAL_POSITIONING,
                                            HORIZONTAL_POSITIONING)
from .py_yolo_layout import get_annotated_prediction


def get_writing_area_properties(writing_area_list, i, min_left, max_right, threshold):
    guess_left = min_left
    guess_right = max_right
    is_single = True
    offset = -1

    writing_area = writing_area_list[i]
    while (i + offset) >= 0:
        if (contains([0, 2], [min_left, 0, max_right, 0], writing_area_list[i + offset], threshold)
                and overlap_vertically(writing_area_list[i + offset], writing_area, threshold)
                and not overlap_horizontally(writing_area_list[i + offset], writing_area, threshold)):
            if writing_area_list[i + offset][0] < writing_area[0] and guess_left < writing_area_list[i + offset][2]:
                guess_left = writing_area_list[i + offset][2]
            elif writing_area[0] <= writing_area_list[i + offset][0] < guess_right:
                guess_right = writing_area_list[i + offset][0]
            is_single = False
        offset -= 1
    offset = 1
    while (i + offset) < len(writing_area_list):
        if (contains([0, 2], [min_left, 0, max_right, 0], writing_area_list[i + offset], threshold)
                and overlap_vertically(writing_area_list[i + offset], writing_area, threshold)
                and not overlap_horizontally(writing_area_list[i + offset], writing_area, threshold)):
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

class UnlocatedBoxes:
    INSIDE_OF_SECTION_POS = 0
    PREVIOUS_OF_SECTION_POS = -1
    NEXT_OF_SECTION_POS = 1
    # # CASES FOR INSIDE_OF_SECTION_POS
    # SINGLE_AREA_IN_SINGLE_SECTION_CASE = 0
    # SINGLE_AREA_IN_SIBLING_SECTION_CASE = 1
    # #SINGLE_AREA_INSIDE_ONE_SIBLING_WITH_SIMILAR_WIDTH_IN_SIBLING_SECTION_CASE = 1
    # #SINGLE_AREA_INSIDE_ONE_SIBLING_WITH_DIFERENT_WIDTH_IN_SIBLING_SECTION_CASE = 2
    # #SINGLE_AREA_IN_MULTI_SIBLINGS_IN_SIBLING_SECTION_CASE = 3
    # SIBLING_AREA_IN_SIBLING_SECTION_CASE = 2
    # SIBLING_AREA_IN_SINGLE_SECTION_CASE = 3
    # # CASES FOR OUTSIDE_OF_SECTION_POS
    # SINGLE_AREA_CASE = 0
    # SIBLING_AREA_CASE = 1


    def __init__(self):
        # self._index_maps_by_pos_type = [
        #     [
        #         {}, # -POS = INSIDE_OF_SECTION_POS, -CASE = SINGLE_AREA_IN_SINGLE_SECTION_CASE
        #         {}, # -POS = INSIDE_OF_SECTION_POS, -CASE = SINGLE_AREA_IN_SIBLING_SECTION_CASE
        #         {}, # -POS = INSIDE_OF_SECTION_POS, -CASE = SIBLING_AREA_IN_SIBLING_SECTION_CASE
        #         {}  # -POS = INSIDE_OF_SECTION_POS, -CASE = SIBLING_AREA_IN_SINGLE_SECTION_CASE
        #     ],
        #     [
        #         {}, # -POS = NEXT_OF_SECTION_POS, -CASE = SINGLE_AREA_CASE
        #         {}  # -POS = NEXT_OF_SECTION_POS, -CASE = SIBLING_AREA_CASE
        #     ],
        #     [
        #         {},  # -POS = PREVIOUS_OF_SECTION_POS, -CASE = SINGLE_AREA_CASE
        #         {}   # -POS = PREVIOUS_OF_SECTION_POS, -CASE = SIBLING_AREA_CASE
        #     ]
        # ]
        # self._unlocated_boxes_by_pos_type = [
        #     [
        #         [], # -POS = INSIDE_OF_SECTION_POS, -CASE = SINGLE_AREA_IN_SINGLE_SECTION_CASE
        #         [], # -POS = INSIDE_OF_SECTION_POS, -CASE = SINGLE_AREA_IN_SIBLING_SECTION_CASE
        #         [], # -POS = INSIDE_OF_SECTION_POS, -CASE = SIBLING_AREA_IN_SIBLING_SECTION_CASE
        #         []  # -POS = INSIDE_OF_SECTION_POS, -CASE = SIBLING_AREA_IN_SINGLE_SECTION_CASE
        #     ],
        #     [
        #         [], # -POS = NEXT_OF_SECTION_POS, -CASE = SINGLE_AREA_CASE
        #         []  # -POS = NEXT_OF_SECTION_POS, -CASE = SIBLING_AREA_CASE
        #     ],
        #     [
        #         [],  # -POS = PREVIOUS_OF_SECTION_POS, -CASE = SINGLE_AREA_CASE
        #         []   # -POS = PREVIOUS_OF_SECTION_POS, -CASE = SIBLING_AREA_CASE
        #     ]
        # ]
        self._index_maps_by_pos = [{}, {}, {}]
        self._unlocated_boxes_by_pos = [[], [], []]

    def set_unlocated_box(self, unlocated_box_properties, referred_section, pos, comp=None):
        index_map = self._index_maps_by_pos[pos]
        unlocated_boxes = self._unlocated_boxes_by_pos[pos]
        if not referred_section in index_map:
            index_map[referred_section] = len(unlocated_boxes)
            if pos == UnlocatedBoxes.INSIDE_OF_SECTION_POS:
                data = dict(referred_section=referred_section, boxes=[], compared_width=[])
            else:
                data = dict(referred_section=referred_section, boxes=[])
            unlocated_boxes.append(data)
        unlocated_boxes[index_map[referred_section]]['boxes'].append(unlocated_box_properties)
        if comp is not None:
            unlocated_boxes[index_map[referred_section]]['compared_width'].append(comp)
        #OLD
        # if pos == 0:
        #     self.set_unlocated_for_inside_type(unlocated_box_properties, section_or_column)
        # else:
        #     self.set_unlocated_for_outside_type(unlocated_box_properties, pos, section_or_column)

    # def set_unlocated_for_inside_type(self, unlocated_box_properties, column):
    #     index_map = self.index_maps_by_type[0]
    #     unlocated_boxes = self.unlocated_boxes_by_type[0]
    #     if not column in index_map:
    #         index_map[column] = len(unlocated_boxes)
    #         unlocated_boxes.append(InsideUnlocatedBoxesToColumn(column))
    #     unlocated_boxes[index_map[column]].boxes.append(unlocated_box_properties)
    #
    # def set_unlocated_for_outside_type(self, unlocated_box_properties, type, section):
    #     index_map = self.index_maps_by_type[type]
    #     unlocated_boxes = self.unlocated_boxes_by_type[type]
    #     if not section in index_map:
    #         index_map[section] = len(unlocated_boxes)
    #         unlocated_boxes.append(OutsideUnlocatedBoxesForSection(section))
    #     unlocated_boxes[index_map[section]].boxes.append(unlocated_box_properties)

    def get_unlocated_boxes_by_pos(self, pos):
        unlocated_boxes = self._unlocated_boxes_by_pos[pos]
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
        return [self._diagonal_points[0][0], self._diagonal_points[0][1], self._diagonal_points[1][0], self._diagonal_points[1][1]]

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

    def can_be_added(self, writing_area_properties, only_width=False):
        pass

    def add_writing_area(self, writing_area_properties):
        pass

    def sort_content(self):
        pass

    def _compare_similar_width_area_to_sibling(self, writing_area_properties, only_width=False):
        pass

    def get_single_sections_as_boxes(self, boxes=None, add_threshold=False):
        pass

    def get_vertical_positioning_vs(self, box):
        pass

    def get_horizontal_positioning_vs(self, box):
        pass

    def get_left_right_boundary_for(self, box):
        pass

    # def get_right_boundary_for(self, p1, p2=None):
    #     pass

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
        self.top_sections=[]
        self.bottom_sections=[]
        self.left_sections=[]
        self.right_sections=[]

    #        self.is_bottom_expandable = True

    @property
    def sections(self):
        return self._sections

    def get_single_sections_as_boxes(self, boxes=None, add_threshold=False):
        if boxes is None:
            boxes = []
        for section in self.sections:
            section.get_single_sections_as_boxes(boxes, add_threshold)
        return boxes

    def __update_top_bottom_left_right_new_section(self, section, edge):
        margin = self.threshold+self.threshold
        if edge==0:
            b_sections = self.left_sections
        elif edge==1:
            b_sections = self.top_sections
        elif edge==2:
            b_sections = self.right_sections
        elif edge==3:
            b_sections = self.bottom_sections
        to_remove=[]
        l = len(b_sections)
        ov = False
        dif = 10000
        for i in range(l):
            if edge == 0:
                d = section.left - b_sections[i].left
                ov = ov or overlap_vertically(section.coordinates, b_sections[i].coordinates, self.threshold)
            elif edge == 1:
                d = section.top - b_sections[i].top
                ov = ov or overlap_horizontally(section.coordinates, b_sections[i].coordinates, self.threshold)
            elif edge == 2:
                d =  b_sections[i].right - section.right
                ov = ov or overlap_vertically(section.coordinates, b_sections[i].coordinates, self.threshold)
            elif edge == 3:
                d =  b_sections[i].bottom - section.bottom
                ov = ov or overlap_horizontally(section.coordinates, b_sections[i].coordinates, self.threshold)
            if ov and d < -margin:
                to_remove.append(i)
            if dif > d:
                dif = d

        if not ov or dif < margin:
            b_sections.append(section)
            for r in reversed(to_remove):
                b_sections.pop(r)


    def add_new_section(self, section):
        self._sections.append(section)
        if len(self.sections)==1:
            self.top_sections.append(section)
            self.bottom_sections.append(section)
            self.left_sections.append(section)
            self.right_sections.append(section)
        else:
            self.__update_top_bottom_left_right_new_section(section,0)
            self.__update_top_bottom_left_right_new_section(section,1)
            self.__update_top_bottom_left_right_new_section(section,2)
            self.__update_top_bottom_left_right_new_section(section,3)

    def _fill_vertical_gaps_and_resize(self):
        pass

    def _fill_horizontal_gaps_and_resize(self):
        pass

    def _resize_left_sections(self):
        pass

    def _resize_top_sections(self):
        pass

    def _resize_right_sections(self):
        pass

    def _resize_bottom_sections(self):
        pass


class MainLayout(StructuredSection):
    def __init__(self, page_boundary, w: int, h: int = 0, threshold=40, proposed_sections=None):
        super().__init__([0, 0, w, h], threshold, page_boundary=page_boundary)
        self.proposed_sections = proposed_sections
        self.list_of_unlocated_boxes = []
        self.unlocated_boxes = UnlocatedBoxes()
        self.__can_use_existing_single_section = True

    def get_single_sections_as_boxes(self, add_threshold=False):
        boxes = []
        return super().get_single_sections_as_boxes(boxes, add_threshold)

    def get_unlocated_boxes(self):
        boxes = []
        for wa in self.list_of_unlocated_boxes:
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

    def _fill_horizontal_gaps_and_resize(self):
        # fill the gaps
        new_sections = []
        for i, section in enumerate(self.sections):
            inc_left = 10000
            inc_right = 10000
            rel_left = 0
            rel_right = 0
            for j, section2 in enumerate(self.sections):
                if j == i:
                    continue
                o, r, v = section2.get_horizontal_positioning_vs(section.coordinates)
                if o:
                    if r > 0:
                        if inc_left > v:
                            inc_left = max(v, 0)
                            rel_left = r
                    elif r < 0:
                        if inc_right > v:
                            inc_right = max(v, 0)
                            rel_right = r
            if rel_left > 0 and inc_left <= self.threshold:
                section.left -= inc_left
            if rel_right < 0 and inc_right <= self.threshold:
                section.right += inc_right//2
            elif rel_right < 0:
                valid_gap = False
                # new_box = [section.left, section.bottom, section.right, section.bottom + inc_bottom]
                # for j, section2 in enumerate(self.sections):
                #     # TODO Buscar marges laterals i ajustar la nova caixa
                #     o, xl, xr = section2.get_left_right_boundary_for(new_box)
                #     if o:
                #         new_box[0] = xl
                #         new_box[2] = xr
                # new_section = SingleSection(self, new_box)
                # new_section.writing_areas.append(new_box)
                # new_sections.append(new_section)
            if type(section) is BigSectionOfSibling:
                # section._resize_left_sections()
                # section._resize_top_sections()
                # section._resize_right_sections()
                # section._resize_bottom_sections()
                section._fill_horizontal_gaps_and_resize()

        for s in new_sections:
            self.add_new_section(s)


    def _fill_vertical_gaps_and_resize(self):
        # fill the gaps
        new_sections = []
        for i, section in enumerate(self.sections):
            inc_top = 10000
            inc_bottom = 10000
            rel_top = 0
            rel_bottom = 0
            for j, section2 in enumerate(self.sections):
                if j == i:
                    continue
                o, r, v = section2.get_vertical_positioning_vs(section.coordinates)
                if o:
                    if r > 0:
                        if inc_top > v:
                            inc_top = v
                            rel_top = r
                    elif r < 0:
                        if inc_bottom > v:
                            inc_bottom = v
                            rel_bottom = r
            if rel_top > 0 and inc_top <= self.threshold:
                section.top -= inc_top
            if rel_bottom < 0 and inc_bottom <= self.threshold:
                section.bottom += inc_bottom//2
            elif rel_bottom < 0:
                new_box = [section.left, section.bottom, section.right, section.bottom + inc_bottom]
                for j, section2 in enumerate(self.sections):
                    # TODO Buscar marges laterals i ajustar la nova caixa
                    o, xl, xr = section2.get_left_right_boundary_for(new_box)
                    if o:
                        new_box[0] = xl
                        new_box[2] = xr
                new_section = SingleSection(self, new_box)
                new_section.writing_areas.append(new_box)
                new_sections.append(new_section)
            if type(section) is BigSectionOfSibling:
                section._resize_left_sections()
                section._resize_top_sections()
                section._resize_right_sections()
                section._resize_bottom_sections()
                section._fill_vertical_gaps_and_resize()

        for s in new_sections:
            self.add_new_section(s)

    def _resize_left_sections(self):
        for section in self.left_sections:
            if section.left < self.page_boundary[0]:
                self.page_boundary[0] = section.left
            if type(section) is BigSectionOfSibling:
                for col in section.siblings:
                    if col.left < self.page_boundary[0]:
                        self.page_boundary[0] = col.left

        for i, section in enumerate(self.left_sections):
            section.left = self.page_boundary[0]
            if type(section) is BigSectionOfSibling:
                if len(section.siblings)>1:
                    section._resize_left_sections()
                elif len(section.siblings)==1:
                    section.siblings[0].left = section.left

    def _resize_right_sections(self):
        for section in self.right_sections:
            if section.right > self.page_boundary[2]:
                self.page_boundary[2] = section.right
            if type(section) is BigSectionOfSibling:
                for col in section.siblings:
                    if col.right > self.page_boundary[2]:
                        self.page_boundary[2] = col.right

        for section in self.right_sections:
            section.right = self.page_boundary[2]
            if type(section) is BigSectionOfSibling:
                if len(section.siblings)>1:
                    section._resize_right_sections()
                elif len(section.siblings)==1:
                    section.siblings[0].right = section.right

    def _resize_top_sections(self):
        for section in self.top_sections:
            if section.top < self.page_boundary[1]:
                self.page_boundary[1] = section.top
            if type(section) is BigSectionOfSibling:
                for col in section.siblings:
                    if col.top > self.page_boundary[1]:
                        self.page_boundary[1] = col.top

        for section in self.top_sections:
            section.top = self.page_boundary[1]
            if type(section) is BigSectionOfSibling:
                if len(section.siblings) > 1:
                    section._resize_top_sections()
                elif len(section.siblings) == 1:
                    section.siblings[0].top = section.top

    def _resize_bottom_sections(self):
        for section in self.bottom_sections:
            if section.bottom > self.page_boundary[3]:
                self.page_boundary[3] = section.bottom
            if type(section) is BigSectionOfSibling:
                for col in section.siblings:
                    if col.bottom > self.page_boundary[3]:
                        self.page_boundary[3] = col.bottom

        for section in self.bottom_sections:
            section.bottom = self.page_boundary[3]
            if type(section) is BigSectionOfSibling:
                if len(section.siblings)>1:
                    section._resize_bottom_sections()
                elif len(section.siblings)==1:
                    section.siblings[0].bottom = section.bottom

    def add_writing_area(self, writing_area_properties):
        # searching predicted section position compatible with the writing area
        ppos = self.search_proposed_section(writing_area_properties)
        if ppos >= 0:
            # compatible predicted section found. Then, the predicted section is added and the writing area is added
            pos = len(self.sections)
            self.add_section_from_proposed_section(ppos, writing_area_properties['is_single'])
            self.sections[pos].top = writing_area_properties['area'][1]
            self.sections[pos].add_writing_area(writing_area_properties)
        else:
            # Don't exist any predicted section compatible.
            # Searching if an existing section is compatible
            pos = self.search_section(writing_area_properties)
            if pos >= 0:
                # Compatible existing
                self.sections[pos].add_writing_area(writing_area_properties)
            else:
                # OLD
                # elif contains([0, 1, 2, 3], self.page_boundary, writing_area_properties['area'], self.threshold):
                # Adding unallocated box. It isn't possible to know the location now. We will reserve the area to resolve later
                self.list_of_unlocated_boxes.append(writing_area_properties)
                # OLD
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

    def sort_content(self, all=False):
        self.sections.sort(key=lambda x: x.top * 10000 + x.left)
        if all:
            for section in self.sections:
                section.sort_content(all)

    def clasify_unlocated_area(self, unlocated_pos):
        ret = False
        pos = -1
        status = -10
        found = False
        only_width = False
        while not found and pos + 1 < len(self.sections):
            found = True
            prev_pos = pos
            prev_status = status
            for i in range(pos + 1, len(self.sections)):
                status = self.get_status_in_section(self.list_of_unlocated_boxes[unlocated_pos]['area'], i)
                if status < 1 and contains([0, 2], self.sections[i].coordinates,
                                           self.list_of_unlocated_boxes[unlocated_pos]['area'], self.threshold):
                    pos = i
                    break
            if prev_pos == pos:
                status = prev_status
                found = True
                only_width = True
            if -1 <= status < -0.5 and pos > -1:
                # Relative pos to section: PREVIOUS
                self.unlocated_boxes.set_unlocated_box(self.list_of_unlocated_boxes[unlocated_pos],
                                                       self.sections[pos],
                                                       UnlocatedBoxes.PREVIOUS_OF_SECTION_POS)
                ret = True
                #OLD
                # if self.list_of_unlocated_boxes[unlocated_pos]['is_single']:
                #     if self.__can_use_existing_single_section and type(self.sections[pos]) is SingleSection:
                #         # afegir a la secció pos
                #         self.sections[pos].top = self.list_of_unlocated_boxes[unlocated_pos]['area'][1]
                #         self.sections[pos].add_writing_area(self.list_of_unlocated_boxes[unlocated_pos])
                #     elif self.__can_use_existing_single_section and pos > 0 and type(
                #             self.sections[pos - 1]) is SingleSection:
                #         # afegir a la secció pos -1
                #         self.sections[pos - 1].bottom = self.list_of_unlocated_boxes[unlocated_pos]['area'][3]
                #         self.sections[pos - 1].add_writing_area(self.list_of_unlocated_boxes[unlocated_pos])
                #     else:
                #         # inserir una nova secció
                #         b = [self.page_boundary[0], self.list_of_unlocated_boxes[unlocated_pos]['area'][1],
                #              self.page_boundary[2], self.list_of_unlocated_boxes[unlocated_pos]['area'][3]]
                #         section = SingleSection(self, b)
                #         section.add_writing_area(self.list_of_unlocated_boxes[unlocated_pos])
                #         self.sections.insert(pos, section)
                #     self.sort_content()
                #     ret = self.__can_use_existing_single_section = True
                # else:
                #     self.unlocated_boxes.set_unlocated_box(self.list_of_unlocated_boxes[unlocated_pos], -1,
                #                                            self.sections[pos])
                #     ret = True
                #     self.__can_use_existing_single_section = False
                #
                #     # Afegim nova secció i si cal, ja l'ajuntarem
                #     b = [self.page_boundary[0], self.unlocated_boxes[unlocated_pos]['area'][1], self.page_boundary[2], self.unlocated_boxes[unlocated_pos]['area'][3]]
                #     section = BigSectionOfSibling(self, b)
                #     col = SingleSection(section, self.unlocated_boxes[unlocated_pos]['area'], is_bottom_expandable=True)
                #     section.add_new_section(col)
                #     section.add_writing_area(self.unlocated_boxes[unlocated_pos])
                #     self.sections.insert(pos, section)
                # ret = True
            elif -0.5 <= status < 0.5 and pos > -1:
                # Relative pos to section: INSIDE
                belong_to, w_comparison, c_pos = self.sections[pos]._compare_similar_width_area_to_sibling(
                    self.list_of_unlocated_boxes[unlocated_pos],
                    only_width)
                if belong_to:
                    if w_comparison > 0:
                        self.unlocated_boxes.set_unlocated_box(self.list_of_unlocated_boxes[unlocated_pos],
                                                               self.sections[pos],
                                                               UnlocatedBoxes.INSIDE_OF_SECTION_POS, w_comparison)
                    else:
                        if c_pos > -1:
                            self.unlocated_boxes.set_unlocated_box(self.list_of_unlocated_boxes[unlocated_pos],
                                                                   self.sections[pos].siblings[c_pos],
                                                                   UnlocatedBoxes.INSIDE_OF_SECTION_POS, w_comparison)
                        else:
                            self.unlocated_boxes.set_unlocated_box(self.list_of_unlocated_boxes[unlocated_pos],
                                                                   self.sections[pos],
                                                                   UnlocatedBoxes.INSIDE_OF_SECTION_POS, w_comparison)
                    ret = True
                else:
                    found = False
                    if not only_width and pos+1 == len(self.sections):
                        pos -= 1
                        only_width = True
            elif 0.5 <= status <= 1 and pos > -1:
                self.unlocated_boxes.set_unlocated_box(self.list_of_unlocated_boxes[unlocated_pos],
                                                       self.sections[pos],
                                                       UnlocatedBoxes.NEXT_OF_SECTION_POS)
                ret = True
            else:
                self.unlocated_boxes.set_unlocated_box(self.list_of_unlocated_boxes[unlocated_pos],
                                                       self.sections[len(self.sections)-1],
                                                       UnlocatedBoxes.NEXT_OF_SECTION_POS)
                ret = True
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

    def process_prev_unlocated_areas(self):
        prev_unlocated_boxes = self.unlocated_boxes.get_unlocated_boxes_by_pos(UnlocatedBoxes.PREVIOUS_OF_SECTION_POS)
        new_sections = []
        for unlocated_box in prev_unlocated_boxes:
            to_locate = []
            to_locate_properties = []
            for unloc_wa in unlocated_box['boxes']:
                to_locate.append(unloc_wa['area'])
                to_locate_properties.append(unloc_wa)
            for i, wa in enumerate(to_locate):
                s = self.sections.index(unlocated_box['referred_section'])
                for nsec in new_sections:
                    if contains([0, 2], nsec.coordinates, wa, self.threshold) and  nsec.top < self.sections[s].top:
                        s =  self.sections.index(nsec)
                min_left = self.sections[s].left
                max_right = self.sections[s].right

                new_section = False
                # wap = get_writing_area_properties(to_locate, i, min_left, max_right, self.threshold)
                if to_locate_properties[i]['is_single']:
                    if ((type(self.sections[s]) is SingleSection or len(self.sections[s].siblings) == 1)
                            and self.sections[s].can_be_added(to_locate_properties[i], only_width=True)):
                        if self.sections[s].top > to_locate[i][1]:
                            self.sections[s].top = to_locate[i][1]
                        self.sections[s].add_writing_area(to_locate_properties[i])
                    elif s>0 and ((type(self.sections[s-1]) is SingleSection or len(self.sections[s-1].siblings) == 1)
                                  and self.sections[s-1].can_be_added(to_locate_properties[i], only_width=True)):
                        if self.sections[s-1].bottom < to_locate[i][3]:
                            self.sections[s-1].bottom = to_locate[i][3]
                        self.sections[s-1].add_writing_area(to_locate_properties[i])
                    else:
                        new_section = True
                # elif wap['is_single']:
                #     if type(self.sections[s]) is BigSectionOfSibling:
                #
                #     else:
                #         new_section = True
                else:
                    if type(self.sections[s]) is BigSectionOfSibling:
                        if self.sections[s].can_be_added(to_locate_properties[i], only_width=True):
                            if self.sections[s].top > to_locate[i][1]:
                                self.sections[s].top = to_locate[i][1]
                            if self.sections[s].can_be_added(to_locate_properties[i]):
                                self.sections[s].add_writing_area(to_locate_properties[i])
                            elif self.sections[s].can_be_inserted_a_column_for(to_locate_properties[i]):
                                # insert
                                section = SingleSection(self.sections[s], to_locate[i])
                                section.add_writing_area(to_locate_properties[i])
                                self.sections[s].add_new_section(section)
                                self.sections[s].sort_content()
                        elif self.sections[s].can_be_inserted_a_column_for(to_locate_properties[i]):
                            # insert
                            section = SingleSection(self.sections[s], to_locate[i])
                            section.add_writing_area(to_locate_properties[i])
                            self.sections[s].add_new_section(section)
                            self.sections[s].sort_content()
                        elif s>0  and self.sections[s-1].can_be_added(to_locate_properties[i], only_width=True):
                            if self.sections[s - 1].bottom < to_locate[i][3]:
                                self.sections[s - 1].bottom = to_locate[i][3]
                            self.sections[s-1].add_writing_area(to_locate_properties[i])
                        elif s> 0 and type(self.sections[s-1]) is BigSectionOfSibling and self.sections[s-1].can_be_inserted_a_column_for(to_locate_properties[i]):
                            # insert
                            section = SingleSection(self.sections[s-1], to_locate[i])
                            section.add_writing_area(to_locate_properties[i])
                            self.sections[s-1].add_new_section(section)
                            self.sections[s-1].sort_content()
                        elif to_locate[i][3] - to_locate[i][1] <= self.threshold+self.threshold+self.threshold:
                            # is little
                            _, _, cp = self.sections[s]._compare_similar_width_area_to_sibling(to_locate_properties[i], only_width=True)
                            if cp > -1:
                                if self.sections[s].top > to_locate[i][1]:
                                    self.sections[s].top = to_locate[i][1]
                                if self.sections[s].siblings[cp].top > to_locate[i][1]:
                                    self.sections[s].siblings[cp].top = to_locate[i][1]
                                self.sections[s].siblings[cp].add_writing_area(to_locate_properties[i])
                                self.sections[s].siblings[cp].sort_content()
                            else:
                                new_section = True
                        else:
                            new_section = True
                    else:
                        # nova secció
                        new_section = True
                if new_section:
                    if to_locate_properties[i]['is_single']:
                        # crear una nova SingleSection i afegir la wa en cas contrari
                        if s == 0:
                            min_top = self.page_boundary[1]
                        else:
                            min_top = min(self.sections[s - 1].bottom, wa[1]) # wa[1] # min(self.sections[s - 1].bottom, wa[1])
                        max_bottom = max(wa[3], self.sections[s].top) # wa[3] # max(wa[3], self.sections[s].top)
                        if max_bottom > self.sections[s].top:
                            self.sections[s].top = max_bottom
                        section = SingleSection(self, [min_left, min_top, max_right, max_bottom])
                        section.add_writing_area(to_locate_properties[i])
                    else:
                        # crear una nova BigSectionOfSibling i una SingleSection com a sibling i afegir-hi la wa en cas contrari
                        if s == 0:
                            min_top = self.page_boundary[1]
                        else:
                            min_top = min(self.sections[s - 1].bottom, wa[1]) # wa[1] # min(self.sections[s - 1].bottom, wa[1])
                        max_bottom = max(wa[3], self.sections[s].top) # wa[3] #max(wa[3], self.sections[s].top)
                        if max_bottom > self.sections[s].top:
                            self.sections[s].top = max_bottom
                        section = BigSectionOfSibling(self,
                                                      [min_left, min_top, max_right, max_bottom])
                        col = SingleSection(section, [to_locate[i][0], min_top, to_locate[i][2], max_bottom])
                        col.add_writing_area(to_locate_properties[i])
                        section.add_new_section(col)
                    new_sections.append(section)
                    self.add_new_section(section)
                    self.sort_content()

    def _insert_a_new_section(self, cp1, cp2, s, box_to_locate_properties):
        # l'àrea ocupa totes les columnes de la secció. Es pot tallar la secció
        section = self.sections[s]
        if cp1 == 0 and cp2 + 1 == len(section.siblings):
            # convertir la secció en dues i inserir entre elles una nova secció
            section.cut(box_to_locate_properties['area'])

            if s + 1 < len(self.sections) and (type(self.sections[s + 1]) is SingleSection
                                               or len(self.sections[s + 1].siblings) == 1):
                new_section = self.sections[s + 1]
                new_section.top = box_to_locate_properties['area'][1]
            else:
                new_section = SingleSection(self, [section.left, box_to_locate_properties['area'][1], section.right,  box_to_locate_properties['area'][3]])
                self.add_new_section(new_section)
            new_section.add_writing_area(box_to_locate_properties)
            new_section.sort_content()
            self.sort_content()
        else:
            to_remove=[]
            for ind_col in range(cp1, cp2+1):
                need_remove = section.siblings[ind_col].cut(box_to_locate_properties['area'])
                if need_remove:
                    to_remove.append(ind_col)
            for ind_col in reversed(to_remove):
                section.siblings.pop(ind_col)

            new_section = SingleSection(self, box_to_locate_properties['area'])
            section.add_new_section(new_section)
            new_section.add_writing_area(box_to_locate_properties)
            new_section.sort_content()
            section.sort_content()

    def process_inside_unlocated_areas(self, image=None, lmodel=None):
        margin = self.threshold + self.threshold + self.threshold
        inside_unlocated_boxes = self.unlocated_boxes.get_unlocated_boxes_by_pos(UnlocatedBoxes.INSIDE_OF_SECTION_POS)
        for unlocated_box in inside_unlocated_boxes:
            to_locate = []
            to_locate_properties = []
            overlapped = []
            for unloc_wa in unlocated_box['boxes']:
                to_locate.append(unloc_wa['area'])
                to_locate_properties.append(unloc_wa)
            for i, wa in enumerate(to_locate):
                if type(unlocated_box['referred_section'].section_container) is BigSectionOfSibling:
                    section = unlocated_box['referred_section'].section_container
                else:
                    section = unlocated_box['referred_section']
                s = self.sections.index(section)
                min_left = section.left
                max_right = section.right

                wap = get_writing_area_properties(to_locate, i, min_left, max_right, self.threshold)
                if to_locate_properties[i]['is_single'] and (type(section) is SingleSection or len(section.siblings) == 1):
                    # no es té en compte wap['is_single'] perquè si to_locate_properties[1]['is_single']=TRUE, wap['is_single] també ha de ser TRUE
                    # Afegir on pertoca
                    if section.top > to_locate[i][1]:
                        section.top = to_locate[i][1]
                    if section.bottom < to_locate[i][3]:
                        section.bottom = to_locate[i][3]
                    section.add_writing_area(to_locate_properties[i])
                elif to_locate_properties[i]['is_single'] and type(section) is BigSectionOfSibling:
                    # cal trencar la secció?
                    if  unlocated_box['compared_width'][i] == 1:
                        # trencar la secció existent i inserir la nova
                        f, cp1, cp2 = section.search_siblings_to_cut_for_bigger_section(to_locate_properties[i])
                        if f:
                            # l'àrea ocupa totes les columnes de la secció. Es pot tallar la secció
                            self._insert_a_new_section(cp1, cp2, s, to_locate_properties[i])
                            # if cp1 == 0 and cp2 + 1 == len(section.siblings):
                            #     # convertir la secció en dues i inserir entre elles una nova secció
                            #     section.cut(to_locate[i])
                            #
                            #     if s + 1 < len(self.sections) and (type(self.sections[s + 1]) is SingleSection
                            #                                        or len(self.sections[s + 1].siblings) == 1):
                            #         new_section = self.sections[s + 1]
                            #         new_section.top = to_locate[i][1]
                            #     else:
                            #         new_section = SingleSection(self, to_locate[i])
                            #         self.add_new_section(new_section)
                            #     new_section.add_writing_area(to_locate_properties[i])
                            #     new_section.sort_content()
                            #     self.sort_content()
                            # else:
                            #     # TODO: ???
                            #     self.list_of_unlocated_boxes.append(to_locate_properties[i])
                        else:
                            # TODO: ???
                            self.list_of_unlocated_boxes.append(to_locate_properties[i])
                    elif unlocated_box['compared_width'][i] == 0 and is_similar_distance(
                            unlocated_box['referred_section'].left,
                            unlocated_box['referred_section'].right,
                            to_locate[i][0],
                            to_locate[i][2],
                            self.threshold*2):
                        # Deduim que no és single i inserim
                        unlocated_box['referred_section'].add_writing_area(to_locate_properties[i])
                    elif (unlocated_box['compared_width'][i] == -1
                          and (i+1<=len(to_locate) or to_locate[i+1][3] - to_locate[i][1] > self.threshold*3)
                          and to_locate[i][3] - to_locate[i][1] <= self.threshold*3):
                        # Deduim que no és single i inserim
                        unlocated_box['referred_section'].add_writing_area(to_locate_properties[i])
                    else:
                        # TODO: ???
                        self.list_of_unlocated_boxes.append(to_locate_properties[i])
                elif not to_locate_properties[i]['is_single'] and unlocated_box["compared_width"][i] == 1:
                    #Assegurar que no és una mala predicció i es tracta de columnes
                    f, cp1, cp2 = section.search_siblings_to_cut_for_bigger_section(to_locate_properties[i])
                    if f:
                        if image is not None and lmodel is not None:
                            fragment_image = image[to_locate[i][1] + 5:to_locate[i][3] + 5,
                                             to_locate[i][0]+5:to_locate[i][2]+5]
                            fragment_predict = get_annotated_prediction(fragment_image, lmodel)
                            # si la columnes tenen la mateixa mida que le columnes predites, es un error i no cal crear secció
                            create_section = False
                            ind_fr=0
                            for pred in fragment_predict:
                                if pred['class_name'] == 'columna':
                                    if cp2 - cp1 < ind_fr:
                                        create_section = True
                                        break
                                    if not is_similar_distance(pred['box'][0], pred['box'][2],
                                                               section.siblings[ind_fr+cp1].left,
                                                               section.siblings[ind_fr+cp1].right, self.threshold*3):
                                        create_section=True
                                        break
                                    ind_fr += 1
                            if create_section:
                                create_section = False
                                ind_fr=0
                                for pred in fragment_predict:
                                    if pred['class_name'] == 'bloque':
                                        if cp2 - cp1 < ind_fr:
                                            create_section = True
                                            break
                                        if not is_similar_distance(pred['box'][0], pred['box'][2],
                                                                   section.siblings[ind_fr+cp1].left,
                                                                   section.siblings[ind_fr+cp1].right, self.threshold):
                                            create_section = True
                                            break
                                        ind_fr += 1
                        else:
                            # l'àrea ocupa totes les columnes de la secció. Es pot tallar la secció
                            create_section = True
                        if create_section:
                            self._insert_a_new_section(cp1, cp2, s, to_locate_properties[i])
                        else:
                            # afegir area a cada columna
                            for ind_col in range(cp1, cp2+1):
                                section.siblings[ind_col].writing_areas.append([section.siblings[ind_col].left, to_locate[i][1], section.siblings[ind_col].right, to_locate[i][3]])
                                section.siblings[ind_col].sort_content()
                    else:
                        # TODO: ???
                        self.list_of_unlocated_boxes.append(to_locate_properties[i])
                elif not to_locate_properties[i]['is_single'] and unlocated_box["compared_width"][i] == -1:
                    if wap['is_single']:
                        # Deduïm que només pot pertànyer a aquesta columna
                        unlocated_box['referred_section'].add_writing_area(to_locate_properties[i])
                    elif image is not None and lmodel is not None:
                        p_over = -1
                        for i in range(len(overlapped)):
                            for over_box in overlapped[i]['overlapped_boxes']:
                                dif_center = abs((wa[0] + wa[2]) / 2 - (over_box[0] + over_box[2]) / 2)
                                dif_start = abs(wa[0] - over_box[0])
                                min_max_h = max(wa[3], over_box[3]) - min(wa[1], over_box[1])
                                sum_h = (wa[3] - wa[1]) + (over_box[3] - over_box[1])
                                if (overlap_vertically(over_box, wa, self.threshold)
                                        or (dif_start <= self.threshold or dif_center <= self.threshold)
                                        and abs(min_max_h - sum_h) <= self.threshold):
                                    p_over = i
                                    break
                        if p_over == -1:
                            overlapped.append({'column':  unlocated_box['referred_section'], 'min_top':wa[1], 'max_bottom':wa[3] , 'overlapped_boxes': [wa]})
                        else:
                            overlapped[p_over]['overlapped_boxes'].append(wa)
                            if overlapped[p_over]['min_top'] > wa[1]:
                                overlapped[p_over]['min_top'] = wa[1]
                            if overlapped[p_over]['max_bottom'] < wa[3]:
                                overlapped[p_over]['max_bottom'] = wa[3]
                    else:
                        self.list_of_unlocated_boxes.append(to_locate_properties[i])
                else:
                    self.list_of_unlocated_boxes.append(to_locate_properties[i])
            if image is not None and lmodel is not None:
                for overlapped_in_column in overlapped:
                    fragment_image = image[overlapped_in_column['min_top'] + 5:overlapped_in_column['max_bottom'] + 5,
                                     overlapped_in_column['column'].left:overlapped_in_column['column'].right]
                    fragment_predict = get_annotated_prediction(fragment_image, lmodel)
                    new_columns = []
                    for pred in fragment_predict:
                        if pred['class_name'] == 'columna':
                            new_columns.append(pred['box'])
                    if len(new_columns) > 1:
                        # cortar e insertar varias singlesection (una por columna nueva) en la seccion existente
                        overlapped_in_column['column'].cut([0, overlapped_in_column['min_top'], 0, overlapped_in_column['max_bottom']])
                        for new_col_box in new_columns:
                            x1 = overlapped_in_column['column'].left + new_col_box[0]
                            y1 = overlapped_in_column['min_top']
                            x2 = x1 + new_col_box[2] - new_col_box[0]
                            y2 = overlapped_in_column['max_bottom']
                            new_column = SingleSection(overlapped_in_column['column'].section_container, [x1, y1, x2, y2])
                            overlapped_in_column['column'].section_container.add_new_section(new_column)
                        overlapped_in_column['column'].section_container.sort_content()
                    else:
                        overlapped_in_column['column'].writing_areas.append([overlapped_in_column['column'].left,
                                                                             overlapped_in_column['min_top'],
                                                                             overlapped_in_column['column'].right,
                                                                             overlapped_in_column['max_bottom']])
                        overlapped_in_column['column'].sort_content()

    def process_post_unlocated_areas(self):
        margin = self.threshold + self.threshold + self.threshold
        next_unlocated_boxes = self.unlocated_boxes.get_unlocated_boxes_by_pos(UnlocatedBoxes.NEXT_OF_SECTION_POS)
        new_sections = []
        for unlocated_box in next_unlocated_boxes:
            to_locate = []
            to_locate_properties = []
            for unloc_wa in unlocated_box['boxes']:
                to_locate.append(unloc_wa['area'])
                to_locate_properties.append(unloc_wa)
            for i in range(len(to_locate)):
                wa = to_locate[i]
                if type(unlocated_box['referred_section'].section_container) is BigSectionOfSibling:
                    section = unlocated_box['referred_section'].section_container
                else:
                    section = unlocated_box['referred_section']
                s = self.sections.index(section)
                min_left = section.left
                max_right = section.right

                new_section = False
                # wap = get_writing_area_properties(to_locate, i, min_left, max_right, self.threshold)
                if to_locate_properties[i]['is_single']:
                    added = False
                    solved = False
                    if (s < len(self.sections) - 1 and type(self.sections[s+1]) is SingleSection
                            and self.sections[s+1].can_be_added(to_locate_properties[i], only_width=True)):
                        if self.sections[s+1].top > to_locate[i][1]:
                            self.sections[s+1].top = to_locate[i][1]
                        self.sections[s+1].add_writing_area(to_locate_properties[i])
                        solved = True
                    elif (len(new_sections) > 0 and type(new_sections[-1]) is SingleSection
                            and new_sections[-1].can_be_added(to_locate_properties[i], only_width=True)):
                        if new_sections[-1].top > to_locate[i][1]:
                            new_sections[-1].top = to_locate[i][1]
                        new_sections[-1].add_writing_area(to_locate_properties[i])
                        added = True
                    if (i == len(to_locate)-1 and (type(self.sections[s]) is SingleSection or len(self.sections[s].siblings) == 1)
                            and self.sections[s].can_be_added(to_locate_properties[i], only_width=True)):
                        if added:
                            #fusionar
                            self.sections[s].top = min(self.sections[s].top, new_sections[-1][1])
                            self.sections[s].bottom = max(self.sections[s].bottom, new_sections[-1][3])
                            new_sections.pop(len(new_sections)-1)
                        elif not solved:
                            #afegir
                            if self.sections[s].bottom < to_locate[i][3]:
                                self.sections[s].bottom = to_locate[i][3]
                            self.sections[s].add_writing_area(to_locate_properties[i])
                    elif not added and not solved:
                        new_section = True
                else:
                    # added = False
                    if s < len(self.sections) - 1 and type(self.sections[s+1]) is BigSectionOfSibling:
                        if self.sections[s+1].can_be_added(to_locate_properties[i], only_width=True):
                            if self.sections[s+1].top > to_locate[i][1]:
                                self.sections[s+1].top = to_locate[i][1]
                                for col in self.sections[s+1].siblings:
                                    col.top = self.sections[s+1].top
                            self.sections[s+1].add_writing_area(to_locate_properties[i])
                        elif self.sections[s+1].can_be_inserted_a_column_for(to_locate_properties[i]):
                            # insert
                            section = SingleSection(self.sections[s+1], to_locate[i])
                            section.add_writing_area(to_locate_properties[i])
                            self.sections[s+1].add_new_section(section)
                            self.sections[s+1].sort_content()
                        elif to_locate[i][3] - to_locate[i][1] <= self.threshold + self.threshold + self.threshold:
                            # is little
                            _, _, cp = self.sections[s+1]._compare_similar_width_area_to_sibling(to_locate_properties[i],
                                                                                              only_width=True)
                            if cp > -1:
                                if self.sections[s+1].bottom < to_locate[i][3]:
                                    self.sections[s+1].bottom = to_locate[i][3]
                                if self.sections[s+1].siblings[cp].bottom < to_locate[i][3]:
                                    self.sections[s+1].siblings[cp].bottom = to_locate[i][3]
                                self.sections[s+1].siblings[cp].add_writing_area(to_locate_properties[i])
                                self.sections[s+1].siblings[cp].sort_content()
                            else:
                                new_section = True
                        else:
                            new_section = True
                    elif len(new_sections) > 0 and type(new_sections[-1]) is BigSectionOfSibling:
                        if new_sections[-1].can_be_added(to_locate_properties[i], only_width=True):
                            if new_sections[-1].top < to_locate[i][1]:
                                new_sections[-1].top = to_locate[i][1]
                            new_sections[-1].add_writing_area(to_locate_properties[i])
                        elif new_sections[-1].can_be_inserted_a_column_for(to_locate_properties[i]):
                            # insert
                            section = SingleSection(new_sections[-1], to_locate[i])
                            section.add_writing_area(to_locate_properties[i])
                            new_sections[-1].add_new_section(section)
                            new_sections[-1].sort_content()
                        elif to_locate[i][3] - to_locate[i][1] <= self.threshold + self.threshold + self.threshold:
                            # is little
                            _, _, cp = new_sections[-1]._compare_similar_width_area_to_sibling(to_locate_properties[i],
                                                                                               only_width=True)
                            if cp > -1:
                                if new_sections[-1].bottom < to_locate[i][3]:
                                    new_sections[-1].bottom = to_locate[i][3]
                                if new_sections[-1].siblings[cp].bottom < to_locate[i][3]:
                                    new_sections[-1].siblings[cp].bottom = to_locate[i][3]
                                new_sections[-1].siblings[cp].add_writing_area(to_locate_properties[i])
                                new_sections[-1].siblings[cp].sort_content()
                            else:
                                new_section = True
                        else:
                            new_section = True
                    #     added = True
                    # if i == len(to_locate)-1 and  type(self.sections[s]) is BigSectionOfSibling:
                    #    if self.sections[s].can_be_added(to_locate_properties[i], only_width=True):
                    #         if self.sections[s].bottom > to_locate[i][3]:
                    #             self.sections[s].bottom = to_locate[i][3]
                    #         self.sections[s].add_writing_area(to_locate_properties[i])
                    #     elif self.sections[s].can_be_inserted_a_column_for(to_locate_properties[i]):
                    #         # insert
                    #         section = SingleSection(self.sections[s], to_locate[i])
                    #         section.add_writing_area(to_locate_properties[i])
                    #         self.sections[s].add_new_section(section)
                    #         self.sections[s].sort_content()
                    #     elif to_locate[i][3] - to_locate[i][1] <= self.threshold + self.threshold + self.threshold:
                    #         # is little
                    #         _, _, cp = self.sections[s]._compare_similar_width_area_to_sibling(to_locate_properties[i],
                    #                                                                            only_width=True)
                    #         if cp > -1:
                    #             if self.sections[s].bottom < to_locate[i][3]:
                    #                 self.sections[s].bottom = to_locate[i][3]
                    #             if self.sections[s].siblings[cp].bottom < to_locate[i][3]:
                    #                 self.sections[s].siblings[cp].bottom = to_locate[i][3]
                    #             self.sections[s].siblings[cp].add_writing_area(to_locate_properties[i])
                    #             self.sections[s].siblings[cp].sort_content()
                    #         else:
                    #             new_section = True
                    #     else:
                    #         new_section = True
                    else:
                        # nova secció
                        new_section = True
                if new_section:
                    if s == len(self.sections)-1:
                        max_bottom = self.bottom
                    else:
                        max_bottom = self.sections[s+1].top
                    if to_locate_properties[i]['is_single']:
                        # crear una nova SingleSection i afegir la wa en cas contrari
                        min_top = wa[1]
                        section = SingleSection(self, [min_left, min_top, max_right, max_bottom])
                        section.add_writing_area(to_locate_properties[i])
                    else:
                        # crear una nova BigSectionOfSibling i una SingleSection com a sibling i afegir-hi la wa en cas contrari
                        min_top = wa[1]
                        section = BigSectionOfSibling(self,
                                                      [min_left, min_top, max_right, max_bottom])
                        ncols =  (max_right-min_left) // ( to_locate_properties[i]['guess_right']-to_locate_properties[i]['guess_left'] )
                        ncols_float =  (max_right-min_left) / ( to_locate_properties[i]['guess_right']-to_locate_properties[i]['guess_left'] )
                        if ncols_float - ncols > 0.4:
                            ncols += 1
                        width = (max_right-min_left) / ncols
                        left = to_locate_properties[i]['guess_left']
                        right = max(left + width, to_locate[i][3])
                        col_boundary = [left, min_top, int(right), max_bottom]
                        col = SingleSection(section, col_boundary)
                        col.add_writing_area(to_locate_properties[i])
                        section.add_new_section(col)
                    new_sections.append(section)
                    self.add_new_section(section)
                    self.sort_content(True)

    @staticmethod
    def build_lauoud_from_sections(page_boundary, sections, writing_area_list, w: int, h: int, threshold=None,
                                   image=None, lmodel=None):
        main_layout = MainLayout(page_boundary, w, h, proposed_sections=sections)
        if threshold is not None:
            main_layout.threshold = threshold

        # The page boundary comes from the YOLO prediction. The prediction is not always correct. Here we try to correct excessive page margins.
        if page_boundary[0] > w * 0.05:
            page_boundary[0] = threshold
        if page_boundary[1] > h * 0.05:
            page_boundary[1] = threshold
        if page_boundary[2] < w * 0.95:
            page_boundary[2] = w - threshold
        if page_boundary[3] < h * 0.95:
            page_boundary[3] = h - threshold

        # calculate min left  and max right areas and removing areas outside the page boundary
        min_left = 100000
        max_right = 0
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

        # Trying to locate all areas in predicted sections. Unlocated areas are saved in UnlocatedBoxes attribute
        writing_area_list.sort(key=lambda x: x[1] * 10000 + x[0])
        for i in range(len(writing_area_list)):
            # Calculating properties for every area (if ares is single (it hasn't siblings), real dimension and guess
            # dimension based into siblings, and so on)
            wap = get_writing_area_properties(writing_area_list, i, min_left, max_right, main_layout.threshold)
            main_layout.add_writing_area(wap)

        # Trying to resolve unallocated boxes
        for i in range(len(main_layout.list_of_unlocated_boxes) - 1, -1, -1):
            # buscar la secció prèvia a l'àrea
            if main_layout.clasify_unlocated_area(i):
                main_layout.list_of_unlocated_boxes.pop(i)

        # recol·locar inconherències en columnes
        main_layout.process_prev_unlocated_areas()
        main_layout.process_inside_unlocated_areas(image, lmodel)
        main_layout.process_post_unlocated_areas()
        main_layout.sort_content(True)
        # # ajustar medidas exteriores
        main_layout._resize_left_sections()
        main_layout._resize_top_sections()
        main_layout._resize_right_sections()
        main_layout._resize_bottom_sections()
        #llenar gaps i ajustar medidas
        main_layout._fill_vertical_gaps_and_resize()
        main_layout._fill_horizontal_gaps_and_resize()


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

    def get_left_right_boundary_for(self, b):
        p = False
        r = l = -1
        if overlap_vertically(self.coordinates, b, self.threshold):
            cpl = -1
            cpr = -1
            for i in range(len(self.siblings)):
                if cpr == -1 and overlap_vertically(self.siblings[i].coordinates, b, self.threshold):
                    cpl = i
                if cpr > -1 and not overlap_vertically(self.siblings[i].coordinates, b, self.threshold):
                    cpr = i
                else:
                    cpr = i

            if cpl > -1:
                l = min(b[0], self.siblings[cpl].right)
            else:
                l = b[0]
            if cpr > -1:
                r = max(b[2], self.siblings[cpr].left)
            else:
                r = b[3]
        return p, l, r

    def get_horizontal_positioning_vs(self, box):
        rel=0
        val=0
        vok = False
        ov = overlap_vertically(box, self.coordinates, self.threshold)
        if ov:
            val = 100000
            for i, col in enumerate(self.siblings):
                ovcl, r_cl, v_cl = col.get_horizontal_positioning_vs(box)
                if ovcl:
                    vok = ovcl
                    if val > v_cl:
                        val = v_cl
                        rel =r_cl
        return vok, rel, val

    def _fill_horizontal_gaps_and_resize(self):
        for id_col, col in enumerate(self.siblings):
            inc_left = 10000
            inc_right = 10000
            rel_left= 0
            rel_right = 0
            for id_col2, col2 in enumerate(self.siblings):
                if id_col2 == id_col:
                    continue
                o, r, v = col2.get_horizontal_positioning_vs(col.coordinates)
                if o:
                    if r > 0:
                        if inc_left > v:
                            inc_left = max(v, 0)
                            rel_left = r
                    elif r < 0:
                        if inc_right > v:
                            inc_right = max(v, 0)
                            rel_right = r
            if rel_left > 0 and inc_left <= self.threshold:
                col.left -= inc_left
            if rel_right < 0 and inc_right <= self.threshold:
                col.right += inc_right // 2
            elif rel_right < 0:
                # TODO nova columna
                valid_gap = False
                # TODO Cal veure si hi ha altres seccions que intersecten i valorar si el gap és correcte o cal ampliar la columna
                if not valid_gap:
                    col.right += inc_right//2

    def get_vertical_positioning_vs(self, box):
        rel=0
        val=0
        vok = False
        ov = overlap_horizontally(box, self.coordinates, self.threshold)
        if ov:
            val = 100000
            for i, col in enumerate(self.siblings):
                ovcl, r_cl, v_cl = col.get_vertical_positioning_vs(box)
                if ovcl:
                    vok = ovcl
                    if val > v_cl:
                        val = v_cl
                        rel =r_cl
        return vok, rel, val

    def _fill_vertical_gaps_and_resize(self):
        for id_col, col in enumerate(self.siblings):
            inc_top = 10000
            inc_bottom = 10000
            rel_top = 0
            rel_bottom = 0
            for id_col2, col2 in enumerate(self.siblings):
                if id_col2 == id_col:
                    continue
                o, r, v = col2.get_vertical_positioning_vs(col.coordinates)
                if o:
                    if r > 0:
                        if inc_top > v:
                            inc_top = v
                            rel_top = r
                    elif r < 0:
                        if inc_bottom > v:
                            inc_bottom = v
                            rel_bottom = r
            if rel_top > 0 and inc_top <= self.threshold:
                col.top -= inc_top
            if rel_bottom < 0 and inc_bottom <= self.threshold:
                col.bottom += inc_bottom // 2
            elif rel_bottom < 0:
                valid_gap = True
                # TODO Cal veure si hi ha altres seccions que intersecten i valorar si el gap és correcte o cal ampliar la columna
                if not valid_gap:
                    col.bottom += inc_bottom//2


    def _resize_left_sections(self):
        for col in self.left_sections:
            ov = False
            inc = 10000
            for j, ls in enumerate(self.section_container.sections):
                if self == ls:
                    continue
                if self != ls:
                    if type(ls) is BigSectionOfSibling:
                        if contains([0,1,2,3], self.coordinates, ls.coordinates, self.threshold):
                            if (overlap_horizontally([min(self.left, col.bottom), col.top, col.left, col.bottom],
                                                    ls.coordinates, min(self.threshold*2.5, 0.15*(ls.right-ls.left)))
                                    and overlap_vertically(col.coordinates, ls.coordinates, self.threshold)):
                                ov = True
                                if inc > col.left - ls.right:
                                    inc = col.left - ls.right
                        else:
                            for cls in ls.left_sections:
                                if (overlap_horizontally([self.left, col.top, col.left, col.bottom], cls.coordinates,
                                                         min(self.threshold*2.5, 0.15*(cls.right-cls.left)))
                                        and overlap_vertically(col.coordinates, cls.coordinates, self.threshold)):
                                    ov = True
                                    if inc > col.left - cls.right:
                                        inc = col.left - cls.right
                    else:
                        if (overlap_horizontally([self.left, col.top, col.left, col.bottom], ls.coordinates,
                                                min(self.threshold * 2.5, 0.15 * (ls.right - ls.left)))
                                and overlap_vertically(col.coordinates, ls.coordinates, self.threshold)):
                            ov = True
                            if inc > col.left - ls.right:
                                inc = col.left - ls.right
            if not ov:
                col.left = self.left
            else:
                col.left -= inc

    def _resize_right_sections(self):
        for col in self.right_sections:
            ov = False
            inc = 10000
            for j, ls in enumerate(self.section_container.sections):
                if self != ls:
                    if type(ls) is BigSectionOfSibling:
                        if contains([0,1,2,3], self.coordinates, ls.coordinates, self.threshold):
                            if (overlap_horizontally([col.right, col.top, self.right, col.bottom], ls.coordinates,
                                                     min(self.threshold*2.5, 0.15*(ls.right-ls.left)))
                                    and overlap_vertically(col.coordinates, ls.coordinates, self.threshold)):
                                ov = True
                                if inc > ls.left - col.right:
                                    inc = ls.left - col.right
                        else:
                            for cls in ls.right_sections:
                                if (overlap_horizontally([col.right, col.top, self.right, col.bottom], cls.coordinates,
                                                         min(self.threshold*2.5, 0.15*(cls.right-cls.left)))
                                        and overlap_vertically(col.coordinates, cls.coordinates, self.threshold)):
                                    ov = True
                                    if inc > cls.left - col.right:
                                        inc = cls.left - col.right
                    else:
                        if (overlap_horizontally([col.right, col.top, self.right, col.bottom], ls.coordinates,
                                                 min(self.threshold*2.5, 0.15*(ls.right-ls.left)))
                                and overlap_vertically(col.coordinates, ls.coordinates, self.threshold)):
                            ov = True
                            if inc > ls.left - col.right:
                                inc = ls.left - col.right
            if not ov:
                col.right = self.right
            else:
                col.right += inc

    def _resize_top_sections(self):
        for col in self.top_sections:
            ov = False
            inc = 10000
            for j, ls in enumerate(self.section_container.sections):
                if self != ls:
                    if type(ls) is BigSectionOfSibling:
                        if contains([0,1,2,3], self.coordinates, ls.coordinates, self.threshold):
                            if (overlap_vertically([col.left, min(col.top, self.top), col.right, col.top],
                                                   ls.coordinates, self.threshold)
                                    and overlap_horizontally(col.coordinates, ls.coordinates,
                                                             min(self.threshold*2.5, 0.15*(ls.right-ls.left)))):
                                ov = True
                                if inc > col.top - ls.bottom:
                                    inc = col.top - ls.bottom

                        else:
                            for cls in ls.top_sections:
                                if (overlap_vertically([col.left, min(col.top, self.top), col.right, col.top],
                                                       cls.coordinates, self.threshold)
                                        and overlap_horizontally(col.coordinates, cls.coordinates,
                                                                 min(self.threshold*2.5, 0.15*(cls.right-cls.left)))):
                                    ov = True
                                    if inc > col.top - cls.bottom:
                                        inc = col.top - cls.bottom
                    else:
                        if (overlap_vertically([col.left, min(col.top,self.top), col.right, col.top], ls.coordinates,
                                               self.threshold)
                                and overlap_horizontally(col.coordinates, ls.coordinates,
                                                         min(self.threshold*2.5, 0.15*(ls.right-ls.left)))):
                            ov = True
                            if inc > col.top - ls.bottom:
                                inc = col.top - ls.bottom
            if not ov:
                col.top = self.top
            else:
                col.top -= inc

    def _resize_bottom_sections(self):
        for col in self.bottom_sections:
            ov = False
            inc = 10000
            for j, ls in enumerate(self.section_container.sections):
                if self != ls:
                    if type(ls) is BigSectionOfSibling:
                        if contains([0,1,2,3], self.coordinates, ls.coordinates, self.threshold):
                            if (overlap_vertically([col.left, col.bottom, col.right, max(self.bottom, col.bottom)],
                                                   ls.coordinates, self.threshold)
                                    and overlap_horizontally(col.coordinates, ls.coordinates,
                                                             min((ls.right - ls.left)*0.15, self.threshold*2.5))):
                                ov = True
                                if inc > ls.top - col.bottom:
                                    inc = ls.top - col.bottom
                        else:
                            for cls in ls.bottom_sections:
                                if (overlap_vertically([col.left, col.bottom, col.right, max(self.bottom, col.bottom)],
                                                       cls.coordinates, self.threshold)
                                        and overlap_horizontally(col.coordinates,cls.coordinates,
                                                                 min((cls.right - cls.left)*0.15, self.threshold*2.5))):
                                    ov = True
                                    if inc > cls.top - col.bottom:
                                        inc = cls.top - col.bottom
                    else:
                        if (overlap_vertically([col.left, col.bottom, col.right, max(self.bottom, col.bottom)],
                                               ls.coordinates, self.threshold)
                                and overlap_horizontally(col.coordinates, ls.coordinates,
                                                         min((ls.right - ls.left)*0.15, self.threshold*2.5))):
                            ov = True
                            if inc > ls.top - col.bottom:
                                inc = ls.top - col.bottom
            if not ov:
                col.bottom = self.bottom
            else:
                col.bottom += inc

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
                self.siblings.remove(col)
                col.section_container = partial_section_2
                partial_section_2.add_new_section(col)
            else:
                col.bottom = boundaries[1]
                partial_column_2.sort_content()
                partial_section_2.add_new_section(partial_column_2)
        if len(self.siblings) == 0:
            self._sections = partial_section_2._sections
            self.top = partial_section_2.top
            self.bottom = partial_section_2.bottom
        else:
            self.bottom = boundaries[1]
            if keep:
                self.section_container.add_new_section(partial_section_2)
                self.section_container.sort_content()

    def sort_content(self, all=False):
        self.siblings.sort(key=lambda x: x.top * 10000 + x.left)
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

    def _compare_similar_width_area_to_sibling(self, writing_area_properties, only_width=False):
        margin = self.threshold + self.threshold
        x1, y1, x2, y2 = writing_area_properties['area']
        area_width = x2 - x1
        dif = margin + 1
        if only_width:
            belongs = True
        else:
            belongs = False
        width_comparison = 1
        pos = -1
        for i, col in enumerate(self.siblings):
            if not only_width:
                belongs = contains([1,3], col.coordinates, writing_area_properties['area'], self.threshold)
            if not belongs or not contains([0,2], col.coordinates, writing_area_properties['area'], self.threshold):
                continue
            d = abs(area_width - (col.right - col.left))
            if d > margin:
                d = abs(writing_area_properties['guess_width'] - (col.left - col.right))
            if d < dif:
                dif = d
            if dif <= margin:
                width_comparison = 0
            else:
                width_comparison = -1
            pos = i
            break
        return belongs, width_comparison, pos

    def _has_area_similar_width(self, writing_area_properties, only_width=False):
        margin = self.threshold + self.threshold
        x1, y1, x2, y2 = writing_area_properties['area']
        area_width = x2 - x1
        dif = margin + 1
        if only_width:
            edges = [0, 2]
        else:
            edges = [0, 1, 2, 3]
        for col in self.siblings:
            if not contains(edges, col.coordinates, writing_area_properties['area'], self.threshold):
                continue
            d = abs(area_width - (col.right - col.left))
            if d > margin:
                d = abs(writing_area_properties['guess_width'] - (col.left - col.right))
            if d < dif:
                dif = d
        return dif <= margin

    def is_area_compatible(self, writing_area_properties, only_width=False):
        ret = self._has_area_similar_width(writing_area_properties, only_width)
        if not ret:
            ret = self._has_area_similar_center(writing_area_properties, only_width)
        return ret

    def search_sibling(self, writing_area_properties, only_width=False):
        pos = -1
        for i, sibling in enumerate(self.siblings):
            if sibling.can_be_added(writing_area_properties, only_width=only_width):
                pos = i
                break
        return pos

    def search_siblings_to_cut_for_bigger_section(self, writing_area_properties):
        # margin = self.threshold + self.threshold + self.threshold
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
                dif = col.left <= box[0] + self.threshold and  box[0] - self.threshold <= col.right
                if dif and col.fits_vertically(box):
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
                    found = True
                    break
            if found:
                exit_loop = True
            else:
                cp2 += 1
                exit_loop = cp2 >= len(self.siblings)
        return found, cp1, cp2

    def can_be_added(self, writing_area_properties, only_width=False):
        result = not writing_area_properties['is_single']
        if only_width:
            edges = [0,2]
        else:
            edges = [0, 1, 2]
            if not self.is_bottom_expandable:
                edges.append(3)
        result = result and contains(edges, self.coordinates, writing_area_properties['area'], self.threshold)
        result = result and self.is_area_compatible(writing_area_properties, only_width)
        pos = self.search_sibling(writing_area_properties, only_width=only_width)
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
            if self.can_be_inserted_a_column_for(writing_area_properties):
                # insert
                column = SingleSection(self, writing_area_properties['area'])
                column.add_writing_area(writing_area_properties)
                self.siblings.append(column)
                self.siblings.sort(key=lambda x: x[0])
                if column.down_fill_level > self.down_fill_level:
                    self.down_fill_level = column.down_fill_level
                if column.upper_fill_level < self.upper_fill_level:
                    self.upper_fill_level = column.upper_fill_level
            else:
                # TODO ?????
                raise "Error trying add writing area to wrong column"
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
        self.bad_size_counter = [0,0,0,0]
        self.counter_limit_to_resize=4

    @property
    def needResize(self, border_num=-1):
        ret = False
        if border_num==-1 or border_num==0:
            ret = ret or self.bad_size_counter[0]
        if border_num==-1 or border_num==1:
            ret = ret or self.bad_size_counter[1]
        if border_num==-1 or border_num==2:
            ret = ret or self.bad_size_counter[2]
        if border_num==-1 or border_num==3:
            ret = ret or self.bad_size_counter[3]
        return ret



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

    def can_be_added(self, writing_area_properties, verify_space=False, only_width=False):
        result = writing_area_properties['is_single'] == (not self._is_column)
        if only_width:
            edges = [0,2]
        else:
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
        if not v and contains([1, 3], [0, self.top, 0, self.bottom], writing_area, self.threshold):
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
            # self.left = x1
            self.bad_size_counter[0] += 1
        if x2 > self.right:
            # self.right = x2
            self.bad_size_counter[2] += 1
        self._writing_areas.append(writing_area_properties['area'])

    def get_single_sections_as_boxes(self, boxes=[], add_threshold=False):
        x1, y1, x2, y2 = self.coordinates
        if type(add_threshold) is bool:
            x1 = max(int(self.page_boundary[0]), x1-self.threshold)
            y1 = max(int(self.page_boundary[1]), y1-self.threshold)
            x2 = min(int(self.page_boundary[2]), x2 + self.threshold)
            y2 = min(int(self.page_boundary[3]), y2 + self.threshold)
        elif type(add_threshold) is int:
            x1 = max(int(self.page_boundary[0]), x1-add_threshold)
            y1 = max(int(self.page_boundary[1]), y1-add_threshold)
            x2 = min(int(self.page_boundary[2]), x2 + add_threshold)
            y2 = min(int(self.page_boundary[3]), y2 + add_threshold)
        boxes.append([x1, y1, x2, y2])

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

    def get_horizontal_positioning_vs(self, box):
        r = 0
        v = -1
        vok = False
        ov = overlap_vertically(box, self.coordinates, self.threshold)
        if ov:
            vok, r, v = get_boxes_non_overlapping_positioning(HORIZONTAL_POSITIONING, box, self.coordinates,
                                                              self.threshold*3)
        return vok, r, v

    def get_vertical_positioning_vs(self, box):
        r = 0
        v = -1
        vok = False
        ov = overlap_horizontally(box, self.coordinates, self.threshold)
        if ov:
            vok, r, v = get_boxes_non_overlapping_positioning(VERTICAL_POSITIONING, box, self.coordinates, self.threshold)
        return vok, r, v


    def get_left_right_boundary_for(self, b):
        p = False
        l = r = -1
        if overlap_vertically(self.coordinates, b, self.threshold):
            l = min(b[0], self.right)
            r = max(b[2], self.left)
            p = True
        return p, l, r

    # def get_right_boundary_for(self, p1, p2=None):
    #     ret = -1
    #     y1, y2, x1, x2 = calculate_coordinates(p1, p2, 0, 0)
    #     if overlap_vertically(self.coordinates, [x1, y1, x2, y2], self.threshold):
    #         ret = self.right
    #     return ret > -1, ret

    def cut(self, boundaries):
        need_remove = False
        partial_section_2 = SingleSection(self.section_container, [self.left, boundaries[3], self.right, self.bottom])
        keep = False
        for i in range(len(self.writing_areas) - 1, -1, -1):
            if self.writing_areas[i][1] >= boundaries[3]:
                partial_section_2.writing_areas.append(self.writing_areas.pop(i))
                keep = True
        self.bottom = boundaries[1]
        if self.bottom -self.top < self.threshold*3 and len(self.writing_areas) == 0:
            if partial_section_2.bottom-partial_section_2.top >= self.threshold*4 or keep:
                self._writing_areas = partial_section_2._writing_areas
                self.top = partial_section_2.top
                self.bottom = partial_section_2.bottom
            else:
                need_remove = True
        else:
            if partial_section_2.bottom-partial_section_2.top >= self.threshold*4 or keep:
                self.section_container.add_new_section(partial_section_2)
                self.section_container.sort_content()
        return need_remove

    def _compare_similar_width_area_to_sibling(self, writing_area_properties, only_width=False):
        margin = self.threshold + self.threshold
        x1, y1, x2, y2 = writing_area_properties['area']
        area_width = x2 - x1
        dif = margin + 1
        if only_width:
            belongs = True
        else:
            belongs = False
        width_comparison = 1
        pos = -1
        if not only_width:
            belongs = contains([1,3], self.coordinates, writing_area_properties['area'], self.threshold)
        if belongs and contains([0,2], self.coordinates, writing_area_properties['area'], self.threshold):
            d = abs(area_width - (self.right - self.left))
            if d > margin:
                d = abs(writing_area_properties['guess_width'] - (self.left - self.right))
            if d < dif:
                dif = d
            if dif <= margin:
                width_comparison = 0
            else:
                width_comparison = -1
        return belongs, width_comparison, pos


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
