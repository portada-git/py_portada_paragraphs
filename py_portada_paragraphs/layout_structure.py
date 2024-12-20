from builtins import staticmethod, enumerate
from typing import Union

from torch._lazy.extract_compiled_graph import force_lazy_device


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
    if type(p1) is int and type(p2) is int:
        x1 = p1
        y1 = p2

    if type(p2) is list or type(p2) is tuple:
        x2, y2 = p2
        x2, y2, _, _ = calculate_coordinates(x2, y2)

    if type(p1) is list or type(p1) is tuple:
        if 4 == len(p1):
            x1, y1, x2, y2 = p1
            x1, y1, x2, y2 = calculate_coordinates(x1, y1, x2, y2)
        else:
            x1, y1 = p1
            x1, y1, _, _ = calculate_coordinates(x1, y1)

    return x1, y1, x2, y2


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

    dif = min(b1y[1], b2y[1]) - max(b1y[0], b2y[0]) - threshold
    return dif > 0


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

    dif = min(b1y[1], b2y[1]) - max(b1y[0], b2y[0]) - threshold
    return dif > 0


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
    for i in range(4):
        dif = 0
        if i < 2:
            if i in edges_in_account:
                dif = box[i] - container[i] + threshold
            elif limit_values is not None:
                dif = box[i] - limit_values[i] + threshold
        else:
            if i in edges_in_account:
                dif = container[i] - box[i] + threshold
            elif limit_values is not None:
                dif = limit_values[i] - box[i] + threshold
        ret = dif >= 0
        if not ret:
            break
    return ret


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

    def __init__(self, p1: Union[int, list] = 0, p2: Union[int, list] = 0, x2=0, y2=0,
                 threshold: Union[int, ThresholdAttribute, None] = 40, container=None):
        if type(threshold) is ThresholdAttribute:
            self._threshold = threshold
        else:
            self._threshold = ThresholdAttribute()
            if type(threshold) is int:
                self.threshold = threshold
        self._diagonal_points = None
        self.diagonal_points = p1, p2, x2, y2
        self.section_container = container

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

    def add_writing_area(self, p1: Union[int, list] = 0, p2: Union[int, list] = 0, x2=0, y2=0,
                         guess_width=-1):
        pass

    def get_compatible_status(self, p1: Union[int, list] = 0, p2: Union[int, list] = 0, x2=0, y2=0):
        ret = 0
        x1, y1, x2, y2 = calculate_coordinates(p1, p2, x2, y2)
        ldif = x1 - self.left + self.threshold
        rdif = self.right - x2 + self.threshold
        if ldif >= 0 and rdif >= 0:
            ret = 0
        elif ldif < 0:
            ret = -1
        else:
            ret = 1
        return ret

    def get_single_sections_as_boxes(self, boxes=[]):
        pass

    # def is_area_in(self, edges_in_account: list, p1: Union[int, list, None], p2: Union[int, list, None] = None, x2=0,
    #                y2=0):
    #     coord = [calculate_coordinates(p1, p2, x2, y2)]
    #     ret = False
    #     for i in edges_in_account:
    #         if i < 2:
    #             dif = coord[i] - self.coordinates[i] + self.threshold
    #         else:
    #             dif = self.coordinates[i] - coord[i] + self.threshold
    #         ret = (dif >= 0)
    #         if not ret:
    #             break
    #     return ret


class StructuredSection(AbstractSection):
    """
    A StructuredSection class implements a structured section which can contain 2 kinds of structures: section stack or
    sibling sections

    Attributtes
    -----------


    Methods
    -------

    """

    def __init__(self, p1: Union[int, list] = 0, p2: Union[int, list] = 0, x2=0, y2=0,
                 threshold: Union[int, ThresholdAttribute, None] = 40, container=None):
        super().__init__(p1, p2, x2, y2, threshold, container)
        self._sections = []
        self.is_rigth_expandable = False
        self.is_bottom_expandable = True

    @property
    def sections(self):
        return self._sections

    def get_single_sections_as_boxes(self, boxes=[]):
        for section in self.sections:
            section.get_single_sections_as_boxes(boxes)
        return boxes

    # def build_layout_structure(self, area_list, pos: int) -> int:
    #     to_parent = False
    #     while not to_parent and pos < len(area_list):
    #         built_sibling = False
    #         if pos + 1 < len(area_list):
    #             built_sibling = overlap_vertically(area_list[pos], area_list[pos + 1], self.threshold)
    #         if built_sibling:
    #             # built a new sibling
    #             sibling_section_list = SiblingSectionList()
    #             self.add_new_section(sibling_section_list)
    #             pos = sibling_section_list.build_layout_structure(area_list, pos)
    #         else:
    #             status = self.get_compatible_status(area_list[pos]) % 100
    #             if status == 0:
    #                 self.add_writing_area(area_list[pos])
    #                 pos += 1
    #             else:
    #                 to_parent = True
    #     return pos
    #
    def add_new_section(self, section=False):
        self._sections.append(section)


class MainLayout(StructuredSection):
    def __init__(self, w: int, threshold=40):
        super().__init__(0, 0, w, 0, threshold)
        self.is_rigth_expandable = False

    def get_single_sections_as_boxes(self):
        boxes = []
        return super().get_single_sections_as_boxes(boxes)

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

    @staticmethod
    def build_layout(writing_area_list, w: int, threshold=None):
        main_layout = MainLayout(w)
        if threshold is not None:
            main_layout.threshold = threshold

        min_left = 100000
        max_right = 0
        for writing_area in writing_area_list:
            if min_left > writing_area[0]:
                min_left = writing_area[0]
            if max_right < writing_area[2]:
                max_right = writing_area[2]

        writing_area_list.sort(key=lambda x: x[1] * 10000 + x[0])
        lef_unexplored = min_left
        top_unexplored = 0
        right_unexplored = max_right
        for i, writing_area in enumerate(writing_area_list):
            added = False
            guess_left = min_left
            guess_right = max_right
            force_sibling = False
            offset = -1
            while (i + offset) >= 0 and overlap_vertically(writing_area_list[i + offset],
                                                           writing_area, main_layout.threshold):
                if writing_area_list[i + offset][0] < writing_area[0] and guess_left < writing_area_list[i + offset][2]:
                    guess_left = writing_area_list[i + offset][2]
                elif writing_area[0] <= writing_area_list[i + offset][0] < guess_right:
                    guess_right = writing_area_list[i + offset][0]
                force_sibling = True
                offset -= 1
            offset = 1
            while (i + offset) < len(writing_area_list) and overlap_vertically(writing_area_list[i + offset],
                                                                               writing_area, main_layout.threshold):
                if writing_area_list[i + offset][0] < writing_area[0] and guess_left < writing_area_list[i + offset][2]:
                    guess_left = writing_area_list[i + offset][2]
                elif writing_area[0] <= writing_area_list[i + offset][0] < guess_right:
                    guess_right = writing_area_list[i + offset][0]
                offset += 1
            guess_width = guess_right - guess_left
            pos = len(main_layout.sections) - 1
            if force_sibling:
                added, lef_unexplored, top_unexplored, right_unexplored = main_layout.sections[pos].add_writing_area(
                    writing_area, guess_width=guess_width, force_sibling=force_sibling)
            else:
                found = False
                while not found and pos >= 0:
                    found = main_layout.sections[pos].is_area_inside(writing_area)
                    if overlap_horizontally(
                            [main_layout.sections[pos].left, 0,  main_layout.sections[pos].right, 0],
                            writing_area, main_layout.threshold):
                        pos = -1
                    pos -= 1
                if found:
                    pos += 1
                    added, lef_unexplored, top_unexplored, right_unexplored = main_layout.sections[pos].add_writing_area(
                        writing_area, guess_width=guess_width)
                if not added:
                    if lef_unexplored + main_layout.threshold > writing_area[0]:
                        lef_unexplored = min_left
                    if right_unexplored - main_layout.threshold < writing_area[2]:
                        right_unexplored = max_right
                    new_section = BigSectionOfSibling(main_layout, lef_unexplored, writing_area[1], right_unexplored,
                                                      top_unexplored,
                                                      threshold=main_layout._threshold)
                    main_layout.add_new_section(new_section)
                    new_section.add_writing_area(writing_area, guess_width=guess_width)

        # main_layout.build_layout_structure(writing_area_list, 0)
        return main_layout


class BigSectionOfSibling(StructuredSection):
    def __init__(self, container: MainLayout, p1: Union[int, list] = 0, p2: Union[int, list] = 0, x2=0, y2=0,
                 threshold: Union[int, ThresholdAttribute, None] = 40):
        super().__init__(p1, p2, x2, y2, threshold, container)
        self._width_sibling = -1

    @property
    def width_sibling(self):
        # if self._width_sibling == -1:
        #     ret = self.width
        # else:
        #     ret = self._width_sibling
        # return ret
        return self._width_sibling

    @property
    def siblings(self):
        return self.sections

    def is_area_compatible(self, pos: int, p1: Union[int, list] = 0, p2: Union[int, list] = 0, x2=0, y2=0,
                           guess_width=-1):
        x1, y1, x2, y2 = calculate_coordinates(p1, p2, x2, y2)
        ret = self._has_area_similar_width(pos, x1, y1, x2, y2, guess_width)
        if not ret:
            ret = self._has_area_similar_center(pos, x1, y1, x2, y2)
            if ret and len(self.siblings) > pos + 1:
                ret = (x2 - self.siblings[pos + 1].left - self.threshold) <= 0
        return ret

    def _has_area_similar_center(self, pos: int, p1: Union[int, list] = 0, p2: Union[int, list] = 0, x2=0, y2=0):
        ret = False
        if pos in range(len(self.siblings)):
            x1, y1, x2, y2 = calculate_coordinates(p1, p2, x2, y2)
            dif = (abs(self.siblings[pos].center - (x1 + x2) / 2) % self.width_sibling) - self.threshold
            ret = dif <= 0
        return ret

    def is_area_inside(self, p1: Union[int, list] = 0, p2: Union[int, list] = 0, x2=0, y2=0):
        x1, y1, x2, y2 = calculate_coordinates(p1, p2, x2, y2)
        edges = [0, 1]
        if not self.is_rigth_expandable:
            edges.append(2)
        if not self.is_bottom_expandable:
            edges.append(3)
        return contains(edges, self.coordinates, (x1, y1, x2, y2), self.threshold)

    def _area_belongs_to_sibling(self, pos: int, p1: Union[int, list] = 0, p2: Union[int, list] = 0, x2=0, y2=0):
        x1, y1, x2, y2 = calculate_coordinates(p1, p2, x2, y2)
        dif = abs(x1 - self.siblings[pos].left) - self.threshold
        if dif > 0:
            dif = abs(self.siblings[pos].center - ((x1+x2)/2)) - self.threshold
        return dif<=0

    def _has_area_similar_width(self, pos: int, p1: Union[int, list] = 0, p2: Union[int, list] = 0, x2=0, y2=0,
                                guess_width=-1):
        margin = self.threshold + self.threshold + self.threshold
        if self._width_sibling == -1:
            dif = 0
        else:
            x1, y1, x2, y2 = calculate_coordinates(p1, p2, x2, y2)
            dif = abs(x2 - x1 - self.width_sibling)
            if dif > margin:
                dif = abs(guess_width - self.width_sibling)
            if dif > margin and pos in range(len(self.siblings)):
                dif = abs(x2 - x1 - self.siblings[pos].max_width)
            if dif > margin and pos in range(len(self.siblings)):
                dif = abs(guess_width - self.siblings[pos].max_width)
        return dif <= margin

    def _search_sibling_pos(self, p1: Union[int, list] = 0, p2: Union[int, list] = 0, x2=0, y2=0):
        x1, y1, x2, y2 = calculate_coordinates(p1, p2, x2, y2)
        pos = 0
        status = 1
        while status == 1 and pos < len(self.siblings):
            status = self.siblings[pos].get_compatible_status(x1, y1, x2, y2)
            pos += 1
        return pos - 1, status

    def _can_insert_sibling(self, pos):
        if pos <= -1 or len(self.siblings) == 0:
            x1 = self.left
            x2 = self.right
        elif pos == len(self.siblings):
            x1 = self.siblings[pos - 1].right
            x2 = self.right
        elif pos == 0:
            x1 = self.left
            x2 = self.siblings[pos].left
        else:
            x1 = self.siblings[pos - 1].right
            x2 = self.siblings[pos].left
        dif = 0 if self.width_sibling == -1 else abs(x2 - x1 - self.width_sibling)
        return dif <= (self.threshold + self.threshold)

    def _insert_area(self, pos, status, p1: Union[int, list] = 0, p2: Union[int, list] = 0, x2=0, y2=0, guess_width=-1,
                     force_sibling=False):
        added = True
        x1, y1, x2, y2 = calculate_coordinates(p1, p2, x2, y2)
        if force_sibling:
            force_insert = not overlap_horizontally([x1,y1,x2,y2], self.siblings[pos].coordinates, self.threshold)
            force_insert = force_insert and True if (pos-status) not in range(len(self.siblings)) else \
                        not overlap_horizontally([x1,y1,x2,y2], self.siblings[pos-status].coordinates, self.threshold)
        else:
            force_insert=False
        if status == -1 and (force_insert or self._can_insert_sibling(pos)):
            single_section = SingleSection(self, x1, self.top, x2, self.top, guess_width,
                                           threshold=self._threshold)
            self.siblings.insert(pos, single_section)
            self.siblings.sort(key=lambda x: x.left)
            single_section.add_writing_area(x1, y1, x2, y2)

        elif force_insert or self._can_insert_sibling(pos + 1):
            single_section = SingleSection(self, x1, self.top, x2, self.top, guess_width,
                                           threshold=self._threshold)
            self.siblings.append(single_section)
            single_section.add_writing_area(x1, y1, x2, y2)
        else:
            added = False

        return added

    def add_writing_area(self, p1: Union[int, list] = 0, p2: Union[int, list] = 0, x2=0, y2=0,
                         guess_width=-1, force_sibling=False):
        x1, y1, x2, y2 = calculate_coordinates(p1, p2, x2, y2)
        # if is_area_expandable:
        #     x1 = self.left
        #     if self.width_sibling==-1:
        #         x2 = self.right
        #     else:
        #         x2 = x1+self.width_sibling
        left_unexplored = -1
        right_unexplored = -1
        top_unexplored = -1
        if not self.is_area_inside(x1, y1, x2, y2):
            added = False
            left_unexplored = self.left
            right_unexplored = self.right
            top_unexplored = self.bottom
        else:
            pos, status = self._search_sibling_pos(x1, y1, x2, y2)
            added = False
            if force_sibling:
                if status == 0:
                    added, left_unexplored, right_unexplored, top_unexplored = \
                        self.siblings[pos].add_writing_area(x1, y1, x2, y2, guess_width=guess_width)
                else:
                    added = self._insert_area(pos, status, x1, y1, x2, y2, guess_width, force_sibling)
            # elif self.is_area_compatible(pos, x1, y1, x2, y2, guess_width):
            #     # Try to add
            #     if status == 0:
            #         # add to pos
            #         added, left_unexplored, right_unexplored, top_unexplored = \
            #             self.siblings[pos].add_writing_area(x1, y1, x2, y2, guess_width=guess_width)
            #     else:
            #         added = self._insert_area(pos, status, x1, y1, x2, y2, guess_width)
            elif status == 0 and self._area_belongs_to_sibling(pos, x1, y1, x2, y2):
                added, left_unexplored, right_unexplored, top_unexplored = \
                     self.siblings[pos].add_writing_area(x1, y1, x2, y2, guess_width=guess_width)
            elif status != 0 and self.is_area_compatible(pos, x1, y1, x2, y2, guess_width):
                added = self._insert_area(pos, status, x1, y1, x2, y2, guess_width)
            if not added:
                if pos == -1:
                    left_unexplored = self.left
                    right_unexplored = self.right
                    top_unexplored = self.bottom
                elif pos == len(self.siblings):
                    left_unexplored = self.right
                    right_unexplored = self.right
                    top_unexplored = self.bottom
                else:
                    left_unexplored = self.siblings[pos].left if status == 1 else self.siblings[pos].right
                    top_unexplored = self.siblings[pos].bottom
                    if pos > 0:
                        right_unexplored = self.siblings[pos - 1].right if status == 1 else self.siblings[pos].right
                    else:
                        right_unexplored = self.right

        return added, left_unexplored, top_unexplored, right_unexplored


# class BigSectionlist(StructuredSection):
# def build_layout_structure(self, area_list, pos: int) -> int:
#     added = False
#     while not added and pos < len(area_list):
#         built_sibling = False
#         if pos + 1 < len(area_list):
#             built_sibling = overlap_vertically(area_list[pos], area_list[pos + 1], self.threshold)
#         if built_sibling:
#             # built a new sibling
#             sibling_section_list = BigSectionOfSibling()
#             self.add_new_section(sibling_section_list)
#             pos = sibling_section_list.build_layout_structure(area_list, pos)
#         else:
#             status = self.get_compatible_status(area_list[pos]) % 100
#             if status == 0:
#                 self.add_writing_area(area_list[pos])
#                 pos += 1
#             else:
#                 to_parent = True
#     return pos


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

    def __init__(self, container: BigSectionOfSibling, p1: Union[int, list] = 0, p2: Union[int, list] = 0, x2=0, y2=0,
                 max_width=-1, threshold: Union[int, ThresholdAttribute, None] = 40):
        super().__init__(p1, p2, x2, y2, threshold=threshold, container=container)
        self._len = 0
        self._suma_center = 0
        self.max_width = max_width
        if self.section_container.width_sibling < self.right - self.left:
            self.section_container._width_sibling = self.right - self.left

    @property
    def guess_left(self):
        ret = min(self.right - self.max_width, self.left)
        if ret < self.section_container.left:
            ret = self.section_container.left
        return ret

    @property
    def guess_right(self):
        ret = max(self.left + self.max_width, self.right)
        if ret > self.section_container.right:
            ret = self.section_container.right
        return ret

    @property
    def center(self):
        return self._suma_center / self._len

    def add_writing_area(self, p1: Union[int, list] = 0, p2: Union[int, list] = 0, x2=0, y2=0,
                         guess_width=-1):
        x1, y1, x2, y2 = calculate_coordinates(p1, p2, x2, y2)
        self.bottom = y2
        self._len += 1
        self._suma_center += (x1 + x2) / 2
        if x1 < self.left:
            self.left = x1
        if x2 > self.right:
            self.right = x2
        if self.section_container.width_sibling < self.width:
            self.section_container._width_sibling = self.width
        if self.section_container.bottom < self.bottom:
            self.section_container.bottom = self.bottom
        return True, -1, -1, -1

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
    main_layout.add_writing_area(10, 10, 90, 50)
    main_layout.add_writing_area(10, 50, 90, 60)
    try:
        main_layout.add_writing_area(10, 50, 90, 60)
    except ValueError as e:
        print(e)

    main_layout.add_writing_area(10, 60, 90, 70)
    main_layout.add_new_section(SingleSection())
    main_layout.add_writing_area(10, 72, 90, 85)
    main_layout.add_writing_area(10, 85, 90, 90)
    main_layout.add_new_section(BigSectionOfSibling())
    main_layout.add_writing_area(10, 91, 40, 98)
    main_layout.add_writing_area(55, 94, 90, 105)
    main_layout.add_new_section(SingleSection())
    main_layout.add_writing_area(10, 104, 90, 120)

    print(main_layout.diagonal_points)

    boxes = main_layout.get_single_sections_as_boxes()

    print(boxes)

    m = MainLayout.build_layout([[10, 10, 90, 50], [10, 50, 100, 60], [10, 60, 90, 70], [10, 72, 90, 85],
                                 [10, 85, 90, 90], [10, 91, 40, 98], [55, 94, 90, 105], [10, 110, 40, 120],
                                 [55, 104, 90, 117], [10, 119, 90, 125], [10, 125, 90, 135]], 105)
    boxes = m.get_single_sections_as_boxes()
    print(boxes)


if __name__ == "__main__":
    test_build_layout()
