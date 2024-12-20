from pathlib import Path
from py_portada_paragraphs import raw_predictions, extract_fragments_and_get_json, MainLayout
import cv2
from doclayout_yolo import YOLOv10
import os


class PortadaParagraphCutter(object):
    def __init__(self, input_path='', model_path=None):
        if len(input_path) > 0:
            self._image_path = input_path
            self._image = cv2.imread(input_path)
        else:
            self.image = None
            self._image_path = ''
        self._yolo_model = None
        self._yolo_model_path = model_path
        self.image_blocks = []
        self.iou_threshold = 0.5
        self.area_ratio_threshold = 0.8
        self.margin = 5

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, val):
        self._image = val
        self.image_blocks = []

    @property
    def image_path(self):
        return self._image_path

    @image_path.setter
    def image_path(self, val):
        self._image_path = val
        self.image = cv2.imread(val)

    @property
    def yolo_model(self):
        return self._yolo_model

    @property
    def yolo_model_path(self):
        return self._yolo_model_path

    @yolo_model_path.setter
    def yolo_model_path(self, val):
        self._yolo_model_path = val
        self._yolo_model = YOLOv10(self._yolo_model_path)

    def __verify_image(self):
        if self.image is None:
            raise Exception("Error: Image is not specified.")

    # @staticmethod
    # def __get_config_content():
    #     #Directory global /etc/

    def save_image(self, image_path=''):
        """
        Save the image from 'self.image' to 'image_path'. By default, image_path is equal to 'self.image_path'
        :param image_path: the image path where save the image
        :return: None
        """
        self.__verify_image()
        if len(image_path) == 0:
            image_path = self.image_path
        cv2.imwrite(image_path, self.image)

    def process_image(self):
        self.__verify_image()
        # get json blocks from arcanum
        blocks = raw_predictions(self.image, self.yolo_model)
        blocks.sort(key=lambda x: x[1] * 10000 + x[0])

        layout = MainLayout.build_layout(blocks, self.image.shape[1], 30)
        boxes = layout.get_single_sections_as_boxes()
        boxes.sort(key=lambda x: x[1] * 10000 + x[0])

        column_images = self.__cut_images_from_blocks(self.image, boxes)

        paragraph_images = []
        for image in column_images:
            ajson = extract_fragments_and_get_json(image, self.yolo_model, iou_threshold=self.iou_threshold,
                                               area_ratio_threshold=self.area_ratio_threshold, margin=self.margin)
            for info_block in ajson:
                paragraph_images.append(info_block["image"])

        file_name = Path(self.image_path).stem
        ext = Path(self.image_path).suffix
        if len(ext) == 0:
            ext = ".jpg"
        self.image_blocks = []
        count = 0
        for image in paragraph_images:
            self.image_blocks.append(
                dict(file_name=file_name, extension=ext, count=count, image=cv2.imencode(ext, image)[1]))
            count = count + 1

    def save_block_images(self, dir_name="", image_name=""):
        self.__verify_image()
        for bi in self.image_blocks:
            if len(image_name) == 0:
                image_name = bi["file_name"]

            if len(dir_name) > 0:
                image_path = Path(dir_name).joinpath(Path(image_name).stem)
            else:
                image_path = Path(image_name)

            with open("{file_name}_{count:03d}{extension}".format(file_name=image_path, count=bi["count"],
                                                                  extension=bi["extension"]), "wb") as bf:
                bf.write(bi["image"])

    @staticmethod
    def __cut_images_from_blocks(image, boxes):
        img_height, img_width, _ = image.shape

        cut_blocks = []
        for box in boxes:
            x1, y1, x2, y2 = box

            # Cut the text block region from the image
            cut_block = image[y1:y2, x1:x2]

            # Append the cut block to the list and transform into image cv2
            cut_blocks.append(cut_block)
        return cut_blocks


if __name__ == "__main__":
    cwd = os.getcwd()
    cwd = Path(cwd).parent.absolute().as_posix()
    filepath = cwd + "/py_portada_paragraphs/modelo/doclayout_yolo_docstructbench_imgsz1024.pt"
    input_dir = cwd + "/demo/input"
    output_dir = cwd + "/demo/output"
    annotated_dir = cwd + "/demo/annotated"
    model = YOLOv10(filepath)

    #process_one(model, "1902_01_05_HAB_DM_00000_U_01_0.jpg", input_dir, output_dir, annotated_dir)
    #process_one(model, "inclinada.jpg", input_dir, output_dir, annotated_dir)
    #process_all(model, input_dir, output_dir, annotated_dir)

    util = PortadaParagraphCutter()

    util.yolo_model_path = filepath
    util.image_path = image_path = os.path.join(input_dir, "inclinada.jpg")
    util.process_image()
    util.save_block_images(output_dir)
