from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, Comment, tostring, ElementTree

from detect.detection import Detection


class PascalVOCAnnotator:

    def __init__(self, devkit_path, classes):

        self.devkit_path = devkit_path
        self.classes = classes

    def write(self, filename, img, objects):

        annotation = Element('annotation')

        self.write_simple_text_node( annotation, 'folder', 'Wallapop')
        self.write_simple_text_node( annotation, 'filename', filename)
        source = SubElement(annotation, 'source')
        self.write_simple_text_node(source, 'database', 'Wallapop')
        self.write_image_node(annotation, img)
        self.write_simple_text_node(annotation, 'segmented', '0')

        print prettify(annotation)


    def write_simple_text_node(self, parent_node, node_name, text):
        node = SubElement(parent_node, node_name)
        node.text = text

    def write_object_node(self, parent_node, img, obj):

        object = SubElement(parent_node, 'object')

        self.write_simple_text_node(object, 'name', self.classes[int(obj.class_id)])
        self.write_simple_text_node(object, 'pose', 'not_used')
        self.write_simple_text_node(object, 'truncated', '0')
        self.write_simple_text_node(object, 'difficult', '0')

        bndbox = SubElement(object, 'bndbox')

        width = img.shape[0]
        height = img.shape[1]

        self.write_simple_text_node(bndbox, 'xmin', obj.xmin * width)
        self.write_simple_text_node(bndbox, 'ymin', obj.ymin * height)
        self.write_simple_text_node(bndbox, 'xmax', obj.xmax * width)
        self.write_simple_text_node(bndbox, 'ymax', obj.ymax * height)



    def write_image_node(self, parent_node, img):

        if len(img.shape) == 3:
            w, h, d = img.shape
        else:
            w, h = img.shape
            d = 1

        size = SubElement(parent_node, 'size')
        self.write_simple_text_node(size, 'width', str(w))
        self.write_simple_text_node(size, 'height', str(h))
        self.write_simple_text_node(size, 'depth', str(d))

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


if __name__ == '__main__':

    det = Detection(8.0, 0.9, 0.1, 0.12, 0.9, 0.4)

    voc = PascalVOCAnnotator("./")

    voc.write("0000001.jpg", "",  [det])
