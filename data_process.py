# -*- coding: utf-8 -*-
"""
数据预处理
"""
import os
import xml.etree.ElementTree as ET

__author__ = 'Liushen'

classes_dic = {'classN': 1, 'classM': 2, 'classI': 3, 'classE': 4,
               'classC': 5, 'classPA': 6, 'classA': 7, 'classPX': 8, 'classNO': 9}


def main():
    input_dir = "annotation_train"
    output_file = "data.txt"

    all_data = []

    for each_file in os.listdir(input_dir):
        if not each_file.endswith(".xml"):
            continue
        tree = ET.parse(os.path.join(input_dir, each_file))
        root = tree.getroot()

        for each_annotation in root.iter('annotation'):
            folder = each_annotation.find('folder').text
            filename = each_annotation.find('filename').text

            for each_obj in each_annotation.iter('object'):
                for each_box in each_obj.iter('bndbox'):
                    all_data.append(((os.path.join(input_dir, ".", folder, filename),
                                      each_box.find('xmin').text,
                                      each_box.find('ymin').text,
                                      each_box.find('xmax').text,
                                      each_box.find('ymax').text,
                                      each_obj.find('name').text)))

    with open(output_file, "w") as fd:
        for d in all_data:
            fd.write("\t".join(d))


if __name__ == '__main__':
    main()
    print("done")
