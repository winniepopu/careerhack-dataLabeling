from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QPoint
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPixmap, QPainter, QColor, QBrush, QPen, QPolygon
import sys
import labeling
from PIL import Image, ExifTags
from PIL.ImageQt import ImageQt
import glob
import json
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import pdb

image_list = []
name_list = []
bbox_list = []
put_in_json = []
run_first = False


class ExampleApp(QtWidgets.QMainWindow, labeling.Ui_MainWindow):
    def __init__(self, parent=None):
        super(ExampleApp, self).__init__(parent)
        self.setupUi(self)
        # read picture
        for filename in glob.glob('./train/train/Input/*.jpg'):  # set input dir
            name_list.append(filename[-17:])
            im = Image.open(filename)
            # check EXIF rotation
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            if im._getexif() != None:
                exif = dict(im._getexif().items())
                if exif[orientation] == 3:
                    im = im.rotate(180, expand=True)

                elif exif[orientation] == 6:
                    im = im.rotate(270, expand=True)

                elif exif[orientation] == 8:
                    im = im.rotate(90, expand=True)

            image_list.append(im)

        # set self.qim,self.resize_ratio,self.cursor,self.rotate_type
        self.qim = ImageQt(image_list[0])
        first = QPixmap.fromImage(self.qim)
        w = self.label.width()
        h = self.label.height()
        w_ratio = w/float(first.width())
        h_ratio = h/float(first.height())
        if w_ratio > h_ratio:
            self.resize_ratio = h_ratio
            # print('h_ratio=',h_ratio)
        else:
            self.resize_ratio = w_ratio
            # print('w_ratio=',w_ratio)
        self.cursor = 0
        self.changeStatus(self, name_list[self.cursor])
        self.rotate_type = 0
        self.d2v = Doc2Vec.load("d2v.model")  # set model dir

        myline_path = './my_line/' + \
            name_list[self.cursor][:-3] + 'json'  # set my_line dir
        self.my_line_list = json.load(open(myline_path, 'r'))
        self.key_word = ''

        # print(name_list[0])

############################################
    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        pic = QPixmap.fromImage(self.qim)
        resize = pic.size().scaled(self.label.size(), QtCore.Qt.KeepAspectRatio)
        qp.drawPixmap(0, 0, resize.width(), resize.height(), pic)
        self.bbox(qp)
        # qp.end()

############################################
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            m_x = event.pos().x()
            m_y = event.pos().y()
            in_bbox = []
            if self.rotate_type == 0:
                in_bbox = list(map(lambda bbox: bbox['boundingBox'][0]*self.resize_ratio <= m_x and bbox['boundingBox'][1]*self.resize_ratio <=
                                   m_y and bbox['boundingBox'][4]*self.resize_ratio >= m_x and bbox['boundingBox'][5]*self.resize_ratio >= m_y, bbox_list))
            elif self.rotate_type == 1:
                in_bbox = list(map(lambda bbox: bbox['boundingBox'][1]*self.resize_ratio <= m_x and self.label.height()-bbox['boundingBox'][0]*self.resize_ratio <=
                                   m_y and bbox['boundingBox'][5]*self.resize_ratio >= m_x and self.label.height()-bbox['boundingBox'][4]*self.resize_ratio >= m_y, bbox_list))
            elif self.rotate_type == 2:
                in_bbox = list(map(lambda bbox: bbox['boundingBox'][1]*self.resize_ratio <= m_x and bbox['boundingBox'][0]*self.resize_ratio <=
                                   m_y and bbox['boundingBox'][5]*self.resize_ratio >= m_x and bbox['boundingBox'][4]*self.resize_ratio >= m_y, bbox_list))
            elif self.rotate_type == 3:
                in_bbox = list(map(lambda bbox: bbox['boundingBox'][0]*self.resize_ratio <= m_x and self.label.height()-bbox['boundingBox'][1]*self.resize_ratio <=
                                   m_y and bbox['boundingBox'][4]*self.resize_ratio >= m_x and self.label.height()-bbox['boundingBox'][5]*self.resize_ratio >= m_y, bbox_list))
            print(m_x, m_y)

            try:
                which_bbox = in_bbox.index(True)
                put_in_json.append(bbox_list[which_bbox])
            except ValueError:
                print('this position dosen\'t have bbox!')

############################################
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Z:
            self.next()
        elif event.key() == QtCore.Qt.Key_X:
            self.reset()
        elif event.key() == QtCore.Qt.Key_A:
            self.last()

############################################
    def next(self):
        self.get_select_vec()
        self.write_json()
        self.cursor += 1
        bbox_list.clear()
        put_in_json.clear()
        if self.cursor >= len(image_list):
            print('all done!')
            self.Cancel()
        self.changeStatus(self, name_list[self.cursor])

        self.qim = ImageQt(image_list[self.cursor])
        first = QPixmap.fromImage(self.qim)
        w = self.label.width()
        h = self.label.height()
        w_ratio = w/float(first.width())
        h_ratio = h/float(first.height())
        if w_ratio > h_ratio:
            self.resize_ratio = h_ratio
        else:
            self.resize_ratio = w_ratio

        self.my_line_list.clear()
        myline_path = './my_line/' + \
            name_list[self.cursor][:-3] + 'json'  # set my_line dir
        self.my_line_list = json.load(open(myline_path, 'r'))

        global run_first
        run_first = False

        self.update()

############################################
    def last(self):
        self.get_select_vec()
        self.write_json()
        self.cursor -= 1
        bbox_list.clear()
        put_in_json.clear()
        if self.cursor < 0:
            print('this is first!')
            return

        self.qim = ImageQt(image_list[self.cursor])
        self.changeStatus(self, name_list[self.cursor])
        first = QPixmap.fromImage(self.qim)
        w = self.label.width()
        h = self.label.height()
        w_ratio = w/float(first.width())
        h_ratio = h/float(first.height())
        if w_ratio > h_ratio:
            self.resize_ratio = h_ratio
        else:
            self.resize_ratio = w_ratio

        self.my_line_list.clear()
        myline_path = './my_line/' + \
            name_list[self.cursor][:-3] + 'json'  # set my_line dir
        self.my_line_list = json.load(open(myline_path, 'r'))

        global run_first
        run_first = False

        self.update()

############################################
    def write_json(self):
        #print(put_in_json)
        if len(put_in_json) == 0:
            return

        # sort the lift of dict according to their value in 'index' (ascending order)
        put_in_json.sort(key=lambda k: k['index'], reverse=False)

        # remove the redundant tokens ('index', 'confidence')
        for bbox in put_in_json:
            if bbox.__contains__('confidence'):
                del bbox['confidence']
            if bbox.__contains__('index'):
                del bbox['index']
        
        output_format = {'text': '', 'elements': []}
        text = ''
        for bbox in put_in_json:
            if ('text' in bbox) == False:
                continue
            text += bbox['text']+' '
            del bbox['text']
            if bbox.__contains__('confidence'):
                del bbox['confidence']
            output_format['elements'].append(bbox)
        output_format['text'] = text[:-1]
        output_dir = './output/' + \
            name_list[self.cursor][:-3] + 'json'  # set output dir
        with open(output_dir, 'w') as outfile:
            json.dump(output_format, outfile)


############################################


    def get_select_vec(self):
        if len(put_in_json) == 0 or len(self.my_line_list) == 0:
            return

        center_x = abs(put_in_json[0]['boundingBox']
                       [0] + put_in_json[0]['boundingBox'][4])/2.0
        center_y = abs(put_in_json[0]['boundingBox']
                       [1] + put_in_json[0]['boundingBox'][5])/2.0
        for line_dict in self.my_line_list:
            if line_dict['boundingBox'][0] <= center_x and \
               line_dict['boundingBox'][1] <= center_y and \
               line_dict['boundingBox'][4] >= center_x and \
               line_dict['boundingBox'][5] >= center_y:
                # print(line_dict['text'])
                infer_test = self.d2v.infer_vector(doc_words=word_tokenize(
                    line_dict['text']), alpha=0.025, steps=500)
                sims = self.d2v.docvecs.most_similar([infer_test], topn=1)
                self.key_word = sims[0][0]

############################################
    def reset(self):
        # print(put_in_json)
        put_in_json.clear()
        print('reset!')

############################################
    def bbox(self, qp):
        json_path = './train/train/Input/' + \
            name_list[self.cursor][:-3] + 'json'  # set input dir
        json_dict = json.load(open(json_path, 'r'))
        json_rotate = json_dict['recognitionResults'][0]['clockwiseOrientation']
        json_line = json_dict['recognitionResults'][0]['lines']
        mark_bbox = self.mark_line(qp, json_rotate)
        mark_first = 0
        global run_first
        qp.setPen(QPen(QColor(200, 0, 0),  1, QtCore.Qt.SolidLine))

        # the index of bbox
        box_index = 0
        for line in json_line:
            for word in line['words']:
                # adding index to dict (used for sorting)
                word['index'] = box_index
                box_index = box_index + 1
                bbox_list.append(word)
                # print(word)
                line_position = word['boundingBox']
                if run_first == False:
                    if mark_bbox == None:
                        pass
                    elif (line_position[0] + line_position[2])/2>=mark_bbox[0] and (line_position[1] + line_position[7])/2>=mark_bbox[1] and (line_position[0] + line_position[2])/2<=mark_bbox[4] and (line_position[1] + line_position[7])/2<=mark_bbox[5]:
                    
                        if (mark_bbox[0] + mark_bbox[2])/2 < (line_position[0] + line_position[2])/2:
                            m_brush = QBrush(QColor(0, 200, 200, 100), QtCore.Qt.SolidPattern)
                            qp.setBrush(m_brush)
                            mark_first = 1
                        
                    else:
                        qp.setBrush(QtCore.Qt.NoBrush)
                        mark_first = 0

                    if mark_first == 1:
                        put_in_json.append(word)

                if abs(json_rotate-360) < 45 or abs(json_rotate) < 45:
                    self.rotate_type = 0
                    points = [QPoint(line_position[0]*self.resize_ratio, line_position[1]*self.resize_ratio), QPoint(line_position[2]*self.resize_ratio, line_position[3]*self.resize_ratio), QPoint(
                        line_position[4]*self.resize_ratio, line_position[5]*self.resize_ratio), QPoint(line_position[6]*self.resize_ratio, line_position[7]*self.resize_ratio)]
                    qp.drawPolygon(QPolygon(points))

                elif abs(json_rotate-90) < 45:
                    self.rotate_type = 1
                    points = [QPoint(line_position[1]*self.resize_ratio, self.label.height()-line_position[0]*self.resize_ratio), QPoint(line_position[3]*self.resize_ratio, self.label.height()-line_position[2]*self.resize_ratio), QPoint(
                        line_position[5]*self.resize_ratio, self.label.height()-line_position[4]*self.resize_ratio), QPoint(line_position[7]*self.resize_ratio, self.label.height()-line_position[6]*self.resize_ratio)]
                    qp.drawPolygon(QPolygon(points))

                elif abs(json_rotate-270) < 45:
                    self.rotate_type = 2
                    points = [QPoint(line_position[1]*self.resize_ratio, line_position[0]*self.resize_ratio), QPoint(line_position[3]*self.resize_ratio, line_position[2]*self.resize_ratio), QPoint(
                        line_position[5]*self.resize_ratio, line_position[4]*self.resize_ratio), QPoint(line_position[7]*self.resize_ratio, line_position[6]*self.resize_ratio)]
                    qp.drawPolygon(QPolygon(points))

                elif abs(json_rotate-180) < 45:
                    self.rotate_type = 3
                    points = [QPoint(line_position[0]*self.resize_ratio, self.label.height()-line_position[1]*self.resize_ratio), QPoint(line_position[2]*self.resize_ratio, self.label.height()-line_position[3]*self.resize_ratio), QPoint(
                        line_position[4]*self.resize_ratio, self.label.height()-line_position[5]*self.resize_ratio), QPoint(line_position[6]*self.resize_ratio, self.label.height()-line_position[7]*self.resize_ratio)]
                    qp.drawPolygon(QPolygon(points))

        run_first = True

############################################
    def mark_line(self,qp,json_rotate): 
        if self.key_word == '':
            return
        #key = word_tokenize(self.key_word)

        most_sim_prob = 0
        most_sim_index = 0
        count = 0
        for i in self.my_line_list:
            infer_test = self.d2v.infer_vector(doc_words=word_tokenize(i['text']),alpha=0.025,steps=500)
            sims = self.d2v.docvecs.most_similar([infer_test],topn=1)
            given = sims[0][0]
            prob = self.d2v.docvecs.similarity(self.key_word,given)
            
            if count == 0:
                most_sim_prob = prob
                most_sim_index = count
            elif prob > most_sim_prob:
                most_sim_prob = prob
                most_sim_index = count
            count+=1
        #print(self.my_line_list[most_sim_index]['text'],most_sim_prob)
        mark_bbox = self.my_line_list[most_sim_index]['boundingBox']
        qp.setPen(QtCore.Qt.NoPen)
        brush = QBrush(QColor(200, 200, 0, 100), QtCore.Qt.SolidPattern)
        qp.setBrush(brush)
        if abs(json_rotate-360)<45 or abs(json_rotate)<45:
            points = [QPoint(mark_bbox[0]*self.resize_ratio, mark_bbox[1]*self.resize_ratio), QPoint(mark_bbox[2]*self.resize_ratio, mark_bbox[3]*self.resize_ratio), QPoint(mark_bbox[4]*self.resize_ratio, mark_bbox[5]*self.resize_ratio), QPoint(mark_bbox[6]*self.resize_ratio, mark_bbox[7]*self.resize_ratio)]
            qp.drawPolygon(QPolygon(points))                  
                    
        elif abs(json_rotate-90)<45:
            points = [QPoint(mark_bbox[1]*self.resize_ratio, self.label.height()-mark_bbox[0]*self.resize_ratio), QPoint(mark_bbox[3]*self.resize_ratio, self.label.height()-mark_bbox[2]*self.resize_ratio), QPoint(mark_bbox[5]*self.resize_ratio, self.label.height()-mark_bbox[4]*self.resize_ratio), QPoint(mark_bbox[7]*self.resize_ratio, self.label.height()-mark_bbox[6]*self.resize_ratio)]
            qp.drawPolygon(QPolygon(points))              
                    
        elif abs(json_rotate-270)<45:
            points = [QPoint(mark_bbox[1]*self.resize_ratio, mark_bbox[0]*self.resize_ratio), QPoint(mark_bbox[3]*self.resize_ratio, mark_bbox[2]*self.resize_ratio), QPoint(mark_bbox[5]*self.resize_ratio, mark_bbox[4]*self.resize_ratio), QPoint(mark_bbox[7]*self.resize_ratio, mark_bbox[6]*self.resize_ratio)]
            qp.drawPolygon(QPolygon(points))             
                    
        elif abs(json_rotate-180)<45:
            points = [QPoint(mark_bbox[0]*self.resize_ratio, self.label.height()-mark_bbox[1]*self.resize_ratio), QPoint(mark_bbox[2]*self.resize_ratio, self.label.height()-mark_bbox[3]*self.resize_ratio), QPoint(mark_bbox[4]*self.resize_ratio, self.label.height()-mark_bbox[5]*self.resize_ratio), QPoint(mark_bbox[6]*self.resize_ratio, self.label.height()-mark_bbox[7]*self.resize_ratio)]
            qp.drawPolygon(QPolygon(points))

        qp.setBrush(QtCore.Qt.NoBrush)
        return mark_bbox

############################################
    def Cancel(self):
        self.close()

############################################


def main():
    app = QApplication(sys.argv)
    form = ExampleApp()
    form.show()
    app.exec_()


if __name__ == '__main__':
    main()
