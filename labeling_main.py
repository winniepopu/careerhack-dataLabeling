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
import numpy as np
from sklearn.cluster import KMeans
import pickle

'''
input dir*2
kmean dir
kmean n_clusters
output dir
embedding input dir
'''

image_list = []
name_list = []
line_list = []
bbox_list = []
put_in_json = []
embedding = np.zeros(1)
w_h = [0,0]

class ExampleApp(QtWidgets.QMainWindow, labeling.Ui_MainWindow):
    def __init__(self, parent=None):
        '''
        self.qim : ImageQt
        self.resize_ratio : image resize ratio
        self.cursor : count how many file has processed
        self.rotate_type : orientation in json
        self.kmean : kmean model
        self.max_count : which class be selected most
        '''
        super(ExampleApp, self).__init__(parent)
        self.setupUi(self)
        #read picture
        for filename in glob.glob('./train/train/Input/*.jpg'):             ##set input dir #./train/train/Input/*.jpg
            name_list.append(filename[-17:])
            im=Image.open(filename)
            #check EXIF rotation
            for orientation in ExifTags.TAGS.keys():
               if ExifTags.TAGS[orientation]=='Orientation':
                   break
            if im._getexif() != None:
                exif=dict(im._getexif().items())
                if exif[orientation] == 3:
                    im=im.rotate(180, expand=True)
                    
                elif exif[orientation] == 6:
                    im=im.rotate(270, expand=True)
                    
                elif exif[orientation] == 8:
                    im=im.rotate(90, expand=True)

            image_list.append(im)

        #set self.qim,self.resize_ratio,self.cursor,self.rotate_type
        self.qim = ImageQt(image_list[0])
        first = QPixmap.fromImage(self.qim)
        w = self.label.width()
        h = self.label.height()
        w_ratio = w/float(first.width())
        h_ratio = h/float(first.height())
        if w_ratio > h_ratio:
            self.resize_ratio = h_ratio
            #print('h_ratio=',h_ratio)
        else:
            self.resize_ratio = w_ratio
            #print('w_ratio=',w_ratio)
        self.cursor = 0 
        self.rotate_type = 0

        with open('./kmean.pickle', 'rb') as f:                                 ## set kmean dir
            self.kmean = pickle.load(f)
        self.max_count = [0] * 10                                               ## set kmean n_clusters         
        
        #print(name_list[0])

############################################
    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        pic = QPixmap.fromImage(self.qim)
        resize = pic.size().scaled(self.label.size(), QtCore.Qt.KeepAspectRatio);
        qp.drawPixmap(0,0,resize.width(),resize.height(), pic)
        self.bbox(qp)
        #qp.end()

############################################
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            m_x = event.pos().x()
            m_y = event.pos().y()
            in_bbox = []
            if self.rotate_type == 0:
                in_bbox = list(map(lambda bbox: bbox['boundingBox'][0]*self.resize_ratio<=m_x and bbox['boundingBox'][1]*self.resize_ratio<=m_y and bbox['boundingBox'][4]*self.resize_ratio>=m_x and bbox['boundingBox'][5]*self.resize_ratio>=m_y, bbox_list))
            elif self.rotate_type == 1:
                in_bbox = list(map(lambda bbox: bbox['boundingBox'][1]*self.resize_ratio<=m_x and self.label.height()-bbox['boundingBox'][0]*self.resize_ratio<=m_y and bbox['boundingBox'][5]*self.resize_ratio>=m_x and self.label.height()-bbox['boundingBox'][4]*self.resize_ratio>=m_y, bbox_list))
            elif self.rotate_type == 2:
                in_bbox = list(map(lambda bbox: bbox['boundingBox'][1]*self.resize_ratio<=m_x and bbox['boundingBox'][0]*self.resize_ratio<=m_y and bbox['boundingBox'][5]*self.resize_ratio>=m_x and bbox['boundingBox'][4]*self.resize_ratio>=m_y, bbox_list))
            elif self.rotate_type == 3:
                in_bbox = list(map(lambda bbox: bbox['boundingBox'][0]*self.resize_ratio<=m_x and self.label.height()-bbox['boundingBox'][1]*self.resize_ratio<=m_y and bbox['boundingBox'][4]*self.resize_ratio>=m_x and self.label.height()-bbox['boundingBox'][5]*self.resize_ratio>=m_y, bbox_list))
            print(m_x,m_y)
            
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

############################################
    def next(self):
        self.max_counter()
        #self.write_json()
        self.cursor += 1
        line_list.clear()
        bbox_list.clear()
        put_in_json.clear()
        if self.cursor >= len(image_list):
            print('all done!')
            self.Cancel()
            
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
        self.update()

############################################   
    def write_json(self):
        if len(put_in_json)==0:
            return
        output_format = {'text':'','elements':[]}
        text = ''
        for bbox in put_in_json:
            text += bbox['text']+' '
            del bbox['text']
            if bbox.__contains__('confidence'):
                del bbox['confidence']
            output_format['elements'].append(bbox)
        output_format['text'] = text[:-1]
        output_dir = './output/' + name_list[self.cursor][:-3] + 'json'                 ##set output dir #./output/
        with open(output_dir, 'w') as outfile:
            json.dump(output_format, outfile)
        
############################################
    def max_counter(self):
        global embedding
        global w_h
        if len(put_in_json)==0:
            return
        center_x = abs(put_in_json[0]['boundingBox'][0] + put_in_json[0]['boundingBox'][4])/2.0
        center_y = abs(put_in_json[0]['boundingBox'][1] + put_in_json[0]['boundingBox'][5])/2.0
        true_or_false_index = np.array(list(map(lambda x:x[0]<=center_x and x[1]<=center_y and x[4]>=center_x and x[5]>=center_y,line_list)))
        true_or_false = np.array(line_list)
        true_or_false = true_or_false[np.where(true_or_false_index == True)]
        if len(true_or_false) == 0:
            return
        line_x = abs(true_or_false[0][0] + true_or_false[0][4])/2.0
        line_x = line_x/w_h[0]
        line_y = abs(true_or_false[0][1] + true_or_false[0][5])/2.0
        line_y = line_y/w_h[1]
        true_embedding_index = np.where(np.array(list(map(lambda x:x[-2] == line_x and x[-1] == line_y,embedding))) == True)
        true_embedding = embedding[true_embedding_index]
        if len(true_embedding) == 0:
            return
        result = self.kmean.predict(true_embedding)
        for class_index in result:
             self.max_count[class_index] += 1

############################################        
    def reset(self):
        #print(put_in_json)
        put_in_json.clear()
        print('reset!')

############################################
    def bbox(self,qp):
        json_path = './train/train/Input/' + name_list[self.cursor][:-3] + 'json'       ##set input dir #./train/train/Input/
        json_dict = json.load(open(json_path, 'r'))
        json_rotate = json_dict['recognitionResults'][0]['clockwiseOrientation']
        json_line = json_dict['recognitionResults'][0]['lines']

        embedding_path = './newinput/' + name_list[self.cursor][:-3] + 'npy'            ##set embedding input dir
        global embedding
        global w_h
        embedding = np.load(embedding_path)
        w_h[0] = json_dict['recognitionResults'][0]['width']
        w_h[1] = json_dict['recognitionResults'][0]['height']

        qp.setPen(QPen(QColor(200, 0, 0),  1, QtCore.Qt.SolidLine))
        for line in json_line:
            line_list.append(line['boundingBox'])
            need_marking = self.marking(line['boundingBox'])
            for word in line['words']:
                bbox_list.append(word)
                line_position = word['boundingBox']

                if need_marking == True:
                    brush = QBrush(QColor(200, 200, 0, 100), QtCore.Qt.SolidPattern)
                    qp.setBrush(brush)

                if abs(json_rotate-360)<45 or abs(json_rotate)<45:
                    self.rotate_type = 0
                    points = [QPoint(line_position[0]*self.resize_ratio, line_position[1]*self.resize_ratio), QPoint(line_position[2]*self.resize_ratio, line_position[3]*self.resize_ratio), QPoint(line_position[4]*self.resize_ratio, line_position[5]*self.resize_ratio), QPoint(line_position[6]*self.resize_ratio, line_position[7]*self.resize_ratio)]
                    qp.drawPolygon(QPolygon(points))                  
                    
                elif abs(json_rotate-90)<45:
                    self.rotate_type = 1
                    points = [QPoint(line_position[1]*self.resize_ratio, self.label.height()-line_position[0]*self.resize_ratio), QPoint(line_position[3]*self.resize_ratio, self.label.height()-line_position[2]*self.resize_ratio), QPoint(line_position[5]*self.resize_ratio, self.label.height()-line_position[4]*self.resize_ratio), QPoint(line_position[7]*self.resize_ratio, self.label.height()-line_position[6]*self.resize_ratio)]
                    qp.drawPolygon(QPolygon(points))              
                    
                elif abs(json_rotate-270)<45:
                    self.rotate_type = 2
                    points = [QPoint(line_position[1]*self.resize_ratio, line_position[0]*self.resize_ratio), QPoint(line_position[3]*self.resize_ratio, line_position[2]*self.resize_ratio), QPoint(line_position[5]*self.resize_ratio, line_position[4]*self.resize_ratio), QPoint(line_position[7]*self.resize_ratio, line_position[6]*self.resize_ratio)]
                    qp.drawPolygon(QPolygon(points))             
                    
                elif abs(json_rotate-180)<45:
                    self.rotate_type = 3
                    points = [QPoint(line_position[0]*self.resize_ratio, self.label.height()-line_position[1]*self.resize_ratio), QPoint(line_position[2]*self.resize_ratio, self.label.height()-line_position[3]*self.resize_ratio), QPoint(line_position[4]*self.resize_ratio, self.label.height()-line_position[5]*self.resize_ratio), QPoint(line_position[6]*self.resize_ratio, self.label.height()-line_position[7]*self.resize_ratio)]
                    qp.drawPolygon(QPolygon(points))

                if need_marking == True:
                    qp.setBrush(QtCore.Qt.NoBrush)
                    need_marking = False

############################################
    def marking(self,line_bbox):
        if self.cursor == 0:
            return False
        global w_h
        global embedding
        line_x = abs(line_bbox[0] + line_bbox[4])/2.0
        line_x = line_x/w_h[0]
        line_y = abs(line_bbox[1] + line_bbox[5])/2.0
        line_y = line_y/w_h[1]     
        true_embedding_index = np.where(np.array(list(map(lambda x:x[-2] == line_x and x[-1] == line_y,embedding))) == True)
        true_embedding = embedding[true_embedding_index]
        if len(true_embedding) == 0:
            return False
        result = self.kmean.predict(true_embedding)
        max_index = np.array(list(map(lambda x:x == max(self.max_count),self.max_count)))
        max_index = np.where(max_index == True)[0]
        for class_index in max_index:
            is_true = np.array(list(map(lambda x:x == class_index,result)))
            if len(np.where(is_true == True)[0]) != 0:
                return True
        return False
                    
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
