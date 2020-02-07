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
import pdb

image_list = []
name_list = []
bbox_list = []
put_in_json = []

class ExampleApp(QtWidgets.QMainWindow, labeling.Ui_MainWindow):
    def __init__(self, parent=None):
        super(ExampleApp, self).__init__(parent)
        self.setupUi(self)
        #read picture
        for filename in glob.glob('./train/train/Input/*.jpg'): ##set input dir 
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
        self.changeStatus(self,name_list[self.cursor])
        
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
        self.write_json()
        self.cursor += 1
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
        self.changeStatus(self,name_list[self.cursor])
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
            output_format['elements'].append(bbox)##set output dir
        output_format['text'] = text[:-1]
        output_dir = './output/' + name_list[self.cursor][:-3] + 'json'
        with open(output_dir, 'w') as outfile:
            json.dump(output_format, outfile)
        

############################################        
    def reset(self):
        #print(put_in_json)
        put_in_json.clear()
        print('reset!')

############################################
    def bbox(self,qp):
        json_path = './train/train/Input/' + name_list[self.cursor][:-3] + 'json'##set input dir 
        json_dict = json.load(open(json_path, 'r'))
        json_rotate = json_dict['recognitionResults'][0]['clockwiseOrientation']
        json_line = json_dict['recognitionResults'][0]['lines']
        qp.setPen(QPen(QColor(200, 0, 0),  1, QtCore.Qt.SolidLine))
        for line in json_line:
            for word in line['words']:
                bbox_list.append(word)
                line_position = word['boundingBox']
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
