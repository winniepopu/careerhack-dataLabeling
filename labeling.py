# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './untitled.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(900, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 500, 800))
        self.label.setText("")
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(650, 100, 115, 30))
        self.pushButton.setObjectName("pushButton")

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(650, 160, 115, 30))
        self.pushButton_2.setObjectName("pushButton_2")

        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(650, 40, 115, 30))
        self.pushButton_3.setObjectName("pushButton_3")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 35))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.pushButton.clicked.connect(MainWindow.next)
        self.pushButton_2.clicked.connect(MainWindow.reset)
        self.pushButton_3.clicked.connect(MainWindow.last)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.label2 = QtWidgets.QLabel(self.centralwidget)
        self.label2.setWordWrap(True)
        self.label2.setGeometry(QtCore.QRect(600, 200, 300, 500))
        self.label2.setFont(QtGui.QFont("Times", 18, QtGui.QFont.Bold))
        # self.label.setText("")/
        # self.label2 = QLabel(self)
        # self.label2.setFixedWidth(400)
        # self.label2.setFixedHeight(400)
        # self.label2.setAlignment(Qt.AlignCenter)
        self.label2.setText("init")

    def show_json(self,selected_jsons):
        self.label2.clear()
        self.label2.setText(selected_jsons)



    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "next"))
        self.pushButton_2.setText(_translate("MainWindow", "reset"))
        self.pushButton_3.setText(_translate("MainWindow", "last"))

    def changeStatus(self, MainWindow, filename):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", filename))
