# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'TestFrm.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(701, 441)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.BtnClose = QtWidgets.QPushButton(self.centralwidget)
        self.BtnClose.setGeometry(QtCore.QRect(450, 350, 75, 23))
        self.BtnClose.setObjectName("BtnClose")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 701, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        # self.BtnClose.clicked.connect(MainWindow.close)#将BtnClose的clicked信号和MainWindow的close槽连接
        self.BtnClose.clicked.connect(self.clickTime)  # 换成绑定自己编写的函数   <-----------
        self.clickCnt = 0  # 记录点击次数                                         <-----------
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.BtnClose.setText(_translate("MainWindow", "Close"))

    # 当点击按钮时触发，修改按钮名                                              <-----------
    def clickTime(self):
        self.clickCnt += 1  # 点击次数递增
        _translate = QtCore.QCoreApplication.translate
        self.BtnClose.setText(_translate("MainWindow", "点击次数 %d" % self.clickCnt))  # 将按钮名改成点击次数


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    formObj = QtWidgets.QMainWindow()  # 注意，这里和我们一开始创建窗体时使用的界面类型相同
    ui = Ui_MainWindow()
    ui.setupUi(formObj)
    formObj.show()
    sys.exit(app.exec_())
