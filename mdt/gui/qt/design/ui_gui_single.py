# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui_single.ui'
#
# Created by: PyQt5 UI code generator 5.4.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(950, 650)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/gui_single/logo.gif"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.MainTabs = QtWidgets.QTabWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.MainTabs.sizePolicy().hasHeightForWidth())
        self.MainTabs.setSizePolicy(sizePolicy)
        self.MainTabs.setObjectName("MainTabs")
        self.runModelTab = QtWidgets.QWidget()
        self.runModelTab.setObjectName("runModelTab")
        self.MainTabs.addTab(self.runModelTab, "")
        self.generateBrainMaskTab = QtWidgets.QWidget()
        self.generateBrainMaskTab.setObjectName("generateBrainMaskTab")
        self.MainTabs.addTab(self.generateBrainMaskTab, "")
        self.generateROIMaskTab = QtWidgets.QWidget()
        self.generateROIMaskTab.setObjectName("generateROIMaskTab")
        self.MainTabs.addTab(self.generateROIMaskTab, "")
        self.generateProtocolTab = QtWidgets.QWidget()
        self.generateProtocolTab.setObjectName("generateProtocolTab")
        self.MainTabs.addTab(self.generateProtocolTab, "")
        self.viewResultsTab = QtWidgets.QWidget()
        self.viewResultsTab.setObjectName("viewResultsTab")
        self.MainTabs.addTab(self.viewResultsTab, "")
        self.verticalLayout.addWidget(self.MainTabs)
        self.loggingTextBox = QtWidgets.QPlainTextEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.loggingTextBox.sizePolicy().hasHeightForWidth())
        self.loggingTextBox.setSizePolicy(sizePolicy)
        self.loggingTextBox.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.loggingTextBox.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.loggingTextBox.setReadOnly(True)
        self.loggingTextBox.setPlainText("")
        self.loggingTextBox.setTabStopWidth(80)
        self.loggingTextBox.setObjectName("loggingTextBox")
        self.verticalLayout.addWidget(self.loggingTextBox)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 950, 27))
        self.menubar.setObjectName("menubar")
        self.menuMenu = QtWidgets.QMenu(self.menubar)
        self.menuMenu.setObjectName("menuMenu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.menuMenu.addSeparator()
        self.menuMenu.addAction(self.actionExit)
        self.menubar.addAction(self.menuMenu.menuAction())

        self.retranslateUi(MainWindow)
        self.MainTabs.setCurrentIndex(3)
        self.actionExit.triggered.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Maastricht Diffusion Toolbox"))
        self.MainTabs.setTabText(self.MainTabs.indexOf(self.runModelTab), _translate("MainWindow", "Run model"))
        self.MainTabs.setTabText(self.MainTabs.indexOf(self.generateBrainMaskTab), _translate("MainWindow", "Generate brain mask"))
        self.MainTabs.setTabText(self.MainTabs.indexOf(self.generateROIMaskTab), _translate("MainWindow", "Generate ROI mask"))
        self.MainTabs.setTabText(self.MainTabs.indexOf(self.generateProtocolTab), _translate("MainWindow", "Generate protocol file"))
        self.MainTabs.setTabText(self.MainTabs.indexOf(self.viewResultsTab), _translate("MainWindow", "View results"))
        self.menuMenu.setTitle(_translate("MainWindow", "&File"))
        self.actionExit.setText(_translate("MainWindow", "&Quit"))
        self.actionExit.setShortcut(_translate("MainWindow", "Ctrl+Q"))

from . import gui_single_rc
