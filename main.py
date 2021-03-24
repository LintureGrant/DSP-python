import numpy as np
from scipy import signal

import FilterDesigner#布局文件
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *
from PyQt5.QtWidgets import *

import sys
import wave
import time
import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

class ProcessFunction(object):  ##这里负责写一些数字信号处理的方法
    def Audio_TimeDomain(self,feature):  ##时域
        f = wave.open(feature.path,"rb")
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        # nchannels通道数
        # sampwidth量化位数
        # framerate采样频率
        # nframes采样点数
        str_data = f.readframes(nframes)
        f.close()
        # 将字符串转换为数组，得到一维的short类型的数组
        wave_data = np.fromstring(str_data, dtype=np.short)
        # 赋值的归一化
        wave_data = wave_data * 1.0 / (max(abs(wave_data)))
        # 整合左声道和右声道的数据
        wave_data = np.reshape(wave_data, [nframes, nchannels])
        # 最后通过采样点数和取样频率计算出每个取样的时间
        time = np.arange(0, nframes) * (1.0 / framerate)

        feature.textBrowser_2.append("AUDIO INFO:   Number of channel: " + str(nchannels))
        feature.textBrowser_2.append("AUDIO INFO:   Sampling Frequency: " + str(framerate)+" Hz")
        feature.textBrowser_2.append("AUDIO INFO:   Sampling number: " + str(nframes))
        feature.textBrowser_2.append("AUDIO INFO:   Sampling duration: " + str(nframes/framerate)+" seconds")

        ax = feature.fig5.add_subplot(111)
        ###进度条显示******
        feature.progressBar.setValue(10)
        #***************
        #调整图像大小
        ax.cla()  # TODO:删除原图，让画布上只有新的一次的图
        #ax.clear()
        ax.plot(time, wave_data[:, 0])
        ax.set_title('Normalized Magnitude')
        ax.set_xlabel('Time [sec]')

        feature.fig5.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
        feature.canvas5.draw()  # TODO:这里开始绘制

        feature.progressBar.setValue(20)

    def Audio_FrequencyDomain(self,feature):
        #*********************STFT图像绘制*****************************
        sampling_freq, audio = wavfile.read(feature.path)
        T = 20  # 短时傅里叶变换的时长 单位 ms
        fs = sampling_freq  # 采样频率是这么多，那么做出来的频谱宽度就是这么多
        N = len(audio)  # 采样点的个数
        audio = audio * 1.0 / (max(abs(audio)))

        # 计算并绘制STFT的大小
        f, t, Zxx = signal.stft(audio, fs, nperseg=T * fs / 1000)

        feature.progressBar.setValue(30)

        ax = feature.fig7.add_subplot(111)
        feature.fig7.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
        ax.cla()  # TODO:删除原图，让画布上只有新的一次的图
        # ax=plt.figure()
        ax.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=0.1)
        ax.set_title('STFT Magnitude')
        #feature.fig6.colorbar(ax=ax)
        #feature.fig6.colorbar(feature.fig6)
        ####还存在的问题是colorbar显示不了的问题####
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Frequency [Hz]')
        feature.canvas7.draw()  # TODO:这里开始绘制
        feature.progressBar.setValue(40)
        #**************************************************************

        # *******************FFT图像绘制*********************************
        fft_signal = np.fft.fft(audio)
        fft_signal = abs(fft_signal)
        # 建立频率轴

        fft_signal = np.fft.fftshift(fft_signal)

        freqInteral = (sampling_freq / len(fft_signal))  ###频率轴的间隔
        Freq = np.arange(-sampling_freq / 2, sampling_freq / 2, sampling_freq / len(fft_signal))
        feature.progressBar.setValue(50)
        #
        highFreq=  (np.argmax(fft_signal[int(len(fft_signal) / 2):len(fft_signal)]) )*freqInteral
        feature.textBrowser_2.append("FFT INFO:   Highest frequency: "+str(highFreq))


        ax = feature.fig6.add_subplot(111)
        # 调整图像大小
        ax.cla()  # TODO:删除原图，让画布上只有新的一次的图
        ax.plot(Freq, fft_signal, color='red')
        ax.set_title('FFT Figure')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Am')
        feature.fig6.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
        feature.canvas6.draw()  # TODO:这里开始绘制

        feature.progressBar.setValue(60)

    def Audio_YuPuDomain(self, feature):
        #*******************语谱图绘制***********************************
        f = wave.open(feature.path, "rb")
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        strData = f.readframes(nframes)  # 读取音频，字符串格式
        waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
        waveData = waveData * 1.0 / (max(abs(waveData)))  # wave幅值归一化
        waveData = np.reshape(waveData, [nframes, nchannels]).T
        f.close()
        feature.progressBar.setValue(70)

        ax = feature.fig8.add_subplot(111)
        feature.fig8.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
        ax.cla()  # TODO:删除原图，让画布上只有新的一次的图
        # ax=plt.figure()
        ax.specgram(waveData[0],Fs = framerate, scale_by_freq = True, sides = 'default')
        ax.set_title('Spectrogram')
        # feature.fig6.colorbar(ax=ax)
        # feature.fig6.colorbar(feature.fig6)
        ####还存在的问题是colorbar显示不了的问题####
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Frequency [Hz]')
        feature.canvas8.draw()  # TODO:这里开始绘制
        feature.progressBar.setValue(80)

    def IIR_Designer(self,feature):
        if str(feature.iirType)=='Butterworth':##巴特沃斯 双线性变换法间接设计模拟滤波器
            print(str(feature.iirType))
            fs=float(feature.fs)
            wp=float(feature.An_wp)*(2*np.pi/fs)
            wst=float(feature.An_wst) * (2 * np.pi / fs)
            omiga_p = (2 * fs) * np.tan(wp / 2)
            omiga_st = (2 * fs) * np.tan(wst / 2)
            feature.Rp=float(feature.Rp)
            feature.As=float(feature.As)
            N, Wn = signal.buttord(omiga_p, omiga_st, feature.Rp, feature.As, True)
            feature.filts = signal.lti(*signal.butter(N, Wn, btype=str(feature.filterType),
                                              analog=True))
            feature.filtz = signal.lti(*signal.bilinear(feature.filts.num, feature.filts.den, fs))

            wz, hz = signal.freqz(feature.filtz.num, feature.filtz.den)

            ax = feature.fig1.add_subplot(111)
            ax.cla()  # TODO:删除原图，让画布上只有新的一次的图
            #ax.clear()
            ax.semilogx(wz * fs / (2 * np.pi), 20 * np.log10(np.abs(hz).clip(1e-15)),
                     label=r'$|H_z(e^{j \omega})|$')
            ax.set_xlabel('Hz')
            ax.set_ylabel('dB')
            ax.set_title('Butterworth')
            feature.fig1.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
            feature.canvas1.draw()  # TODO:这里开始绘制

        if str(feature.iirType) == 'Chebyshev I':  ##切比雪夫一型
            print(str(feature.iirType))
            fs=float(feature.fs)
            wp=float(feature.An_wp)*(2*np.pi/fs)
            wst=float(feature.An_wst) * (2 * np.pi / fs)
            omiga_p = (2 * fs) * np.tan(wp / 2)
            omiga_st = (2 * fs) * np.tan(wst / 2)
            feature.Rp=float(feature.Rp)
            feature.As=float(feature.As)
            N, Wn = signal.cheb1ord(omiga_p, omiga_st, feature.Rp, feature.As, True)
            filts = signal.lti(*signal.cheby1(N, 0.1,Wn, btype=str(feature.filterType),
                                              analog=True))##切比雪夫是还有一个纹波参数
            filtz = signal.lti(*signal.bilinear(filts.num, filts.den, fs))

            wz, hz = signal.freqz(filtz.num, filtz.den)

            ax = feature.fig1.add_subplot(111)
            ax.cla()  # TODO:删除原图，让画布上只有新的一次的图
            #ax.clear()
            ax.semilogx(wz * fs / (2 * np.pi), 20 * np.log10(np.abs(hz).clip(1e-15)),
                     label=r'$|H_z(e^{j \omega})|$')
            ax.set_xlabel('Hz')
            ax.set_ylabel('dB')
            ax.set_title('Chebyshev I')
            feature.fig1.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
            feature.canvas1.draw()  # TODO:这里开始绘制

    def apply_IIR(self,feature):
        f = wave.open(feature.path,"rb")
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        # nchannels通道数
        # sampwidth量化位数
        # framerate采样频率
        # nframes采样点数
        str_data = f.readframes(nframes)
        f.close()
        # 将字符串转换为数组，得到一维的short类型的数组
        wave_data = np.fromstring(str_data, dtype=np.short)
        # 赋值的归一化
        wave_data = wave_data * 1.0 / (max(abs(wave_data)))
        # 整合左声道和右声道的数据
        wave_data = np.reshape(wave_data, [nframes, nchannels])

        # 最后通过采样点数和取样频率计算出每个取样的时间
        time = np.arange(0, nframes) * (1.0 / framerate)
        #print(time)
        t = np.linspace(0, nframes/ framerate, nframes, endpoint=False)


        ##feature.precessed_Audio = feature.filtz.output(wave_data[:, 0], time, X0=None)
        u = (np.cos(2 * np.pi * 4 * t) + 0.6 * np.sin(2 * np.pi * 40 * t) +
             0.5 * np.cos(2 * np.pi * 80 * t))
        print(type(u))
        feature.tout ,  feature.yout ,  feature.xout =  signal.lsim(feature.filts, wave_data[:, 0] , t ,X0=None)
        #
        #feature.tout, feature.yout, feature.xout =feature.filts.output(wave_data[:, 0], t, X0=None)
        print(feature.yout)
        ax = feature.fig2.add_subplot(111)
        #调整图像大小
        ax.cla()  # TODO:删除原图，让画布上只有新的一次的图
        #ax.clear()
        ax.plot(feature.tout, feature.yout)
        ax.set_title('Passed Filter')
        ax.set_xlabel('Time [sec]')

        feature.fig2.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
        feature.canvas2.draw()  # TODO:这里开始绘制

        ##绘制出时域的图像之后，再到频率分析
        #FFT变换#
        fft_signal = np.fft.fft(feature.yout)
        fft_signal = np.fft.fftshift(abs(fft_signal))
        # 建立频率轴
        Freq = np.arange(-framerate / 2, framerate / 2, framerate / len(fft_signal))

        ####绘图######
        ax = feature.fig4.add_subplot(111)
        # 调整图像大小
        ax.cla()  # TODO:删除原图，让画布上只有新的一次的图
        ax.plot(Freq, fft_signal, color='red')
        ax.set_title('FFT Figure')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Am')
        feature.fig4.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
        feature.canvas4.draw()  # TODO:这里开始绘制


        #feature.precessed_Audio=feature.filtz.output(wave_data,time,X0=None)#求系统的零状态响应
        # feature.precessed_Audio =feature.precessed_Audio.tostring()
        # feature.process_flag=1#标志位为1，代表处理好了，否则的话就代表没有



#MyMainForm类里面负责写窗体的一些逻辑控制以及方法调用
#
#
class MyMainForm(QMainWindow, FilterDesigner.Ui_FilterDesigner):#因为界面py文件和逻辑控制py文件分开的，所以在引用的时候要加上文件名再点出对象
    def __init__(self, parent=None):
        super(MyMainForm, self).__init__(parent)#从父类哪里继承下来
        self.Process=ProcessFunction()#process对象包含了所有的信号处理函数及其画图

        self.setupUi(self)#setupUi是Ui_FilterDesigner类里面的一个方法，这里的self是两个父类的子类的一个实例
        self.progressBar.setValue(0)#进度条初始化为0
        #**************播放器的设定**********************
        self.isPlay=0
        self.player = QMediaPlayer()
        self.player_IIR = QMediaPlayer()#定义两个对象出来，这个负责播放处理过后的
        self.horizontalSlider_2.sliderMoved[int].connect(lambda: self.player.setPosition(self.horizontalSlider_2.value()))
        self.horizontalSlider_2.setStyle(QStyleFactory.create('Fusion'))

        self.timer = QTimer(self)
        self.timer.start(1000)##定时器设定为1s，超时过后链接到playRefresh刷新页面
        self.timer.timeout.connect(self.playRefresh)##



        #**************菜单栏的事件绑定*******************
        self.action_2.triggered.connect(self.onFileOpen)##菜单栏的action打开文件
        self.actionExit.triggered.connect(self.close)#菜单栏的退出action
        self.Timelayout_()##时间域的四个图窗布局
        self.Iirlayout_()##IIR设计界面的四个图窗布局


        #**************第一个界面的事件绑定配置*************
        self.dial.setValue(20)#默认音量大小为20
        self.dial.valueChanged.connect(self.changeVoice)##音量圆盘控制事件绑定,如果值被改变就调起事件
        self.pushButton_analyse.clicked.connect(self.Analyse_btn_start)  # 给pushButton_3添加一个点击事件
        self.pushButton_3.clicked.connect(self.palyMusic)

        #**************第二个界面的事件绑定配置*************
        self.pushButton.clicked.connect(self.desigenIIR)#点击开始设计IIR滤波器按钮之后，调用函数
        self.pushButton_2.clicked.connect(self.applyIIR)#点击应用滤波器
        self.horizontalSlider_4.sliderMoved[int].connect(lambda: self.player_IIR.setPosition(self.horizontalSlider_4.value()))
        self.horizontalSlider_4.setStyle(QStyleFactory.create('Fusion'))

    def Timelayout_(self):
        self.fig5 = plt.figure()
        self.canvas5 = FigureCanvas(self.fig5)
        layout = QVBoxLayout()  # 垂直布局
        layout.addWidget(self.canvas5)
        self.graphicsView_5.setLayout(layout)  # 设置好布局之后调用函数

        self.fig6 = plt.figure()
        self.canvas6 = FigureCanvas(self.fig6)
        layout = QVBoxLayout()  # 垂直布局
        layout.addWidget(self.canvas6)
        self.graphicsView_6.setLayout(layout)  # 设置好布局之后调用函数

        self.fig7 = plt.Figure()
        self.canvas7 = FigureCanvas(self.fig7)
        layout = QVBoxLayout()  # 垂直布局
        layout.addWidget(self.canvas7)
        self.graphicsView_7.setLayout(layout)  # 设置好布局之后调用函数

        self.fig8 = plt.Figure()
        self.canvas8 = FigureCanvas(self.fig8)
        layout = QVBoxLayout()  # 垂直布局
        layout.addWidget(self.canvas8)
        self.graphicsView_8.setLayout(layout)  # 设置好布局之后调用函数

    def Iirlayout_(self):
        self.fig1 = plt.figure()
        self.canvas1 = FigureCanvas(self.fig1)
        layout = QVBoxLayout()  # 垂直布局
        layout.addWidget(self.canvas1)
        self.graphicsView.setLayout(layout)  # 设置好布局之后调用函数

        self.fig2 = plt.figure()
        self.canvas2 = FigureCanvas(self.fig2)
        layout = QVBoxLayout()  # 垂直布局
        layout.addWidget(self.canvas2)
        self.graphicsView_2.setLayout(layout)  # 设置好布局之后调用函数

        self.fig3 = plt.Figure()
        self.canvas3 = FigureCanvas(self.fig3)
        layout = QVBoxLayout()  # 垂直布局
        layout.addWidget(self.canvas3)
        self.graphicsView_3.setLayout(layout)  # 设置好布局之后调用函数

        self.fig4 = plt.Figure()
        self.canvas4 = FigureCanvas(self.fig4)
        layout = QVBoxLayout()  # 垂直布局
        layout.addWidget(self.canvas4)
        self.graphicsView_4.setLayout(layout)  # 设置好布局之后调用函数

    ###############对应的一些触发方法######################
    def onFileOpen(self): ##打开文件
        self.path, _ = QFileDialog.getOpenFileName(self, '打开文件', '', '音乐文件 (*.wav)')

        if self.path:##选中文件之后就选中了需要播放的音乐，并同时显示出来
            self.isPlay=0#每次打开文件的时候就需要暂停播放，无论是否在播放与否
            self.player.pause()
            self.horizontalSlider_2.setMinimum(0)
            self.horizontalSlider_2.setMaximum(self.player.duration())
            self.horizontalSlider_2.setValue(self.horizontalSlider_2.value() + 1000)
            self.horizontalSlider_2.setSliderPosition(0)
            self.label_17.setText("Current File:  "+os.path.basename(self.path))
            self.player.setMedia(QMediaContent(QUrl(self.path)))##选中需要播放的音乐

    def Analyse_btn_start(self):##这里对应的是打开文件，并点击按钮
        try:
            if self.path:##要必须在打开文件之后才允许进行处理
                self.textBrowser_2.append("*********This file :"+str(os.path.basename(self.path))+"*********")
                self.progressBar.setValue(0)  ##每次允许处理时进度条归0
                self.Process.Audio_TimeDomain(self)  ##把实例传入进去
                self.Process.Audio_FrequencyDomain(self)
                self.Process.Audio_YuPuDomain(self)
                self.progressBar.setValue(100)
                self.textBrowser_2.append("Analyse Succeed!")
                self.textBrowser_2.append("---------  "+str(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))+"  ---------")

        except Exception as e:
            print(e)
            self.textBrowser_2.setText("There are some errors occuring when programme trying to open file")

    def palyMusic(self):
        try:
            if self.path:#这个path是当前的路径，如果path变了，那么就意味着更换了文件
                if not self.isPlay:##如果isPaly=0，那就说明播放器并没有打开，且此时按下了播放按钮，就开始播放
                    self.horizontalSlider_2.setValue(0)
                    self.player.play()
                    self.isPlay=1##播放之后同时置为1，代表播放器目前正在播放
                else:
                    self.player.pause()
                    self.isPlay = 0  ##暂停之后同时置为0，代表播放器目前没有播放
        except Exception as e:
            print(e)
            self.textBrowser_2.setText("There are some errors occuring when playing audio")

    def playRefresh(self):
        if self.isPlay:
            #print(self.player.duration())
            self.horizontalSlider_2.setMinimum(0)
            self.horizontalSlider_2.setMaximum(self.player.duration())
            self.horizontalSlider_2.setValue(self.horizontalSlider_2.value() + 1000)
        self.label_14.setText(time.strftime('%M:%S', time.localtime(self.player.position() / 1000)))
        self.label_15.setText(time.strftime('%M:%S', time.localtime(self.player.duration() / 1000)))

    def changeVoice(self):
        #print(self.dial.value())
        self.player.setVolume(self.dial.value())

    def desigenIIR(self):
        ###获取到输入参数：滤波器四个指标
        try:
            self.An_wp=self.lineEdit_3.text()
            self.An_wst=self.lineEdit_2.text()
            self.Rp=self.lineEdit.text()
            self.As=self.lineEdit_4.text()

            self.fs=self.lineEdit_5.text()

            self.filterType=self.comboBox_2.currentText()
            self.iirType=self.comboBox_3.currentText()
            self.Process.IIR_Designer(self)
        except Exception as e:
            print(e)

    def applyIIR(self):
        try:
            self.process_flag=0
            self.Process.apply_IIR(self)
        except Exception as e:
            print(e)
        # if self.process_flag:#如果处理好了
        #     print(self.process_flag)
            #self.player.setMedia(QMediaContent(self.precessed_Audio))  ##选中需要播放的音乐
            # self.isPlay = 0
            # self.player.pause()#暂停另外的播放器
            # self.player_IIR.pause()
            # self.horizontalSlider_4.setMinimum(0)
            # self.horizontalSlider_4.setMaximum(self.player_IIR.duration())
            # self.horizontalSlider_4.setValue(self.horizontalSlider_4.value() + 1000)
        #     self.horizontalSlider_4.setSliderPosition(0)
        #     self.label_21.setText("Processed Audio: " + os.path.basename(self.path))
        #     self.player.setMedia(self.precessed_Audio)  ##选中需要播放的音乐



if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainForm()
    myWin.show()
    sys.exit(app.exec_())
