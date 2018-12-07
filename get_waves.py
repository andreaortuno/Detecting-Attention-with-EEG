import matplotlib.pyplot as plt
import sys; sys.path.append('..') # help python find cyton.py relative to scripts folder
from openbci import cyton as bci
import logging
import time
import numpy as np
from scipy import signal
import threading

plt.ion()
class DynamicUpdate():
    #Suppose we know the x range
    min_x = 0
    max_x = 1000
    step = 10

    def on_launch(self):
        #Set up plot
        self.fig = plt.figure()

        self.ax1 = self.fig.add_subplot()

        [self.delta, self.theta, self.alpha, self.beta, self.gamma] = plt.bar([x for x in range(5)],  [0,0,0,0,0])

        # self.ax1 = self.fig.add_subplot(8,1,1)
        # self.ax2 = self.fig.add_subplot(8,1,2)
        # self.ax3 = self.fig.add_subplot(8,1,3)
        # self.ax4 = self.fig.add_subplot(8,1,4)
        # self.ax5 = self.fig.add_subplot(8,1,5)
        # self.ax6 = self.fig.add_subplot(8,1,6)
        # self.ax7 = self.fig.add_subplot(8,1,7)
        # self.ax8 = self.fig.add_subplot(8,1,8)
        #
        # self.lines1, = self.ax1.plot([],[])
        # self.lines2, = self.ax2.plot([],[])
        # self.lines3, = self.ax3.plot([],[])
        # self.lines4, = self.ax4.plot([],[])
        # self.lines5, = self.ax5.plot([],[])
        # self.lines6, = self.ax6.plot([],[])
        # self.lines7, = self.ax7.plot([],[])
        # self.lines8, = self.ax8.plot([],[])
        # #Autoscale on unknown axis and known lims on the other
        # self.ax1.set_autoscaley_on(True)
        # self.ax2.set_autoscaley_on(True)
        # self.ax3.set_autoscaley_on(True)
        # self.ax4.set_autoscaley_on(True)
        # self.ax5.set_autoscaley_on(True)
        # self.ax6.set_autoscaley_on(True)
        # self.ax7.set_autoscaley_on(True)
        # self.ax8.set_autoscaley_on(True)
        #
        # self.ax1.set_xlim(self.min_x, self.max_x)
        # self.ax2.set_xlim(self.min_x, self.max_x)
        # self.ax3.set_xlim(self.min_x, self.max_x)
        # self.ax4.set_xlim(self.min_x, self.max_x)
        # self.ax5.set_xlim(self.min_x, self.max_x)
        # self.ax6.set_xlim(self.min_x, self.max_x)
        # self.ax7.set_xlim(self.min_x, self.max_x)
        # self.ax8.set_xlim(self.min_x, self.max_x)

        #Other stuff
        #self.ax1.grid()
        ...

    def plot_data(self, sample):
        global bci_data
        Scale_Factor= (4.5/ 24) /(2^23 - 1)
        data = [i * Scale_Factor for i in sample.channel_data]
        bci_data.append(data)
        filtered_data = self.filters(bci_data)

        #Update data (with the new _and_ the old points)
        try:
            self.lines1.set_ydata(np.array(filtered_data)[:,0][-self.max_x:])
            self.lines1.set_xdata(range(len(np.array(filtered_data)[:,0][-self.max_x:])))
            self.lines2.set_ydata(np.array(filtered_data)[:,1][-self.max_x:])
            self.lines2.set_xdata(range(len(np.array(filtered_data)[:,1][-self.max_x:])))
            self.lines3.set_ydata(np.array(filtered_data)[:,2][-self.max_x:])
            self.lines3.set_xdata(range(len(np.array(filtered_data)[:,2][-self.max_x:])))
            self.lines4.set_ydata(np.array(filtered_data)[:,3][-self.max_x:])
            self.lines4.set_xdata(range(len(np.array(filtered_data)[:,3][-self.max_x:])))
            self.lines5.set_ydata(np.array(filtered_data)[:,4][-self.max_x:])
            self.lines5.set_xdata(range(len(np.array(filtered_data)[:,4][-self.max_x:])))
            self.lines6.set_ydata(np.array(filtered_data)[:,5][-self.max_x:])
            self.lines6.set_xdata(range(len(np.array(filtered_data)[:,5][-self.max_x:])))
            self.lines7.set_ydata(np.array(filtered_data)[:,6][-self.max_x:])
            self.lines7.set_xdata(range(len(np.array(filtered_data)[:,6][-self.max_x:])))
            self.lines8.set_ydata(np.array(filtered_data)[:,7][-self.max_x:])
            self.lines8.set_xdata(range(len(np.array(filtered_data)[:,7][-self.max_x:])))

        except:
            self.lines1.set_ydata(np.array(filtered_data)[:,0])
            self.lines1.set_xdata(range(len(np.array(filtered_data)[:,0])))
            self.lines2.set_ydata(np.array(filtered_data)[:,1])
            self.lines2.set_xdata(range(len(np.array(filtered_data)[:,1])))
            self.lines3.set_ydata(np.array(filtered_data)[:,2])
            self.lines3.set_xdata(range(len(np.array(filtered_data)[:,2])))
            self.lines4.set_ydata(np.array(filtered_data)[:,3])
            self.lines4.set_xdata(range(len(np.array(filtered_data)[:,3])))
            self.lines5.set_ydata(np.array(filtered_data)[:,4])
            self.lines5.set_xdata(range(len(np.array(filtered_data)[:,4])))
            self.lines6.set_ydata(np.array(filtered_data)[:,5])
            self.lines6.set_xdata(range(len(np.array(filtered_data)[:,5])))
            self.lines7.set_ydata(np.array(filtered_data)[:,6])
            self.lines7.set_xdata(range(len(np.array(filtered_data)[:,6])))
            self.lines8.set_ydata(np.array(filtered_data)[:,7])
            self.lines8.set_xdata(range(len(np.array(filtered_data)[:,7])))


        #Need both of these in order to rescale
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax3.relim()
        self.ax3.autoscale_view()
        self.ax4.relim()
        self.ax4.autoscale_view()
        self.ax5.relim()
        self.ax5.autoscale_view()
        self.ax6.relim()
        self.ax6.autoscale_view()
        self.ax7.relim()
        self.ax7.autoscale_view()
        self.ax8.relim()
        self.ax8.autoscale_view()
        #We need to draw *and* flush
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    #Example
    def __call__(self):
        def Data_stream(sample):
            #self.plot_data(sample)
            self.get_alpha(sample)
        self.on_launch()
        global bci_data
        bci_data = []
        #connect to board
        port = 'COM5'
        baud = 115200
        logging.basicConfig(filename="test.log", format='%(asctime)s -%(levelname)s : %(message)s', level=logging.DEBUG)
        logging.info('--------------LOG START ------------')
        board = bci.OpenBCICyton(port=port, scaled_output=False, log=True)
        print("Board Instantiated")
        board.ser.write(str.encode('v'))
        time.sleep(10)
        board.start_streaming(Data_stream)
        board.print_bytes_in()
        return xdata, ydata

    def filters(self, bci_data):
        def bandpass(start, stop, data, fs = 200):
            bp_Hz = np.array([start, stop])
            b, a = signal.butter(5, bp_Hz / (fs / 2.0), btype='bandpass')
            return signal.lfilter(b, a, data, axis=0)

        def notch(val, data, fs= 200):
            notch_freq_Hz = np.array([float(val)])
            for freq_Hz in np.nditer(notch_freq_Hz):
                bp_stop_Hz = freq_Hz + 3.0 * np.array([-1, 1])
                b, a = signal.butter(3, bp_stop_Hz / (fs / 2.0), 'bandstop')
                fin = data = signal.lfilter(b, a, data)
            return fin
        data = np.array(bci_data)
        fs = 1000

        notch_channels = []
        for i in range(8):
            notch_channels.append(notch(60,data[:,i], fs = fs))

        #applied bandpass filter = 5-50 and notch = 60
        bandpass_notch_channels = []
        band = (15,50)
        for i in range(8):
            bandpass_notch_channels.append(bandpass(band[0],band[1],notch_channels[i], fs = fs))
        return np.array(bandpass_notch_channels).T

    def get_alpha(self, sample):

        def rose_funtion(waves):
            alpha = waves[0]
            beta = waves[1]
            print(alpha, beta)

        fs = 250                               # Sampling rate (250 Hz)
        time = 60  # 60 sec of data b/w 0.0-100.0
        num_vals = int((1/fs)/time)

        global bci_data
        Scale_Factor= (4.5/ 24) /(2^23 - 1)/1000000
        data = [i * Scale_Factor for i in sample.channel_data]
        bci_data.append(data)
        filtered_data = self.filters(bci_data)
        if len(filtered_data[:,0]) >  num_vals:
            filtered_data = filtered_data[-num_vals:]
        # Get real amplitudes of FFT (only in postive frequencies)
        fft_vals1 = np.absolute(np.fft.rfft(filtered_data[:,0]))

        # Get frequencies for amplitudes in Hz
        fft_freq1 = np.fft.rfftfreq(len(filtered_data[:,0]), 1.0/fs)

        # Define EEG bands
        eeg_bands = {'Delta': (0, 4),
                     'Theta': (4, 8),
                     'Alpha': (8, 12),
                     'Beta': (12, 30),
                     'Gamma': (30, 45)}

        # Take the mean of the fft amplitude for each EEG band
        eeg_band_fft = dict()
        for band in eeg_bands:
            freq_ix = np.where((fft_freq1 >= eeg_bands[band][0]) &
                               (fft_freq1 <= eeg_bands[band][1]))[0]
            eeg_band_fft[band] = np.mean(fft_vals1[freq_ix])

        labels = eeg_bands.keys()
        self.delta.set_height(eeg_band_fft['Delta'])
        self.theta.set_height(eeg_band_fft['Theta'])
        self.alpha.set_height(eeg_band_fft['Alpha'])
        self.beta.set_height(eeg_band_fft['Beta'])
        self.gamma.set_height(eeg_band_fft['Gamma'])

        thread = threading.Thread(target=rose_funtion, args= ([eeg_band_fft['Alpha'], eeg_band_fft['Beta']],))
        thread.start()

        #plt.bar([x for x in range(len(labels))], values)
        plt.xticks([x for x in range(len(labels))], ([y for y in labels]))
        #plt.ylim(0, max([eeg_band_fft['Delta'],eeg_band_fft['Theta'],eeg_band_fft['Alpha'],eeg_band_fft['Beta'],eeg_band_fft['Gamma']]))

        plt.ylim(0, 1)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

d = DynamicUpdate()
d()
