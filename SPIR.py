import os

from datetime import datetime
from astropy.io import fits
from time import sleep
from matplotlib import pyplot as plt
import numpy as np

VERSION = 0.01
click_event = None

def onclick(event):
    global click_event 
    click_event = event

class SPIR:

    def __init__(self):
        self.loaded_files = []
        self.data = []
        self.headers = []

        self.mount_pointing = []
        self.airmass = []

        self.target = None

        self.latest_index = 0

        self.print_splash()
        self.set_data_dir()

        plt.style.use('dark_background')

        self.fig, self.axs = plt.subplots(2, 2)
        self.axs[1, 0].remove()
        self.axs[1, 0] = self.fig.add_subplot(2, 2, 3, projection='3d')

        self.axs[0, 0].set_title('Latest Image')
        self.axs[0, 1].set_title('SNR & Relative Flux')
        self.axs[1, 0].set_title('Mount Pointing')
        self.axs[1, 1].set_title('Airmass')
        self.fig.tight_layout(pad=3.0)
        self.fig.set_figheight(8)
        self.fig.set_figwidth(14)
        self.fig.canvas.mpl_connect('button_press_event', onclick)

        #man = plt.get_current_fig_manager()
        #man.canvas.set_window_title('SPIR v' + str(VERSION))

        plt.ion()
        plt.show()

        self.run()

    def print_splash(self):
        splash = open("splash.txt", "r")
        splash_lines = splash.readlines()
    
        print("===========================================================================================================================================")
        for l in splash_lines:
            print(l, end="")
        print("\nSPIR Version:", VERSION)
        print("===========================================================================================================================================")

    def set_data_dir(self):
        raw_path = input("\nSet data path: ")
        clean_path = raw_path.replace("\\", "/")
        assert os.path.exists(clean_path), "Entered path \"" + raw_path + "\" cannot be found!"
        self.data_path = clean_path

    def load_new_files(self):
        global click_event
        
        self.file_list = []
        for file in os.listdir(self.data_path):
            if file.endswith(".fit"):
                self.file_list.append(file)

        for file in self.file_list:
            if not self.target and len(self.loaded_files) > 0:
                if click_event != None:
                    if click_event.inaxes == self.axs[0, 0]:
                        self.target = [click_event.xdata, click_event.ydata]
                        self.display()
                        confirm = input("\nConfirm target placement(y/n): ")
                        if not confirm.capitalize() == "Y":
                            self.target = None
                            click_event = None
                            self.display()
                else:
                    return
            if not file in self.loaded_files:
                try:
                    img = fits.open(self.data_path + "/" + file)
                    self.loaded_files.append(file)
                    self.data.append(img[0].data)
                    self.headers.append(img[0].header)

                    theta = np.radians(90 - self.headers[self.latest_index]['CENTALT'])
                    phi = np.radians(self.headers[self.latest_index]['CENTAZ'] + 90)
                    x = np.cos(phi)*np.sin(theta)
                    y = np.sin(phi)*np.sin(theta)
                    z = np.cos(theta)
                    self.mount_pointing.append([x, y, z])

                    img.close()
                    # print("Loaded:", file)
                    self.display()
                except:
                    pass # file creation may not be finished.

    def display(self):
        self.latest_index = len(self.data) - 1
        
        latest_data = self.data[self.latest_index]
        self.axs[0, 0].cla()
        self.axs[0, 0].imshow(latest_data, vmin=np.percentile(latest_data, 1), vmax=np.percentile(latest_data, 99))
        self.axs[0, 0].set_title('Latest Image: ' + self.loaded_files[self.latest_index])
        if self.target:
            self.draw_target()
        
        self.draw_mount_pointing()

        self.draw_airmass()

        self.latest_index = len(self.data) - 1

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def draw_target(self):
        self.axs[0, 0].scatter(self.target[0], self.target[1], s=120, facecolors='none', edgecolors='r')
        self.axs[0, 0].scatter(self.target[0], self.target[1], s=300, facecolors='none', edgecolors='w')
        self.axs[0, 0].scatter(self.target[0], self.target[1], s=600, facecolors='none', edgecolors='w')

    def draw_mount_pointing(self):
        self.axs[1, 0].cla()
        self.axs[1, 0].xaxis.pane.fill = False
        self.axs[1, 0].yaxis.pane.fill = False
        self.axs[1, 0].zaxis.pane.fill = False

        self.axs[1, 0].set_title('Mount Pointing: ALT: ' + 
                                 str(round(self.headers[self.latest_index]['CENTALT'], 4)) +
                                 ' AZ: ' + str(round(self.headers[self.latest_index]['CENTAZ'], 4)))
        u, v = np.mgrid[0:2*np.pi:20j, 0:0.5*np.pi:10j]
        x = np.cos(u)*np.sin(v)
        y = np.sin(u)*np.sin(v)
        z = np.cos(v)
        self.axs[1, 0].plot_wireframe(x, y, z, color="r", linewidths=0.5)
        self.axs[1, 0].quiver(0, 0 ,0, 1, 0, 0, color="r")
        self.axs[1, 0].grid(False)
        self.axs[1, 0].set_xticks([])
        self.axs[1, 0].set_yticks([])
        self.axs[1, 0].set_zticks([])
        self.axs[1, 0].set_box_aspect([1.0, 1.0, 0.5])

        self.axs[1, 0].quiver(0, 0 ,0, 
                              self.mount_pointing[self.latest_index][0],
                              self.mount_pointing[self.latest_index][1],
                              self.mount_pointing[self.latest_index][2], color="g")

    def draw_airmass(self):
        if self.latest_index == 0:
            return
        airmass = []
        times = []
        for i in range(self.latest_index):
            airmass.append(self.headers[i]['AIRMASS'])
            date_obj = datetime.strptime(self.headers[i]['DATE-OBS'], "%Y-%m-%dT%H:%M:%S.%f")
            times.append(self.latest_index * self.headers[0]['EXPTIME'])
        #print(airmass)
        self.axs[1, 1].cla()
        self.axs[1, 1].set_aspect("auto")
        self.axs[1, 1].set_title('Airmass')
        #self.axs[1, 1].set_yscale('log')
        self.axs[1, 1].plot(airmass)

    def run(self):
        print("Watching data path...")
        while True:
            self.load_new_files()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            sleep(0.05)

SPIR()