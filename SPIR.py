import os

from astropy.io import fits
from time import sleep
from matplotlib import pyplot as plt
import numpy as np

VERSION = 0.01

class SPIR:

    def __init__(self):
        self.loaded_files = []
        self.data = []
        self.headers = []

        self.latest_index = 0

        self.print_splash()
        self.set_data_dir()

        self.fig, self.axs = plt.subplots(2, 2)
        self.axs[1, 0].remove()
        self.axs[1, 0] = self.fig.add_subplot(2, 2, 3, projection='3d')
        self.axs[1, 0].grid(False)
        self.axs[1, 0].set_xticks([])
        self.axs[1, 0].set_yticks([])
        self.axs[1, 0].set_zticks([])
        self.axs[1, 0].set_box_aspect([1.0, 1.0, 0.5])

        self.axs[0, 0].set_title('Latest Image')
        self.axs[0, 1].set_title('Relative Flux')
        self.axs[1, 0].set_title('Mount Pointing')
        self.axs[1, 1].set_title('SNR & Airmass')
        self.fig.tight_layout(pad=3.0)
        self.fig.set_figheight(8)
        self.fig.set_figwidth(14)

        u, v = np.mgrid[0:2*np.pi:20j, 0:0.5*np.pi:10j]
        x = np.cos(u)*np.sin(v)
        y = np.sin(u)*np.sin(v)
        z = np.cos(v)
        self.axs[1, 0].plot_wireframe(x, y, z, color="r", linewidths=0.5)

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
        self.file_list = []
        for file in os.listdir(self.data_path):
            if file.endswith(".fit"):
                self.file_list.append(file)

        for file in self.file_list:
            if not file in self.loaded_files:
                try:
                    img = fits.open(self.data_path + "/" + file)
                    self.loaded_files.append(file)
                    self.data.append(img[0].data)
                    self.headers.append(img[0].header)
                    img.close()
                    print("Loaded:", file)
                except:
                    pass # file creation may not be finished.

    def display(self):
        if len(self.data) == 0 or self.latest_index == len(self.data) - 1:
            return
        
        self.latest_index = len(self.data) - 1
        
        latest_data = self.data[self.latest_index]
        self.axs[0, 0].imshow(latest_data, vmin=np.percentile(latest_data, 1), vmax=np.percentile(latest_data, 99))
        self.axs[0, 0].set_title('Latest Image: ' + self.loaded_files[self.latest_index])

        theta = np.radians(90 - self.headers[self.latest_index]['CENTALT'])
        phi = np.radians(self.headers[self.latest_index]['CENTAZ'])
        print(self.headers[self.latest_index]['CENTALT'], self.headers[self.latest_index]['CENTAZ'])
        x = np.cos(phi)*np.sin(theta)
        y = np.sin(phi)*np.sin(theta)
        z = np.cos(phi)

        self.axs[1, 0].scatter(x, y, z, color="g")
        self.axs[1, 0].quiver(0, 0 ,0, x, y, z, color="b")

        self.latest_index = len(self.data) - 1

    def run(self):
        print("Watching data path...")
        while True:
            self.load_new_files()
            self.display()

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            sleep(0.1)

SPIR()