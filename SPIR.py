import astropy.units as u
import matplotlib.pyplot as plt 
import numpy as np
import os

from astropy.io import fits
from astropy.stats import sigma_clipped_stats 
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from glob import glob
from matplotlib.widgets import Button, Slider
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats
from time import sleep

GJ182_COORDS_J2000 = ['04h59m34.8342878424s +01d47m00.669818328s']
REF_STAR_0_COORDS_J2000 = ['04h59m30.9509965632	+01d45m24.447571200s'] #TYC 85-1241-1

# should_recompute = True
frame_percent = 100
update_frame_indicator = False

# def trigger_recompute(event):
#     global should_recompute
#     should_recompute = True

def set_frame_view(percent):
    global frame_percent
    global update_frame_indicator
    frame_percent = percent
    update_frame_indicator = True

class Telescope:

    def __init__(self, data_path, data_file):
        self.data_path = data_path
        self.data_files = glob(self.data_path + '/data/*.fits')

        self.exp_indicies = []
        self.exp_times_epoch = []
        self.target_offsets = []
        self.ref_offsets = []
        self.target_flux = []
        self.ref_flux = []
        self.snr = []
        
        f = open(data_file, "r")
        for i, line in enumerate(f):
            if i == 0:
                continue
            data = line.split(",")

            self.exp_indicies.append(int(data[0]))
            self.exp_times_epoch.append(float(data[1]))
            self.target_offsets.append([float(data[2]), float(data[3])])
            self.ref_offsets.append([float(data[4]), float(data[5])])
            self.target_flux.append(float(data[6]))
            self.ref_flux.append(float(data[7]))
            self.snr.append(float(data[8]))

        self.latest_frame = None
        self.target_star_apt = None
        self.target_sky_apt = None
        self.ref_star_apt = None
        self.ref_sky_apt = None

        self.current_target = [0, 0]
        self.current_ref = [0, 0]

        self.current_index = 0

    def set_latest_frame(self):
        global frame_percent
        new_index = int((len(self.exp_indicies) - 1) * (frame_percent / 100.0))
        self.current_index = self.exp_indicies[new_index]
        fits_file = fits.open(self.data_files[self.current_index])[0]
        self.latest_frame = fits_file.data

        star_r = 8 * 4
        sky_r_in = star_r + 5
        sky_r_out = sky_r_in + 40

        self.current_target = self.target_offsets[new_index]
        self.current_ref = self.ref_offsets[new_index]

        self.target_star_apt = CircularAperture(self.current_target, r=star_r)
        self.target_sky_apt = CircularAnnulus(self.current_target, r_in=sky_r_in, r_out=sky_r_out)

        self.ref_star_apt = CircularAperture(self.current_ref, r=star_r)
        self.ref_sky_apt = CircularAnnulus(self.current_ref, r_in=sky_r_in, r_out=sky_r_out)

        self.sliced_times = self.exp_times_epoch[new_index-300:new_index]
        self.diff_flux = np.divide(self.target_flux[new_index-300:new_index], self.ref_flux[new_index-300:new_index]) 
        self.diff_flux = np.divide(self.diff_flux, np.median(self.diff_flux))

        self.snr_sliced = self.snr[new_index-300:new_index]

class SPIR:

    BASE_PATH = '../GJ182/data/'
    DATE = '12_7_23'

    def __init__(self):
        self.apollo = Telescope(self.BASE_PATH + 'apollo_r/' + self.DATE, "../apollo_r_data.txt")
        self.artemis = Telescope(self.BASE_PATH + 'artemis_g/' + self.DATE, "../artemis_g_data.txt")
        self.leto = Telescope(self.BASE_PATH + 'leto_i/' + self.DATE, "../leto_i_data.txt")

        self.init_figure()
        self.run()

    def init_figure(self):
        plt.style.use('dark_background')

        self.fig, self.axs = plt.subplots(3, 3)

        self.fig.tight_layout(pad=2.0)
        self.fig.set_figheight(8)
        self.fig.set_figwidth(14)

        self.cycle_figure()

        # axs_button = plt.axes([0.0, 0.0, 0.1, 0.05])
        # self.recompute_button = Button(ax=axs_button, label='RECOMPUTE', color='midnightblue', hovercolor='royalblue')

        # self.recompute_button.on_clicked(trigger_recompute)

        self.axs_slider = plt.axes([0.01, 0.25, 0.0225, 0.63])
        self.time_slider = Slider(
            ax=self.axs_slider,
            label='Exp. N%',
            valmin=0,
            valmax=100,
            valinit=100,
            orientation="vertical"
        )

        self.time_slider.on_changed(set_frame_view)
        
        plt.ion()
        plt.show()

    def cycle_figure(self):

        self.axs[0, 0].cla()
        self.axs[1, 0].cla()
        self.axs[2, 0].cla()
        
        self.axs[0, 1].cla()
        self.axs[1, 1].cla()
        self.axs[2, 1].cla()
        
        self.axs[0, 2].cla()
        self.axs[1, 2].cla()
        self.axs[2, 2].cla()
        
        self.axs[0, 0].set_title('Apollo r\' frame: ' + str(self.apollo.current_index) + ' ' + str([round(self.apollo.current_target[0]), round(self.apollo.current_target[1])]))
        self.axs[1, 0].set_title('Artemis g\' frame: ' + str(self.artemis.current_index) + ' ' + str([round(self.artemis.current_target[0]), round(self.artemis.current_target[1])]))
        self.axs[2, 0].set_title('Leto i\' frame: ' + str(self.leto.current_index) + ' ' + str([round(self.leto.current_target[0]), round(self.leto.current_target[1])]))

        self.axs[0, 1].set_title('Apollo r\' Relative Flux vs Exposure Count')
        self.axs[1, 1].set_title('Artemis g\' Relative Flux vs Exposure Count')
        self.axs[2, 1].set_title('Leto i\' Relative Flux vs Exposure Count')

        self.axs[0, 2].set_title('Apollo r\' SNR vs Exposure Count')
        self.axs[1, 2].set_title('Artemis g\' SNR vs Exposure Count')
        self.axs[2, 2].set_title('Leto i\' SNR vs Exposure Count')

    def plot_latest_frames(self):
        if type(self.apollo.latest_frame) != type(None):
            self.axs[0, 0].imshow(self.apollo.latest_frame, cmap='viridis', vmin=np.percentile(self.apollo.latest_frame, 1), vmax=np.percentile(self.apollo.latest_frame, 99))
            self.plot_apetures(self.apollo, self.axs[0, 0])
        
        if type(self.artemis.latest_frame) != type(None):
            self.axs[1, 0].imshow(self.artemis.latest_frame, cmap='viridis', vmin=np.percentile(self.artemis.latest_frame, 1), vmax=np.percentile(self.artemis.latest_frame, 99))
            self.plot_apetures(self.artemis, self.axs[1, 0])

        if type(self.leto.latest_frame) != type(None):
            self.axs[2, 0].imshow(self.leto.latest_frame, cmap='viridis', vmin=np.percentile(self.leto.latest_frame, 1), vmax=np.percentile(self.leto.latest_frame, 99))
            self.plot_apetures(self.leto, self.axs[2, 0])

    def plot_apetures(self, telescope, axis):
        telescope.target_star_apt.plot(color='red', ax=axis)
        telescope.target_sky_apt.plot(color='red', linestyle='--', ax=axis)
        telescope.ref_star_apt.plot(color='white', ax=axis)
        telescope.ref_sky_apt.plot(color='white', linestyle='--', ax=axis)

    def plot_relative_fluxes(self):
        # self.axs[0, 1].set_xlim(self.apollo.sliced_times[0], self.apollo.sliced_times[-1])
        # self.axs[1, 1].set_xlim(self.artemis.sliced_times[0], self.artemis.sliced_times[-1])
        # self.axs[2, 1].set_xlim(self.leto.sliced_times[0], self.leto.sliced_times[-1])

        self.axs[0, 1].plot(self.apollo.sliced_times, self.apollo.diff_flux)
        self.axs[0, 1].plot(self.artemis.sliced_times, self.artemis.diff_flux)
        self.axs[0, 1].plot(self.leto.sliced_times, self.leto.diff_flux)

    def plot_snr(self):
        self.axs[0, 2].plot(self.apollo.sliced_times, self.apollo.snr_sliced)
        self.axs[1, 2].plot(self.artemis.sliced_times, self.artemis.snr_sliced)
        self.axs[2, 2].plot(self.leto.sliced_times, self.leto.snr_sliced)

    def plot_frame_indicator(self):
        # self.axs[0, 1].axvline(x = self.apollo.current_index, color = 'white', linestyle = '--')
        # self.axs[1, 1].axvline(x = self.artemis.current_index, color = 'white', linestyle = '--')
        # self.axs[2, 1].axvline(x = self.leto.current_index, color = 'white', linestyle = '--')

        # self.axs[0, 2].axvline(x = self.apollo.current_index, color = 'white', linestyle = '--')
        # self.axs[1, 2].axvline(x = self.artemis.current_index, color = 'white', linestyle = '--')
        # self.axs[2, 2].axvline(x = self.leto.current_index, color = 'white', linestyle = '--')

        pass

    def redraw(self):
        self.cycle_figure()
        self.plot_latest_frames()
        self.plot_relative_fluxes()
        self.plot_snr()
        self.plot_frame_indicator()

    def run(self):
        global update_frame_indicator

        self.apollo.set_latest_frame()
        self.artemis.set_latest_frame()
        self.leto.set_latest_frame()

        self.redraw()

        while True:
            if update_frame_indicator:
                self.apollo.set_latest_frame()
                self.artemis.set_latest_frame()
                self.leto.set_latest_frame()

                self.redraw()
                update_frame_indicator = False

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

SPIR()