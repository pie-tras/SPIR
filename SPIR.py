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

should_recompute = True
frame_percent = 100
update_frame_indicator = False

def trigger_recompute(event):
    global should_recompute
    should_recompute = True

def set_frame_view(percent):
    global frame_percent
    global update_frame_indicator
    frame_percent = percent
    update_frame_indicator = True

class Telescope:

    def __init__(self, data_path, target_guess=[500, 500], ref_guess=[400, 400], drift_correct_freq=10, fwhm_est=8.0):
        self.data_path = data_path
        self.data_files = []
        self.dark_files = []
        self.bias_files = []
        self.flat_files = []

        self.computed_index = 0
        self.view_frame = 0

        self.latest_frame = None
        self.latest_header = None
        self.target_coords = target_guess
        self.ref_coords = ref_guess
        self.target_star_apt = None
        self.target_sky_apt = None
        self.ref_star_apt = None
        self.ref_sky_apt = None
        self.target_flux = []
        self.ref_flux = []
        self.target_rel_flux = []
        self.ref_rel_flux = []
        self.corrected_rel_flux = []
        self.snr = []
        self.median_dark = None
        self.median_bias = None
        self.dark_rate = None
        self.read_noise_adu = None

        self.latest_view_frame = None

        self.target_coord_hist = []
        self.ref_coord_hist = []

        self.target_guess = target_guess
        self.ref_guess = ref_guess
        self.drift_correct_freq = drift_correct_freq
        self.fwhm_est = fwhm_est

        self.load_wcs()
        self.load_noise_files()
        self.load_new_files()

    def load_wcs(self):
        try:
            wcs_header = fits.open(self.data_path + '/wcs.fits')[0].header
            w = WCS(wcs_header)

            target_sky_coord = SkyCoord(GJ182_COORDS_J2000, unit=(u.hourangle, u.deg), obstime="J2000")
            ref_sky_coord = SkyCoord(REF_STAR_0_COORDS_J2000, unit=(u.hourangle, u.deg), obstime="J2000")

            target_x, target_y = w.world_to_pixel(target_sky_coord)
            ref_x, ref_y = w.world_to_pixel(ref_sky_coord)

            self.target_coords = [target_x[0], target_y[0]]
            self.ref_coords = [ref_x[0], ref_y[0]]
        except:
            print("[WARNING]: WCS HEADER NOT FOUND DEFAULTING TO GUESSED COORDS")
            print("Telescope: " + self.data_path)
            print("Target: " + str(self.target_coords))
            print("Ref: " + str(self.ref_coords))
            
    def load_noise_files(self):

        dark_file_list = glob(self.data_path + '/noise/*-Dark-*.fit')
        bias_file_list = glob(self.data_path + '/noise/*-Bias-*.fit')
        flat_file_list = glob(self.data_path + '/noise/*-FlatField-*.fit')

        for f in dark_file_list:
            if not f in self.dark_files:
                self.dark_files.append(f)
                
        for f in bias_file_list:
            if not f in self.bias_files:
                self.bias_files.append(f)

        for f in flat_file_list:
            if not f in self.flat_files:
                self.flat_files.append(f)

        bias_imgs = []
        for f in self.bias_files:
            fits_file = fits.open(f)[0]
            bias_imgs.append(fits_file.data)

        self.median_bias = np.median(bias_imgs, axis=0)

        dark_imgs = []
        for f in self.dark_files:
            fits_file = fits.open(f)[0]
            dark_imgs.append(fits_file.data)

        self.median_dark = np.median(dark_imgs, axis=0)

        dark_header =  fits.open(self.dark_files[0])[0].header

        self.gain = dark_header['GAINADU']
        t_dark = dark_header['EXPTIME']

        self.dark_rate = (self.gain * (self.median_dark - self.median_bias))/ t_dark

        self.read_noise_adu = np.mean(np.std(bias_imgs, axis=0))

    def load_new_files(self):
        data_file_list = glob(self.data_path + '/data/*.fits')
 
        for f in data_file_list:
            if not f in self.data_files:
                self.data_files.append(f)

    def compute_new(self):
        print("Current apeture locations on: " + self.data_path)
        print("Target: " + str(self.target_coords))
        print("Ref: " + str(self.ref_coords) + "\n")

        for i in range(self.computed_index, len(self.data_files) - 1):
            fits_file = fits.open(self.data_files[i])[0]
            
            self.latest_frame = fits_file.data
            self.latest_view_frame = fits_file.data
            self.latest_header = fits_file.header
            
            if i % self.drift_correct_freq == 0 or i == 0:
                self.update_apetures()

            self.target_coord_hist.append(self.target_coords)
            self.ref_coord_hist.append(self.ref_coords)

          #  self.compute_sigma_light()
            self.compute_fluxes()
            self.compute_relative_fluxes()

            self.computed_index = i + 1
            self.view_frame = i

    def set_latest_frame(self):
        global frame_percent
        new_index = int((self.computed_index - 1) * (frame_percent / 100.0))
        fits_file = fits.open(self.data_files[new_index])[0]
        self.latest_view_frame = fits_file.data
        self.view_frame = new_index

    def update_apetures(self):
        stars = self.find_stars(self.latest_frame)

        if stars:
            self.target_coords = self.find_closest_coords(stars, self.target_coords)
            self.ref_coords = self.find_closest_coords(stars, self.ref_coords)

            star_r = self.fwhm_est * 4
            sky_r_in = star_r + 5
            sky_r_out = sky_r_in + 20

            self.target_star_apt = CircularAperture(self.target_coords, r=star_r)
            self.target_sky_apt = CircularAnnulus(self.target_coords, r_in=sky_r_in, r_out=sky_r_out)

            self.ref_star_apt = CircularAperture(self.ref_coords, r=star_r)
            self.ref_sky_apt = CircularAnnulus(self.ref_coords, r_in=sky_r_in, r_out=sky_r_out)

    def find_stars(self, data):
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        #print((mean, median, std))  
        daofind = DAOStarFinder(fwhm=self.fwhm_est, threshold=5.*std)
        stars = daofind(data - median)  
        if stars:
            for col in stars.colnames:  
                if col not in ('id', 'npix'):
                    stars[col].info.format = '%.2f'
            #sources.pprint(max_width=200)  
            #print("Found", len(stars), "stars.")

            return stars
        else:
            return None
        
    def find_closest_coords(self, stars, target):
        min_dis = 100000
        next_target = target

        for star in stars:
            x = star['xcentroid']
            y = star['ycentroid']

            dis = np.sqrt((x - target[0])**2 + (y - target[1])**2)

            if dis < min_dis:
                min_dis = dis
                next_target = [x, y]

        if dis > 10:
            next_target = target

        return next_target
    
    def compute_sigma_light(self):
        exp_time = self.latest_header['EXPTIME']
   
        light_shape = np.shape(self.latest_frame)
        n_light = self.gain*(self.latest_frame - self.median_bias[0:light_shape[0], 0:light_shape[1]]) - exp_time * self.dark_rate[0:light_shape[0], 0:light_shape[1]]

        read_noise_electrons = self.read_noise_adu * self.gain
        test_val = read_noise_electrons**2  + (exp_time * self.dark_rate[0:light_shape[0], 0:light_shape[1]]) + n_light
        # for y, row in enumerate(test_val):
        #     for x, val in enumerate(row):
        #         if val < 0:
        #             test_val[y][x] = 0

        #print(test_val)
        sigma_n_light = np.sqrt(np.abs(test_val))
        return sigma_n_light
    
    def compute_fluxes(self):

        sigma_light = self.compute_sigma_light()

        target_apt_flux, target_apt_uncertainty = self.target_star_apt.do_photometry(self.latest_frame, error=sigma_light)
        target_sky_stats = ApertureStats(self.latest_frame, self.target_sky_apt)
        target_sky_per_pixel = target_sky_stats.median
        target_sky_flux = self.target_star_apt.area*target_sky_per_pixel
        target_flux = target_apt_flux[0] - target_sky_flux
        self.target_flux.append(target_apt_flux[0] - target_sky_flux)

        ref_apt_flux, ref_apt_uncertainty = self.ref_star_apt.do_photometry(self.latest_frame, error=sigma_light)
        ref_sky_stats = ApertureStats(self.latest_frame, self.ref_sky_apt)
        ref_sky_per_pixel = ref_sky_stats.median
        ref_sky_flux = self.ref_star_apt.area*ref_sky_per_pixel
        self.ref_flux.append(ref_apt_flux[0] - ref_sky_flux)
       
        star_uncertainty = target_apt_uncertainty[0]
        self.snr.append(target_flux/star_uncertainty)

    def compute_relative_fluxes(self):
        self.target_rel_flux = self.target_flux/np.median(self.target_flux)
        self.ref_rel_flux = self.ref_flux/np.median(self.ref_flux)

        corrected_flux = np.divide(self.target_flux, self.ref_flux)

        self.corrected_rel_flux = corrected_flux/np.median(corrected_flux)

class SPIR:

    BASE_PATH = '../GJ182/data/'
    DATE = '12_3_23'

    def __init__(self):
        self.apollo = Telescope(self.BASE_PATH + 'apollo_r/' + self.DATE)
        self.artemis = Telescope(self.BASE_PATH + 'artemis_g/' + self.DATE, fwhm_est=7)
        self.leto = Telescope(self.BASE_PATH + 'leto_i/' + self.DATE, target_guess=[704, 515], ref_guess=[800, 667], fwhm_est=13)

        self.init_figure()
        self.run()

    def init_figure(self):
        plt.style.use('dark_background')

        self.fig, self.axs = plt.subplots(3, 3)

        self.fig.tight_layout(pad=2.0)
        self.fig.set_figheight(8)
        self.fig.set_figwidth(14)

        self.cycle_figure()

        axs_button = plt.axes([0.0, 0.0, 0.1, 0.05])
        self.recompute_button = Button(ax=axs_button, label='RECOMPUTE', color='midnightblue', hovercolor='royalblue')

        self.recompute_button.on_clicked(trigger_recompute)

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
        
        self.axs[0, 0].set_title('Apollo r\' frame')
        self.axs[1, 0].set_title('Artemis g\' frame')
        self.axs[2, 0].set_title('Leto i\' frame')

        self.axs[0, 1].set_title('Apollo r\' Relative Flux vs Exposure Count')
        self.axs[1, 1].set_title('Artemis g\' Relative Flux vs Exposure Count')
        self.axs[2, 1].set_title('Leto i\' Relative Flux vs Exposure Count')

        self.axs[0, 2].set_title('Apollo r\' SNR vs Exposure Count')
        self.axs[1, 2].set_title('Artemis g\' SNR vs Exposure Count')
        self.axs[2, 2].set_title('Leto i\' SNR vs Exposure Count')

    def plot_latest_frames(self):
        if type(self.apollo.latest_view_frame) != type(None):
            self.axs[0, 0].imshow(self.apollo.latest_view_frame, cmap='viridis', vmin=np.percentile(self.apollo.latest_view_frame, 1), vmax=np.percentile(self.apollo.latest_view_frame, 99))
            self.plot_apetures(self.apollo, self.axs[0, 0])
        
        if type(self.artemis.latest_view_frame) != type(None):
            self.axs[1, 0].imshow(self.artemis.latest_view_frame, cmap='viridis', vmin=np.percentile(self.artemis.latest_view_frame, 1), vmax=np.percentile(self.artemis.latest_view_frame, 99))
            self.plot_apetures(self.artemis, self.axs[1, 0])

        if type(self.leto.latest_view_frame) != type(None):
            self.axs[2, 0].imshow(self.leto.latest_view_frame, cmap='viridis', vmin=np.percentile(self.leto.latest_view_frame, 1), vmax=np.percentile(self.leto.latest_view_frame, 99))
            self.plot_apetures(self.leto, self.axs[2, 0])

    def plot_apetures(self, telescope, axis):
        global frame_percent
        
        if frame_percent == 100:
            telescope.target_star_apt.plot(color='red', ax=axis)
            telescope.target_sky_apt.plot(color='red', linestyle='--', ax=axis)
            telescope.ref_star_apt.plot(color='white', ax=axis)
            telescope.ref_sky_apt.plot(color='white', linestyle='--', ax=axis)
        else:
            star_r = telescope.fwhm_est * 4
            sky_r_in = star_r + 5
            sky_r_out = sky_r_in + 20

            target_star_apt = CircularAperture(telescope.target_coord_hist[telescope.view_frame], r=star_r)
            target_sky_apt = CircularAnnulus(telescope.target_coord_hist[telescope.view_frame], r_in=sky_r_in, r_out=sky_r_out)

            ref_star_apt = CircularAperture(telescope.ref_coord_hist[telescope.view_frame], r=star_r)
            ref_sky_apt = CircularAnnulus(telescope.ref_coord_hist[telescope.view_frame], r_in=sky_r_in, r_out=sky_r_out)

            target_star_apt.plot(color='red', ax=axis)
            target_sky_apt.plot(color='red', linestyle='--', ax=axis)
            ref_star_apt.plot(color='white', ax=axis)
            ref_sky_apt.plot(color='white', linestyle='--', ax=axis)

    def plot_relative_fluxes(self):
        self.axs[0, 1].plot(self.apollo.corrected_rel_flux)
        self.axs[1, 1].plot(self.artemis.corrected_rel_flux)
        self.axs[2, 1].plot(self.leto.corrected_rel_flux)

    def plot_snr(self):
        self.axs[0, 2].plot(self.apollo.snr)
        self.axs[1, 2].plot(self.artemis.snr)
        self.axs[2, 2].plot(self.leto.snr)

    def plot_frame_indicator(self):
        self.axs[0, 1].axvline(x = self.apollo.view_frame, color = 'white', linestyle = '--')
        self.axs[1, 1].axvline(x = self.artemis.view_frame, color = 'white', linestyle = '--')
        self.axs[2, 1].axvline(x = self.leto.view_frame, color = 'white', linestyle = '--')

        self.axs[0, 2].axvline(x = self.apollo.view_frame, color = 'white', linestyle = '--')
        self.axs[1, 2].axvline(x = self.artemis.view_frame, color = 'white', linestyle = '--')
        self.axs[2, 2].axvline(x = self.leto.view_frame, color = 'white', linestyle = '--')

    def redraw(self):
        self.cycle_figure()
        self.plot_latest_frames()
        self.plot_relative_fluxes()
        self.plot_snr()
        self.plot_frame_indicator()

    def run(self):
        global should_recompute
        global update_frame_indicator

        while True:
            self.apollo.load_new_files()
            self.artemis.load_new_files()
            self.leto.load_new_files()

            if should_recompute:
                self.apollo.compute_new()
                self.artemis.compute_new()
                self.leto.compute_new()

                self.redraw()
                should_recompute = False
                print("Finished Recompute.")

            if update_frame_indicator:
                self.apollo.set_latest_frame()
                self.artemis.set_latest_frame()
                self.leto.set_latest_frame()

                self.redraw()
                update_frame_indicator = False

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

SPIR()