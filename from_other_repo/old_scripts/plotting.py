"""Plotting
"""

import datetime
import io
from urllib.request import urlopen, Request

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import pandas as pd

def image_spoof(self, tile):
    """Thanks to Joshua Hrisko
    https://makersportal.com/blog/2020/4/24/geographic-visualizations-in-python-with-cartopy
    """
    # get the url of the street map API
    url = self._image_url(tile)
    req = Request(url)
    req.add_header('User-agent','Anaconda 3')
    fh = urlopen(req)
    im_data = io.BytesIO(fh.read())
    fh.close()
    # open image with PIL, set format, reformat for Cartopy
    img = Image.open(im_data)
    img = img.convert(self.desired_tile_form)
    return (img, self.tileextent(tile), 'lower')


