# SlideLoader
*SlideLoader* is a Python 3.9+ package for loading whole slide images (WSIs) and reading tiles. 
The aim is to provide a unified interface for loading WSIs in DICOM format using 
the [wsidicom](https://github.com/imi-bigpicture/wsidicom) library and for other 
formats using the [OpenSlide](https://github.com/openslide) library.

## Installing *SlideLoader*
*SlideLoader* can be installed from GitHub:
```console
$ pip install git+https://github.com/RTLucassen/slideloader
```
OpenSlide binaries can be downloaded from [here](https://openslide.org/). 
For Python >= 3.8 on Windows, the path to the OpenSlide bin-folder must be provided 
before importing OpenSlide. Specify the path in `src/slideloader/slideloader.py` 
after installation. For example:
```
OPENSLIDE_PATH = r'C:\Users\user\path\to\openslide-win64-20221111\bin'
```

## Example
A minimal example of how *SlideLoader* can be used for loading WSIs.
```
from slideloader import SlideLoader

# initialize SlideLoader instance
loader = SlideLoader()

# define the path and load the WSI
path = r'project/images/image.svs'
loader.load_slide(path)

# get the WSI properties
properties = loader.get_properties()

# get the dimensions of the WSI (in pixels)
dimensions = loader.get_dimensions(magnification=5.0)

# get the WSI
image = loader.get_image(magnification=5.0)

# get a single tile from the WSI
tile = loader.get_tile(
    magnification=5.0, 
    location=(0, 0), 
    shape=(256, 256),
) 
# get multiple tiles (same size) from the WSI
tiles = loader.get_tiles(
    magnification=5.0, 
    locations=[(0, 0), (1000, 1000)], 
    shapes=(256, 256),
) 
# get multiple tiles (different sizes) from the WSI
tiles = loader.get_tiles(
    magnification=5.0, 
    locations=[(0, 0), (1000, 1000)], 
    shapes=[(256, 256), (1024, 256)],
) 
```
