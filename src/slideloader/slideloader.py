#    Copyright 2022 Ruben T Lucassen, UMC Utrecht, The Netherlands 
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
Utility class for loading whole slide images (WSIs) and reading tiles.
The aim is to provide a unified interface for loading WSIs in DICOM format using 
the wsidicom library and for other formats using the OpenSlide library.
"""

import concurrent.futures
import os
from math import floor, ceil, log2
from pathlib import Path
from typing import Any, Union, Sequence

import numpy as np
import pydicom
import SimpleITK as sitk
import wsidicom
from skimage import img_as_ubyte
from skimage.transform import resize
from tqdm import tqdm


OPENSLIDE_PATH = None

# import openslide
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    if OPENSLIDE_PATH is None:
        raise FileNotFoundError('Specify the path to the OpenSlide bin folder.')
    else:
        with os.add_dll_directory(OPENSLIDE_PATH):
            import openslide
else:
    import openslide


class SlideLoader():
    """
    Class for loading whole slide images and reading tiles.
    """

    __settings = {
        'multithreading': True,
        'progress_bar': False,
        'chunk_shape': (1024, 1024),
        'max_difference': 0.025,
        'anti-aliasing': False,
        'interpolation_order': 2,
    }

    def __init__(self, settings: dict[str, Any] = {}) -> None:
        """
        Initialize SlideLoader instance.

        Args:
            settings:  Dictionary with names of settings and corresponding values.
                Default values are used for each setting if unspecified.
        """
        # configure settings
        for key in settings:
            self.__settings[key] = settings[key]

        # initialize slide loader objects for non-DICOM and DICOM images
        self.__openslide_loader = OpenSlideLoader(
            self.__settings, 
        )
        self.__dicomslide_loader = DicomSlideLoader(            
            self.__settings, 
        )
        self.__loader = None

    def load_slide(self, paths: Union[str, Path, Sequence[Union[str, Path]]]) -> None:
        """
        Load whole slide image.
        
        Args:
            path:  Path to whole slide image.
        """
        # format paths
        if isinstance(paths, (str, Path)):
            paths = [paths]
        paths = [Path(path) for path in paths]

        # check if the paths are valid
        if not len(paths):
             raise ValueError('No paths were provided.')
        elif len(paths) == 1:
            path = paths[0]
            # check if the extension is from a DICOM image
            if path.suffix == '.dcm':
                self.__dicomslide_loader.load_slide(path)
                self.__loader = self.__dicomslide_loader
            else:
                self.__openslide_loader.load_slide(path)
                self.__loader = self.__openslide_loader
        elif len(paths) != len(set(paths)):
            raise ValueError('Duplicate paths were provided.')
        # only DICOM images should be provided as multiple paths
        else:
            cases = set([path.name.split('.')[0] for path in paths])
            extensions = set([path.suffix for path in paths])
            if len(extensions) > 1:
                raise ValueError('At least two paths end with different file types.')
            elif extensions.pop() != '.dcm':
                raise ValueError(('Only multiple paths to the same DICOM image '
                    'at different magnifications can be loaded at once.'))
            elif len(cases) > 1:
                raise ValueError('Atleast two paths are from different DICOM images.')
            else:
                self.__dicomslide_loader.load_slide(paths)
                self.__loader = self.__dicomslide_loader

    def get_properties(self) -> dict[str, Any]:
        return self.__loader.get_properties()

    def get_dimensions(self, magnification: float) -> tuple[int, int]:
        """
        Return the dimensions of the whole slide image at the specified magnification.
        """
        return self.__loader.get_dimensions(magnification)

    def get_tile(
        self, 
        magnification: float, 
        location: tuple[int, int],
        shape: tuple[int, int] = (256, 256), 
        channels_last: bool = True,
    ) -> np.ndarray:
        """
        Get a tile from the whole slide image based on the specified
        magnification, location, and shape.
        
        Args:
            magnification:  Magnification at which the whole slide image is loaded.
            location:  Location of top left pixel as (x, y) at the specified magnification.
            shape:  Shape of the tile in pixels as (height, width).
            channels_last:  Specifies if the channels dimension of the output tensor 
                is last. If False, the channels dimension is the second dimension.
        
        Returns:
            tile:  Whole slide image tile [uint8] as (height, width, channel) 
                for channels last or (channel, height, width) for channels first.
        """
        return self.__loader.get_tile(magnification, location, shape, channels_last)

    def get_tiles(
        self, 
        magnification: float, 
        locations: list[tuple[int, int]],
        shapes: Union[tuple[int, int], list[tuple[int, int]]] = (256, 256), 
        channels_last: bool = True,
    ) -> list[np.ndarray]:
        """
        Get one or more tiles from the whole slide image based on the specified 
        magnification, location, and shape. The tiles are read using multithreading 
        if this was enabled in the settings.

        Args:
            magnification:  Magnification at which the whole slide image is loaded.
            locations:  Locations of top left pixel as (x, y) at the specified magnification.
            shapes:  Shape of the tiles in pixels as (height, width).
            channels_last:  Specifies if the channels dimension of the output tensor 
                is last. If False, the channels dimension is the second dimension.
        
        Returns:
            tile: whole slide image tiles [uint8] as (height, width, channel) 
                  for channels last or (channel, height, width) for channels first.
        """
        return self.__loader.get_tiles(magnification, locations, shapes, channels_last)

    def get_image(
        self, 
        magnification: float, 
        channels_last: bool = True,
    ) -> np.ndarray:
        """
        Get the whole slide image at the specified magnification.     

        Args:
            magnification:  Magnification at which the whole slide image is loaded.
            channels_last:  Indicates if the channels dimension should be last.
                If False, the channels dimension is the second dimension.
        
        Returns:
            image:  Whole slide image [uint8] as (height, width, channel) 
                for channels last or as (channel, height, width) for channels first.
        """ 
        return self.__loader.get_image(magnification, channels_last)

    def write_image(
        self, 
        magnification: float,
        output_path: Union[Path, str], 
    ) -> None:
        """
        Write a copy of the whole slide image at the specified magnification 
        to the output_path as a .png or .tif tile.     

        Args:
            magnification:  Magnification at which the whole slide image is loaded.
            output_path:  Path where image is saved.
        """ 
        # get the image
        image = self.__loader.get_image(
            magnification=magnification, 
            channels_last=True,
        )
        # check if the shape of the image is valid
        for length in image.shape:
            if length <= 0:
                raise ValueError('Invalid image shape.')
        # check if the image extensions is valid
        output_path = Path(output_path)
        if output_path.suffix.lower() in ['.png', '.tif', '.tiff']:
            sitk.WriteImage(sitk.GetImageFromArray(image[None, ...]), output_path)
        else:
            raise ValueError('Invalid image type.')   

    def close(self):
        self.__loader.close()


class OpenSlideLoader():
    """
    Class for loading whole slide images (WSIs) and reading tiles using the 
    OpenSlide ImageSlide class.
    """

    __settings = {
        'multithreading': True,
        'progress_bar': False,
        'chunk_shape': (1024, 1024),
        'max_difference': 0.025,
        'anti-aliasing': False,
        'interpolation_order': 2,
    }

    def __init__(self, settings: dict[str, Any] = {}) -> None:
        """
        Args:
            settings:  Dictionary with names of settings and corresponding values.
                Default values are used for each setting if unspecified.
        """
        # initialize instance attributes
        self.__slide = None
        self.__properties = None

        # configure settings
        for key in settings:
            self.__settings[key] = settings[key]  

    def load_slide(self, path: Union[str, Path]) -> None:
        """
        Load whole slide image slide.

        Args:
            path:  Path to whole slide image.
        """
        # initialize instance variable with path to slide and load the slide
        self.__slide = openslide.open_slide(path)

        # retrieve the native magnification
        power = openslide.PROPERTY_NAME_OBJECTIVE_POWER
        native_magnification = float(self.__slide.properties[power])

        # calculate the magnification for all downsample levels
        magnification_levels = []
        for downsample_level in self.__slide.level_downsamples:
            magnification_levels.append(native_magnification/downsample_level)

        # get the dimensions for the slide at all downsample levels
        dimensions = []
        for dimension in list(self.__slide.level_dimensions):
            dimensions.append(tuple(list(dimension)[::-1]))

        # resolution as (x, y) in micrometer / pixel  
        # dimensions as (height, width) in pixels
        self.__properties = {
            'vendor': self.__slide.properties['openslide.vendor'],
            'native_resolution': (                                     
                float(self.__slide.properties['openslide.mpp-x']), 
                float(self.__slide.properties['openslide.mpp-y']),
            ), 
            'native_magnification': native_magnification,
            'magnification_levels': magnification_levels,
            'downsample_levels': list(self.__slide.level_downsamples),
            'dimensions': dimensions,    
        }   

    def get_properties(self) -> dict[str, Any]:
        return self.__properties

    def get_dimensions(self, magnification: float) -> tuple[int, int]:
        """
        Get the whole slide image dimensions at a specified magnification.

        Args:
            magnification:  The magnification factor to get the dimensions for.

        Returns:
            corrected_dimensions:  Dimensions of the whole slide image at the 
                specified magnification.
        """
        # determine the best level and correction factor
        level, correction_factor = self.__best_level_and_correction(magnification)
        # get the dimensions at the level and apply the correction
        dimensions = self.__properties['dimensions'][level]
        corrected_dimensions = (
            floor(dimensions[0]/correction_factor),
            floor(dimensions[1]/correction_factor),
        )
        return corrected_dimensions

    def __best_level_and_correction(self, magnification: float) -> tuple[int, float]:
        """
        Determine the best level of the image pyramid for getting tiles and
        calculate the correction factor for resizing to the exact magnification.

        Args:
            magnification:  The desired magnification factor.
        
        Returns:
            level:  Index of downsampling level with a magnification factor that
                is closest but larger than the desired magnification factor.
            correction_factor:  Magnification factor for the best level to
                downsample from divided by the desired magnification level.
        """
        # check the difference between the requested magnification and 
        # the magnifications available in the whole slide image
        abs_differences = []
        for magnification_level in self.__properties['magnification_levels']:
            abs_differences.append(abs(magnification-magnification_level))

        # check if the smallest difference can be considered negligible
        if min(abs_differences) < self.__settings['max_difference']:
            level = abs_differences.index(min(abs_differences))
            correction_factor = 1
        else:
            # get the best downsample level for resizing the image
            # to the desired magnification level
            level = self.__slide.get_best_level_for_downsample(
                self.__properties['native_magnification']/magnification,
            )
            # calculate the correction factor 
            correction_factor = (
                self.__properties['magnification_levels'][level]/magnification
            )

        return level, correction_factor

    def __read_tile(
        self, 
        level: int, 
        correction_factor: float,
        native_location: tuple[int, int], 
        shape: tuple[int, int],
    ) -> np.ndarray:
        """
        Read a tile from the whole slide image.
        
        Args:
            level:  Downsample level used for loading the tile.
            correction_factor:  Magnification factor for the best level to
                downsample from divided by the desired magnification level.
            native_location:  Location of top left pixel as (x, y) at the 
                native magnification.
            shape:  Shape of tile in pixels as (height, width) at the specified 
                magnification.

        Returns:
            tile:  Whole slide image tile.
        """
        # correct the tile shape if necessary
        if correction_factor == 1:
            corrected_shape = (shape[1], shape[0])
        else:
            corrected_shape = (
                round(shape[1]*correction_factor), 
                round(shape[0]*correction_factor),
            )
        # load the image tile at the specified magnification level
        tile = self.__slide.read_region(
            location=native_location, 
            level=level, 
            size=corrected_shape,
        )
        tile = np.array(tile)[..., 0:3]
        
        # resize the tile if necessary
        if correction_factor != 1:
            tile = resize(
                image=tile, 
                output_shape=shape, 
                anti_aliasing=self.__settings['anti-aliasing'], 
                order=self.__settings['interpolation_order'],
            )
            tile = img_as_ubyte(tile)
        
        # remove first axis if necessary
        if len(tile.shape) == 4:
            tile = tile[0, ...]
        
        return tile

    def get_tile(
        self, 
        magnification: float, 
        location: tuple[int, int],
        shape: tuple[int, int] = (256, 256), 
        channels_last: bool = True,
    ) -> np.ndarray:
        """
        Get a tile from the whole slide image based on the specified
        magnification, location, and shape.
        
        Args:
            magnification:  Magnification at which the whole slide image is loaded.
            location:  Location of top left pixel as (x, y) at the specified magnification.
            shape:  Shape of the tile in pixels as (height, width).
            channels_last:  Specifies if the channels dimension of the output tensor 
                is last. If False, the channels dimension is the second dimension.
        
        Returns:
            tile:  Whole slide image tile [uint8] as (height, width, channel) 
                for channels last or (channel, height, width) for channels first.
        """
        # the method for getting multiple tiles is used for getting only one tile
        tile = self.get_tiles(magnification, [location], shape, channels_last)[0]
        
        return tile

    def get_tiles(
        self, 
        magnification: float, 
        locations: list[tuple[int, int]],
        shapes: Union[tuple[int, int], list[tuple[int, int]]] = (256, 256), 
        channels_last: bool = True,
    ) -> list[np.ndarray]:
        """
        Get one or more tiles from the whole slide image based on the specified 
        magnification, location, and shape. The tiles are read using multithreading 
        if this was enabled in the settings.

        Args:
            magnification:  Magnification at which the whole slide image is loaded.
            locations:  Locations of top left pixel as (x, y) at the specified magnification.
            shape:  Shape of the tile in pixels as (height, width).
            channels_last:  Specifies if the channels dimension of the output tensor 
                is last. If False, the channels dimension is the second dimension.
        
        Returns:
            tile:  Whole slide image tiles [uint8] as (height, width, channel) 
                for channels last or (channel, height, width) for channels first.
        """
        # check if a slide has been loaded
        if self.__slide is None:
            raise ValueError('A slide must be loaded first.')
    
        # check if the specified magnification is valid
        upper_threshold = (self.__properties['native_magnification'] + 
                           self.__settings['max_difference'])
        if not (0.0 < magnification <= upper_threshold):
            message = ('The argument for `magnification` is invalid '
                       '(`magnification` must be in between 0.0 and '
                       f'{upper_threshold:0.3f}x).')
            raise ValueError(message)
        
        # determine the best level and correction factor
        level, correction_factor = self.__best_level_and_correction(magnification)

        # check if locations is a list 
        if not isinstance(locations, list):
            raise TypeError('The argument for `locations` must be a list of tuples.')
        # check if atleast one position was provided
        if not len(locations):
            return []     
        # check if a single shape was provided 
        if not isinstance(shapes, list):
            shapes = [shapes]
        # if only one shape was provided, duplicate the shape for each location
        if len(shapes) == 1:
            shapes *= len(locations)
        # check if the number of shapes and locations are equal
        elif len(shapes) != len(locations):
            raise ValueError('The number of tile shapes and location are unequal.')
  
        # calculate the downsample factor
        downsample_factor = self.__properties['native_magnification']/magnification

        native_locations = []
        # loop over combinations of tile location and shapes
        for location, shape in zip(locations, shapes):
            # check if the specified tile location and shape are valid
            dimensions = self.__properties['dimensions'][level]
            if not (0 <= round(location[0]*correction_factor) <= dimensions[1]):
                raise ValueError('Top left location is invalid.')
            if not (0 <= round(location[1]*correction_factor) <= dimensions[0]):
                raise ValueError('Top left location is invalid.')
            if not (0 <= round((location[0]+shape[1])*correction_factor) <= dimensions[1]):
                raise ValueError('Bottom right location is invalid.')
            if not (0 <= round((location[1]+shape[0])*correction_factor) <= dimensions[0]):
                raise ValueError('Bottom right location is invalid.')
            
            # calculate native locations
            native_locations.append((
                round(location[0]*downsample_factor),
                round(location[1]*downsample_factor),
            ))

        # only use multithreading in case of more than one tile
        N_tiles = len(locations)
        if self.__settings['multithreading'] and N_tiles > 1:
            # prepare tile reading function for multithreading
            read_tile = lambda native_location, shape: self.__read_tile(
                level=level, 
                native_location=native_location,
                shape=shape,  
                correction_factor=correction_factor,
            )
            # use multithreading for speedup in loading tiles
            with concurrent.futures.ThreadPoolExecutor() as executor:
                if self.__settings['progress_bar']:
                    tiles = list(tqdm(
                        executor.map(read_tile, native_locations, shapes), 
                        total=N_tiles,
                    ))
                else:
                    tiles = list(executor.map(read_tile, native_locations, shapes))
        else:
            # define tile iterator
            tile_iterator = zip(native_locations, shapes)
            if self.__settings['progress_bar']:
                tile_iterator = tqdm(tile_iterator, total=N_tiles)  

            tiles = []
            for native_location, shape in tile_iterator:
                # load the image tiles at the specified magnification level
                tiles.append(
                    self.__read_tile(
                        level=level,
                        native_location=native_location, 
                        shape=shape,
                        correction_factor=correction_factor,
                    ),
                )
        # change position of channels dimension
        if not channels_last:
            tiles = [np.transpose(tile, (2,0,1)) for tile in tiles]

        return tiles

    def get_image(
        self, 
        magnification: float, 
        channels_last: bool = True,
    ) -> np.ndarray:
        """
        Get the whole slide image at the specified magnification.        
        
        Args:
            magnification:  Magnification at which the whole slide image is loaded.
            channels_last:  Indicates if the channels dimension should be last.
                If False, the channels dimension is the second dimension.
        
        Returns:
            image:  Whole slide image [uint8] as (height, width, channel) 
                for channels last or as (channel, height, width) for channels first.
        """ 
        # check if a slide has been loaded
        if self.__slide is None:
            raise ValueError('A slide must be loaded first.')
        
        # check if the specified magnification is valid
        upper_threshold = (self.__properties['native_magnification'] + 
                           self.__settings['max_difference'])
        if not (0.0 < magnification <= upper_threshold):
            message = ('The argument for `magnification` is invalid '
                       '(`magnification` must be in between 0.0 and '
                       f'{upper_threshold:0.3f}x).')
            raise ValueError(message)
        
        # determine the best level and correction factor
        level, correction_factor = self.__best_level_and_correction(magnification)

        # determine the final image size
        height, width = self.get_dimensions(magnification)

        # check if the magnification value is valid
        if height <= 0 or width <= 0:
            raise ValueError('The argument for `magnification` is too small.')
        
        # load the image from the exact image pyramid level if available
        if ((magnification < min(self.__properties['magnification_levels'])) or 
            (correction_factor == 1)):
            # load the entire image at the specified downsample level
            image = self.__slide.read_region(
                location=(0,0),
                level=level, 
                size=tuple(list(self.__properties['dimensions'][level])[::-1]),
            )
            # remove the alpha channel if present
            image = np.array(image)[..., 0:3]

            # resize the image to the desired shape
            if magnification < min(self.__properties['magnification_levels']):
                image = resize(
                    image=image, 
                    output_shape=(height, width), 
                    anti_aliasing=self.__settings['anti-aliasing'], 
                    order=self.__settings['interpolation_order'],
                )
                image = img_as_ubyte(image)
        else:
            # determine locations and shapes of tiles
            chunk_shape = self.__settings['chunk_shape']
            locations = []
            shapes = []
            for y in range(ceil(height/chunk_shape[0])):
                for x in range(ceil(width/chunk_shape[1])):
                    locations.append((x*chunk_shape[1], y*chunk_shape[0]))
                    shapes.append((
                        min(height, (y+1)*chunk_shape[0])-(y*chunk_shape[0]),
                        min(width, (x+1)*chunk_shape[1])-(x*chunk_shape[1]),
                    ))
            # get tiles from the image at the requested magnification
            tiles = self.get_tiles(
                magnification=magnification, 
                locations=locations,
                shapes=shapes,
            )
            # stitch the tiles together to obtain full image
            image = np.zeros((height, width, 3), dtype=np.uint8)
            # add values of tiles to image
            for (x, y), shape, tile in zip(locations, shapes, tiles):
                image[y:y+shape[0], x:x+shape[1], :] = tile

        # change position of channels dimension
        if not channels_last:
            image = np.transpose(image, (2,0,1))

        return image   
    
    def close(self):
        self.__slide.close()


class DicomSlideLoader():
    """
    Class for loading and tiling DICOM whole slide images (WSIs) using the 
    WsiDicom class from the wsidicom package.
    """

    __settings = {
        'multithreading': True,
        'progress_bar': False,
        'chunk_shape': (1024, 1024),
        'max_difference': 0.025,
        'anti-aliasing': False,
        'interpolation_order': 2,
    }

    def __init__(self, settings: dict[str, Any] = {}) -> None:
        """
        Args:
            settings:  Dictionary with names of settings and corresponding values.
                Default values are used for each setting if unspecified.
        """
        # initialize instance attributes
        self.__slide = None
        self.__properties = None

        # configure settings
        for key in settings:
            self.__settings[key] = settings[key]

    def load_slide(self, paths: Union[str, Path, Sequence[Union[str, Path]]]) -> None:
        """
        Load whole slide image slide.

        Args:
            path:  Path to whole slide image.
        """
        # add path to list
        if isinstance(paths, (str, Path)):
            paths = [paths]
        # initialize instance variable with path to slide and load the slide
        self.__slide = wsidicom.WsiDicom.open(paths)

        # retrieve the native magnification
        mpp = self.__slide.levels.base_level.mpp
        if mpp.height != mpp.width:
            raise ValueError('Pixel height and width do not match.')
        native_magnification = 10.0/mpp.height

        # calculate the magnification for all downsample levels
        magnification_levels = []
        downsample_levels = []
        dimensions = []
        for _, level in self.__slide.levels._levels.items():
            magnification_levels.append(10.0/level.mpp.height)
            downsample_levels.append(native_magnification/magnification_levels[-1])
            dimensions.append((level.size.height, level.size.width))

        # resolution as (x, y) in micrometer / pixel  
        # dimensions as (height, width) in pixels
        self.__properties = {
            'vendor': pydicom.dcmread(sorted(paths)[0])[0x0008,0x0070].value,
            'native_resolution': (mpp.width, mpp.height), 
            'native_magnification': native_magnification,
            'magnification_levels': magnification_levels,
            'downsample_levels': downsample_levels,
            'dimensions': dimensions,    
        }   

    def get_properties(self) -> dict[str, Any]:
        return self.__properties

    def get_dimensions(self, magnification: float) -> tuple[int, int]:
        """
        Return the dimensions of the whole slide image at the specified magnification.
        
        Args:
            magnification:  The magnification factor to get the dimensions for.

        Returns:
            corrected_dimensions:  Dimensions of the whole slide image at the 
                specified magnification.
        """
        # determine the best level and correction factor
        level, correction_factor = self.__best_level_and_correction(magnification)
        # get the dimensions at the level and apply the correction
        dimensions = self.__properties['dimensions'][level]
        corrected_dimensions = (
            floor(dimensions[0]/correction_factor),
            floor(dimensions[1]/correction_factor),
        )
        return corrected_dimensions

    def __best_level_and_correction(self, magnification: float) -> tuple[float, float]:
        """
        Determine the best level of the image pyramid for getting tiles and
        calculate the correction factor for resizing to the exact magnification.
        
        Args:
            magnification:  The desired magnification factor.
        
        Returns:
            level:  Index of downsampling level with a magnification factor that
                is closest but larger than the desired magnification factor.
            correction_factor:  Magnification factor for the best level to
                downsample from divided by the desired magnification level.
        """
        # check the difference between the requested magnification and 
        # the magnifications available in the whole slide image
        differences = []
        for magnification_level in self.__properties['magnification_levels']:
            differences.append(magnification-magnification_level)

        # check if the smallest difference can be considered negligible
        abs_differences = list(map(abs, differences))
        if min(abs_differences) < self.__settings['max_difference']:
            level = abs_differences.index(min(abs_differences))
            correction_factor = 1
        else:
            # get the best downsample level for resizing the image
            # to the desired magnification level
            level = differences.index(
                max([diff for diff in differences if diff < 0]),
            )
            # calculate the correction factor 
            correction_factor = (
                self.__properties['magnification_levels'][level]/magnification
            )

        return level, correction_factor

    def __read_tile(
        self, 
        level: int, 
        location: tuple[int, int], 
        shape: tuple[int, int],
        correction_factor: float,
    ) -> np.ndarray:
        """
        Read a tile from the whole slide image.
        
        Args:
            level:  Downsample level used for loading the tile.
            correction_factor:  Magnification factor for the best level to
                downsample from divided by the desired magnification level.
            location:  Location of top left pixel as (x, y) at the specified 
                magnification.
            shape:  Shape of tile in pixels as (height, width) at the specified 
                magnification.
        
        Returns:
            tile:  Whole slide image tile.
        """
        read_settings = [(level, correction_factor)]
        for fallback_level in list(range(level))[::-1]:
            fallback_factor = (
                self.__properties['magnification_levels'][fallback_level]
                / self.__properties['magnification_levels'][level] 
                * correction_factor
            )
            read_settings.append((fallback_level, fallback_factor))
        
        for level, correction_factor in read_settings:
            # correct the tile location and shape if necessary
            if correction_factor == 1:
                corrected_location = location
                corrected_shape = (shape[1], shape[0])
            else:
                corrected_location = (
                    round(location[0]*correction_factor), 
                    round(location[1]*correction_factor),
                )
                corrected_shape = (
                    round(shape[1]*correction_factor), 
                    round(shape[0]*correction_factor),
                )
            # load the image tile at the specified magnification level
            try:
                tile = self.__slide.read_region(
                    location=corrected_location, 
                    level=round(log2(self.__properties['downsample_levels'][level])), 
                    size=corrected_shape,
                )
            except wsidicom.errors.WsiDicomFileError as error:
                continue
            else:
                tile = np.array(tile)[..., 0:3]
                # resize the tile if necessary
                if correction_factor != 1:
                    tile = resize(
                        image=tile, 
                        output_shape=shape, 
                        anti_aliasing=self.__settings['anti-aliasing'], 
                        order=self.__settings['interpolation_order'],
                    )
                    tile = img_as_ubyte(tile)
                
                # remove first axis if necessary
                if len(tile.shape) == 4:
                    tile = tile[0, ...]
            
                return tile
        
        raise error

    def get_tile(
        self, 
        magnification: float, 
        location: tuple[int, int],
        shape: tuple[int, int] = (256, 256), 
        channels_last: bool = True,
    ) -> np.ndarray:
        """
        Get a tile from the whole slide image based on the specified
        magnification, location, and shape.
        
        Args:
            magnification:  Magnification at which the whole slide image is loaded.
            location:  Location of top left pixel as (x, y) at the specified magnification.
            shape:  Shape of the tile in pixels as (height, width).
            channels_last:  Specifies if the channels dimension of the output tensor 
                is last. If False, the channels dimension is the second dimension.
        
        Returns:
            tile:  Whole slide image tile [uint8] as (height, width, channel) 
                for channels last or (channel, height, width) for channels first.
        """
        # the method for getting multiple tiles is used for getting only one tile
        tile = self.get_tiles(magnification, [location], shape, channels_last)[0]
        return tile

    def get_tiles(
        self, 
        magnification: float, 
        locations: list[tuple[int, int]],
        shapes: Union[tuple[int, int], list[tuple[int, int]]] = (256, 256), 
        channels_last: bool = True,
    ) -> list[np.ndarray]:
        """
        Get one or more tiles from the whole slide image based on the specified 
        magnification, location, and shape. The tiles are read using multithreading 
        if this was enabled in the settings.

        Args:
            magnification:  Magnification at which the whole slide image is loaded.
            locations:  Locations of top left pixel as (x, y) at the specified magnification.
            shape:  Shape of the tile in pixels as (height, width).
            channels_last:  Specifies if the channels dimension of the output tensor 
                is last. If False, the channels dimension is the second dimension.
        
        Returns:
            tile:  Whole slide image tiles [uint8] as (height, width, channel) 
                for channels last or (channel, height, width) for channels first.
        """
        # check if a slide has been loaded
        if self.__slide is None:
            raise ValueError('A slide must be loaded first.')
        
        # check if the specified magnification is valid
        upper_threshold = (self.__properties['native_magnification'] 
                           + self.__settings['max_difference'])
        if not (0.0 < magnification <= upper_threshold):
            message = ('The argument for `magnification` is invalid '
                       '(`magnification` must be in between 0.0 and '
                       f'{upper_threshold:0.3f}x).')
            raise ValueError(message)
        
        # determine the best level and correction factor
        best_level, best_correction_factor = self.__best_level_and_correction(magnification)
        
        # check if locations is a list 
        if not isinstance(locations, list):
            raise TypeError('The argument for `locations` must be a list of tuples.')
        # check if atleast one position was provided
        if not len(locations):
            return []     
        # check if a single shape was provided 
        if not isinstance(shapes, list):
            shapes = [shapes]
        # if only one shape was provided, duplicate the shape for each location
        if len(shapes) == 1:
            shapes *= len(locations)
        # check if the number of shapes and locations are equal
        elif len(shapes) != len(locations):
            raise ValueError('The number of tile shapes and location are unequal.')
 
        # loop over combinations of tile location and shapes
        tile_iterator = zip(locations, shapes)
        for location, shape in tile_iterator:
            # check if the specified tile location and shape are valid
            dimensions = self.__properties['dimensions'][best_level]
            if not (0 <= round(location[0]*best_correction_factor) <= dimensions[1]):
                raise ValueError('Top left location is invalid.')
            if not (0 <= round(location[1]*best_correction_factor) <= dimensions[0]):
                raise ValueError('Top left location is invalid.')
            if not (0 <= round((location[0]+shape[1])*best_correction_factor) <= dimensions[1]):
                raise ValueError('Bottom right location is invalid.')
            if not (0 <= round((location[1]+shape[0])*best_correction_factor) <= dimensions[0]):
                raise ValueError('Bottom right location is invalid.')

        # only use multithreading in case of more than one tile
        N_tiles = len(locations)
        if self.__settings['multithreading']:
            # prepare tile reading function for multithreading
            read_tile = lambda location, shape: self.__read_tile(
                level=best_level, 
                location=location, 
                shape=shape, 
                correction_factor=best_correction_factor, 
            )
            # use multithreading for speedup in loading tiles
            with concurrent.futures.ThreadPoolExecutor() as executor:
                if self.__settings['progress_bar']:
                    tiles = list(tqdm(
                        executor.map(read_tile, locations, shapes), 
                        total=N_tiles,
                    ))
                else:
                    tiles = list(executor.map(read_tile, locations, shapes))
        else:
            # define tile iterator
            tile_iterator = zip(locations, shapes)
            if self.__settings['progress_bar']:
                tile_iterator = tqdm(tile_iterator, total=N_tiles)                
            
            tiles = []
            for location, shape in tile_iterator:
                # load the image tiles at the specified magnification level
                tiles.append(
                    self.__read_tile(
                        level=best_level, 
                        location=location, 
                        shape=shape,
                        correction_factor=best_correction_factor,
                    ),
                )

        # change position of channels dimension
        if not channels_last:
            tiles = [np.transpose(tile, (2,0,1)) for tile in tiles]
        
        return tiles

    def get_image(
        self, 
        magnification: float, 
        channels_last: bool = True,
    ) -> np.ndarray:
        """
        Get the whole slide image at the specified magnification.     

        Args:
            magnification:  Magnification at which the whole slide image is loaded.
            channels_last:  Indicates if the channels dimension should be last.
                If False, the channels dimension is the second dimension.
        
        Returns:
            image:  Whole slide image [uint8] as (height, width, channel) 
                for channels last or as (channel, height, width) for channels first.
        """  
        # check if a slide has been loaded
        if self.__slide is None:
            raise ValueError('A slide must be loaded first.')
        
        # check if the specified magnification is valid
        upper_threshold = (self.__properties['native_magnification'] 
                           + self.__settings['max_difference'])
        if not (0.0 < magnification <= upper_threshold):
            message = ('The argument for `magnification` is invalid '
                       '(`magnification` must be in between 0.0 and '
                       f'{upper_threshold:0.3f}x).')
            raise ValueError(message)
        
        # determine the best level and correction factor
        level, correction_factor = self.__best_level_and_correction(magnification)

        # determine the final image size
        height, width = self.get_dimensions(magnification)

        # check if the magnification value is valid
        if height <= 0 or width <= 0:
            raise ValueError('The argument for `magnification` is too small.')

        # load the image from the exact image pyramid level if available
        read_tiles = True
        if ((magnification < min(self.__properties['magnification_levels'])) 
            or (correction_factor == 1)):
            # load the entire image at the specified downsample level
            try:
                image = self.__slide.read_region(
                    location=(0,0),
                    level=round(log2(self.__properties['downsample_levels'][level])), 
                    size=tuple(list(self.__properties['dimensions'][level][::-1])),
                )
            except wsidicom.errors.WsiDicomFileError:
                pass
            else:
                # remove the alpha channel if present
                image = np.array(image)[..., 0:3]

                # resize the image to the desired shape
                if magnification < min(self.__properties['magnification_levels']):
                    image = resize(
                        image=image, 
                        output_shape=(height, width), 
                        anti_aliasing=self.__settings['anti-aliasing'], 
                        order=self.__settings['interpolation_order'],
                    )
                    image = img_as_ubyte(image)
                
                read_tiles = False
        
        if read_tiles:
            # determine locations and shapes of tiles
            chunk_shape = self.__settings['chunk_shape']
            locations = []
            shapes = []
            for y in range(ceil(height/chunk_shape[0])):
                for x in range(ceil(width/chunk_shape[1])):
                    locations.append((x*chunk_shape[1], y*chunk_shape[0]))
                    shapes.append((
                        min(height, (y+1)*chunk_shape[0])-(y*chunk_shape[0]),
                        min(width, (x+1)*chunk_shape[1])-(x*chunk_shape[1]),
                    ))

            # get tiles from the image at the requested magnification
            tiles = self.get_tiles(
                magnification=magnification, 
                locations=locations,
                shapes=shapes,
            )
            # stitch the tiles together to obtain full image
            image = np.zeros((height, width, 3), dtype=np.uint8)
            # add values of tiles to image
            for (x, y), shape, tile in zip(locations, shapes, tiles):
                image[y:y+shape[0], x:x+shape[1], :] = tile

        # change position of channels dimension
        if not channels_last:
            image = np.transpose(image, (2,0,1))

        return image
    
    def close(self):
        self.__slide.close()