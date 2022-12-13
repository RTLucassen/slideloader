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
Utility class for loading and tiling whole slide images.
"""

import os
import math
import pydicom
import wsidicom
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from typing import Any, Union
from tqdm import tqdm
from math import ceil, floor
from skimage import img_as_ubyte
from skimage.transform import resize

OPENSLIDE_PATH = None

# import openslide
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    if OPENSLIDE_PATH is None:
        raise IOError('Specify the path to the OpenSlide bin folder.')
    else:
        with os.add_dll_directory(OPENSLIDE_PATH):
            import openslide
else:
    import openslide


class SlideLoader():
    """
    Class for loading and tiling whole slide images (WSIs).
    """

    __settings = {
        'inspection_mode': False,
        'progress_bar': True,
        'extraction_tile_shape': (1024, 1024),
        'percentual_diff_threshold': 0.01,
        'anti-aliasing': False,
        'interpolation_order': 2,
    }

    def __init__(
        self,
        settings: dict[str, Any] = {}, 
        multithreading: bool = True
    ) -> None:
        """
        Args:
            settings: dictionary with names of settings and values.
                      default values are used for each setting if unspecified.
            multithreading: indicates if tiles are loaded using multithreading.
        """
        # configure settings
        self.__dicom = False
        for key in settings:
            self.__settings[key] = settings[key]

        # initialize slide loader objects for non-dicom and dicom images
        self.__openslide_loader = OpenSlideLoader(
            self.__settings, 
            multithreading,
        )
        self.__dicomslide_loader = DicomSlideLoader(            
            self.__settings, 
            multithreading,
        )

    def load_slide(self, paths: Union[str, list[str]]) -> None:
        """
        Load whole slide image slide.
        
        Args:
            path: path to whole slide image.
        """
        # check if only a single path was specified
        if isinstance(paths, str):
            extension = os.path.splitext(paths)[1]
            # check if the extension is from a dicom image
            if extension == '.dcm':
                self.__dicom = True
                self.__dicomslide_loader.load_slide(paths)
            else:
                self.__dicom = False
                self.__openslide_loader.load_slide(paths)
        elif not len(paths):
             raise ValueError('No paths were provided.')
        elif len(paths) == 1:
            extension = os.path.splitext(paths[0])[1]
            # check if the extension is from a dicom image
            if extension == '.dcm':
                self.__dicom = True
                self.__dicomslide_loader.load_slide(paths[0])
            else:
                self.__dicom = False
                self.__openslide_loader.load_slide(paths[0])
        elif len(paths) != len(set(paths)):
            raise ValueError('Duplicate paths were provided.')
        else:
            cases = set([os.path.split(path)[1].split('.')[0] for path in paths])
            extensions = set([os.path.splitext(path)[1] for path in paths])
            if len(extensions) > 1:
                raise ValueError('At least two paths end with different file types.')
            elif extensions.pop() != '.dcm':
                raise ValueError(('Only multiple paths to the same DICOM image '
                    'at different magnifications can be loaded in at once.'))
            elif len(cases) > 1:
                raise ValueError('At least two paths are from different DICOM images.')
            else:
                self.__dicom = True
                self.__dicomslide_loader.load_slide(paths)

    def get_properties(self) -> dict:
        if self.__dicom:
            return self.__dicomslide_loader.get_properties()
        else:
            return self.__openslide_loader.get_properties()
    
    def get_image(
        self, 
        magnification: float, 
        channels_last: bool = True,
    ) -> np.ndarray:
        """
        Args:
            magnification: magnification at which the image is loaded.
            channels_last: indicates if the channels dimension should be last.
                           if False, the channels dimension is the second dimension.
        Returns:
            image: whole slide image [uint8] in grayscale or with RGB color 
                   channels as (row, column, channel) by default.
        """
        if self.__dicom:
            return self.__dicomslide_loader.get_image(magnification, channels_last)
        else:
            return self.__openslide_loader.get_image(magnification, channels_last)

    def get_tiles(
        self, 
        magnification: float, 
        tile_shape: tuple = (256, 256), 
        overlap: tuple = (0.0, 0.0),  
        channels_last: bool = True,
        include_partial_tiles: bool = True,
    ) -> np.ndarray:
        """
        Args:
            magnification: magnification at which the image is loaded.
            tile_shape: shape of tile in pixels as (row, column).
            overlap: overlap faction between extracted tiles as (row, column).
            channels_last: specifies if the channels dimension of the output tensor is last.
                           if False, the channels dimension is the second dimension.
            include_partial_tiles: specifies if partial tiles at the border are included.
        
        Returns:
            tiles: whole slide image tiles with RGB color channels as 
                   (tile, row, column, channel).
            information: tile information containing the position as (row, column)
                         and pixel locations of top left corner as (x, y).
        """
        if self.__dicom:
            return self.__dicomslide_loader.get_tiles(
                magnification,
                tile_shape,
                overlap,
                channels_last,
                include_partial_tiles,
            )
        else:
            return self.__openslide_loader.get_tiles(
                magnification,
                tile_shape,
                overlap,
                channels_last,
                include_partial_tiles,
            )

class OpenSlideLoader():
    """
    Class for loading and tiling whole slide images (WSIs) 
    using the OpenSlide ImageSlide class.
    """

    __settings = {
        'inspection_mode': False,
        'progress_bar': True,
        'extraction_tile_shape': (1024, 1024),
        'percentual_diff_threshold': 0.01,
        'anti-aliasing': False,
        'interpolation_order': 2,
    }

    def __init__(
        self,
        settings: dict[str, Any] = {}, 
        multithreading: bool = True
    ) -> None:
        """
        Args:
            settings: dictionary with names of settings and values.
                      default values are used for each setting if unspecified.
            multithreading: indicates if tiles are loaded using multithreading.
        """
        # initialize instance attributes
        self.__slide = None
        self.__properties = None

        # configure settings
        self.__multithreading = multithreading
        for key in settings:
            self.__settings[key] = settings[key]  

    def load_slide(self, path: str) -> None:
        """
        Load whole slide image slide.
        Args:
            path: path to whole slide image.
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
        # dimensions as (row, column) in pixels
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

    def get_properties(self) -> dict:
        return self.__properties

    def get_image(
        self, 
        magnification: float, 
        channels_last: bool = True,
    ) -> np.ndarray:
        """
        Args:
            magnification: magnification at which the image is loaded.
            channels_last: indicates if the channels dimension should be last.
                           if False, the channels dimension is the second dimension.
        Returns:
            image: whole slide image [uint8] in grayscale or with RGB color 
                   channels as (row, column, channel) by default.
        """
        # check if a slide has been loaded
        if self.__slide is None:
            raise AssertionError('A slide must be loaded first.')
        
        # check if the specified magnification is valid
        upper_threshold = (self.__properties['native_magnification'] * 
            (1+self.__settings['percentual_diff_threshold']))
        if magnification < 0 or magnification > upper_threshold:
            message = ('The argument for `magnification` is invalid '
                       '(`magnification` must be in between 0.0 and '
                       f'{upper_threshold:0.3f}x).')
            raise ValueError(message)
        
        # check the difference between the requested magnification and 
        # the magnifications available in the pyramid image
        percentual_diff = []
        for mag_level in self.__properties['magnification_levels']:
            percentual_diff.append(abs((magnification-mag_level)/mag_level))
        
        # check if the smallest difference can be considered negligible
        if min(percentual_diff) < self.__settings['percentual_diff_threshold']:
            level = percentual_diff.index(min(percentual_diff))
            selected_magnification = magnification
            correction_factor = 1
        else:
            # get the best downsample level for resizing the image 
            # to the desired magnification level
            downsample_factor = self.__properties['native_magnification']/magnification
            level = self.__slide.get_best_level_for_downsample(downsample_factor)
            selected_magnification = self.__properties['magnification_levels'][level]
            correction_factor = magnification/selected_magnification
        
        # load the downsampled image from the pyramid representation if available
        if correction_factor == 1:
            # load the entire image at the specified downsample level
            image = self.__slide.read_region(
                location=(0,0),
                level=level, 
                size=tuple(list(self.__properties['dimensions'][level])[::-1]),
            )
            # remove the alpha channel if present
            image = np.array(image)[..., 0:3]
        else:
            # extract tiles from the image at the requested magnification
            tiles, information = self.get_tiles(
                magnification=magnification, 
                tile_shape=self.__settings['extraction_tile_shape'],
                include_partial_tiles=True,
                overlap=(0,0)
            )
            length_y, length_x = information['tile_shape']
            # stitch the tiles together to obtain full image
            image = np.zeros((
                length_y*(information['positions'][-1][0]+1), 
                length_x*(information['positions'][-1][1]+1), 
                3), dtype=np.uint8
            )
            # add values of tiles to image
            for i, tile in enumerate(tiles):
                top_left_x, top_left_y = information['locations'][i]
                bottom_left_y = top_left_y+length_y
                top_right_x = top_left_x+length_x
                image[top_left_y:bottom_left_y, top_left_x:top_right_x] = tile
            # remove the zero-padding added to partial tiles
            image = image[
                :np.count_nonzero(np.sum(image, axis=(1,2))), 
                :np.count_nonzero(np.sum(image, axis=(0,2))),
            ]

        # change position of channels dimension
        if not channels_last:
            image = np.transpose(image, (2,0,1))

        return image           

    def get_tiles(
        self, 
        magnification: float, 
        tile_shape: tuple = (256, 256), 
        overlap: tuple = (0.0, 0.0),  
        channels_last: bool = True,
        include_partial_tiles: bool = True,
    ) -> np.ndarray:
        """
        Args:
            magnification: magnification at which the image is loaded.
            tile_shape: shape of tile in pixels as (row, column).
            overlap: overlap faction between extracted tiles as (row, column).
            channels_last: specifies if the channels dimension of the output tensor is last.
                           if False, the channels dimension is the second dimension.
            include_partial_tiles: specifies if partial tiles at the border are included.
        Returns:
            tiles: whole slide image tiles with RGB color channels as 
                   (tile, row, column, channel).
            information: tile information containing the position as (row, column)
                         and pixel locations of top left corner as (x, y).
        """
        # check if a slide has been loaded
        if self.__slide is None:
            raise AssertionError('A slide must be loaded first.')
    
        # check if the specified magnification is valid
        upper_threshold = (self.__properties['native_magnification'] * 
            (1+self.__settings['percentual_diff_threshold']))
        if magnification < 0 or magnification > upper_threshold:
            message = ('The argument for `magnification` is invalid '
                       '(`magnification` must be in between 0.0 and '
                       f'{upper_threshold:0.3f}x).')
            raise ValueError(message)
        
        # check the difference between the requested magnification and 
        # the magnifications available in the pyramid image
        percentual_diff = []
        for mag_level in self.__properties['magnification_levels']:
            percentual_diff.append(abs((magnification-mag_level)/mag_level))

        # check if the smallest difference can be considered negligible
        if min(percentual_diff) < self.__settings['percentual_diff_threshold']:
            level = percentual_diff.index(min(percentual_diff))
            correction_factor = 1
        else:
            # get the best downsample level for resizing the image
            # to the desired magnification level
            level = self.__slide.get_best_level_for_downsample(
                self.__properties['native_magnification']/magnification,
            )
            # calculate the correction factor 
            correction_factor = self.__properties['magnification_levels'][level]/magnification
        
        # correct the tile shape if necessary
        if correction_factor == 1:
            corrected_tile_shape = tile_shape
        else:
            corrected_tile_shape = (
                tile_shape[0]*correction_factor, 
                tile_shape[1]*correction_factor,
            )         
        # get the image dimensions at the specified magnification level
        dimensions = self.__properties['dimensions'][level]
        # calculate the downsample factor
        downsample_factor = (self.__properties['native_magnification'] /
            self.__properties['magnification_levels'][level])
        # calculate the stride for both spatial dimensions
        stride = (
            corrected_tile_shape[0]*(1-overlap[0]), 
            corrected_tile_shape[1]*(1-overlap[1])
        )
        
        # calculate the number of tiles in each row and column
        if include_partial_tiles:
            tiles_per_row = ceil(dimensions[0]/stride[0])
            tiles_per_column = ceil(dimensions[1]/stride[1])
        else:
            tiles_per_row = floor((dimensions[0]-corrected_tile_shape[0])/stride[0])
            tiles_per_column = floor((dimensions[1]-corrected_tile_shape[1])/stride[1])
        N_tiles = tiles_per_row * tiles_per_column
       
        # initialize lists with the position (row, column) 
        positions = []
        for row in range(tiles_per_row):
            for column in range(tiles_per_column):
                positions.append((row, column))
        
        locations = []
        native_locations = []
        for position in positions:
            # add location of the top left pixel (x, y) for all tiles to a list
            locations.append(
                (round(position[1]*stride[1]/correction_factor),
                round(position[0]*stride[0]/correction_factor)),
            )
            # add location of the top left pixel (x, y) for all tiles to a list
            # in the native magnification image
            native_locations.append(
                (round(position[1]*stride[1]*downsample_factor),
                round(position[0]*stride[0]*downsample_factor)),
            )

        # store tile information in a dictionary 
        information = {
            'tile_shape': tile_shape, 
            'locations': locations, 
            'positions': positions,
        }

        if self.__multithreading:
            # prepare tile reading function for multi-threading
            read_tile = lambda native_location: self.__read_tile(
                level=level, 
                tile_shape=tile_shape, 
                native_location=native_location, 
                corrected_tile_shape=corrected_tile_shape, 
                correction_factor=correction_factor, 
                add_axis=True,
            )
            # use multithreading for speedup in loading tiles
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # convert the clips to separate frames
                if self.__settings['progress_bar']:
                    collected_tiles = tqdm(
                        executor.map(read_tile, native_locations), 
                        total=N_tiles,
                    )
                else:
                    collected_tiles = executor.map(read_tile, native_locations)
                # save retrieved tiles in allocated memory
                tiles = np.concatenate(list(collected_tiles), axis=0)
        else:
            # allocate memory to store tiles in array
            tiles = np.zeros((N_tiles, tile_shape[0], tile_shape[1], 3), dtype=np.uint8)

            if self.__settings['progress_bar']:
                tile_iterator = tqdm(range(N_tiles))
            else:
                tile_iterator = range(N_tiles)

            for i in tile_iterator:
                # load the image tile at the specified magnification level
                tile = self.__read_tile(
                    level=level, 
                    tile_shape=tile_shape, 
                    native_location=native_locations[i], 
                    corrected_tile_shape=corrected_tile_shape, 
                    correction_factor=correction_factor,
                )
                tiles[i, ...] = tile
                
        if self.__settings['inspection_mode']:
            # allocate memory for creating the inspection image
            inspection_image = np.zeros((
                round((tiles_per_row-1)*stride[0]/correction_factor+tile_shape[0]),
                round((tiles_per_column-1)*stride[1]/correction_factor+tile_shape[1]), 
                3,
            ))
            print('Shape of inspection image: ', inspection_image.shape, '\n')

            for i in range(N_tiles):
                # retrieve the tile
                tile = tiles[i, ...] 
                # retrieve the coordinate of the top left pixel
                top_left_x, top_left_y = information['locations'][i] 
                # add the tile to the inspection image
                inspection_image[top_left_y:top_left_y+tile_shape[0], 
                                 top_left_x:top_left_x+tile_shape[1], :] += tile

            # show inspection image
            plt.imshow(inspection_image/np.max(inspection_image, axis=(0,1)))
            plt.show()   

        # change position of channels dimension
        if not channels_last:
            tiles = np.transpose(tiles, (0,3,1,2))
        
        return tiles, information

    def __read_tile(
        self, 
        level: int, 
        tile_shape: tuple,
        native_location: tuple, 
        corrected_tile_shape: tuple,
        correction_factor: float,
        add_axis: bool = False
    ) -> np.ndarray:
        """
        Read tile from pyramid image.
        
        Args:
            native_location: location of top left pixel (x, y) in the native magnification image.
            level: downsample level used for loading the tile.
            tile_shape: shape of tile in pixels as (row, column).
            corrected_tile_shape: shape of tile in the native magnification image in pixels as (row, column).
            correction_factor: best magnification (to downsample from) divided by the selected magnification. 
            add_axis: specifies if an additional axis in the first position should be added.
        Returns:
            tile: extracted image tile.
        """
        # load the image tile at the specified magnification level
        tile = self.__slide.read_region(
            location=native_location, 
            level=level, 
            size=(
                round(corrected_tile_shape[1]), 
                round(corrected_tile_shape[0]),
            ),
        )
        tile = np.array(tile)[..., 0:3]
        # resize the tile if necessary
        if correction_factor != 1:
            tile = resize(
                image=tile, 
                output_shape=tile_shape, 
                anti_aliasing=self.__settings['anti-aliasing'], 
                order=self.__settings['interpolation_order'],
            )
            tile = img_as_ubyte(tile)
        # add additional axis in the first position
        if add_axis:
            tile = tile[None, ...]
        
        return tile


class DicomSlideLoader():
    """
    Class for loading and tiling DICOM whole slide images (WSIs) 
    using the WsiDicom class from the wsidicom package.
    """

    __settings = {
        'inspection_mode': False,
        'progress_bar': True,
        'extraction_tile_shape': (1024, 1024),
        'percentual_diff_threshold': 0.01,
        'anti-aliasing': False,
        'interpolation_order': 2,
    }

    def __init__(
        self,
        settings: dict[str, Any] = {}, 
        multithreading: bool = True
    ) -> None:
        """
        Args:
            settings: dictionary with names of settings and values.
                      default values are used for each setting if unspecified.
            multithreading: indicates if tiles are loaded using multithreading.
        """
        # initialize instance attributes
        self.__slide = None
        self.__properties = None

        # configure settings
        self.__multithreading = multithreading
        for key in settings:
            self.__settings[key] = settings[key]
        
    def load_slide(self, paths: Union[str, list[str]]) -> None:
        """
        Load whole slide image slide.
        Args:
            path: path to whole slide image.
        """
        # add path to list
        if isinstance(paths, str):
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
        # dimensions as (row, column) in pixels
        self.__properties = {
            'vendor': pydicom.dcmread(sorted(paths)[0])[0x0008,0x0070].value,
            'native_resolution': (mpp.width, mpp.height), 
            'native_magnification': native_magnification,
            'magnification_levels': magnification_levels,
            'downsample_levels': downsample_levels,
            'dimensions': dimensions,    
        }   

    def get_properties(self) -> dict:
        return self.__properties

    def get_image(
        self, 
        magnification: float, 
        channels_last: bool = True,
    ) -> np.ndarray:
        """
        Args:
            magnification: magnification at which the image is loaded.
            channels_last: indicates if the channels dimension should be last.
                           if False, the channels dimension is the second dimension.
        Returns:
            image: whole slide image [uint8] in grayscale or with RGB color 
                   channels as (row, column, channel) by default.
        """
        # check if a slide has been loaded
        if self.__slide is None:
            raise AssertionError('A slide must be loaded first.')
        
        # check if the specified magnification is valid
        upper_threshold = (self.__properties['native_magnification'] * 
            (1+self.__settings['percentual_diff_threshold']))
        if magnification < 0 or magnification > upper_threshold:
            message = ('The argument for `magnification` is invalid '
                       '(`magnification` must be in between 0.0 and '
                       f'{upper_threshold:0.3f}x).')
            raise ValueError(message)
        
        # check the difference between the requested magnification and 
        # the magnifications available in the pyramid image
        percentual_diff = []
        abs_percentual_diff = []
        for mag_level in self.__properties['magnification_levels']:
            percentual_diff.append((magnification-mag_level)/mag_level)
            abs_percentual_diff.append(abs((magnification-mag_level)/mag_level))
        
        # check if the smallest difference can be considered negligible
        if min(abs_percentual_diff) < self.__settings['percentual_diff_threshold']:
            level = abs_percentual_diff.index(min(abs_percentual_diff))
            selected_magnification = magnification
            correction_factor = 1
        else:
            # get the best downsample level for resizing the image 
            # to the desired magnification level
            level = percentual_diff.index(max([diff for diff in percentual_diff if diff < 0]))
            selected_magnification = self.__properties['magnification_levels'][level]
            correction_factor = magnification/selected_magnification
        
        # load the downsampled image from the pyramid representation if available
        if correction_factor == 1:
            # load the entire image at the specified downsample level
            image = self.__slide.read_region(
                location=(0,0),
                level=round(math.log2(self.__properties['downsample_levels'][level])), 
                size=tuple(list(self.__properties['dimensions'][level][::-1])),
            )
            # remove the alpha channel if present
            image = np.array(image)[..., 0:3]
        else:
            # extract tiles from the image at the requested magnification
            tiles, information = self.get_tiles(
                magnification=magnification, 
                tile_shape=self.__settings['extraction_tile_shape'],
                include_partial_tiles=True,
                overlap=(0,0)
            )
            length_y, length_x = information['tile_shape']
            # stitch the tiles together to obtain full image
            image = np.zeros((
                length_y*(information['positions'][-1][0]+1), 
                length_x*(information['positions'][-1][1]+1), 
                3), dtype=np.uint8
            )
            # add values of tiles to image
            for i, tile in enumerate(tiles):
                top_left_x, top_left_y = information['locations'][i]
                bottom_left_y = top_left_y+length_y
                top_right_x = top_left_x+length_x
                image[top_left_y:bottom_left_y, top_left_x:top_right_x] = tile
            # remove the zero-padding added to partial tiles
            image = image[
                :np.count_nonzero(np.sum(image, axis=(1,2))), 
                :np.count_nonzero(np.sum(image, axis=(0,2))),
            ]

        # change position of channels dimension
        if not channels_last:
            image = np.transpose(image, (2,0,1))

        return image           

    def get_tiles(
        self, 
        magnification: float, 
        tile_shape: tuple = (256, 256), 
        overlap: tuple = (0.0, 0.0),  
        channels_last: bool = True,
        include_partial_tiles: bool = True,
    ) -> np.ndarray:
        """
        Args:
            magnification: magnification at which the image is loaded.
            tile_shape: shape of tile in pixels as (row, column).
            overlap: overlap faction between extracted tiles as (row, column).
            channels_last: specifies if the channels dimension of the output tensor is last.
                           if False, the channels dimension is the second dimension.
            include_partial_tiles: specifies if partial tiles at the border are included.
        Returns:
            tiles: whole slide image tiles with RGB color channels as 
                   (tile, row, column, channel).
            information: tile information containing the position as (row, column)
                         and pixel locations of top left corner as (x, y).
        """
        # check if a slide has been loaded
        if self.__slide is None:
            raise AssertionError('A slide must be loaded first.')
        
        # check if the specified magnification is valid
        upper_threshold = (self.__properties['native_magnification'] * 
            (1+self.__settings['percentual_diff_threshold']))
        if magnification < 0 or magnification > upper_threshold:
            message = ('The argument for `magnification` is invalid '
                       '(`magnification` must be in between 0.0 and '
                       f'{upper_threshold:0.3f}x).')
            raise ValueError(message)
        
        # check the difference between the requested magnification and 
        # the magnifications available in the pyramid image
        percentual_diff = []
        abs_percentual_diff = []
        for mag_level in self.__properties['magnification_levels']:
            percentual_diff.append((magnification-mag_level)/mag_level)
            abs_percentual_diff.append(abs((magnification-mag_level)/mag_level))

        # check if the smallest difference can be considered negligible
        if min(abs_percentual_diff) < self.__settings['percentual_diff_threshold']:
            level = abs_percentual_diff.index(min(abs_percentual_diff))
            correction_factor = 1
        else:
            # get the best downsample level for resizing the image
            # to the desired magnification level
            level = percentual_diff.index(max([diff for diff in percentual_diff if diff < 0]))
            # calculate the correction factor 
            correction_factor = self.__properties['magnification_levels'][level]/magnification
        
        # correct the tile shape if necessary
        if correction_factor == 1:
            corrected_tile_shape = tile_shape
        else:
            corrected_tile_shape = (
                tile_shape[0]*correction_factor, 
                tile_shape[1]*correction_factor,
            )         
        # get the image dimensions at the specified magnification level
        dimensions = self.__properties['dimensions'][level]
        # calculate the stride for both spatial dimensions
        stride = (
            corrected_tile_shape[0]*(1-overlap[0]), 
            corrected_tile_shape[1]*(1-overlap[1])
        )
        # calculate the number of tiles in each row and column
        if include_partial_tiles:
            tiles_per_row = ceil(dimensions[0]/stride[0])
            tiles_per_column = ceil(dimensions[1]/stride[1])
        else:
            tiles_per_row = floor((dimensions[0]-corrected_tile_shape[0])/stride[0])
            tiles_per_column = floor((dimensions[1]-corrected_tile_shape[1])/stride[1])
        N_tiles = tiles_per_row * tiles_per_column
       
        # initialize lists with the position (row, column) 
        positions = []
        for row in range(tiles_per_row):
            for column in range(tiles_per_column):
                positions.append((row, column))
        
        locations = []
        native_locations = []
        for position in positions:
            # add location of the top left pixel (x, y) for all tiles to a list
            locations.append(
                (round(position[1]*stride[1]/correction_factor),
                round(position[0]*stride[0]/correction_factor)),
            )
            # add location of the top left pixel (x, y) for all tiles to a list
            # in the native magnification image
            native_locations.append(
                (round(position[1]*stride[1]),
                round(position[0]*stride[0])),
            )

        # store tile information in a dictionary 
        information = {
            'tile_shape': tile_shape, 
            'locations': locations, 
            'positions': positions,
        }

        if self.__multithreading:
            # prepare tile reading function for multi-threading
            read_tile = lambda native_location: self.__read_tile(
                level=level, 
                tile_shape=tile_shape, 
                native_location=native_location, 
                corrected_tile_shape=corrected_tile_shape, 
                correction_factor=correction_factor, 
                add_axis=True,
            )
            # use multithreading for speedup in loading tiles
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # convert the clips to separate frames
                if self.__settings['progress_bar']:
                    collected_tiles = tqdm(
                        executor.map(read_tile, native_locations), 
                        total=N_tiles,
                    )
                else:
                    collected_tiles = executor.map(read_tile, native_locations)
                # save retrieved tiles in allocated memory
                tiles = np.concatenate(list(collected_tiles), axis=0)
        else:
            # allocate memory to store tiles in array
            tiles = np.zeros((N_tiles, tile_shape[0], tile_shape[1], 3), dtype=np.uint8)

            if self.__settings['progress_bar']:
                tile_iterator = tqdm(range(N_tiles))
            else:
                tile_iterator = range(N_tiles)

            for i in tile_iterator:
                # load the image tile at the specified magnification level
                tile = self.__read_tile(
                    level=level, 
                    tile_shape=tile_shape, 
                    native_location=native_locations[i], 
                    corrected_tile_shape=corrected_tile_shape, 
                    correction_factor=correction_factor,
                )
                tiles[i, ...] = tile
                
        if self.__settings['inspection_mode']:
            # allocate memory for creating the inspection image
            inspection_image = np.zeros((
                round((tiles_per_row-1)*stride[0]/correction_factor+tile_shape[0]),
                round((tiles_per_column-1)*stride[1]/correction_factor+tile_shape[1]), 
                3,
            ))
            print('Shape of inspection image: ', inspection_image.shape, '\n')

            for i in range(N_tiles):
                # retrieve the tile
                tile = tiles[i, ...] 
                # retrieve the coordinate of the top left pixel
                top_left_x, top_left_y = information['locations'][i] 
                # add the tile to the inspection image
                inspection_image[top_left_y:top_left_y+tile_shape[0], 
                                 top_left_x:top_left_x+tile_shape[1], :] += tile

            # show inspection image
            plt.imshow(inspection_image/np.max(inspection_image, axis=(0,1)))
            plt.show()   

        # change position of channels dimension
        if not channels_last:
            tiles = np.transpose(tiles, (0,3,1,2))
        
        return tiles, information

    def __read_tile(
        self, 
        level: int, 
        tile_shape: tuple,
        native_location: tuple, 
        corrected_tile_shape: tuple,
        correction_factor: float,
        add_axis: bool = False
    ) -> np.ndarray:
        """
        Read tile from pyramid image.
        
        Args:
            native_location: location of top left pixel (x, y) in the native magnification image.
            level: downsample level used for loading the tile.
            tile_shape: shape of tile in pixels as (row, column).
            corrected_tile_shape: shape of tile in the native magnification image in pixels as (row, column).
            correction_factor: best magnification (to downsample from) divided by the selected magnification. 
            add_axis: specifies if an additional axis in the first position should be added.
        Returns:
            tile: extracted image tile.
        """
        # calculate corrections for when the tile exceeds the edge of the image
        height_correction = max(native_location[1]+round(corrected_tile_shape[0]) -
            self.__properties['dimensions'][level][0], 0)
        width_correction = max(native_location[0]+round(corrected_tile_shape[1]) -
            self.__properties['dimensions'][level][1], 0)

        # load the image tile at the specified magnification level
        tile = self.__slide.read_region(
            location=native_location, 
            level=round(math.log2(self.__properties['downsample_levels'][level])), 
            size=(
                round(corrected_tile_shape[1])-width_correction, 
                round(corrected_tile_shape[0])-height_correction,
            ),
        )
        tile = np.array(tile)[..., 0:3]
        tile = np.pad(tile, ((0,height_correction), (0,width_correction), (0,0)))

        # resize the tile if necessary
        if correction_factor != 1:
            tile = resize(
                image=tile, 
                output_shape=tile_shape, 
                anti_aliasing=self.__settings['anti-aliasing'], 
                order=self.__settings['interpolation_order'],
            )
            tile = img_as_ubyte(tile)
        # add additional axis in the first position
        if add_axis:
            tile = tile[None, ...]
        
        return tile