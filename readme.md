

Code for the ABRF/LMRG Image Analysis Study ([link](https://sites.google.com/view/lmrg-image-analysis-study))

## Requirements
- python 3.x and the usual stuff (napari, skimage, scipy, numpy, pandas) and tifffile for tiff import
- all of the tiff image stacks downloaded into the ```/data``` folder

## Usage
Running ```do_seg.py``` will execute the segmentation for all of the image stacks (and the calibration images) and export results to file.

If you set ```view=True``` in the ```__main__``` part of the script, a napari viewer is spawned for each segmentation, allowing inspection of the results.

Results are exported to csv files (```/results``` folder) and the parameters used for each segmentation are exported to a json file.

## Gallery




