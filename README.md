# CellQuant
These functions are meant to help you learn about cells in images once masks have been generated for each cell.

(To genearate cell masks you may want to check out: https://github.com/SydShafferLab/DeepCellHelper)

## List of funtions

```python
MaskCheck(img_path, mask_path, channel=1, min_brightness=.15):
```
MaskCheck outlines the masks on top of the original image to check if the masks are doing a good job captuing the cell (outputs a jpeg). 
As input it takes the path to the raw tif image (img_path), and the path to the mask file you want to check which should be a tif file where each mask has a unique intensity(mask_path).
Optionally, you can enter which channel you want the mask to be overlayed on (default is 1 (uses base 1)). If the output image brightness does not look righ you can also change the optional min_brighness parameter.

Example output:

<img src="https://github.com/gharmange/CellQuant/blob/main/Images/MaskCheck_example.png" width="250" height="250">

```python
MaskPos(Mask_path)
```
MaskPos generates a numpy array and csv file containing the x and y coordinates of each mask in a mask file (input mask file should be a tif file where each mask has a unique intensity, funtion takes the path to the mask as input). The csv file will be output
in the direcory of the mask file with "_Cellxy.csv" at the end of the file name.

```python
MaskQuant(Image_path,Mask_path,CHANNELS)
```
MaskQuant generates a table containing the mean and sum intensities within each mask. It takes in the path to the raw image you are quantifying (Image_path), the path to the mask you are quantifying (Mask_path) (input mask file should be a tif file where each mask has a unique intensity),
and a list of the channels of the image you want to quantify (CHANNELS). The CHANNELS list is base 1, for example to just quantify channel 1 input: [1], to quantify channel 1 and 3 input: [1,3].

```python
TestRad(path,radius_um,objective)
```
One metric that may be of interest is how dense your cells are. We find that a good way to calculate density for each cell is to determine how many cells are within a given radius of each cell. To do quickly test what radius
gives good density results for your cells use TestRad to test different radiuses. This function takes the path to xy cordinates of masks geneated by the MaskPos function (path), the radius you would like to try in um (radius_um),
and what objective was used to take the image (objective) (can input '10x' or '20x'). NOTE: the accuracy of the um value depends on the scope used and may need to be changed in the funtion if not using the Shaffer Lab scope. The ouput of this function is a png showing the density value of each cell as a heatmap.

```python
DensityQuant(path,radius_um,objective)
```
DensityQuant generates a density metric for each cell by determining how many cells are with a given radius of each cell. This function takes the path to a folder containing xy cordinates of masks geneated by the MaskPos function (path),
the radius you would like to try in um (radius_um), and what objective was used to take the image (objective) (can input '10x' or '20x'). NOTE: the accuracy of the um value depends on the scope used and may need to be changed in the funtion if not using the Shaffer Lab scope.
The ouput of this function is a png showing the density value of each cell as a heatmap.

```python
PlotDensity(path)
```
PlotDesity generates a scatter plot showing the density of each cell using a heatmap. The input to this function is a table containing the density metric of each cell created with the DensityQuant function.


```python
CytoNucRatios(Image_path, Mask_path_Nuc, Mask_path_cyto,CHANNELS)
```
makeCytoMask generates a mask to sample the cytoplasm of cells based on a nuclear mask. The input to this function is: the path to the nuclear mask file (nucPath) (input mask file should be a tif file where each mask has a unique intensity). (Function made by Shivani Nellore)

```python
CytoNucRatios(Image_path, Mask_path_Nuc, Mask_path_cyto,CHANNELS)
```
CytoNucRatios calculates the cytoplasm to nuclear ratio (both mean and median) as well as records each cells nuclear and cytoplasm size. The inputs to this function are: the path to the raw image in the tif format (Image_path), the path to the nuclear masks (Mask_path_Nuc),
the path tho the cytoplasm mask (Mask_path_cyto), and the list of channels you would like to run cytoplasm to nuclear ratio on (CHANNELS). The Mask files should be a tif files where each mask has a unique intensity, and the matched nuclear and cytolasm masks should have the 
same intensity values. The CHANNELS list is base 1, for example to just quantify channel 1 input: [1], to quantify channel 1 and 3 input: [1,3]. (function written in collaboration with Shivani Nellore)
