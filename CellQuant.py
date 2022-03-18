from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import tifffile
import cv2 as cv
from skimage import measure



# This function plots an outline of the mask onto the image to check how well the mask is capturing objects of interest
#code for finding the outline take from Cellori (https://github.com/SydShafferLab/Cellori)
def MaskCheck(img_path, mask_path, channel=1, min_brightness=.15):

    #load mask
    mask=tifffile.imread(mask_path)

    if len(mask.shape) == 2:
        mask = mask
    elif len(mask.shape) == 3:
        mask = mask[...,0]
    elif len(mask.shape) == 4:
        mask = mask[0,...,0]
    else:
        print("mask has unsupported dimensions")


    #load image
    image = tifffile.imread(img_path)
    imgShape = len(image.shape)
    if imgShape == 2:
        pass
    else:
        image = image[channel-1,...]

    #normalize tiff image
    norm_image = cv.normalize(image, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

    # brighten up image if necessary (code taken from: https://stackoverflow.com/questions/57030125/automatically-adjusting-brightness-of-image-with-opencv)
    cols, rows = norm_image.shape
    brightness = np.sum(norm_image) / (255 * cols * rows)

    ratio = brightness / min_brightness
    if ratio >= 1:
        print("no Scale")
    else:
        # Otherwise, adjust brightness to get the target brightness
        norm_image = cv.convertScaleAbs(norm_image, alpha = 1 / ratio, beta = 0)

    #convert scaled gray scale image to color
    col_norm_image = cv.cvtColor(norm_image, cv.COLOR_GRAY2BGR)

    #convert mask to outline( Code from: https://github.com/SydShafferLab/Cellori)
    REGIONS = measure.regionprops(mask,cache=False)

    outlines = np.zeros(mask.shape,dtype=bool)

    for region in REGIONS:
        c_mask = region.image.astype(np.uint8)
        contours = cv.findContours(c_mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
        contours = np.concatenate(contours[0],axis=0).squeeze().T
        outlines[contours[1] + region.slice[0].start,contours[0] + region.slice[1].start] = 1

    # add mask to gray scale image
    #mask color
    color = [0, 0, 255]
    #transparency of mask
    alpha = 1

    #determine index where you want the mask
    mask_ind=np.where(outlines > 0)
    out = col_norm_image.copy()
    img_layer = col_norm_image.copy()
    img_layer[mask_ind] = color

    out = cv.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)

    # write image with mask in directory of image file
    outdir = (mask_path.split('.tif')[0]+'_IMG_overlay.jpg')
    cv.imwrite(outdir,out)
    return(image)



##function that finds center point of each nucleus given a nuclear mask tif file
def NucPos(NucMask_path):

    #load masks
    if os.path.isfile(NucMask_path):
        NucMask = tifffile.imread(NucMask_path)
    else:
        print("No Nuclear Mask")
        exit()

    # find the center point of each nucleus
    X = []
    Y = []

    UNUC = np.unique(NucMask)[1:]
    # Define funtion that rapidly gets all indicies
    def get_indices_pandas(data):
        d = data.ravel()
        f = lambda x: np.unravel_index(x.index, data.shape)
        return pd.Series(d).groupby(d).apply(f)

    #get all indicies of nuclei
    indicies = get_indices_pandas(NucMask)

    for nuc in UNUC:
        y, x = indicies[nuc]
        #y, x = np.where(NucMask == nuc)
        X.append(np.mean(x))
        Y.append(1-np.mean(y))

    # export data
    X = np.array(X).reshape((len(X),1))
    Y = np.array(Y).reshape((len(Y),1))

    NucDat = np.hstack((X,Y))
    NucDat = pd.DataFrame(NucDat, columns = ['X','Y'])
    outpath =  NucMask_path.split('_nuc_mask.tif')[0]
    outpath = str(outpath+'_Cellxy.csv')
    NucDat.to_csv(outpath,index=False)
    return(NucDat)






##function that quantifies intensity withing the cytoplasm mask provided by
def MaskQuant(Image_path,Mask_path,CHANNELS):

    #load images and set up variables
    AllMask = tifffile.imread(Mask_path)[0,...,0]
    AllMask = np.array(AllMask)
    RawIm = tifffile.imread(Image_path)
    CHANNELS = np.array(CHANNELS)-1
    UMASK = np.unique(AllMask)[1:]

    #get sum and mean intensity within each mask for all channels specified
    All_sumInt = []
    All_meanInt = []

    for channel in CHANNELS:
        cIm = RawIm[channel,...]

        MaskID = []

        tmp_sumInt = []
        tmp_meanInt = []

        for mask in UMASK:
            allpix = cIm[AllMask == mask]
            tmp_sumInt.append(np.sum(allpix))
            tmp_meanInt.append(np.mean(allpix))
            MaskID.append(mask)
        All_sumInt.append(tmp_sumInt)
        All_meanInt.append(tmp_meanInt)

    #put data into data frames
    sum_dfnames = []
    mean_dfnames = []
    for channel in range(0,len(CHANNELS)):
        All_sumInt[channel] = np.array(All_sumInt[channel])
        All_meanInt[channel] = np.array(All_meanInt[channel])

        sum_dfnames.append(str("sum_Channel"+str(CHANNELS[channel]+1)))
        mean_dfnames.append(str("mean_Channel"+str(CHANNELS[channel]+1)))

    All_sumInt = pd.DataFrame(data=All_sumInt, index= sum_dfnames).T

    All_meanInt = pd.DataFrame(data=All_meanInt, index= mean_dfnames).T

    Alldat = pd.concat([All_meanInt, All_sumInt], axis=1)

    Alldat.insert(0, "MaskID", MaskID)

    #save data
    outpath = Mask_path.split('_mask.tif')[0]
    outpath = str(outpath+'_quant.csv')
    Alldat.to_csv(outpath,index=False)
    return Alldat


#Test what the a radius looks like in your pyplot

def TestRad(path,radius_um,objective):
    #set up parameters
    print("Density functions assumes images are from the Shaffer Lab Scope with 2x2 binning, if not using this scope edit the RESOLUTIONS dictionary in this funtion with the correct um/pixel values for each objective")
    RESOLUTIONS = {'10x':1.29, '10X':1.29,'20x':.65,'20X':.65}
    resolution = RESOLUTIONS[objective]
    radius_pixel = radius_um/resolution

    # get all the xy files for a plate
    well = np.genfromtxt(path, delimiter=',', skip_header= 1)

    #set outpath
    outpath =  path.split('_Cellxy.csv')[0]
    outpath = str(outpath+'_Radius_test.png')

    #plot
    fig, ax = plt.subplots(figsize=(15,15))
    ax.scatter(well[:,0],well[:,1], s=1)
    cir = plt.Circle((np.mean(well[:,0]), np.mean(well[:,1])), radius_pixel, color='r',fill=True)
    ax.set_aspect('equal')
    ax.add_patch(cir)
    plt.savefig(outpath, dpi=300)
    plt.close()




#Determine how densly packed cells are in each wellId
#path is the path to a folder containing files with x and y coordinates of nuclei in file called Cellxy_*.csv radius is the number of pixes around that cell to look for cells (CHANGE TO DISTANCE BASED ON OBJECTIVE)
def DensityQuant(path,radius_um,objective):

    #set up parameters
    print("Density functions assumes images are from the Shaffer Lab Scope with 2x2 binning, if not using this scope edit the RESOLUTIONS dictionary in this funtion with the correct um/pixel values for each objective")
    RESOLUTIONS = {'10x':1.29, '10X':1.29,'20x':.65,'20X':.65}
    resolution = RESOLUTIONS[objective]
    radius_pixel = radius_um/resolution

    # get all the xy files for a plate
    FILES = glob.glob(path + "/*_Cellxy.csv")
    WELLS=[]

    for file in FILES:
        WELLS.append(np.genfromtxt(file, delimiter=',', skip_header= 1))

    #add column for different density metrics for each well
    for w in range(len(WELLS)):
        cols = np.zeros((len(WELLS[w]),2))
        WELLS[w] = np.hstack((WELLS[w], cols))

    #Look at density based on counting cells within a radius around each cells


    #Determine how many cells are within a given radius of a cell
    Means = []
    Medians = []
    WellID = []
    for w in range(len(WELLS)):
        well = WELLS[w]
        inRadius = []
        for i in range(len(well)):
            xcenter = well[i][0]
            ycenter = well[i][1]
            # in radius is  a list containing the number of surounding cells for each cell
            inRadius.append(0)
            #get range of pixels surounding cells
            subsetter = np.where((well[:,0]>(xcenter-radius_pixel)) & (well[:,0] < (xcenter+radius_pixel)) & (well[:,1]>(ycenter-radius_pixel)) & (well[:,1] < (ycenter+radius_pixel)))
            wellSubset = well[subsetter]

            #determine how many cells are within the set radius around each cell
            for j in range(len(wellSubset)):
                xcurrent = wellSubset[j][0]
                ycurrent = wellSubset[j][1]
                if ((xcurrent - xcenter)**2 + (ycurrent - ycenter)**2) < radius_pixel**2:
                    inRadius[i] +=1

        #convert inRadius to numpy array
        inRadius = np.array(inRadius)

        # since cell is counting itself, remove 1 from each element in inRadius
        inRadius = inRadius-1

        #create file with single cell information
        X = WELLS[w][:,0].reshape((len(WELLS[w][:,0]),1))
        Y = WELLS[w][:,1].reshape((len(WELLS[w][:,1]),1))
        inRadius = inRadius.reshape((len(inRadius),1))


        scDensDat = np.hstack((X,Y,inRadius))
        scDensDat = pd.DataFrame(scDensDat, columns = ['X','Y',str('cellsIn'+str(radius_um)+'umRadius')])
        scoutpath =  FILES[w].split('_Cellxy.csv')[0]
        scoutpath = str(scoutpath+'_SingleCellDensity.csv')
        scDensDat.to_csv(scoutpath,index=False)

        #append wellId mean and median
        WellID.append(str("well" + str(w)))
        Means.append(np.mean(inRadius))
        Medians.append(np.median(inRadius))

    #create csv with well and then mean and median
    WellID = np.array(WellID).reshape((len(WellID),1))
    Means = np.array(Means).reshape((len(Means),1))
    Medians = np.array(Medians).reshape((len(Medians),1))

    DensDat = np.hstack((WellID,Means,Medians))
    DensDat = pd.DataFrame(DensDat, columns = ['WellID','Mean','Median'])
    outpath = str(path+'/WholeWellDensities.csv')
    DensDat.to_csv(outpath, index=False)




#plot Densities to check if they look correct
def PlotDensity(path):
    #import data
    dat = np.genfromtxt(path, delimiter=',', skip_header= 1)
    X = dat[:,0]
    Y = dat[:,1]
    density = dat[:,2]

    #set outpath
    outpath =  path.split('_SingleCellDensity.csv')[0]
    outpath = str(outpath + '_DensityPlot.png')

    #plot
    fig, ax = plt.subplots(figsize=(15, 15))
    sc = plt.scatter(X,Y, c=density, cmap="seismic", s=1)
    ax.set_aspect('equal')
    plt.colorbar(sc)
    plt.savefig(outpath,dpi=300)
    plt.close()


## Functions created by Shivani Nellore

#create a ring around the nucleaus to be used as a proxy for a cytoplasm mask
def makeCytoMask(nucPath):
    # read in nuc mask
    nucMaskArray = tifffile.imread(nucPath)[0,...,0]
    nucMaskArray = nucMaskArray.astype("float32")

    # create kernel -- increase kernel size, increases the dilation magnitude
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))

    # create an dilation plot from the nuclear mask
    dn = cv2.dilate(nucMaskArray, kernel)

    # save dilation plot
    dilatePath = nucPath.split('_nuc_mask.tif')[0]
    dilatePath = str(dilatePath+'_dilated.tif')
    cv2.imwrite(dilatePath, dn)

    # Load the dilation and Nuc masks and their pixel maps
    input_imageD = Image.open(dilatePath)
    input_imageN = Image.open(nucPath)
    pixel_mapDil = input_imageD.load()
    pixel_mapNuc = input_imageN.load()

    # create new image mask using dilation mask perimeters
    img = Image.new(input_imageD.mode, input_imageD.size)
    pixelsNew = img.load()

    # get the pixel width and heigh of the of the dilation image:
    width, height = input_imageD.size

    # for all non 0 pixels of Nuc Mask, make black in dilated image and load to new mask
    for i in range(width):
        for j in range(height):
            if(pixel_mapNuc[i,j] != 0):
                pixelsNew[i,j] = (0)
            else:
                pixelsNew[i,j] = (pixel_mapDil[i,j])

    # Save and show new cyto mask
    cytoPath = nucPath.split('_nuc_mask.tif')[0]
    cytoPath = str(cytoPath+'_cyto_mask.tif')
    img.save(cytoPath)
    plt.imshow(img)
    return cytoPath

#calculate the cytoplasm to nuclear ratio
    def CytoNucRatios(Image_path, Mask_path_Nuc, Mask_path_cyto,CHANNELS):
    # read in Nuc Mask
    AllMaskNuc=tifffile.imread(Mask_path_Nuc)

    if len(AllMaskNuc.shape) == 2:
        AllMaskNuc = AllMaskNuc
    elif len(AllMaskNuc.shape) == 3:
        AllMaskNuc = AllMaskNuc[0,...]
    elif len(AllMaskNuc.shape) == 4:
        AllMaskNuc = AllMaskNuc[0,...,0]
    else:
        print("mask has unsupported dimensions")
    AllMaskNuc = np.array(AllMaskNuc)

    # read in Cyto Mask
    AllMaskCyto = np.array(AllMaskCyto)

    #read in raw image mask
    RawIm = tifffile.imread(Image_path)

    # UMASKNuc = np.unique(AllMaskNuc)[1:] ==> don't need because cyto will be smaller
    UMASKCyto = np.unique(AllMaskCyto)[1:]

    #get sum and mean intensity within each mask for all channels specified
    All_sumRatio = []
    All_meanRatio = []

    if CHANNELS != 1:
        for channel in CHANNELS:
            cIm = RawIm[channel,...]

            allpixTotalNuc = cIm[AllMaskNuc != 0]
            allMaskTotalNuc = AllMaskNuc[AllMaskNuc != 0]

            allpixTotalCyto = cIm[AllMaskCyto != 0]
            allMaskTotalCyto = AllMaskCyto[AllMaskCyto != 0]

            MaskID = []

            tmp_sumRatio = []
            tmp_meanRatio = []

            mask = 1
            # for mask in UMASK:
            for mask in UMASKCyto:
                allpixNuc = allpixTotalNuc[allMaskTotalNuc == mask]
                allpixCyto = allpixTotalCyto[allMaskTotalCyto == mask]
                sumRatio = np.sum(allpixCyto)/np.sum(allpixNuc)
                meanRatio = np.mean(allpixCyto)/np.mean(allpixNuc)
                tmp_sumRatio.append(sumRatio)
                tmp_meanRatio.append(meanRatio)
                MaskID.append(mask)
                mask += 1

            All_sumRatio.append(tmp_sumRatio)
            All_meanRatio.append(tmp_meanRatio)
    else:
        channel = 1
        cIm = RawIm[channel,...]
        MaskID = []

        allpixTotalNuc = cIm[AllMaskNuc != 0]
        allMaskTotalNuc = AllMaskNuc[AllMaskNuc != 0]

        allpixTotalCyto = cIm[AllMaskCyto != 0]
        allMaskTotalCyto = AllMaskCyto[AllMaskCyto != 0]
        print(allMaskTotalCyto)

        tmp_sumRatio = []
        tmp_meanRatio = []

        mask = 1
        # for mask in UMASK:
        for mask in UMASKCyto:
            allpixNuc = allpixTotalNuc[allMaskTotalNuc == mask]
            allpixCyto = allpixTotalCyto[allMaskTotalCyto == mask]
            sumRatio = np.sum(allpixCyto)/np.sum(allpixNuc)
            meanRatio = np.mean(allpixCyto)/np.mean(allpixNuc)
            tmp_sumRatio.append(sumRatio)
            tmp_meanRatio.append(meanRatio)
            MaskID.append(mask)
            mask += 1
            #print(time.time())

        All_sumRatio.append(tmp_sumRatio)
        All_meanRatio.append(tmp_meanRatio)

    #put data into data frames
    sum_dfnames = []
    mean_dfnames = []
    for channel in range(0,CHANNELS):
        All_sumRatio[channel] = np.array(All_sumRatio[channel])
        All_meanRatio[channel] = np.array(All_meanRatio[channel])

        sum_dfnames.append(str("sum_Channel"+str(channel+1)))
        mean_dfnames.append(str("mean_Channel"+str(channel+1)))

    All_sumRatio = pd.DataFrame(data=All_sumRatio, index= sum_dfnames).T

    All_meanRatio = pd.DataFrame(data=All_meanRatio, index= mean_dfnames).T

    Alldat = pd.concat([All_meanRatio, All_sumRatio], axis=1)

    Alldat.insert(0, "MaskID", MaskID)

    #save data
    outpath = Mask_path_Nuc.split('_mask.tif')[0]
    outpath = str(outpath+'_CytoToNucratios.csv')
    Alldat.to_csv(outpath,index=False)
    return Alldat
