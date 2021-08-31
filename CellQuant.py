from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

##function that finds center point of each nucleus given a nuclear mask tif file
def NucPos(NucMask_path):

    #load masks
    if os.path.isfile(NucMask_path):
        NucMask = Image.open(NucMask_path)
        NucMask = np.array(NucMask)
        #get dimension of y axis
        dim_y = NucMask.shape[0]
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






##function that quantifies intensity withing the cytoplasm mask provided by
def CytoQuant(mask_dir):
    CytoMask_path = str(mask_dir+"/cyto_mask.tif")
    CytoRaw_path = str(mask_dir+"/raw_cytoplasm.tif")

    if os.path.isfile(CytoMask_path):
        CytoMask = Image.open(CytoMask_path)
        CytoMask = np.array(CytoMask)

        dim_y = CytoMask.shape[0]

        CytoRaw = Image.open(CytoRaw_path)
        CytoRaw  = np.array(CytoRaw)
    else:
        print("No Cytoplasm Mask")
        exit()

    #get intensities in cytoplasm
    UCYTO = np.unique(CytoMask)[1:]

    cyto_sumInt = []
    cyto_meanInt = []
    CytoID = []

    for cyto in UCYTO:
        allpix = CytoRaw[CytoMask == cyto]
        cyto_sumInt.append(np.sum(allpix))
        cyto_meanInt.append(np.mean(allpix))
        CytoID.append(cyto)

    #reshape data
    cyto_sumInt = np.array(cyto_sumInt).reshape((len(cyto_sumInt),1))
    cyto_meanInt = np.array(cyto_meanInt).reshape((len(cyto_meanInt),1))


    CytoID = np.array(CytoID).reshape((len(CytoID),1))
    cyto_sumInt = np.array(cyto_sumInt).reshape((len(cyto_sumInt),1))
    cyto_meanInt = np.array(cyto_meanInt).reshape((len(cyto_meanInt),1))

    CytoDat = np.hstack((CytoID,cyto_sumInt,cyto_meanInt))
    CytoDat = pd.DataFrame(CytoDat, columns = ['cytoID','cyto_sumInt','cyto_meanInt'])
    outpath = str(mask_dir+"/cyto_quant.csv")
    CytoDat.to_csv(outpath,index=False)


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


    #need to add funtion for nuclear intensity as well as get nuclear to cytoplasm intensity ratio


    ## Filter out and split up cytoplasms
    #Remove cytoplasms that had no nuclei
    #for mask in UCYTO:
    #    mask_num = NucMask[CytoMask == mask]
    #    if np.mean(mask_num) <= 0:
    #        CytoMask[CytoMask == mask] = 0
        #    print("removed cytoplasm mask", mask, "because it had no nucleus")

    #split cytoplasms that have multiple nuclei
