from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

##function that finds center point of each nucleus given a nuclear mask tif file
def NucPos(mask_dir):

    ##get masks
    #build path
    NucMask_path = str(mask_dir+"/nuc_mask.tif")

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
    NucID = []

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
        Y.append(np.mean(y))
        NucID.append(nuc)

    # export data
    NucID = np.array(NucID).reshape((len(NucID),1))
    X = np.array(X).reshape((len(X),1))
    Y = np.array(Y).reshape((len(Y),1))

    NucDat = np.hstack((NucID,X,Y))
    NucDat = pd.DataFrame(NucDat, columns = ['nucID','X','Y'])
    outpath = str(mask_dir+"/nuc_quant.csv")
    NucDat.to_csv(outpath)






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
    if os.path.isfile(CytoMask_path):
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
    CytoDat.to_csv(outpath)




#Determine how densly packed cells are in each wellId
#path is the path to a folder containing files with x and y coordinates of nuclei in file called Cellxy_*.csv radius is the number of pixes around that cell to look for cells (CHANGE TO DISTANCE BASED ON OBJECTIVE)
def DensityQuant(path,radius):

    # get all the xy files for a plate
    FILES = glob.glob(path + "/Cellxy_*.csv")
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
            subsetter = np.where((well[:,0]>(xcenter-radius)) & (well[:,0] < (xcenter+radius)) & (well[:,1]>(ycenter-radius)) & (well[:,1] < (ycenter+radius)))
            wellSubset = well[subsetter]

            #determine how many cells are within the set radius around each cell
            for j in range(len(wellSubset)):
                xcurrent = wellSubset[j][0]
                ycurrent = wellSubset[j][1]
                if ((xcurrent - xcenter)**2 + (ycurrent - ycenter)**2) < radius**2:
                    inRadius[i] +=1

        #convert inRadius to numpy array
        inRadius = np.array(inRadius)

        # since cell is counting itself, remove 1 from each element in inRadius
        inRadius = inRadius-1

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
    outpath = str(path+"/density.csv")
    DensDat.to_csv(outpath)




    #need to add funtion for nuclear intensity as well as get nuclear to cytoplasm intensity ratio


    ## Filter out and split up cytoplasms
    #Remove cytoplasms that had no nuclei
    #for mask in UCYTO:
    #    mask_num = NucMask[CytoMask == mask]
    #    if np.mean(mask_num) <= 0:
    #        CytoMask[CytoMask == mask] = 0
        #    print("removed cytoplasm mask", mask, "because it had no nucleus")

    #split cytoplasms that have multiple nuclei
