import suspect
import matplotlib.pyplot as plt
import numpy as np
import os

import pydicom
import struct
import warnings

import nibabel as nib
import ants
import functools

from multiprocessing import Pool

def fit(f):
    print(f)
    os.system("/Users/voelzkey/Desktop/CodeFortran/LCModel/lcmodel < %s" %f)
    
def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def hamming(x):
    a = 25/46
    return a-(1-a)*np.cos(np.arange(x)*2*np.pi/x)

def hamming2D(x,y):
    X = hamming(x)
    Y = hamming(y)
    return np.outer(X,Y)

def hamming3D(x,y,z):
    X = hamming(x)
    Y = hamming(y)
    Z = hamming(z)
    return functools.reduce(np.multiply.outer, (X, Y, Z))

def filterHamming(inp):
    data = inp.copy()
    x,y = data.shape[:2]
    H = hamming2D(x,y)
    data = np.fft.ifftshift(data, axes=(0,1))
    data = np.fft.ifft2(data, axes=(0,1))
    data = np.fft.fftshift(data, axes=(0,1))

    dataF = data * H[...,np.newaxis]

    dataF = np.fft.ifftshift(dataF, axes=(0,1))
    dataF = np.fft.fft2(dataF, axes=(0,1))
    dataF = np.fft.fftshift(dataF, axes=(0,1))

    return dataF

def filterHamming3D(inp):
    data = inp.copy()
    x,y,z = data.shape[:3]
    H = hamming3D(x,y,z)
    print(H.shape)
    data = np.fft.ifftshift(data, axes=(0,1,2))
    data = np.fft.ifft2(data, axes=(0,1,2))
    data = np.fft.fftshift(data, axes=(0,1,2))

    dataF = data * H[...,np.newaxis]

    dataF = np.fft.ifftshift(dataF, axes=(0,1,2))
    dataF = np.fft.fft2(dataF, axes=(0,1,2))
    dataF = np.fft.fftshift(dataF, axes=(0,1,2))

    return dataF

# Define some helper functions, copyied from the suspect library

ima_types = {
    "floats": ["NumberOfAverages", "RSatPositionSag", "PercentPhaseFieldOfView", "RSatOrientationSag", "MixingTime",
               "PercentPhaseFieldOfView", "RSatPositionCor", "InversionTime", "RepetitionTime", "VoiThickness",
               "TransmitterReferenceAmplitude", "ImageOrientationPatient", "SliceThickness", "RSatOrientationTra",
               "PixelBandwidth", "SAR", "PixelSpacing", "ImagePositionPatient", "VoiPosition", "SliceLocation",
               "FlipAngle", "VoiInPlaneRotation", "VoiPhaseFoV", "SliceMeasurementDuration", "HammingFilterWidth",
               "RSatPositionTra", "MagneticFieldStrength", "VoiOrientation", "PercentSampling", "EchoTime",
               "VoiReadoutFoV", "RSatThickness", "RSatOrientationCor", "ImagingFrequency", "TriggerTime", "dBdt",
               "TransmitterCalibration", "PhaseGradientAmplitude", "ReadoutGradientAmplitude",
               "SelectionGradientAmplitude", "GradientDelayTime", "dBdt_max", "t_puls_max", "dBdt_thresh",
               "dBdt_limit", "SW_korr_faktor", "Stim_lim", "Stim_faktor"],
    "integers": ["Rows", "Columns", "DataPointColumns", "SpectroscopyAcquisitionOut-of-planePhaseSteps",
                 "EchoPartitionPosition", "AcquisitionMatrix", "NumberOfFrames", "EchoNumbers", "RealDwellTime",
                 "EchoTrainLength", "EchoLinePosition", "EchoColumnPosition", "SpectroscopyAcquisitionDataColumns",
                 "SpectroscopyAcquisitionPhaseColumns", "SpectroscopyAcquisitionPhaseRows", "RfWatchdogMask",
                 "NumberOfPhaseEncodingSteps", "DataPointRows", "UsedPatientWeight", "NumberOfPrescans",
                 "Stim_mon_mode", "Operation_mode_flag", "CoilId", "MiscSequenceParam", "MrProtocolVersion",
                 "ProtocolSliceNumber"],
    "strings": ["ReferencedImageSequence", "ScanningSequence", "SequenceName", "ImagedNucleus", "TransmittingCoil",
                "PhaseEncodingDirection", "VariableFlipAngleFlag", "SequenceMask", "AcquisitionMatrixText",
                "MultistepIndex", "DataRepresentation", "SignalDomainColumns", "k-spaceFiltering", "ResonantNucleus",
                "ImaCoilString", "FrequencyCorrection", "WaterReferencedPhaseCorrection", "SequenceFileOwner",
                "CoilForGradient", "CoilForGradient2", "PositivePCSDirections", ],
}

CSA1 = 0
CSA2 = 1

def read_csa_header(csa_header_bytes):
    # two possibilities exist here, either this is a CSA2 format beginning with an SV10 string, or a CSA1 format which
    # doesn't. in CSA2 after the "SV10" are four junk bytes, then the number of tags in a uint32 and a delimiter uint32
    # containing the value 77. in CSA1 there is just the number of tags and the delimiter. after that the two formats
    # contain the same structure for each tag, but the definition of the size of the items in each tag is different
    # between the two versions
    if csa_header_bytes[:4] == "SV10".encode('latin-1'):
        num_tags, delimiter = struct.unpack("<II", csa_header_bytes[8:16])
        header_offset = 16
        header_format = CSA2
    else:
        num_tags, delimiter = struct.unpack("<II", csa_header_bytes[:8])
        header_offset = 8
        header_format = CSA1
    # now we can iteratively read the tags and the items inside them
    csa_header = {}
    for i in range(num_tags):
        name, vm, vr, syngo_dt, nitems, delimiter = struct.unpack("<64si4siii",
                                                                  csa_header_bytes[header_offset:(header_offset + 84)])
        header_offset += 84
        # the name of the tag is 64 bytes long, but the string we want is null-terminated inside, so extract the
        # real name by taking only bytes up until the first 0x00
        name = name.decode('latin-1')
        name = name.split("\x00", 1)[0]
        # read all the items inside this tag
        item_list = []
        for j in range(nitems):
            sizes = struct.unpack("<4L", csa_header_bytes[header_offset:(header_offset + 16)])
            header_offset += 16
            if header_format == CSA2:
                item_length = sizes[1]
                if (header_offset + item_length) > len(csa_header_bytes):
                    item_length = len(csa_header_bytes) - header_offset
            elif header_format == CSA1:
                item_length = sizes[0]
            item, = struct.unpack("<%ds" % item_length,
                                  csa_header_bytes[header_offset:(header_offset + item_length)])
            item = item.decode('latin-1')
            item = item.split("\x00", 1)[0]
            if item_length > 0:
                if name in ima_types["floats"]:
                    item = float(item)
                elif name in ima_types["integers"]:
                    item = int(item)
                elif name in ima_types["strings"]:
                    pass
                else:
                    warnings.warn("Unhandled name {0} with vr {1} and value {2}".format(name, vr, item))
                item_list.append(item)
            header_offset += item_length
            header_offset += (4 - (item_length % 4)) % 4  # move the offset to the next 4 byte boundary
        if len(item_list) == 1:
            item_list = item_list[0]
        csa_header[name] = item_list
    return csa_header

def header(file):
    dataset = pydicom.dicomio.read_file(file)

    xx = 0x0010
    header_index = 0
    while (0x0029, xx) in dataset:
        if dataset[0x0029, xx].value == "SIEMENS CSA HEADER":
            header_index = xx
        xx += 1
    # check that we have found the header
    if header_index == 0:
        raise KeyError("Could not find header index")
    # now we know which tag contains the CSA image header info: (0029, xx10)
    csa_header_bytes = dataset[0x0029, 0x0100 * header_index + 0x0010].value
    return read_csa_header(csa_header_bytes)

def write_control(path,name,suffix):

    with open(os.path.join(path,name+"_"+suffix+".control"), 'w') as f:
        f.write(" $LCMODL\n")
        f.write(" OWNER='DZNE Bonn'\n")
        f.write(" key = 210387309\n")
        f.write(" Title='%s'\n"%name)
        f.write(" HZPPPM=2.972168e+02, DELTAT=3.600000e-04, NUNFIL=800\n")
        f.write(" FILBAS='/Users/voelzkey/Desktop/CodeMatlab/reko_dzne/Basis_Set_FID_MRSI_v2.0/fid_1.300000ms.basis'\n")
        f.write(" DOREFS(1) = T\n")
        f.write(" DOREFS(2) = F\n")
        f.write(" WSMET = 'DSS'\n")
        f.write(" WSPPM = 0.0\n")
        f.write(" N1HMET = 9\n")
        f.write(" SUBBAS = T\n")
        f.write(" NEACH = 99\n")
        f.write(" WDLINE(6) = 0\n")
        f.write(" PPMST = 4.2\n")
        f.write(" PPMEND = 1.8\n")
        f.write(" DEGZER = 0\n")
        f.write(" SDDEGZ = 999\n")
        f.write(" NDCOLS = 1\n")
        f.write(" NDROWS = 1\n")
        f.write(" NDSLIC = 1\n")
        f.write(" DEGPPM = 0\n")
        f.write(" SDDEGP = 1\n")
        f.write(" NSIMUL = 0\n")
        f.write(" NOMIT = 5\n")
        f.write(" CHOMIT(1) = 'Cho'\n")
        f.write(" CHOMIT(2) = 'Act'\n")
        f.write(" CHOMIT(3) = 'mm3'\n")
        f.write(" CHOMIT(4) = 'mm4'\n")
        f.write(" CHOMIT(5) = 'Glc_B'\n")
        f.write(" LTABLE = 7\n")
        f.write(" LCSV = 0\n")
        f.write(" LCOORD = 0\n")
        f.write(" FILTAB='%s'\n" %os.path.join(path,name+"_"+suffix+".table"))
        f.write(" FILRAW='%s'\n" %os.path.join(path,name+"_"+suffix+".RAW"))
        f.write(" FILPS='%s'\n"  %os.path.join(path,name+"_"+suffix+".ps"))
        #f.write(" FILCOO='%s'\n" %"%s.coord"%os.path.join(path,name))
        f.write(" $END\n")

def write_raw(path,name,suffix,data):
    with open(os.path.join(path,name+"_"+suffix+".RAW"), 'w') as f:
            f.write(" $SEQPAR\n")
            f.write(" hzpppm=2.972169e+02\n")
            f.write(" $END\n")
            f.write(" $NMID\n")
            f.write(" fmtdat='(2e14.5)'\n")
            f.write(" $END\n")
            for s in data[:800]:
                f.write("  %.5e  %.5e\n" % (s.real,s.imag))


class mrsi_data():
    def __init__(self,path,site,sub,ses):
        self.path = path
        self.site = site
        self.sub  = sub
        self.ses  = ses

        self.path_in  = os.path.join(path,site,sub,ses,"mrsi")
        self.path_mrsi  = os.path.join(path,"derivatives",site,sub,ses,"mrsi")
        self.path_maps = os.path.join(path,"derivatives",site,sub,ses,"mrsi","maps")
        self.path_lcm = os.path.join(path,"derivatives",site,sub,ses,"mrsi","lcm")

        print(self.path_lcm)

        mkdir(self.path_mrsi)
        mkdir(self.path_lcm)
        mkdir(self.path_maps)
        self.load_data()

    def load_data(self):
        load3D = []
        zPos =[]

        for f in sorted(os.listdir(self.path_in)):
            if ".DS_Store" in f:
                pass
            else:
                data = suspect.io.load_siemens_dicom(os.path.join(self.path_in,f))
                load3D.append(np.reshape(np.array(data),(44,44,1024)))
                self.head = header(os.path.join(self.path_in,f))
                zPos.append(self.head["SliceLocation"])
        load3D = np.array(load3D)
        zPos = np.array(zPos)
        load3D = load3D[np.argsort(zPos)]
        self.shape = load3D.shape
        self.shape = (self.shape[1],self.shape[2],self.shape[0])
        self.data = filterHamming3D(load3D)

    def create_dummyNII(self):
        dummy = np.zeros(self.shape)
        name  = os.path.join(self.path_maps,"dummy.nii")

        img = nib.Nifti1Image(dummy, np.identity(4))
        nib.save(img, name)

        img = ants.image_read(name)
    
        (x,y,z) = self.head['VoiPosition']
        self.spacing     = (5.,5.,5.)
        self.origin = (x+5*self.shape[0]/2,y+5*self.shape[1]/2,z-5*self.shape[2]/2)

        img.set_origin(self.origin)
        img.set_spacing(self.spacing)
        ants.image_write(img,name)
        return name

    def write_lcm(self,mask):
        brain = ants.image_read(mask)>.5
        brain = brain[::-1,::-1]
        #brain = np.ones(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    suffix = "%i_%i_%i"%(i,j,k)
                    name = self.sub
                    title = "%s_%s"%(self.site,self.sub)
                    if brain[i,j,k]:

                        write_control(self.path_lcm,name,suffix)
                        write_raw(self.path_lcm,name,suffix,self.data[k,j,i])

                        print((i,j,k))

    def call_lcm(self,p):
        print("open call_lcm")
        files = [os.path.join(self.path_lcm,f) for f in os.listdir(self.path_lcm) if ".control" in f]
        print(self.path_lcm)
        print(len(files))

        with Pool(p) as P:
            P.map(fit, files)

    def save(self,name,data):
            img = nib.Nifti1Image(data, np.identity(4))
            nib.save(img, name)
            
            nii = ants.image_read(name)
            nii.set_origin(self.origin)
            nii.set_spacing(self.spacing)
            ants.image_write(nii,name)

    def save_nii(self):
    
        self.NAA  = np.zeros(self.shape)
        self.CR   = np.zeros(self.shape)
        self.GABA = np.zeros(self.shape)
        self.GLX  = np.zeros(self.shape)
        self.CHO  = np.zeros(self.shape)
        self.THG  = np.zeros(self.shape)
        self.INS  = np.zeros(self.shape)
        self.ASP  = np.zeros(self.shape)
        self.TAU  = np.zeros(self.shape)
        self.SNR  = np.zeros(self.shape)
        self.FWHM = np.zeros(self.shape)
        name = self.sub
        for x in range(44):
            for y in range(44):
                for z in range(27):
                    try:
                        t = open("%s/%s_%i_%i_%i.table"%(self.path_lcm,name,x,y,z),"r")
                        c = t.readlines()
                        for line in c:
                            if "NAA+NAAG" in line:
                                self.NAA[x,y,z] = float(line.split()[0])
                            elif " Cr+PCr" in line:
                                self.CR[x,y,z] = float(line.split()[0])    
                            elif "GABA" in line:
                                line =line.replace("+GABA"," GABA")
                                line =line.replace("-GABA"," GABA")
                                self.GABA[x,y,z] = float(line.split()[0])
                            elif "Glu+Gln" in line:
                                self.GLX[x,y,z] = float(line.split()[0])
                            elif "GPC+PCh" in line:
                                self.CHO[x,y,z] = float(line.split()[0])
                            elif "TwoHG" in line:
                                self.THG[x,y,z] = float(line.split()[0])
                            elif "Ins" in line:
                                self.INS[x,y,z] = float(line.split()[0])
                            elif "Asp" in line:
                                line =line.replace("+Asp"," Asp")
                                line =line.replace("-Asp"," Asp")
                                self.ASP[x,y,z] = float(line.split()[0])
                            elif "Tau" in line:
                                self.TAU[x,y,z] = float(line.split()[0])
                            elif "S/N =" in line:
                                self.SNR[x,y,z] = float(line.split()[-1])
                                self.FWHM[x,y,z] = float(line.split()[2])
                    except:
                        pass

        prefix = "%s_%s_%s_mrsi_" % (self.site,self.sub,self.ses)

        self.save(os.path.join(self.path_maps,prefix+"tnaa.nii"), self.NAA)
        self.save(os.path.join(self.path_maps,prefix+"tcr.nii"), self.CR)
        self.save(os.path.join(self.path_maps,prefix+"gaba.nii"), self.GABA)
        self.save(os.path.join(self.path_maps,prefix+"glx.nii"), self.GLX)
        self.save(os.path.join(self.path_maps,prefix+"tcho.nii"), self.CHO)
        self.save(os.path.join(self.path_maps,prefix+"2HG.nii"), self.THG)
        self.save(os.path.join(self.path_maps,prefix+"ins.nii"), self.INS)
        self.save(os.path.join(self.path_maps,prefix+"asp.nii"), self.ASP)
        self.save(os.path.join(self.path_maps,prefix+"tau.nii"), self.TAU)
        self.save(os.path.join(self.path_maps,prefix+"snr.nii"), self.SNR)
        self.save(os.path.join(self.path_maps,prefix+"fwhm.nii"), self.FWHM)
