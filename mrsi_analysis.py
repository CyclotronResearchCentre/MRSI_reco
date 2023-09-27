from t1data import T1_image
from mrsidata import mrsi_data
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This is the MRSI analysis tool for the SCAIFIELD project. It assumes that data is already structured\n It reads in all .DCM (or .IMA) files, applies a k-space filter and performs spectral quantification for all voxels within brain mask (MPRAGE needed for this)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', help='path #BIDS folder')
    parser.add_argument('--site', help='site')
    parser.add_argument('--sub',  help='subject name')
    parser.add_argument('--ses',  help='session name')
    parser.add_argument('--p'  ,  help='number of kernels that LCMODEL may run on', default=2, required=False)

    args = parser.parse_args()
    path = args.path
    site = args.site
    sub  = args.sub
    ses  = args.ses
    p    = args.p

    mrsi = mrsi_data(path,site,sub,ses)
    name_dummy = mrsi.create_dummyNII()
    
    t1 = T1_image(path,site,sub,ses)
    t1.calc_brainMask()
    name_msk = t1.register_toMRSI(name_dummy)

    mrsi.write_lcm(name_msk)
    mrsi.call_lcm(p)
    mrsi.save_nii()