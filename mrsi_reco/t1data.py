import os
import subprocess
import ants
import antspynet

class T1_image():
    def __init__(self,path,site,sub,ses,container=None):
        self.path = path
        self.site = site
        self.sub  = sub
        self.ses  = ses
        self.container = container

        self.prefix = "%s_%s_%s_mprage_LR"%(site,sub,ses)

        self.rawImage = ants.image_read(os.path.join(path,site,sub,ses,"magn",'%s.nii'%self.prefix))
        self.path_out = os.path.join(path,"derivatives",site,sub,ses,"magn")

    def calc_brainMask(self):
        brain =antspynet.brain_extraction(self.rawImage, modality="t1")
        print(os.path.join(self.path_out,"%s_brain.nii"%self.prefix))
        ants.image_write(brain*self.rawImage,os.path.join(self.path_out,"%s_brain.nii"%self.prefix))
        ants.image_write(brain,os.path.join(self.path_out,"%s_brain-msk.nii"%self.prefix))

    def register_toMRSI(self,mrsi_file):
        inp = os.path.join(os.path.join(self.path_out,"%s_brain-msk.nii"%self.prefix))
        out = os.path.join(os.path.join(self.path_out,"%s_mrsi-msk.nii.gz"%self.prefix))

        cmd = [
            "flirt",
            "-in", inp,
            "-ref", mrsi_file,
            "-out", out,
            "-applyxfm",
            "-usesqform",
            "-verbose", "3",
        ]

        if self.container:
            study_root = os.path.abspath(self.path)
            cmd = [
                "singularity", "exec",
                "-B", f"{study_root}:{study_root}",
                self.container,
            ] + cmd

        subprocess.run(cmd, check=True)
        return out
