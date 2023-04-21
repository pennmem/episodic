#!/home1/ddiwik/anaconda2/envs/shadowfox/bin/python
#$-N grptest
#$-cwd
#$-pe python-distributed 1
import os
import time
from glob import glob
import numpy as np

def build_subj_file(swarm_size):
    subjs = ['LTP093', 'LTP106', 'LTP115', 'LTP117', 'LTP122', 'LTP123', 'LTP133', 'LTP138', 'LTP207', 'LTP210','LTP228', 'LTP229', 'LTP236', 'LTP246', 'LTP249', 'LTP251', 'LTP258', 'LTP259', 'LTP260','LTP265', 'LTP269', 'LTP273', 'LTP278', 'LTP279', 'LTP280', 'LTP283', 'LTP285', 'LTP287', 'LTP293', 'LTP295', 'LTP296', 'LTP297', 'LTP299', 'LTP301', 'LTP302', 'LTP303', 'LTP304', 'LTP305', 'LTP306', 'LTP307', 'LTP309', 'LTP310', 'LTP311', 'LTP312', 'LTP314', 'LTP316', 'LTP317', 'LTP318', 'LTP320', 'LTP321', 'LTP322', 'LTP323', 'LTP324', 'LTP325', 'LTP327', 'LTP328', 'LTP330', 'LTP331', 'LTP334', 'LTP336', 'LTP338', 'LTP339', 'LTP340', 'LTP342', 'LTP343', 'LTP344', 'LTP346', 'LTP347', 'LTP348', 'LTP349', 'LTP353', 'LTP355', 'LTP357', 'LTP359', 'LTP361', 'LTP362', 'LTP364', 'LTP366']
    subjs = [subj for subj in subjs for i in xrange(swarm_size)]
    np.savetxt("subjects_ltpFR2.txt", subjs,fmt='%s')
    return

def main(swarm_size = 90, iterations = 30):
    cwd = os.getcwd()
    #Hard coded a lot of stuff if this is ever used by some1 other than me fix it
    subj_count = 78
    #build_subj_file(swarm_size)
    for i in xrange(subj_count):
        os.system("rm *xfile*")
        os.system("rm *pfile*")
        os.system("rm *vfile*")
        os.system("rm rmses*")
        os.system("rm rg_iter*")
        os.system("rm rp_iter*")
        os.system("python noise_maker.py {} {}".format(swarm_size, iterations))
        os.system("pgo pso_par_cmr2.py {}".format(swarm_size))
        os.system("rm imdone.txt")
        while True:
            if os.path.isfile(os.path.join(cwd, 'imdone.txt')):
                break
if __name__ == "__main__": main()