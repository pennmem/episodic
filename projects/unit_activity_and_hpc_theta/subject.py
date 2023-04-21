
import numpy as np 
from scipy import stats as scps
from os import path
from session import session
import os
import re
from scipy.spatial import distance
from numpy.lib.recfunctions import append_fields
from CategoryReader import get_elec_cat


class subject(object):
    
    def __init__(self,index=None,name=None,subjectList=None):
        if index is not None:
            self.index    = index
            self.name     = subjectList[index]
        elif name is not None:
            self.name = name
            self.index = subjectList.index(name)
        print(self.name)
        self.sessions = None
        self.spath    = None
        self.events   = None
        self.SCS      = None
        self.ATBc     = None
        self.PTBc     = None
        self.seegpath = None
        self.mpChans  = None
        self.bpPairs  = None
        self.talPath  = '/data/eeg/'+self.name+'/tal/'
        self.axis     = None
        if 'FR' in self.name:
            self.linefreq = [48., 52.]
        else:
            self.linefreq = [58., 62.] 
  
    def concatenate(self,eeg,sessionList,perm = False):
        self.spath    = [sessi for sessi in sessionList if self.name in sessi]
        self.sessions = [session(ind,eeg,sessionList).get_events() for ind in [sessionList.index(sessi) for sessi in self.spath]]
        self.sessions = [sessi.calc_transition_dist(append = True) for sessi in self.sessions]
        
        sess2calc = [sessi for sessi in self.sessions if len(sessi.ATBc)>0]
        self.SCS  = [item for sessi in sess2calc for triallist in sessi.SCS for item in triallist]
        self.TCS  = [item for sessi in sess2calc for triallist in sessi.TCS for item in triallist]
        self.aSCS  = np.array(self.SCS).mean(axis = 0) 
        self.aTCS  = np.array(self.TCS).mean(axis = 0)
        self.axis = self.sessions[0].axis 
        
        if perm:
            SCS = []
            TCS = []
            for permi in range(perm):
                pSCS = []
                pTCS = []
                for ind,sessi in enumerate(sess2calc):
                    perm_SCS,perm_TCS = sessi.calc_transition_dist(perm = True) 
                    
                    pSCS.append([item for sublist in perm_SCS for item in sublist])
                    pTCS.append([item for sublist in perm_TCS for item in sublist])
                    
                SCS.append(np.mean([item for sublist in pSCS for item in sublist],axis = 0))
                TCS.append(np.mean([item for sublist in pTCS for item in sublist],axis = 0))

            np.savez ('/scratch/nherweg/DP3/behavior/'+self.name+'_permCS.npz',SCS = SCS,TCS = TCS)
            
        ATBc = np.sum(np.array([[count for count in sessi.ATBc] for sessi in sess2calc]),axis = 0)
        PTBc = np.sum(np.array([[count for count in sessi.PTBc] for sessi in sess2calc]),axis = 0)
        self.CRP  = ATBc/PTBc
        
        
        dt = self.sessions[0].events.dtype.descr# this is now a modifiable list, can't modify numpy.dtype
        dt[:17] = [('subject', '<U256'), ('session', '<U256'), ('trial', '<i8'), ('type', '<U256'), ('serialPos', '<f8'), ('item', '<U256'), ('store', '<U256'), ('storeX', '<f8'), ('store'+self.axis, '<f8'), ('presX', '<f8'), 
                                           ('pres'+self.axis, '<f8'), ('itemno', '<f8'), ('recalled', '<f8'), ('intruded', '<f8'), ('finalrecalled', '<f8'), ('rectime', '<f8'), ('intrusion', '<f8')]
        sessevents = [sessi.events.astype(np.dtype(dt)) for sessi in self.sessions]
        
        if len(sessevents) == 1:
            subevents = sessevents[0]
        elif len(sessevents) > 1:
            subevents = np.rec.array(np.concatenate(tuple(sessevents),axis = 0)) 
        else: raise ValueError(self.name)
        
        # Check for repeated words and append a filter to the events
        words = subevents[subevents.type == 'WORD']
        if len(words['item']) > len(np.unique(words['item'])):
            print(str(len(words['item'])-len(np.unique(words['item'])))+' repeats for '+self.name+'!')
            #raise ValueError('Repeat for '+self.name+'!')
        else:
            print('No repeats for '+self.name+'.')
            
        repeated_word = [0 if (eventi.type == 'WORD') & (subevents[:ind][subevents[:ind].type == 'WORD']['item'].tolist().count(eventi['item']) == 0)  
                        else 1 if (eventi.type == 'WORD') & (subevents[:ind][subevents[:ind].type == 'WORD']['item'].tolist().count(eventi['item']) > 0)  
                        else -999 for ind,eventi in enumerate(subevents)]
        subevents  = append_fields(subevents,'repeated_word',np.array(repeated_word),usemask=False, asrecarray=True)    
        
        self.p_rec = float(len(words[words.recalled == 1]))/float(len(words))
        self.events = subevents
        
        self.seegpath = [sessi.eegpath for sessi in self.sessions]
        
        return(self)
    
    def get_stores(self, inc = 'all'):
        
        # Filter events
        filtered_events = self.events[self.events['type'] != 'STORE_FAM']
        
        # Get store names
        if type(inc) == str and inc == 'all':
            ident = [ii for ii in np.unique(filtered_events.store[(filtered_events.store != '[]') & (filtered_events.store != 'NaN') & (filtered_events.store != '-999')])]
        else:
            ident = [ii for ii in np.unique(filtered_events.store[(filtered_events.store != '[]') & (filtered_events.store != 'NaN') & (filtered_events.store != '-999')]) if ii in inc]
   
        # Get store locations
        x = [np.unique(filtered_events.storeX[np.where(filtered_events.store == stname)[0]]) for stname in ident] 
        z = [np.unique(filtered_events['store'+self.axis][np.where(filtered_events.store == stname)[0]]) for stname in ident]
       
        # Make sure each store only has one location
        if np.any([len(xi) != 1 for xi in x]):
            print(x)
            raise ValueError('No consistent storeX location for subject: '+self.name + '\n'+ np.array(x).tostring())
        if np.any([len(zi) != 1 for zi in z]):
            print(z)
            raise ValueError('No consistent storeZ location for subject: '+self.name + '\n'+ np.array(x).tostring())
        x = [xi[0] for xi in x]
        z = [zi[0] for zi in z]
       
        return(ident,x,z)
    
    def get_dist_mat(self,binned = False,method = None, k = 1, inc = 'all'):    

        # Get store information
        _,x,z = self.get_stores(inc = inc)
        
        # Calculate distances between all stores visited in this session
        d = distance.squareform(distance.pdist(np.column_stack((x,z)), metric='euclidean'))
        d = np.triu(1-(d/np.max(d)),k=k) # 1 is closest, 0 is farthest
  
        if method == 'percentile':
            d_p = np.zeros_like(d)
            for row in range(d.shape[0]):
                for col in range(d.shape[1]):
                    d_p[row,col] = scps.percentileofscore(np.unique(d[d>0]),d[row,col])        
            d = d_p
            if k != 0:
                raise ValueError('Chose k = 0 with method percentile')

        elif method == 'binned':
            binsize = len(np.unique(d))/3
            bins = [-1,np.unique(d)[binsize-1],np.unique(d)[(2*binsize)-1],1]
            #bins =  [-1,1/3.,2/3.,1]
            d = np.digitize(d,bins,right = True)
   
        return(d)
    
    def get_channel_info(self):
        
        if 'LTP' in self.name:
            channels = []
            for thepath in self.seegpath:
                channels.append(np.sort([f.split('.')[-1] for f in os.listdir(thepath) if re.search('.*\.[0-9]+$', f)]))
                f = open(thepath+'bad_chan.txt')
                txtIn = f.read()
                badleads = [ii for ii in txtIn if ii != '\n' and ii != ' ']
                f.close
                if len(badleads) > 0:
                    badleads = txtIn
                    print('Excluding channels: ' + ','.join(badleads))
                else:
                    if all(np.array_equal(x,channels[0]) for x in channels):
                        self.mpChans = np.array([chani for chani in channels[0] if chani.lstrip('0') not in badleads])
                    else:
                        raise ValueError('Channels do not match for:'+ self.name)

        else:
            f = open(self.talPath+'MNIcoords.txt','r')
            txtIN = f.read()
            f.close
            
            monopolar_channels = np.array([elec.split()[0].zfill(3) for elec in txtIN.split('\n') if elec != ''])
            chan_names = np.array([elec.split()[1].lower() for elec in txtIN.split('\n') if elec != ''])

            f = open(self.talPath+'MNIcoords_bp.txt','r')
            txtIN = f.read()
            f.close()
            loc = [(elec.split()[0].split('-')[0].zfill(3),elec.split()[0].split('-')[1].zfill(3)) for elec in txtIN.split('\n') if elec != '']
            bipolar_pairs = np.rec.array(np.array(loc,dtype=[('ch0', 'S3'), ('ch1', 'S3')]))

            # Exclude bad channels
            gf        = open(self.talPath+'good_leads.txt','r')
            goodleads = gf.read().split('\n')
            gf.close()
            categories = get_elec_cat(self.name)
            badleads = []
            
            if path.isfile(self.talPath+'bad_leads.txt'):
                bf       = open(self.talPath+'bad_leads.txt','r')
                badleads = bf.read().split('\n')
                badleads = [bl for bl in badleads if bl != '']
                bf.close()
                
                
            if categories is not None and ('bad ch' in categories.keys() or 'brain lesions' in categories.keys()):
                if 'bad ch' in categories.keys() and len(categories['bad ch'].tolist()) > 0:
                    for ch in categories['bad ch'].tolist():
                        if ch not in ['',' ','-','NONE']:
                            badleads.append(monopolar_channels[chan_names == ch.lower()][0].lstrip('0'))
                if 'brain lesions' in categories.keys() and len(categories['brain lesions'].tolist()) > 0:
                    for ch in categories['brain lesions'].tolist():
                        if ch not in ['',' ','-','NONE']:
                            badleads.append(monopolar_channels[chan_names == ch.lower()][0].lstrip('0'))
                if len(badleads)>0:
                    badleads = np.unique(badleads).tolist()
     
            self.bpPairs = np.rec.array(np.array([pairi for pairi in bipolar_pairs if (pairi[0].decode("utf-8").lstrip('0') in goodleads) and (pairi[1].decode("utf-8").lstrip('0') in goodleads) and (pairi[0].decode("utf-8").lstrip('0') not in badleads) and (pairi[1].decode("utf-8").lstrip('0') not in badleads)],dtype=[('ch0', 'S3'), ('ch1', 'S3')]))
            self.mpChans = np.array([chani for chani in monopolar_channels if (chani.lstrip('0') in goodleads) and (chani.lstrip('0') not in badleads)])

        return(self)
    
    def define_rois(self):
        roi = {'PHG': 0,'HC': 0,'MTL': 0,'PC_MPC': 0,'LTC': 0,'MPFC': 0,'LPFC': 0,'LPC': 0}
        
        f = open(self.talPath+'MNIcoords_bp_HarvardOxford.txt','r')
        elecs = f.read().split('\n')
        f.close
        
        roi['PHG']   = [tuple((channeli.split(' ')[0].split('-')[0].zfill(3),channeli.split(' ')[0].split('-')[1].zfill(3))) for channeli in elecs if any([
                            ' '.join(channeli.split(' ')[5:])  == 'Parahippocampal Gyrus, posterior division', 
                            ' '.join(channeli.split(' ')[5:])  == 'Parahippocampal Gyrus, anterior division'])] 
        roi['HC']    = [tuple((channeli.split(' ')[0].split('-')[0].zfill(3),channeli.split(' ')[0].split('-')[1].zfill(3))) for channeli in elecs if any([
                            ' '.join(channeli.split(' ')[5:])  == 'Left Hippocampus', 
                            ' '.join(channeli.split(' ')[5:])  == 'Right Hippocampus'])] 
        roi['MTL']   = [tuple((channeli.split(' ')[0].split('-')[0].zfill(3),channeli.split(' ')[0].split('-')[1].zfill(3))) for channeli in elecs if any([
                            ' '.join(channeli.split(' ')[5:])  == 'Parahippocampal Gyrus, posterior division', 
                            ' '.join(channeli.split(' ')[5:])  == 'Parahippocampal Gyrus, anterior division',
                            ' '.join(channeli.split(' ')[5:])  == 'Left Hippocampus',
                            ' '.join(channeli.split(' ')[5:])  == 'Right Hippocampus'])] 
        roi['LTC']   = [tuple((channeli.split(' ')[0].split('-')[0].zfill(3),channeli.split(' ')[0].split('-')[1].zfill(3))) for channeli in elecs if any([
                            ' '.join(channeli.split(' ')[5:]) == 'Inferior Temporal Gyrus, anterior division',
                            ' '.join(channeli.split(' ')[5:]) == 'Inferior Temporal Gyrus, posterior division',
                            ' '.join(channeli.split(' ')[5:]) == 'Inferior Temporal Gyrus, temporooccipital part',
                            ' '.join(channeli.split(' ')[5:]) == 'Middle Temporal Gyrus, anterior division', 
                            ' '.join(channeli.split(' ')[5:]) == 'Middle Temporal Gyrus, posterior division', 
                            ' '.join(channeli.split(' ')[5:]) == 'Middle Temporal Gyrus, temporooccipital part',
                            ' '.join(channeli.split(' ')[5:]) == 'Superior Temporal Gyrus, posterior division',
                            ' '.join(channeli.split(' ')[5:]) == 'Superior Temporal Gyrus, anterior division',
                            ' '.join(channeli.split(' ')[5:]) == 'Temporal Fusiform Cortex, anterior division',
                            ' '.join(channeli.split(' ')[5:]) == 'Temporal Fusiform Cortex, posterior division',
                            ' '.join(channeli.split(' ')[5:]) == 'Temporal Pole'])] 
        roi['MPFC']   = [tuple((channeli.split(' ')[0].split('-')[0].zfill(3),channeli.split(' ')[0].split('-')[1].zfill(3))) for channeli in elecs if any([
                            ' '.join(channeli.split(' ')[5:]) == 'Cingulate Gyrus, anterior division',
                            ' '.join(channeli.split(' ')[5:]) == 'Frontal Medial Cortex', 
                            ' '.join(channeli.split(' ')[5:]) == 'Paracingulate Gyrus'])]
        roi['LPFC']   = [tuple((channeli.split(' ')[0].split('-')[0].zfill(3),channeli.split(' ')[0].split('-')[1].zfill(3))) for channeli in elecs if any([
                            ' '.join(channeli.split(' ')[5:]) == 'Superior Frontal Gyrus',
                            ' '.join(channeli.split(' ')[5:]) == 'Middle Frontal Gyrus',
                            ' '.join(channeli.split(' ')[5:]) == 'Inferior Frontal Gyrus, pars triangularis',
                            ' '.join(channeli.split(' ')[5:]) == 'Inferior Frontal Gyrus, pars opercularis'])] 
        roi['PC_MPC'] = [tuple((channeli.split(' ')[0].split('-')[0].zfill(3),channeli.split(' ')[0].split('-')[1].zfill(3))) for channeli in elecs if any([
                            ' '.join(channeli.split(' ')[5:]) == 'Precuneous Cortex',
                            ' '.join(channeli.split(' ')[5:]) == 'Cingulate Gyrus, posterior division'])] 
        roi['LPC']    = [tuple((channeli.split(' ')[0].split('-')[0].zfill(3),channeli.split(' ')[0].split('-')[1].zfill(3))) for channeli in elecs if any([
                            ' '.join(channeli.split(' ')[5:]) == 'Superior Parietal Lobule',
                            ' '.join(channeli.split(' ')[5:]) == 'Angular Gyrus',
                            ' '.join(channeli.split(' ')[5:]) == 'Supramarginal Gyrus, posterior division', 
                            ' '.join(channeli.split(' ')[5:]) == 'Supramarginal Gyrus, anterior division'])] 
        self.roi = roi
        
        return(self)
    def determine_micro_recording_type(self):
        
        micfile_list = [item for sublist in [self.events[ni] for ni in self.events.dtype.names if 'micfile' in ni] for item in sublist]
        micfile = np.unique([thefile for thefile in micfile_list if thefile != '' and thefile != '[]'])
        
        if len(micfile) > 0 and 'CSC' in micfile[0] and 'eeg.noreref' not in micfile[0]:
            recording = 'neuralynx'
        elif len(micfile) > 0 and '.ns' in micfile[0] and 'eeg.noreref' not in micfile[0]: 
            recording = 'blackrock'
        elif 'eeg.noreref' not in micfile[0]:
            recording = 'split_channel'
        else: 
            print (micfile)
            raise ValueError('Data doesn''t seem to be aligned.')
        
        return recording
    
    def get_micro_labels(self, localization_field = None, recording = 'blackrock',return_leads = False):
        
        from scipy.io import loadmat
        
        anatomy_table = loadmat('/home1/nherweg/projects/DP3/T_anatomy_lookup_all.mat')['anatomy_table']
        fields = anatomy_table.dtype.descr

        labels = {ei[np.where([li == ('micro_ch_number', '|O') for li in fields])[0][0]][0][0]:ei[np.where([li == (localization_field, '|O') for li in fields])[0][0]][0] for ei in anatomy_table[0] if self.name in ei[0]}
        leads = {ei[np.where([li == ('micro_ch_number', '|O') for li in fields])[0][0]][0][0]:ei[np.where([li == ('electrode_name', '|O') for li in fields])[0][0]][0] for ei in anatomy_table[0] if self.name in ei[0]}
      
        if len(labels) == 0:
            wires = []
            labels = []
            leads = []
            with open('/home1/nherweg/projects/DP3/wm_loc_macro_extrap.txt','r') as in_file:
                for line in in_file:
                    if self.name in line:
                        print(line.split(' '))
                        channels = line.split(' ')[1].split('-')
                        if self.name[0]=='R' and self.name[-1]=='J' and recording == 'blackrock':
                            for chan in range(1,9):
                                wires.append(line.split(' ')[2]+str(chan))
                                labels.append(line.split(' ')[-1].split('\n')[0])
                                leads.append(line.split(' ')[2])
                        elif recording == 'neuralynx':
                            for chan in range(int(channels[0]),int(channels[1])+1):
                                wires.append(str(chan))
                                labels.append(line.split(' ')[-1].split('\n')[0])
                                leads.append(line.split(' ')[2])
                        else:
                            for chan in range(int(channels[0]),int(channels[1])+1):
                                wires.append('chan'+str(chan))
                                labels.append(line.split(' ')[-1].split('\n')[0])
                                leads.append(line.split(' ')[2])
            labels = {bi:li for bi,li in zip(wires,labels)}    
            leads = {bi:li for bi,li in zip(wires,leads)}
        if return_leads:
            return labels, leads
        else:
            return labels
    
    
    def find_non_mtl_micros(self, recording = 'blackrock'):
        
        non_mtl_leads = []
        labels = self.get_micro_labels(localization_field = 'joel', recording = recording)

        if len(labels) > 0:
            for chan in labels.keys():
                if not np.any('MTL' in labels[chan] or 'amy' in labels[chan].lower() or 'HC' in labels[chan] or 'hipp' in labels[chan] or 'subiculum' in labels[chan] or 'PHG' in labels[chan] or 'EC' in labels[chan]):
                    non_mtl_leads.append(str(chan).zfill(2))
        print ('Non-MTL leads detected:')
        print (non_mtl_leads)  
        
        return non_mtl_leads
    
    def find_bad_micros(self):

        if path.isfile('/data/eeg/'+self.name+'/tal/bad_micleads.txt'):
                bad_leads_txt = open('/data/eeg/'+self.name+'/tal/bad_micleads.txt','r')
                bad_leads = bad_leads_txt.read().split('\n')
                bad_leads_txt.close()
                bad_leads = [lead for lead in bad_leads if lead is not '']
                bad_leads.append('ainp')
        else:
            bad_leads = ['ainp']
        print ('Bad leads detected:')
        print (bad_leads)
        
        return bad_leads