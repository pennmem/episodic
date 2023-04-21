
from ptsa.data.readers import BaseEventReader
import numpy as np
from scipy.spatial import distance
import scipy.stats as scps
from numpy.lib.recfunctions import append_fields
from glob import glob

    
class session(object):
    """This class handles session specific information for deliveryPerson 2-4.

    Attributes:
    index: index of the session in the sessionList
    path: path to the events file in /data/events
    subj: subject name
    eeg: bool indicating whether events should be read with (0) or without events (1) without eeg
    eegpath: path to scalp eeg data in /data/eeg, read using the method get_events()
    hasmicro: bool indicating whether a subject has micro wire data, read using the method get_events()
    axis: do the events refer to second spatial dimension as Y or Z, read using the method get_events()
    events: events structure, read using the method get_events()
    snumber: number of session as read from the events. This is checked against the numeral in the events.mat name. Read using the method get_events()
    """
    def __init__(self,index,eeg,sessionList):
        self.index    = index
        self.path     = sessionList[index]
        self.subj     = self.path.split('/')[-1][:self.path.split('/')[-1].find('_e')]
        self.eeg      = eeg
        self.eegpath  = None
        self.hasmicro = None
        self.comp2rej = None
        self.axis     = None
        
    def get_events(self,exc = None):
        
        # Read events
        if 'LTP' in self.path:
            events = BaseEventReader(filename=self.path, use_reref_eeg = True, eliminate_events_with_no_eeg=self.eeg).read()
        else:
            events = BaseEventReader(filename=self.path, use_reref_eeg = False, eliminate_events_with_no_eeg=self.eeg).read()
        
        eeg_file_info = events['eegfile']
        if self.subj == 'R1017J':
            eeg_file_info = ['/data/eeg'+ii if '/data/eeg' not in ii and ii is not '' else '' for ii in eeg_file_info ]
            events['eegfile'] = eeg_file_info
            
        if self.subj == 'R1027J':
            eeg_file_info = ['/data'+ii if '/data' not in ii and ii is not '' else '' for ii in eeg_file_info ]
            events['eegfile'] = eeg_file_info
            
        if 'DBoy3' in self.path or 'DBoy4' in self.path:
            self.axis = 'Z'
        else:
            self.axis = 'Y'
        
        if 'micfile' in events.dtype.names:
            self.hasmicro = True
        else: 
            self.hasmiscro = False
            events = append_fields(events,['micfile','micoffset'],[np.array(['' for e in events], dtype = '<U256'),np.ones(len(events))*-1],usemask=False,asrecarray = True)
        
        # Remove columns not of interest
        eoi         = ['subject','session','trial','type','serialPos','item','store','storeX',
                      'store'+self.axis,'presX','pres'+self.axis,'itemno','recalled','intruded','finalrecalled',
                      'rectime','intrusion','mstime','eegfile','eegoffset']
        if 'DBoy4' in self.path:
            eoi.extend(['correctPointingDirection','submittedPointingDirection'])

        if exc != 'micro':
            eoi = eoi + [ni for ni in events.dtype.names if 'micfile' in ni or 'micoffset' in ni]
        
        # Fix dtype information for storeY location from i to f
        dt = events.dtype.descr# this is now a modifiable list, can't modify numpy.dtype
        for ind,di in enumerate(dt):
            if di[0] == 'store'+self.axis:
                dt[ind] = ('store'+self.axis,'<f8')
          
        events = events.astype(np.dtype(dt))
        self.events = np.array(events[eoi])
        
        # Fix formatting of German stores in old version
        vec_stores = self.events['store']

        if np.any([len(si.split(' '))>1 for si in vec_stores if si != 'FITNESS STUDIO' and si != 'FAST FOOD' and si != 'JUWELIER GESCHAFT']):
            vec_stores[vec_stores == 'KOSTUMLADEN'] = 'DEN KOSTUMLADEN'
        if np.any(vec_stores == u'DAS JUWELIER GESCH\xc4FT'):
            vec_stores[vec_stores == u'DAS JUWELIER GESCH\xc4FT'] = 'DAS JUWELIER GESCHAFT'
        if np.any(vec_stores == u'DAS MUSIKGESCH\xc4FT'):
            vec_stores[vec_stores == u'DAS MUSIKGESCH\xc4FT'] = 'DAS MUSIKGESCHAFT'
        if np.any(vec_stores == u'DAS CAF\xc9'):
            vec_stores[vec_stores == u'DAS CAF\xc9'] = 'DAS CAFE'
        if np.any(vec_stores == u'DAS EISCAF\xc9'):
            vec_stores[vec_stores == u'DAS EISCAF\xc9'] = 'DAS EISCAFE'
        if np.any(vec_stores == u'DIE B\xc4CKEREI'):
            vec_stores[vec_stores == u'DIE B\xc4CKEREI'] = 'DIE BACKEREI'
        if np.any(vec_stores == u'DEN KOST\xdcMLADEN'):
            vec_stores[vec_stores == u'DEN KOST\xdcMLADEN'] = 'DEN KOSTUMLADEN'
        vec_stores                              = np.array(['_'.join(si.split(' ')) for si in vec_stores])
        self.events['store']                    = vec_stores
        
        # Append encoding and recall times
        enc_mstime =  [-999. if ev['type'] != 'REC_WORD'
                       else events[(events['item'] == ev['item']) & (events['type'] == 'WORD') & (events['trial'] == ev['trial'])]['mstime'].tolist()[0] 
                       if len(events[(events['item'] == ev['item']) & (events['type'] == 'WORD') & (events['trial'] == ev['trial'])]['mstime']) == 1 
                       else -999. 
                       if len(events[(events['item'] == ev['item']) & (events['type'] == 'WORD') & (events['trial'] == ev['trial'])]['mstime']) == 0
                       else float('nan') for ev in events]
        
        if np.any(np.isnan(enc_mstime),axis = 0):
            raise ValueError('Wort repeat within list detected.')
        
        # Append recall filter
        repeat       = [0 if (eventi.type == 'REC_WORD') & ([ii for ii in self.events[:ind]['item']].count(eventi['item']) < 2) 
                        else 1 if (eventi.type == 'REC_WORD') & (eventi.intrusion == 0) & ([ii for ii in self.events[:ind]['item']].count(eventi['item']) > 1) 
                        else -999 for ind,eventi in enumerate(self.events)]
        self.events  = append_fields(self.events,'repeat',np.array(repeat),usemask=False, asrecarray=True)
        singleRecall = [0 if (eventi.type == 'REC_WORD') & (len([ii for ii in self.events[(self.events.trial == eventi.trial) & (self.events.type == 'REC_WORD') & (self.events.intrusion == 0) & (self.events['repeat'] == 0)]]) > 1)
                        else 1 if (eventi.type == 'REC_WORD') & (eventi.intrusion == 0) & (eventi['repeat'] == 0) & (len([ii for ii in self.events[(self.events.trial == eventi.trial) & (self.events.type == 'REC_WORD') & (self.events.intrusion == 0) & (self.events['repeat'] == 0)]]) < 2)
                        else -999 for eventi in self.events]
        surrVocalization = [-999 if eventi.type != 'REC_WORD'
                            else 0 if (len(events[:ind][events[:ind].type == 'REC_WORD']) == 0) & (len(events[:ind][events[:ind].type == 'REC_WORD_VV']) == 0)
                            or events[:ind][(events[:ind].type == 'REC_WORD') | (events[:ind].type == 'REC_WORD_VV')][-1].mstime < eventi.mstime - 1750
                            else 1 for ind, eventi in enumerate(self.events)]
        output_positions = []
        outP = 1
        triali = 0
        for itemi in self.events:
            if itemi.type == 'REC_WORD' and itemi['intrusion'] < 1 and itemi['repeat'] < 1:
                if itemi['trial'] > triali:
                    outP = 1
                    triali = itemi['trial']
                output_positions.append(outP)
                outP += 1
            else: output_positions.append(-999)
                
        self.events  = append_fields(self.events,['singleRecall','surrVocalization','enc_mstime','outputPosition'],[np.array(singleRecall),np.array(surrVocalization),np.array(enc_mstime),np.array(output_positions)],usemask=False, asrecarray=True)
        
        self.snumber = self.events[0][1]
        if 'sess'+str(self.snumber) not in self.path:
            raise ValueError('Filename and session number do not match for session: '+self.path)
        
        if 'LTP' in self.subj:
            self.eegpath = '/data/eeg/scalp/ltp/db2.5/'+self.subj+'/session_'+str(self.snumber)+'/eeg/eeg.reref/'
            f = open('/home1/nherweg/projects/DP3/analyses/comp2rej.txt')
            txtIn = f.read().split('\n')
            comp2rej = [s.split(' ')[1].split(',') for s in txtIn if self.path in s]
            if len(comp2rej) == 1:
                self.comp2rej = [int(c) for c in comp2rej[0]]
            elif len(comp2rej) == 0:
                print('No components selected')
            else: raise ValueError('Something is wrong with comp2rej.txt. Failed to process '+self.path)
        return(self)
    
    def get_logfile_path(self):

        # Tentative log file
        print('Checking for /data/eeg/' + self.subj + '/behavioral/*oy*/session_'+ str(self.snumber) + '/log.txt')
        logfile = glob('/data/eeg/' + self.subj + '/behavioral/*oy*/session_'+ str(self.snumber) + '/log.txt')
        if len(logfile) == 1:
            logfile = logfile[0]
            file_type = 'txt'
        else:
            print('Checking for /data/eeg/' + self.subj + '/behavioral/*OY*/session_'+ str(self.snumber) + '/session.jsonl')
            logfile = glob('/data/eeg/' + self.subj + '/behavioral/*OY*/session_'+ str(self.snumber) + '/session.jsonl')
            if len(logfile) == 1:
                logfile = logfile[0]
                file_type = 'json'
            else:
                raise ValueError('Couldn''t locate log file')

        # Read order of early stores from log
        ordered_stores_log = []
       
        with open(logfile, 'r') as in_file:
            
            for line in in_file: 

                #if 'Player transform' not in line and 'Sync pulse begin' not in line and 'pointer transform' not in line:
                if 'object presentation begins' in line:
                    ordered_stores_log.append('_'.join(line.split('"store name":"')[1].split('"')[0].split(' ')))
                if 'Trial Event' in line and 'STORE_TARGET_STARTED' in line: 
                    ordered_stores_log.append(line.split('\t')[4])
                #elif 'PLANNER_START' in line:
                #    ordered_stores_log.append(line.split('\t')[3])
                elif len(line.split('\t')) > 2 and line.split('\t')[2] == 'SIMPLESOUND_LOADFILE' and 'beephigh.wav' not in line.split('\t')[-1]:
                    ordered_stores_log.append(line.split('\t')[-1].split('/')[-2].upper())
                elif 'showImage' in line.split('\t')[-1] and 'coll_box' not in line.split('\t')[-2] and line.split('\t')[-3] == 'VROBJECT_COLLISIONCALLBACK':
                    ordered_stores_log.append(line.split('\t')[-2].upper())
  
        if np.setdiff1d(ordered_stores_log, self.events['store'][self.events['store']!='-999']).size>0:
            print(np.setdiff1d(ordered_stores_log, self.events['store'][self.events['store']!='-999']))
            print(self.events['store'][self.events['store']!='-999'])
            print(ordered_stores_log)
            raise ValueError('Stores don''t match.')
            
        if self.subj == 'FR429' and self.snumber == 0:
            ordered_stores_log = ordered_stores_log [1:12]
        else:
            ordered_stores_log = ordered_stores_log [:11]
        
        ordered_stores_events = self.events[(self.events['type'] == 'DELIV_START') | (self.events['type'] == 'INST') | (self.events['type'] == 'INST_FAM')]['store']
        ordered_stores_events = ordered_stores_events[ordered_stores_events != '-999'].tolist()[:41]
        # Compare order to order in events
        if len(ordered_stores_log) == 11 and np.all([si == sj for si,sj in zip(ordered_stores_events,ordered_stores_log)]): 
            return logfile
        else: 
            print(len(ordered_stores_log))
            print([si == sj for si,sj in zip(ordered_stores_events,ordered_stores_log)])
            print(ordered_stores_events)
            print(ordered_stores_log)
            raise ValueError('Logfile doesn''t match events')
    
    def get_stores(self):
        
        # Filter events
        filtered_events = self.events[self.events['type'] != 'STORE_FAM']
        
        # Get store names
        ident = [ii for ii in np.unique(filtered_events.store[(filtered_events.store != '[]') & (filtered_events.store != 'NaN') & (filtered_events.store != '-999')])]
        
        # Get store locations
        x = [np.unique(filtered_events.storeX[np.where(filtered_events.store == stname)[0]]) for stname in ident] 
        z = [np.unique(filtered_events['store'+self.axis][np.where(filtered_events.store == stname)[0]]) for stname in ident]
        
        # Make sure each store only has one location
        if np.any([len(xi) != 1 for xi in x]):
            print(ident)
            print(x)
            raise ValueError('No consistent storeX location for session: '+self.path)
        if np.any([len(zi) != 1 for zi in z]):
            print(ident)
            print(z)
            raise ValueError('No consistent storeZ location for session: '+self.path)
        x = [xi[0] for xi in x]
        z = [zi[0] for zi in z]
       
        return(ident,x,z)
    
    def get_dist_mat(self,binned = False, k = 1, x = None, z = None, normalize = True):  
       
        if x is None:
            # Get store information
            _,x,z = self.get_stores()
        
        # Calculate distances between all stores visited in this session
        d = distance.squareform(distance.pdist(np.column_stack((x,z)), metric='euclidean'))
        if normalize:
            d = np.triu(1-(d/np.max(d)),k=k) # 1 is closest and itself, 0 is farthest
        else: 
            d = np.triu(d,k=k)

        if binned:
            binsize = len(np.unique(d))//3
            bins = [-1,np.unique(d)[binsize-1],np.unique(d)[(2*binsize)-1],1]
            #bins =  [-1,1/3.,2/3.,1]
            d = np.digitize(d,bins,right = True)
            
        return(d)
    
    def calc_transition_dist(self, append = False, perm = False):

        REC   = [[linei.trial,np.int(linei.serialPos),self.events[(self.events['item'] == linei['item']) & (self.events['type'] == 'WORD')]['mstime'][0],np.int(self.get_stores()[0].index(linei.store))]
                for linei in self.events if (linei.type == 'REC_WORD' and linei.intrusion==0 and linei['repeat'] != 1 and linei.singleRecall != 1)] 

        if len(REC)>0:
            # Split data up in trials
            RECT  = [[linei for linei in REC if linei[0]==triali] for triali in [ii for ii in np.unique([RECi[0] for RECi in REC])]] # Trial number,serialPos,encoding time,storeIdent

            if perm:
                RECT = [np.random.permutation(trial).tolist() for trial in RECT]
                RECT = [[[int(entry)  if ind != 2 else entry for ind,entry in enumerate(transition)] for transition in trial] for trial in RECT]

            # Calculate normalized and binned spatial distance for each actual transition
            ATB   = [self.get_dist_mat(binned = True)[np.min([RECTi[transition][3],RECTi[transition+1][3]]),np.max([RECTi[transition][3],RECTi[transition+1][3]])] 
                       for RECTi in RECT for transition in range(len(RECTi)-1)] #list of n trials per session
            ATL    = [[self.get_dist_mat()[np.min([RECTi[transition][3],RECTi[transition+1][3]]),np.max([RECTi[transition][3],RECTi[transition+1][3]])] 
                       for transition in range(len(RECTi)-1)] for RECTi in RECT] #list of n trials per session of n transistions per trial
            ATL_time = [[abs(RECTi[transition+1][2]-RECTi[transition][2])/1000. for transition in range(len(RECTi)-1)] for RECTi in RECT]#list of n trials per session of n transistions per trial time in s

            # Calculate normalized and binned spatial distance for all possible transitions
            PTL      = []
            PTB      = []
            PTL_time = []
            for RECTi in RECT:#loop over trials
                PT      = []
                PT_time = []
                # Initialize possible transistions too all stores and times presented on given trial
                PoR      = range(len(self.get_stores()[0]))
                PoR      = np.sort([i for i in PoR if i in 
                           [np.int(self.get_stores()[0].index(s)) for s in self.events[(self.events['trial'] == RECTi[0][0]) & (self.events['type'] == 'WORD')]['store'].tolist()]]).tolist()
                PoR_time = self.events[(self.events['trial'] == RECTi[0][0]) & (self.events['type'] == 'WORD')]['mstime'].tolist()
                for transition in range(len(RECTi)-1):
                    # Present recall
                    PrR      = RECTi[transition][3] 
                    PrR_time = RECTi[transition][2]
                    # Next recall
                    NeR      = RECTi[transition+1][3]
                    NeR_time = RECTi[transition+1][2]
                    if transition == 0:
                        PoR.remove(PrR)#remove current recall from possible transitions
                        PoR_time.remove(PrR_time)
                    PTB.append(list(set([self.get_dist_mat(binned = True)[np.min([PoRi,PrR]),np.max([PoRi,PrR])] for PoRi in PoR])))#list(set(#taking all possible transisition instead of just the unique set should correct for the unequal number of stores per bin and associated unequal probability of recalling from a certain bin 
                    PoR.remove(NeR)#remove next recall from possible transitions for calculation of percentile score
                    PoR_time.remove(NeR_time)
                    if len(PoR)>0: # if subject recalls all item calculation of SCS not possible for last recall
                        PT.append([self.get_dist_mat()[np.min([PoRi,PrR]),np.max([PoRi,PrR])] for PoRi in PoR])#distance for possible transisitons in list transitions x possible transition
                        PT_time.append([abs(PrR_time - PoRi_time)/1000. for PoRi_time in PoR_time])
                PTL.append(PT)# appends PT over trials 
                PTL_time.append(PT_time)
            PTB = [PTBii for PTBi in PTB for PTBii in PTBi]#creates flat version of PTB per session 
      
            # Calculate percentile score
            SCS = [[scps.percentileofscore(PTLii,ATLii) for ATLii,PTLii in zip(ATLi,PTLi)] for ATLi, PTLi in zip(ATL,PTL)]
            TCS = [[100.-scps.percentileofscore(PTLii,ATLii) for ATLii,PTLii in zip(ATLi,PTLi)] for ATLi, PTLi in zip(ATL_time,PTL_time)]
 
            if perm:
                return SCS,TCS
            else:
                
                self.SCS = SCS
                self.TCS = TCS
                
                # Sum bin counts up irrespective of trials 
                self.ATBc = np.array([ATB.count(bini) for bini in range(1,4)], dtype = float)
                self.PTBc = np.array([PTB.count(bini) for bini in range(1,4)], dtype = float)
                
                if append:

                    # Reformat transisition distance to match events
                    postDist = np.array([[lineiR[0],lineiR[1],lineiA] for RECTi,ATLi in zip(RECT,ATL) for lineiR,lineiA in  zip(RECTi[:-1],ATLi)]) 
                    preDist  = np.array([[lineiR[0],lineiR[1],lineiA] for RECTi,ATLi in zip(RECT,ATL) for lineiR,lineiA in  zip(RECTi[1:] ,ATLi)]) 

                    postPerc = np.array([[lineiR[0],lineiR[1],lineiA] for RECTi,ATLi in zip(RECT,self.SCS) for lineiR,lineiA in  zip(RECTi[:-1],ATLi)]) 
                    prePerc  = np.array([[lineiR[0],lineiR[1],lineiA] for RECTi,ATLi in zip(RECT,self.SCS) for lineiR,lineiA in  zip(RECTi[1:] ,ATLi)]) 
                    
                    postDist_time = np.array([[lineiR[0],lineiR[1],lineiA] for RECTi,ATLi in zip(RECT,ATL_time) for lineiR,lineiA in  zip(RECTi[:-1],ATLi)]) 
                    preDist_time  = np.array([[lineiR[0],lineiR[1],lineiA] for RECTi,ATLi in zip(RECT,ATL_time) for lineiR,lineiA in  zip(RECTi[1:] ,ATLi)]) 

                    postPerc_time = np.array([[lineiR[0],lineiR[1],lineiA] for RECTi,ATLi in zip(RECT,self.TCS) for lineiR,lineiA in  zip(RECTi[:-1],ATLi)]) 
                    prePerc_time  = np.array([[lineiR[0],lineiR[1],lineiA] for RECTi,ATLi in zip(RECT,self.TCS) for lineiR,lineiA in  zip(RECTi[1:] ,ATLi)]) 

                    self.events = append_fields(self.events,'postDist',np.array([postDist[(postDist[:,0] == linei.trial) & (postDist[:,1] == linei.serialPos)][0][2] 
                         if postDist[(postDist[:,0] == linei.trial) & (postDist[:,1] == linei.serialPos)].shape[0] == 1 else float('nan') 
                         for linei in self.events]),usemask=False, asrecarray=True) 
                    self.events = append_fields(self.events,'preDist',np.array([preDist[(preDist[:,0] == linei.trial) & (preDist[:,1] == linei.serialPos)][0][2] 
                         if preDist[(preDist[:,0] == linei.trial) & (preDist[:,1] == linei.serialPos)].shape[0] == 1 else float('nan') 
                         for linei in self.events]),usemask=False, asrecarray=True) 
                    self.events = append_fields(self.events,'postPerc',np.array([postPerc[(postPerc[:,0] == linei.trial) & (postPerc[:,1] == linei.serialPos)][0][2] 
                         if postPerc[(postPerc[:,0] == linei.trial) & (postPerc[:,1] == linei.serialPos)].shape[0] == 1 else float('nan') 
                         for linei in self.events]),usemask=False, asrecarray=True) 
                    self.events = append_fields(self.events,'prePerc',np.array([prePerc[(prePerc[:,0] == linei.trial) & (prePerc[:,1] == linei.serialPos)][0][2] 
                         if prePerc[(prePerc[:,0] == linei.trial) & (prePerc[:,1] == linei.serialPos)].shape[0] == 1 else float('nan') 
                         for linei in self.events]),usemask=False, asrecarray=True) 

                    self.events = append_fields(self.events,'postDist_time',np.array([postDist_time[(postDist_time[:,0] == linei.trial) & (postDist_time[:,1] == linei.serialPos)][0][2] 
                         if postDist_time[(postDist_time[:,0] == linei.trial) & (postDist_time[:,1] == linei.serialPos)].shape[0] == 1 else float('nan') 
                         for linei in self.events]),usemask=False, asrecarray=True) 
                    self.events = append_fields(self.events,'preDist_time',np.array([preDist_time[(preDist_time[:,0] == linei.trial) & (preDist_time[:,1] == linei.serialPos)][0][2] 
                         if preDist_time[(preDist_time[:,0] == linei.trial) & (preDist_time[:,1] == linei.serialPos)].shape[0] == 1 else float('nan') 
                         for linei in self.events]),usemask=False, asrecarray=True) 
                    self.events = append_fields(self.events,'postPerc_time',np.array([postPerc_time[(postPerc_time[:,0] == linei.trial) & (postPerc_time[:,1] == linei.serialPos)][0][2] 
                         if postPerc_time[(postPerc_time[:,0] == linei.trial) & (postPerc_time[:,1] == linei.serialPos)].shape[0] == 1 else float('nan') 
                         for linei in self.events]),usemask=False, asrecarray=True) 
                    self.events = append_fields(self.events,'prePerc_time',np.array([prePerc_time[(prePerc_time[:,0] == linei.trial) & (prePerc_time[:,1] == linei.serialPos)][0][2] 
                         if prePerc_time[(prePerc_time[:,0] == linei.trial) & (prePerc_time[:,1] == linei.serialPos)].shape[0] == 1 else float('nan') 
                         for linei in self.events]),usemask=False, asrecarray=True) 
        else:
            self.ATBc = []
            self.PTBc = []
            self.SCS  = []
            self.TCS  = []
            
            if append: 
                self.events = append_fields(self.events,'postDist',np.ones(len(self.events)) * float('nan'),usemask=False, asrecarray=True) 
                self.events = append_fields(self.events,'preDist',np.ones(len(self.events)) * float('nan'),usemask=False, asrecarray=True) 
                self.events = append_fields(self.events,'postPerc',np.ones(len(self.events)) * float('nan'),usemask=False, asrecarray=True) 
                self.events = append_fields(self.events,'prePerc',np.ones(len(self.events)) * float('nan'),usemask=False, asrecarray=True) 
                self.events = append_fields(self.events,'postDist_time',np.ones(len(self.events)) * float('nan'),usemask=False, asrecarray=True) 
                self.events = append_fields(self.events,'preDist_time',np.ones(len(self.events)) * float('nan'),usemask=False, asrecarray=True) 
                self.events = append_fields(self.events,'postPerc_time',np.ones(len(self.events)) * float('nan'),usemask=False, asrecarray=True) 
                self.events = append_fields(self.events,'prePerc_time',np.ones(len(self.events)) * float('nan'),usemask=False, asrecarray=True) 
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
            
    def get_mtl_micros(self,subi):
    
        micfile_list = [item for sublist in [self.events[ni] for ni in self.events.dtype.names if 'micfile' in ni] for item in sublist]
        micfile = np.unique([thefile for thefile in micfile_list if thefile != '' and thefile != '[]'])
        
        recording = self.determine_micro_recording_type()
            
        bad_leads     = subi.find_bad_micros()
        non_mtl_leads = subi.find_non_mtl_micros(recording = recording)
        print(non_mtl_leads)
        if recording == 'blackrock':
              
            from os                    import path            
            import sys
            sys.path.append(path.abspath('/home1/nherweg/toolbox/brPY'))
            from brpylib import NsxFile
            
            nsx_file      = NsxFile(micfile[0])
            config        = nsx_file.getdata('all', 1, 1, 1)
            samplerate    = config['samp_per_s']

            if self.subj[:2] == 'FR':
                channels      = [config['elec_ids'][ind]                          for ind,hdr_ind in enumerate(config['ExtendedHeaderIndices']) 
                                 if np.all([bad_chan not in nsx_file.extended_headers[hdr_ind]['ElectrodeLabel'] for bad_chan in bad_leads]) and nsx_file.extended_headers[hdr_ind]['ElectrodeLabel'] not in non_mtl_leads]
                channel_names = [nsx_file.extended_headers[hdr_ind]['ElectrodeLabel'] for hdr_ind in config['ExtendedHeaderIndices'] #nsx_file[0].extended_headers[hdr_ind]['ElectrodeLabel']
                                 if np.all([bad_chan not in nsx_file.extended_headers[hdr_ind]['ElectrodeLabel'] for bad_chan in bad_leads]) and nsx_file.extended_headers[hdr_ind]['ElectrodeLabel'] not in non_mtl_leads]
            else:
                channels      = [config['elec_ids'][ind]                          for ind,hdr_ind in enumerate(config['ExtendedHeaderIndices']) 
                                 if 'chan' not in nsx_file.extended_headers[hdr_ind]['ElectrodeLabel'] and np.all([bad_chan not in nsx_file.extended_headers[hdr_ind]['ElectrodeLabel'] for bad_chan in bad_leads]) and nsx_file.extended_headers[hdr_ind]['ElectrodeLabel'] not in non_mtl_leads]
                channel_names = [nsx_file.extended_headers[hdr_ind]['ElectrodeLabel'] for hdr_ind in config['ExtendedHeaderIndices']
                                 if 'chan' not in nsx_file.extended_headers[hdr_ind]['ElectrodeLabel'] and np.all([bad_chan not in nsx_file.extended_headers[hdr_ind]['ElectrodeLabel'] for bad_chan in bad_leads]) and nsx_file.extended_headers[hdr_ind]['ElectrodeLabel'] not in non_mtl_leads]
            print ([nsx_file.extended_headers[hdr_ind]['ElectrodeLabel'] for hdr_ind in config['ExtendedHeaderIndices']])
            orig_channels = None
        elif recording == 'neuralynx':
            
            from alignment_tools import get_params
            
            samplerate, _ = get_params(micfile[0],'neuralynx')
            labels = subi.get_micro_labels(localization_field = 'joel',recording = 'neuralynx')
            orig_channels = np.unique([fi.split('CSC')[-1].split('.')[0] for fi in glob(micfile[0]+'*.ncs')]).tolist()
            print(orig_channels)
            print(labels.keys())
            # We want these sorted
            channels = np.unique([int(chan) for chan in orig_channels if chan.lstrip('0') in labels.keys() and chan not in non_mtl_leads and np.all(bad_chan not in chan for bad_chan in bad_leads)]).astype(str).tolist()
            print(channels)
            channel_names = channels
            
        elif recording == 'split_channel':
            basename = micfile[0].split('/')[:-1]
            print ('/'.join(basename)+'/params.txt')
            with open('/'.join(basename)+'/params.txt', 'rt') as in_file: 
                for line in in_file:
                    if 'samplerate' in line:
                        samplerate = np.float(line.split(' ')[-1])
                        break
                    else: raise ValueError('Couldn''t detect samplerate information.')
            orig_channels = np.unique([fi.split('.')[-1] for fi in glob('/'.join(basename)+'/*') if 'txt' not in fi]).tolist()
            channels = [chan for chan in orig_channels if chan not in non_mtl_leads and 'chan'+chan.lstrip('0') not in non_mtl_leads and np.all(bad_chan not in chan for bad_chan in bad_leads)]
            channel_names = channels

        print ('Processing channels:')
        print (channel_names)
        
        return micfile,channels,channel_names,bad_leads,non_mtl_leads,orig_channels,samplerate
        
    def identify_nav_epochs(self,micro = True):
        
        # Identify navigation epochs (defined as being within a list and not interrupted by a change in file)
        start  = []
        finish = []
        searching_finish = 0
        
        if micro == True:
            for ind_ev,ev in enumerate(self.events):
                micfields = [ty for ty in ev.dtype.names if 'micfile' in ty and not np.all(self.events[ty] == '')]
                if ('INST' in ev['type'] or 'DELIV_START' in ev['type'] or 'pointing finished' in ev['type']) and np.all([ev[micfieldi] != '' for micfieldi in micfields]) and np.all([ev[micfieldi] != '[]' for micfieldi in micfields]) and not searching_finish and (ind_ev == len(self.events)-1 or np.all([ev[keyw] == self.events[ind_ev+1][keyw] for keyw in [ty for ty in ev.dtype.names if 'micfile' in ty]])):
                    start.append({keyw: ev[keyw] for keyw in [ty for ty in ev.dtype.names if 'mic' in ty or 'mstime' in ty or 'trial' in ty or 'session' in ty]})#[ev['micoffset'],ev['micfile'],ev['mstime']]) 
                    searching_finish = 1
                elif searching_finish and (ev['type'] == 'REC_START' or ev['type'] == 'SESS_STARTED' or ev['type'] == 'pointing begins' or ind_ev == len(self.events)-1 or np.any([ev[keyw] != self.events[ind_ev+1][keyw] for keyw in [ty for ty in ev.dtype.names if 'micfile' in ty]])):
                    finish.append({keyw: ev[keyw] for keyw in [ty for ty in ev.dtype.names if 'mic' in ty or 'mstime' in ty or 'trial' in ty or 'session' in ty]})#[ev['micoffset'],ev['micfile'],ev['mstime']])
                    searching_finish = 0
                    t_min = ((finish[-1]['mstime']-start[-1]['mstime'])/1000.)/60.
                    print('Time in min: '+ str(int(t_min)))
                    if t_min > 30:
                        del(start[-1])
                        del(finish[-1])
                        print('Long list deleted')
                        t_min = ((finish[-1]['mstime']-start[-1]['mstime'])/1000.)/60.
                        print('Time in min of last list: '+ str(int(t_min)))
                        
        else:
            for ind_ev,ev in enumerate(self.events):
                if ('INST' in ev['type'] or 'DELIV_START' in ev['type'] or 'pointing finished' in ev['type']) and not searching_finish:
                    start.append({keyw: ev[keyw] for keyw in [ty for ty in ev.dtype.names if 'mstime' in ty or 'trial' in ty or 'session' in ty]})
                    searching_finish = 1
                elif searching_finish and (ev['type'] == 'REC_START' or ev['type'] == 'SESS_STARTED' or ev['type'] == 'pointing begins' or ind_ev == len(self.events)-1):
                    finish.append({keyw: ev[keyw] for keyw in [ty for ty in ev.dtype.names if 'mstime' in ty or 'trial' in ty or 'session' in ty]})
                    searching_finish = 0
                    t_min = ((finish[-1]['mstime']-start[-1]['mstime'])/1000.)/60.
                    print('Time in min: '+ str(int(t_min)))
                    if t_min > 30:
                        del(start[-1])
                        del(finish[-1])
                        print('Long list deleted')
                        t_min = ((finish[-1]['mstime']-start[-1]['mstime'])/1000.)/60.
                        print('Time in min of last list: '+ str(int(t_min)))
            
        print ('Found '+str(len(start))+' navigation periods.')
        
        return start,finish
    
    