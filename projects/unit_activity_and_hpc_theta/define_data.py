
from glob import glob
import numpy as np
from os import path

def define_data(eeg, micros,scalp,dBoy25,DBoy3_1,include_learning, DBoy4, more_than_one_sess):
    
    sessionList = []
    if dBoy25:
        sessionList.extend(np.array(glob('/data/events/dBoy25/*.mat')).tolist())
    if scalp:
        sessionList.extend(np.array(glob('/data/events/dBoy25_scalp/*events_sess*.mat')).tolist())
    if DBoy3_1:
        sessionList.extend(np.array(glob('/data/events/DBoy3_1/delivery/*.mat')).tolist())
        if include_learning:
            sessionList.extend(np.array(glob('/data/events/DBoy3_1/learning/*.mat')).tolist())
    if DBoy4:
        sessionList.extend(np.array(glob('/data/events/DBoy4/*.mat')).tolist())
            
    subExc      = ['R1089P','R1093J','TJ030','FR433','TJ056','FR432','FR437','TJ039_2','R1310J','FR430','FR450','FR451']

                # 'FR437','FR435'-sess0 log file has multiple 'SESS_START' entries, event creation does not work
                # FR432 no annotations in events
                # 'R1089P','R1093J' short testing session interrupted & technical issues
                # 'TJ030' has empty log files 
                # 'TJ039_2' second session was run with different store locations, first session was run with different montage therefore this sub is excluded as a whole
                # 'TJ056' wav file exists for one list only
                # R1310J words repeated and logging stopped after list 1 --> just 3 lists in total could be used
                # 'FR450' wav files noisy
                # 'FR451' too few data
                
    sessExc     = ['/data/events/dBoy25/FR423_events_sess1.mat'] # second session was run with different store locations

    if eeg:
        subExc.extend(('R1019J', 'R1030J'))
                # 'R1019J'  'R1030J' less than 9 events with EEGdata in R or F condition
                # FR454_learning has empty eeglog.txt
                
        sessExc.extend(('/data/events/dBoy25/FR426_events_sess1.mat',   # no complete list EEG data
                        '/data/events/dBoy25/FR423_events_sess1.mat',   # recorded with different store locations
                        '/data/events/dBoy25/FR434_events_sess1.mat',
                        '/data/events/dBoy25/FR434_events_sess2.mat'))  # missing EEG Data 
    
    sessionList = [sessi for sessi in sessionList if sessi.split('/')[-1][:sessi.split('/')[-1].find('_e')] not in subExc and sessi not in sessExc and 'DO_NOT_USE' not in sessi]
    subjectList = np.unique([sessi.split('/')[-1][:sessi.split('/')[-1].find('_e')] for sessi in sessionList]).tolist()
    
    # Search for data not present in /data/events
    #subSearch      = glob('/data/eeg/*/*/deliveryBoy*')+glob('/data/eeg/*/*/dB*')+glob('/data/eeg/*/*/db*')+ \
    #                 glob('/data/eeg/*/*/*/deliveryBoy*')+glob('/data/eeg/*/*/*/dB*')+glob('/data/eeg/*/*/*/db*')+ \
    #                 glob('/data/eeg/*/deliveryBoy*')+ glob('/data/eeg/*/dB*')+glob('/data/eeg/*/db*')
    #subjectListExt = list(set([re.search('[A-Z]+[0-9]+_*[0-9]*',potSubi).group(0) for potSubi in subSearch if re.search('[A-Z]+[0-9]+_*[0-9]*',potSubi) is not None]))
    #missedSub      = [subi for subi in subjectListExt if subi not in subjectList and subi not in subExc]
    #print('You missed the following subjects:', missedSub)

    # Optionally filter out sessions without microelectrode recordings
    if micros:
        noMicros = [subi for subi in subjectList if not path.exists('/data/eeg/'+subi+'/npt') 
                and not path.exists('/data/eeg/'+subi+'/spikes') 
                and not path.exists('/data/eeg/'+subi+'/wave_clus') 
                and not path.exists('/data/eeg/'+subi+'/raw/microeeg') 
                and not path.exists('/data/eeg/'+subi+'/raw/micro')
                and not path.exists('/data/eeg/'+subi+'/micro')
                and len(glob('/data/eeg/'+subi+'/raw/*/micro')) == 0
                and len(glob('/data/eeg/'+subi+'/raw/*/*/nlx')) == 0
                and len(glob('/data/eeg/'+subi+'/*/*/*/*/*.ncs')+glob('/data/eeg/'+subi+'/*/*/*/*.ncs')+glob('/data/eeg/'+subi+'/*/*/*.ncs')) == 0 
                and len(glob('/data/eeg/'+subi+'/raw/*.emg')) == 0 ]
        print('noMicros')
        print(noMicros)
        exc_subs_micro = ['FR423', 'FR424', 'TJ039', 'TJ005_1', 'R1068J','R1017J','TJ027']#TJ027 wire bundles in WM, R1017J wires can not be mapped to leads >> no locs, R1027J now aligned
        subjectList = [subi for subi in subjectList if subi not in noMicros and subi not in exc_subs_micro]#FR423 no splitted sync files; 424 no splitted files; TJ039, TJ005_1 no nlx data for dperson; R1068J  does not align; 
        sessionList = [sessi for sessi in sessionList for subi in subjectList if subi+'_e' in sessi and subi not in exc_subs_micro and 'FR436_events_sess1' not in sessi]#'FR436_events_sess1' no micro data
    
    if more_than_one_sess:
        sub_for_sess = [sessi.split('/')[-1].split('_events')[0] for sessi in sessionList]
        unique, counts = np.unique(sub_for_sess, return_counts=True)
        n_sess = dict(zip(unique, counts))
        subjectList = [subi for subi in subjectList if n_sess[subi]>1] 
        sessionList = [sessi for sessi in sessionList if n_sess[sessi.split('/')[-1].split('_events')[0]]>1]
        if micros:
            # FR436 has only one session with micro data
            subjectList = [subi for subi in subjectList if subi != 'FR436']
            sessionList = [sessi for sessi in sessionList if 'FR436' not in sessi]
        
    subjectList.sort()    
    sessionList.sort()

    return(subjectList,sessionList)