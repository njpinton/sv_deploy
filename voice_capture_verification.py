import librosa
import sounddevice as sd
import os
from configuration import get_config
import numpy as np
import tensorflow as tf
from utils_deployment import *
import sys
import pandas as pd
from datetime import datetime

config = get_config()

def main():
    print("Who are you?")
    name = input()
    print("Hi %s, Are you ready? (Press enter to continue. Records immediately)" %name)
    print("The quick brown fox jumps over the lazy dog.")

    ready = input()
    duration = config.duration
    print("Recording . . . (Plays immediately after recording)")
    recording = sd.rec(int(duration * config.sr), samplerate=config.sr, channels=1)
    sd.wait()
    recording = np.reshape(recording,duration*config.sr)
    print("Done Recording.")

    print("Playing recording. . .")
    sd.play(recording,config.sr)
    sd.wait()
    print("Done.")

    write_wav(name,recording,config.sr,enroll=False)
    write_npy(name,recording)
    # enroll_path = os.path.join("enroll_voice",name)
    # create_folder(enroll_path)
    print("shape of recording array: ",recording.shape)
    utterances_spec = preprocess(recording)
    try:
        verif,score = verify(utterances_spec,name)
    except Exception as e:
        print("Please repeat verification process.")
        exit()

    
    docname = 'verification.csv'
    if not os.path.isfile(docname):
        pd.DataFrame(columns=['claimant','accept/reject','score','date','time']).to_csv(docname,index=False)
        
    df = pd.read_csv(docname,index_col=0)
    now = datetime.now()

    df.append({'claimant':name,'accept/reject':verif,'score':score,'date':now.date(),'time':now.time(),
                },ignore_index=True).to_csv(docname,index=False)

if __name__ == '__main__':
    main()
