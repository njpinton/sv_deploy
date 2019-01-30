import librosa
import sounddevice as sd
import os
from configuration import get_config
import numpy as np
import tensorflow as tf
import sys
from utils_deployment import *
import tables
import pandas as pd
from datetime import datetime


config = get_config()

def main():
    
    print("Who are you?")
    name = input()


    user_exist_path=os.path.join("MODEL/{}.npy".format(name))
    if os.path.isfile(user_exist_path):
        print("User with the same name already enrolled. Do you want to continue?")
        key = input()

        if str2bool(key) == False:
            sys.exit("Thank you for patronizing us.")
        
    print("Hi %s, Are you ready? (Press enter to continue. Records immediately)" %name)
    print("The quick brown fox jumps over the lazy dog.")
    ready = input()

    print("Recording . . . (Plays immediately after recording)")
    recording = sd.rec(int(config.duration * config.sr), samplerate=config.sr, channels=1)
    sd.wait()
    recording = np.reshape(recording,config.duration*config.sr)
    print("Done Recording.")
    
    
    write_wav(name,recording,config.sr)
    write_npy(name,recording)
    
    print("Playing recording. . .")
    sd.play(recording,config.sr)
    sd.wait()
    print("Done playing.")

    utterances_spec = preprocess(recording)
    try:
        enrolled = enroll(utterances_spec,name)
    except Exception as e:
        print("Please repeat enrollment process.")
    enrolled_voice_path = 'enrolled_voice_models.csv'
#   save_model_to_df(enrolled_voice_path,name,enrolled)
    
    docname = 'enroll.csv'
    if not os.path.isfile(docname):
        print("Savefile does not exist. Creating savefile, ", docname)
        pd.DataFrame(columns=['name','date','time']).to_csv(docname,index=False)
        
    df = pd.read_csv(docname)
    now = datetime.now()
    
    df.append({'name':name,'date':now.strftime("%Y-%m-%d"),'time':now.strftime("%H:%M:%S")},
              ignore_index=True).to_csv(docname,index=False)

if __name__ == '__main__': 
    main()