import tensorflow as tf
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
import random
from configuration import get_config
from sklearn.metrics.pairwise import cosine_similarity
import soundfile as sf
import itertools
from datetime import datetime
import argparse


config = get_config()


def save_spectrogram_tisv():
    """ Full preprocess of text independent utterance. The log-mel-spectrogram is saved as numpy file.
        Each partial utterance is splitted by voice detection using DB
        and the first and the last 180 frames from each partial utterance are saved. 
        Need : utterance data set (VTCK)
    """
    print("start text independent utterance feature extraction")
    os.makedirs(config.train_path, exist_ok=True)   # make folder to save train file
    os.makedirs(config.test_path, exist_ok=True)    # make folder to save test file

    enroll_folder = 'verif_tisv5'
    create_folder(enroll_folder)

    utter_min_len = (config.tisv_frame * config.hop + config.window) * config.sr    # lower bound of utterance length
    total_speaker_num = len(os.listdir(audio_path))
    train_speaker_num= (total_speaker_num//10)*9            # split total data 90% train and 10% test
    print("total speaker number : %d"%total_speaker_num)
    # print("train : %d, test : %d"%(train_speaker_num, total_speaker_num-train_speaker_num))

    labels = []
    for i, folder in enumerate(os.listdir(audio_path)):
        speaker_path = os.path.join(audio_path, folder)     # path of each speaker
        print("%dth speaker processing..."%i)
        print(folder)
        utterances_spec = []
        k=0
        for utter_name in os.listdir(speaker_path):
            utter_path = os.path.join(speaker_path, utter_name)         # path of each utterance
            utter, sr = librosa.core.load(utter_path, config.sr)        # load utterance audio
            intervals = librosa.effects.split(utter, top_db=20)         # voice activity detection
            for interval in intervals:
                if (interval[1]-interval[0]) > utter_min_len:           # If partial utterance is sufficient long,
                    utter_part = utter[interval[0]:interval[1]]         # save first and last 180 frames of spectrogram.
                    S = librosa.core.stft(y=utter_part, n_fft=config.nfft,
                                          win_length=int(config.window * sr), hop_length=int(config.hop * sr))
                    S = np.abs(S) ** 2
                    mel_basis = librosa.filters.mel(sr=config.sr, n_fft=config.nfft, n_mels=40)
                    S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances

                    utterances_spec.append(S[:, :config.tisv_frame])    # first 180 frames of partial utterance
                    utterances_spec.append(S[:, -config.tisv_frame:])   # last 180 frames of partial utterance

        labels.append('{} {}'.format(i,folder))

        utterances_spec = np.array(utterances_spec)
        print(utterances_spec.shape)

        np.save(os.path.join(enroll_folder, "speaker%d.npy"%(i)), utterances_spec)

    fname = 'labels_verif.txt'
    with open(fname,'w') as f:
        for item in labels:
            f.write(item+'\n')
            

def preprocess(utter,top_db=config.top_db):
    """ returns utterance numpy array of users recording as an input"""
    utter_min_len = (config.tisv_frame * config.hop + config.window) * config.sr  # lower bound of utterance length
    sr = config.sr

    utterances_spec = []
    intervals = librosa.effects.split(utter, top_db=top_db)
    
    utter_part = []
    for interval in intervals:
        utter_part = np.append(utter_part,utter[interval[0]:interval[1]])
    
    S = librosa.core.stft(y=utter_part, n_fft=config.nfft,
        win_length=int(config.window * sr), hop_length=int(config.hop * sr))
    S = np.abs(S) ** 2
    mel_basis = librosa.filters.mel(sr=config.sr, n_fft=config.nfft, n_mels=40)
    S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
    
    utterances_spec.append(S[:, :config.tisv_frame])    # first 180 frames of partial utterance
    utterances_spec.append(S[:, -config.tisv_frame:])   # last 180 frames of partial utterance

    utterances_spec = np.array(utterances_spec)
    print(utterances_spec.shape)
    return utterances_spec

def create_folder(path):
    """ creates a directory of the path if the directory does not exist """
    if not os.path.exists(path):
        os.makedirs(path)

def enroll(utters,speaker,path=config.model_path,speaker_model_path = 'MODEL'):
    """ passes the utterance array of speaker to the network and extract it to a (1,64) feature array """
    tf.reset_default_graph()

    # draw graph
    enroll = tf.placeholder(shape=[None, config.N*config.M, 40], dtype=tf.float32) # enrollment batch (time x batch x n_mel)
    batch = enroll
    # embedding lstm (3-layer default)
    with tf.variable_scope("lstm"):
        lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=config.hidden, num_proj=config.proj) for i in range(config.num_layer)]
        lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)    # make lstm op and variables
        outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=batch, dtype=tf.float32, time_major=True)   # for TI-VS must use dynamic rnn
        embedded = outputs[-1]                            # the last ouput is the embedded d-vector
        embedded = normalize(embedded)

    # enrollment embedded vectors (speaker model)
    enroll_embed = normalize(tf.reduce_mean(tf.reshape(embedded[:config.N*config.M, :], shape= [config.N, config.M, -1]), axis=1))  
    saver = tf.train.Saver(var_list=tf.global_variables())

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # load model
        # print("model path :", path)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=os.path.join(path, "Check_Point"))
        ckpt_list = ckpt.all_model_checkpoint_paths
        loaded = 0
        for model in ckpt_list:
            if config.model_num == int(model[-1]):    # find ckpt file which matches configuration model number
                # print("ckpt file is loaded !", model)
                loaded = 1
                saver.restore(sess, model)  # restore variables from selected ckpt file
                break

        if loaded == 0:
            raise AssertionError("ckpt file does not exist! Check config.model_num or config.model_path.")

        try:
            utter_index = np.random.randint(0, utters.shape[0], config.M)   # select M utterances per speaker
            utter_batch = utters[utter_index]     # each speakers utterance [M, n_mels, frames] is appended
            utter_batch = np.transpose(utter_batch, axes=(2,0,1))     # transpose [frames, batch, n_mels]
            # print(utter_batch.shape)
            enroll = sess.run(enroll_embed, feed_dict={enroll:utter_batch})
#           print('enroll shape: ',enroll.shape)
            print('Enrolled: ',speaker)

            create_folder(speaker_model_path)
            np.save(os.path.join(speaker_model_path,'%s.npy'%speaker),enroll)
        except Exception as e:
            print(e)
    return enroll


def verify(utters,speaker,thres=config.threshold,path=config.model_path,speaker_model_folder = 'MODEL'):
    tf.reset_default_graph()
    # draw graph
    verif = tf.placeholder(shape=[None, config.M, 40], dtype=tf.float32) # verif batch (time x batch x n_mel)
    batch = verif
    # embedding lstm (3-layer default)
    with tf.variable_scope("lstm"):
        lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=config.hidden, num_proj=config.proj) for i in range(config.num_layer)]
        lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)    # make lstm op and variables
        outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=batch, dtype=tf.float32, time_major=True)   # for TI-VS must use dynamic rnn
        embedded = outputs[-1]                            # the last ouput is the embedded d-vector
        embedded = normalize(embedded)
    
    verif_embed = normalize(tf.reduce_mean(tf.reshape(embedded[:config.N*config.M, :], shape= [config.N, config.M, -1]), axis=1))   

    saver = tf.train.Saver(var_list=tf.global_variables())

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # load model
        # print("model path :", path)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=os.path.join(path, "Check_Point"))
        ckpt_list = ckpt.all_model_checkpoint_paths
        loaded = 0
        for model in ckpt_list:
            if config.model_num == int(model[-1]):    # find ckpt file which matches configuration model number
                print("ckpt file is loaded !", model)

                loaded = 1
                saver.restore(sess, model)  # restore variables from selected ckpt file
                break

        if loaded == 0:
            raise AssertionError("ckpt file does not exist! Check config.model_num or config.model_path.")

        try:
            utter_index = np.random.randint(0, utters.shape[0], config.M)   # select M utterances per speaker
            utter_batch = utters[utter_index]     # each speakers utterance [M, n_mels, frames] is appended
            utter_batch = np.transpose(utter_batch, axes=(2,0,1))     # transpose [frames, batch, n_mels]
            print(utter_batch.shape)
            verif = sess.run(verif_embed, feed_dict={verif:utter_batch})
            # print('verif shape:',verif.shape)

        except Exception as e:
            print(e)


        for enrolled_speaker in os.listdir(speaker_model_folder):
            # print(enrolled_speaker[:-4])
            if enrolled_speaker[:-4] == speaker:
                speaker_model_path = os.path.join(speaker_model_folder,enrolled_speaker)
                # print(speaker_model_path)
    try:
        score = get_score(verif,speaker_model_path)
        print('confidence: ', str(score*100))

        if score > thres:
            print('Speaker verified. You are %s.'%speaker)
            verification = 'accepted'
        else:
            print('Speaker rejected. Please try again.')
            verification = 'rejected'
    except Exception as e:
        print(e)
    return verification, score

def normalize(x):
    """ normalize the last dimension vector of the input matrix
    :return: normalized input
    """
    return x/tf.sqrt(tf.reduce_sum(x**2, axis=-1, keep_dims=True)+1e-6)

# def get_score(speaker_feat,speaker,path='MODEL/model.npy'):
#     model = np.load(path)
#     # print('model: ', model[[speaker]])
#     score = cosine_similarity(speaker_feat,model[[speaker]])
#     # print('model shape: ', model.shape)
#     return score

def get_score(speaker_feat,speaker_model_path):
    """ compare the extracted feature of the speaker to the stored feature array of the speaker
    returns: score/similarity of the two features 
    """
    model = np.load(speaker_model_path)
    # print('model: ', model[[speaker]])
    score = cosine_similarity(speaker_feat,model)
    # print('model shape: ', model.shape)
    return score

def write_npy(path,rec,enroll=True):
    if enroll == True:
        enroll_npypath = os.path.join("enroll_npy/{}".format(path))
        enroll_path=os.path.join("enroll_voice/{}".format(path))
        create_folder(enroll_npypath)
        
        i = 0
        for file in os.listdir(enroll_path):
            i+=1
        np.save(os.path.join(enroll_npypath,"{}{}.npy".format(path,i)),rec)

    if enroll == False:
        verify_npypath = os.path.join("verify_npy/{}".format(path))
        verify_path=os.path.join("verify_voice/{}".format(path))
        create_folder(verify_npypath)
        
        i = 0
        for file in os.listdir(verify_path):
            i+=1
        np.save(os.path.join(verify_npypath,"{}{}.npy".format(path,i)),rec)
    
def write_wav(path,rec,sr,enroll=True):
    if enroll == True:
        enroll_path = os.path.join("enroll_voice/{}".format(path))
        create_folder(enroll_path)
        i = 0
        for file in os.listdir(enroll_path):
            i += 1
        sf.write("{}/{}{}.wav".format(enroll_path,path,i),rec,sr)
        
    if enroll == False:
        verify_path = os.path.join("verify_voice/{}".format(path))
        create_folder(verify_path)
        i = 0
        for file in os.listdir(verify_path):
            i += 1
        sf.write("{}/{}{}.wav".format(verify_path,path,i),rec,sr)

def save_model_to_df(enrolled_voice_path,name,enrolled):
    '''
    not yet complete something is wrong with the frame
    '''
    if not os.path.isfile(enrolled_voice_path):
        df = pd.DataFrame(columns=['name','model'])
        df.to_csv(enrolled_voice_path,index=False)
    df = pd.read_csv(enrolled_voice_path)
    print(type(enrolled))
    df.append({'name':name,'model':[np.array(enrolled)]},ignore_index=True).to_csv(enrolled_voice_path,index=False)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    now = datetime.now()

    plt.savefig("plots/conf_{}.png".format(now.strftime("%H%M%S")))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == "__main__":
    random_batch()
    w= tf.constant([1], dtype= tf.float32)
    b= tf.constant([0], dtype= tf.float32)
    embedded = tf.constant([[0,1,0],[0,0,1], [0,1,0], [0,1,0], [1,0,0], [1,0,0]], dtype= tf.float32)
    sim_matrix = similarity(embedded,w,b,3,2,3)
    loss1 = loss_cal(sim_matrix, type="softmax",N=3,M=2)
    loss2 = loss_cal(sim_matrix, type="contrast",N=3,M=2)
    with tf.Session() as sess:
        print(sess.run(sim_matrix))
        print(sess.run(loss1))
        print(sess.run(loss2))