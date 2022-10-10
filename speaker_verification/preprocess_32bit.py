import numpy as np
import ctypes
import os
import json
import shutil
import random
import pickle


##############################################
# Constants.
CHECKPOINTS_DIR = 'checkpoints'
NUM_FRAME = 49
NUM_FBANK = 40
NUM_SPEC_PER_SPEAKER = 200 # 화자 당 평균 발화 수가 300, 현재 128*128=16384 /16000=1초 정도의 입력으로 받고 있으니.
HOP_LENGTH = 128

NUM_ENROLL_UTT = 5
NUM_TRUE_UTT = 10
NUM_FALSE_UTT = 10
##############################################


# 음성 하나를 spectrogram으로 변환 (c++)
def convert(file_path):
    # prepare c function
    libc = ctypes.CDLL('C:/Users/LeeJunghun/Desktop/lingometer/arduino/Arduino_TensorFlowLite/examples/micro_speech/feature_provider.so')
    make_spectrogram_c = libc.PopulateFeatureData_c
    make_spectrogram_c.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_int16), ctypes.POINTER(ctypes.c_int8))

    # load pcm
    with open(file_path, 'rb') as f:
        buf = f.read()
        pcm = np.frombuffer(buf, dtype = 'int16')
    pcm_copy = pcm.copy()  # original file is readonly so cannot apply np.ctypeslib.as_ctypes
    pcm_len = len(pcm)
    c_x= np.ctypeslib.as_ctypes(pcm_copy)  # convert to c type

    # make output array
    spec_t_dim = int((((pcm_len/16)-30)/20)+1)
    spec_f_dim = 40
    y = [0]*(spec_t_dim*spec_f_dim)  
    c_y = (ctypes.c_int8*len(y))(*y)  # convert to c type

    # call spectrogram function
    spec_len = make_spectrogram_c(pcm_len,c_x,c_y)
    if spec_len!=len(y): raise Exception("not corresponding spectrogram dimension")
    spec = np.array(c_y)
    spec = np.reshape(spec,(spec_t_dim,spec_f_dim))
    if spec_t_dim<10: raise Exception("too short to stabilize noise reduction error")
    spec = spec[10:,:]  # clip noise reduction error
    spec = np.expand_dims(spec, axis=-1) # [time][freq(40)][chnnel(1)]
    return spec


# 모든 화자의 음성을 spectrogram으로 변환
# 너무 많은 음성을 한번에 처리하면 c++쪽에서 에러가 난다. 메모리와 관련된 것으로 추정
def convert_all(pcm_dir, json_dir, cache_dir, already_converted_json_name=None):
    for json_filename in os.listdir(json_dir):
        print(json_filename)
        if already_converted_json_name and json_filename<=already_converted_json_name: continue
        pcm_dirname = json_filename[:-5]
        with open(json_dir+'/'+json_filename, "r", encoding="utf-8") as json_file: info = json.load(json_file)
        for utt_info in info['document'][0]["utterance"]:
            utt_id = utt_info['id']
            spk_id = utt_info['speaker_id']
            pcm_filepath = pcm_dir+'/'+pcm_dirname+'/'+utt_id+'.pcm'
            try: spec = convert(pcm_filepath)
            except: print('convert error: ',pcm_filepath); continue
            
            spk_dir = cache_dir+'/'+spk_id
            if not os.path.exists(spk_dir): os.makedirs(spk_dir)
            npy_filepath = spk_dir+'/'+utt_id+".npy"
            np.save(npy_filepath, spec)


# split dataset to train:validation:test=60:20:20
def dataset_split(cache_dir):
    # speaker dir list
    speakers = os.listdir(cache_dir)
    
    # split speakers
    random.shuffle(speakers)
    split_idx = len(speakers)//5
    speakers_train = speakers[split_idx*2:]
    speakers_validation = speakers[split_idx:split_idx*2]
    speakers_test = speakers[:split_idx]
    
    # make dir for save
    train_dir = cache_dir+'/train'
    test_dir = cache_dir+'/test'
    validation_dir = cache_dir+'/validation'
    os.makedirs(train_dir)
    os.makedirs(test_dir)
    os.makedirs(validation_dir)
    
    # move to saving dir
    for speaker in speakers_train: shutil.move(cache_dir+'/'+speaker, train_dir)
    for speaker in speakers_test: shutil.move(cache_dir+'/'+speaker, test_dir)
    for speaker in speakers_validation: shutil.move(cache_dir+'/'+speaker, validation_dir)

    print(f'number of train/val/test speakers: {len(speakers_train)} / {len(speakers_validation)} / {len(speakers_test)}')


def make_specs_of_speakers(input_dir):
    specs_of_speakers = []
    for speaker in os.listdir(input_dir):
        X_speaker = []
        
        for utt in os.listdir(input_dir+'/'+speaker):
            spec = np.load(input_dir+'/'+speaker+'/'+utt)
            if spec.shape[0]<NUM_FRAME : continue      # 최소길이보다 스펙트로그램이 짧다면 pass

            for i in range(0,spec.shape[0]-NUM_FRAME+1,NUM_FRAME): X_speaker.append(spec[i:i+NUM_FRAME,:])
            if len(X_speaker)>=NUM_SPEC_PER_SPEAKER: break  # 최소 스펙트로그램 개수를 채웠다면 break
        
        # 최소 스펙트로그램 개수를 넘겼다면 specs_of_speakers에 추가
        if len(X_speaker)<NUM_SPEC_PER_SPEAKER: continue
        specs_of_speakers.append(np.array(X_speaker[:NUM_SPEC_PER_SPEAKER]))

    specs_of_speakers = np.array(specs_of_speakers)
    return specs_of_speakers


def generate_train(input_dir, output_dir):

    specs_of_speakers = make_specs_of_speakers(input_dir)
    num_speaker = len(specs_of_speakers)
    print('train dataset generated. shape:', specs_of_speakers.shape)

    # save
    with open(output_dir+'/'+f"train_cpp_{num_speaker}_{NUM_SPEC_PER_SPEAKER}_{NUM_FRAME}_{NUM_FBANK}.pickle",'wb') as f:
        pickle.dump(specs_of_speakers,f)


# validation 또는 frame level test를 위한 데이터셋을 생성
def generate_val(input_dir, output_dir, is_test=False):
    
    specs_of_speakers = make_specs_of_speakers(input_dir)
    num_speaker = len(specs_of_speakers)
    
    # sample specs for validation
    batchs = []
    for speaker in range(num_speaker):
        batch = []

        enroll_true_utt = specs_of_speakers[speaker][np.random.choice(NUM_SPEC_PER_SPEAKER, NUM_ENROLL_UTT+NUM_TRUE_UTT, replace=False)]
        batch.extend(enroll_true_utt)
        
        others = np.random.choice(num_speaker-1, NUM_FALSE_UTT, replace=False)
        false_utt = []
        for i in range(NUM_FALSE_UTT):
            if others[i]==speaker: others[i]=num_speaker-1
            false_utt.append(specs_of_speakers[others[i]][np.random.choice(NUM_SPEC_PER_SPEAKER,1)[0]])
        batch.extend(false_utt)

        batch = np.array(batch)
        batchs.append(batch)
    batchs = np.array(batchs)

    if is_test:
      print('frame level test dataset generated. shape:',batchs.shape)
      with open(output_dir+'/'+f"test_cpp_{num_speaker}_{NUM_ENROLL_UTT+NUM_TRUE_UTT+NUM_FALSE_UTT}_{NUM_FRAME}_{NUM_FBANK}.pickle",'wb') as f:
        pickle.dump(batchs,f)
    else:
      print('validataion dataset generated. shape:',batchs.shape)
      with open(output_dir+'/'+f"val_cpp_{num_speaker}_{NUM_ENROLL_UTT+NUM_TRUE_UTT+NUM_FALSE_UTT}_{NUM_FRAME}_{NUM_FBANK}.pickle",'wb') as f:
        pickle.dump(batchs,f)


def generate_test_utt(input_dir, output_dir):
    
    # 모든 화자의 발화 로드, 최소개수가 안되는 화자는 제외
    utts_of_speakers = []
    for speaker in os.listdir(input_dir):
        X_speaker = []

        for utt in os.listdir(input_dir+'/'+speaker):
            spec = np.load(input_dir+'/'+speaker+'/'+utt)
            if spec.shape[0]<NUM_FRAME : continue

            X_speaker.append(spec)
            if len(X_speaker)>=NUM_SPEC_PER_SPEAKER: break

        if len(X_speaker)<NUM_SPEC_PER_SPEAKER: continue
        utts_of_speakers.append(X_speaker[:NUM_SPEC_PER_SPEAKER])
    num_speaker = len(utts_of_speakers)
    
    # sample utterances for test
    batchs = []
    for speaker in range(num_speaker):
        batch = []

        for utt_idx in np.random.choice(NUM_SPEC_PER_SPEAKER, NUM_ENROLL_UTT+NUM_TRUE_UTT, replace=False): batch.append(utts_of_speakers[speaker][utt_idx])
        
        others = np.random.choice(num_speaker-1, NUM_FALSE_UTT, replace=False)
        for i in range(NUM_FALSE_UTT):
            if others[i]==speaker: others[i]=num_speaker-1
            batch.append(utts_of_speakers[others[i]][np.random.choice(NUM_SPEC_PER_SPEAKER,1)[0]])

        batchs.append(batch)

    print('utterance level dataset generated. len:',len(batchs))
    with open(output_dir+'/'+f"testUtt_cpp_{num_speaker}_{NUM_ENROLL_UTT+NUM_TRUE_UTT+NUM_FALSE_UTT}.pickle",'wb') as f:
        pickle.dump(batchs,f)


#convert_all('C:/Users/LeeJunghun/Desktop/lingometer/data/NIKL_DIALOGUE_2020_PCM_v1.2_part4/pcm','C:/Users/LeeJunghun/Desktop/lingometer/data/NIKL_DIALOGUE_2020_PCM_v1.2_part4/json','C:/Users/LeeJunghun/Desktop/lingometer/data/npy_cpp','SDRW2000002074.json')
#dataset_split('C:/Users/LeeJunghun/Desktop/lingometer/data/npy_cpp')
generate_train('C:/Users/LeeJunghun/Desktop/lingometer/data/npy_cpp/train','C:/Users/LeeJunghun/Desktop/lingometer/data/dataset_cpp')