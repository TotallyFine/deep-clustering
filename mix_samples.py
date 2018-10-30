'''
Script to mix two testing samples
'''
import librosa
import numpy as np

print 'import finish'
# provide the wav name and mix
# speech1 = '/media/nca/data/raw_data/speech_train_r/FCMM0/TRAIN_DR2_FCMM0_SI1957.WAV'
# speech2 = '/media/nca/data/raw_data/speech_train_r/FKLC0/TRAIN_DR4_FKLC0_SX355.WAV'
speech1 = '/home/j/TSP/8k-G712/FA/FA01_01.wav'
speech2 = '/home/j/TSP/8k-G712/MA/MA01_01.wav'

data1, _ = librosa.load(speech1, sr=8000)
print 'load data1 finish'
print 'data1, shape: ', data1.shape, ' type: ', data1.dtype
data2, _ = librosa.load(speech2, sr=8000)
print 'load data2 finish'
mix = data1[:len(data2)] + data2[:len(data1)]
print 'mix shape: ', mix.shape
librosa.output.write_wav('mix.wav', mix, 8000)
print 'output finish'
