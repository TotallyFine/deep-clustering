# Deep Clustering
在[deep cluster](https://arxiv.org/abs/1508.04306)的基础上做的综合课程设计，代码基于[3]。

## 环境
Python 2 以及如下包:
  * tensorflow
  * numpy
  * scikit-learn
  * matplotlib
  * librosa
  
## 文件组织
一共有如下的文件：
1. GlobalConstant.py: 全局变量，一些设置都在这里。同时还有一些设置在对应运行的文件内部。
2. datagenerator.py: 将wav文件转换为pkl文件，还进行了stft
3. datagenerator2.py: 读取pkl文件，并分成batch送给神经网络来学习
4. model.py: 定义神经网络的结构
5. train\_net.py: 训练神经网络
6. mix\_samples.py: 将两个声音文件进行混合来测试神经网络
7. AudioSampleReader.py: 在测试阶段将wav文件转换成由帧组成的一块一块数据，再送给训练好的神经网络来测试
8. visualization\_of\_samples.py: 给训练好的神经网络输入测试数据，然后得到嵌入向量，再对嵌入向量进行PCA，然后可视化PCA。
9. audio\_test.py: 利用训练好的神经网络进行分离两个混合的人声

## 短时傅立叶变换简单介绍
声音转换成数据结构就是一个list，假如length为(300,)，其中每个元素就是采样得到的信号值。短时傅立叶变换利用一个滑动窗口，在这个list上进行移动，把list分成很多个片段，然后对每个片段的信号做傅立叶变换。所以短时傅立叶变换之后的数据大小就可能为(100, 3)=>(FRAME\_PER\_SAMPLES, NEFF)。

## 数据集
使用的数据集是TSP数据集，TSP数据集由12个男人和12个女人的声音组成，语音内容是Harvard sentence，这24个人大部分都是native speaker（加拿大人）。Harvard sentence中，一共72个list，每个list10句话。这些人每人读6个list，所以是男人12个将6个list读完，这样就是72*10短语音。女人12个没人读6个list，这样也是72*10个语音。

原始的录音文件是48k Hz，做实验8k就可以。

TSP数据集中的wav文件以说话者进行组织，如MA/代表Male A读的所有句子构成一个文件夹，FA代表Female A。还有CA代表Child A。FE26\_09.wav代表Female E list-26 sentence-09。一般来说建议使用FK，FL，MK，ML作为测试集，剩下的（不包括CA CB）作为训练集。

因为种种原因数据的组织者丢失了ME，所以一共1378个wav文件。

在使用这份代码的时候，将TSP每个文件夹中的txt文件先移除。

## 训练过程及神经网络
### 训练过程
1. 首先是组织好数据，将同一个说话者的wav文件放在同一个目录下，同时分好训练集和测试集，并且按照```some_dir/train_data/spearker_id/speech_files.wav```的格式来组织。
2. 然后更改datagenerator.py中__main__部分的data_dir，改成上面的路径```some_dir/train_data/```。分别让data_dir等于训练集的路径和测试集的路径来运行，并且注意更改生成的pkl文件的名称，在第144行，训练集和测试集进行区分。datagenerator.py会在根目录产生```./val.pkl```，这个文件存储的数据结构是一个列表，列表中包含的是多个样本，每个样本使用字典的数据结构来表达：
```python
sample_dict = {'Sample': sample_mix,
                'VAD': VAD,
                'Target': Y}
```
其中
```python
sample_mix.shape = (FRAMES_PER_SAMPLE, NEFF) # 见GlobalConstont.py
sample_mix.dtype = np.float32

VAD.shape = (FRAMES_PER_SAMPLE, NEFF)
VAD.dtype = np.bool # 这个bool是为了节省存储空间，在train_net.py中又恢复到了np.int64

Y.shape = (FRAMES_PER_SAMPLE, NEFF, 2)
Y.dtype = np.bool
```
datagenerator2.py的作用就只是从pkl文件读取并分成batch送给神经网络。
3. 将train_net.py中的全局变量```pkl_list val_list```进行更改，这两个list的内容改成之前生成过的测试集和训练集pkl文件的的绝对路径。
4. 在命令行中运行net_train.py: ```python net_train.py```。代码会在根目录下生成```./train/model.ckpt```文件，这个文件就是保存的模型。
5. 更改audio_test.py中第88行为之前保存的模型的绝对路径，然后运行```python audio_test.py```。看第233与234行，代码会生成这两个声音文件，代表分离的两个声音。

### 神经网络
这里主要说一下神经网络的输入和输出，在神经网络之前的数据变换上面已经大概说过了。代码里的神经网络开始是4层的BLSTM，开始输入的数据```in_data(看train_net.py第52行)```的shape为```(batch_size=128, FRAMES_PER_SAMPLE=100, NEFF=129)```也就是```(batch_size, time_step_size, input_vec_size)```。

一共有4个双向的LSTM(BLSTM参看[2],LSTM输出参见[1])，每一层的LSTM的输出outputs都为```(outputs_fw, outputs_bw)```, outputs_fw与outputs_bw的shape相同为```(batch_size=128, FRAMES_PER_SAMPLE=100, n_hidden=300)```。第一层的输出outputs都做了concat所以之后的BLSTM的输入就为```(batch_size=128, FRAMES_PER_SAMPLE=100, hidden=600)```。

4层BLSTM之后是全连接神经网络，它的输出为```(batch_size, EMBBEDDING_D * NEFF)```，然后再做tanh激活，最后reshape为```(batch_size, NEFF=129, EMBBEDDING_D=40)```就是网络的最终输出。

## 参考
[1] [BasicLSTMCell中num_units参数解释](https://blog.csdn.net/notHeadache/article/details/81164264)

[2] [tensorflow学习笔记(三十九) : 双向rnn (BiRNN)](https://blog.csdn.net/u012436149/article/details/71080601)

[3] [deep-clustering](www.github.com/zhr1201/deep-clustering)
