"""
Generate speaker embeddings and metadata for training

这个就是从语音中提取出说话人的信息

所以这个可以改成在inference的时候进行推理得到id_emb的
"""
import os
import pickle
from model_bl import D_VECTOR
from collections import OrderedDict
import numpy as np
import torch

C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
c_checkpoint = torch.load('checkpoints/3000000-BL.ckpt')
new_state_dict = OrderedDict()
for key, val in c_checkpoint['model_b'].items():
    new_key = key[7:]
    new_state_dict[new_key] = val
C.load_state_dict(new_state_dict)
num_uttrs = 10
# len_crop = 128 // 2

len_crop = 128

# Directory containing mel-spectrograms
rootDir = './spmel'
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)


speakers = []
for speaker in sorted(subdirList):
    print('Processing speaker: %s' % speaker)
    utterances = []
    utterances.append(speaker)
    _, _, fileList = next(os.walk(os.path.join(dirName, speaker)))
    
    # make speaker embedding
    assert len(fileList) >= num_uttrs  # 这个人的音频不能少于10个, 少于10个就不能识别出来说话的人是谁
    idx_uttrs = np.random.choice(len(fileList), size=num_uttrs, replace=False)  # 随机抽出10个mel的文件
    candidates = np.delete(np.arange(len(fileList)), idx_uttrs)  # 最后得到一些候选的id
    embs = []
    for i in range(num_uttrs):
        tmp = np.load(os.path.join(dirName, speaker, fileList[idx_uttrs[i]]))

        # choose another utterance if the current one is too short
        while tmp.shape[0] < len_crop:
            idx_alt = np.random.choice(candidates)
            tmp = np.load(os.path.join(dirName, speaker, fileList[idx_alt]))
            candidates = np.delete(candidates, np.argwhere(candidates == idx_alt))  # 被使用过的也同样删除
        if tmp.shape[0] == len_crop:
            left = 0
        else:
            left = np.random.randint(0, tmp.shape[0]-len_crop)  # 这里是随机一个位置
        melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).cuda()
        emb = C(melsp)
        embs.append(emb.detach().squeeze().cpu().numpy())

    # 这里是从每段话中截取了一段, 并求了个音频的平均值来得到了属于这个人的emb
    utterances.append(np.mean(embs, axis=0))
    
    # create file list
    for fileName in sorted(fileList):
        utterances.append(os.path.join(speaker, fileName))
    speakers.append(utterances)
    
with open(os.path.join(rootDir, 'train.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)

