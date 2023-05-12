#### update

- 5.11 将expression嵌入到LIA中去, 一般是融入到下采样的末端
- 5.12 设计表情的监督loss
    去除3DMM的投影顶点的监督
    可以将监督信息中的landmark与五官(除了鼻子之外)的权重调低, 面部的外轮廓不用, 因为不使用target中的expression而是reference的表情
    与判别器相关loss无需加上expression image的, expression image只要提供表情向量的监督就行了(类似simswap的id监督)
    加上了id loss, source和recon_drive对应的id loss
    
    通过逻辑分析之后, 这个FM loss待定
    后期还要加上是否要加上高层特征的FM, 但是要加入这个的话, 就要将experssion_ref也加入到D中提取特征, 但是不用做d_loss或者g_loss, 使用得到的向量再去做FM, 
    但是这里有个疑问, 就是表情并不是高层的语义反而是低层的, 然后加上了这个loss就会导致recon的结果不像本人的, 会偏向表情提供者的。


#### generate datasets
```
python video_analysis_and_data_generation.py
```

#### Run
train audio2landmark
```
python models/train_audio2landmark.py
```

test audio2landmark
```
python models/eval_audio2landmark.py
```
![Image text](examples/audio2landmark_test1.png)


train audio2landmark
```
python models/train_landmark2face_LIA.py
```

test audio2landmark
```
python models/train_landmark2face_LIA.py --phrase test
```
![Image text](examples/landmark2face.gif)


#### 技术总结
```
当前的audio2landmark使用了两个模块，一个Audio2landmark_content和一个Audio2landmark_speaker_aware。
一个针对是content一个是针对说话的人的emb的。
但是content的分支应该是一厢情愿, 并没有什么显示的规则来约束, 要说有约束那就是因为Audio2landmark_speaker_aware中确实有emb的输入
emb是通过一个AutoVC的预训练模型得到, 这样等于是在audio中增强了属于speaker的信息, 虽然是这样也不能说content中就没有id emb的信息了

就是content中有很大的可能还耦合着id emb中的信息, 没有解耦开来

content: 对标的应该是表情和口型
id emb: 对标的应该是head pose和人脸面部的轮廓五官的位置

但是五官主要是通过第一帧的图片来控制, head pose也是无法与音频进行强相关的(单训练一个人一段视频应该是ok的, 但是在别的场景进行推理泛化性就会极差)
```















