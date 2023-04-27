#### update

- 2023.3.24 编写数据的生成
- 2023.3.25 编写数据集加载
- 2023.3.26 编写网络部分并训练起来
- 2023.3.26 编写学习率策略与学习率热身策略
- 2023.3.27 编写推理代码, 并验证了没有错

- 2023.4.01 将xujiajian的数据预处理与加载的代码看了一遍
- 2023.4.03 将数据预处理与加载重构了一遍
- 2023.4.05 将网络结构的代码看完并理解
- 2023.4.07 完成了train_landmark2face的编写, 但是从效果上看并不好, loss下降到一定程度之后就不再下降了, 查看了一下原因是判别器太小了
- 2023.4.09 使用了xujiajian的预训练模型, 并且对比了一下判别器的大小, 从而对齐了判别器, 就是说到此为止所有都是对齐的

- 2023.4.10 把完全对齐且用了预训练模型的代码, 使用同样的数据集在服务器上进行训练, 训练了一天的效果也不理想
- 2023.4.11 参考LIA来编写image2image阶段的代码

- 2023.4.13 直接参考训练的代码编写image2image的inference代码



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
```















