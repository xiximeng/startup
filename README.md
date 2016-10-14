# 实验室技能表 v0.10

### Table of Contents

- [Deep Learning](#deep-learning)

  - [Courses & Books](#courses--books)

  - [Paper Reading](#paper-reading)

- [Computer Science](#computer-science)

  - [Python](#python)

- [Deep Reinforcement Learning](#deep-reinforcement-learning)

## Deep Learning

### Courses & Books

- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)

  Stanford 的课程，知乎上有人用中文翻译完了官方的
  [notes](https://zhuanlan.zhihu.com/p/21930884?refer=intelligentunit)

  同时，课程作业是用 Python 来实现的，可以帮助熟悉 Python

  使用服务器的 [iPython Notebook](http://jupyter-notebook.readthedocs.io/en/latest/)
  或者在自己的电脑上安装一个都是比较方便的

  关于学习 Python，可以参考后面的章节 [Python](#python)

- [Neural Networks for Machine Learning](https://www.coursera.org/learn/neural-networks)

  Geoffrey Hinton 在 Coursera 上的课程

- [Deep Learning](http://www.deeplearningbook.org/)
  by Ian Goodfellow, Yoshua Bengio and Aaron Courville

  一本关于 Deep Learning 的书，写得非常全面细致，暂时只推荐这一本

### Paper Reading

#### ImageNet Classification

做 cv 怎么都绕不开的一方向就是 Classification，可以说是很多研究方向的基石

- ImageNet classification with deep convolutional neural networks
  [[pdf]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

  AlexNet，关于此论文，有很多人写过论文笔记，可以自行搜索，比如
  [1](http://www.gageet.com/2014/09140.php) & [2](http://zhangliliang.com/2014/07/01/paper-note-alexnet-nips2012/)

- Very deep convolutional networks for large-scale image recognition
  [[pdf]](https://arxiv.org/pdf/1409.1556.pdf)

  VGG-Net，上篇和此篇都是在 cv 里面引用量很高而且绕不开的论文

  很多文章还会使用或者部分使用这二者里面的网络结构

  但由于业界更迭很快，二者也可以算作是比较旧的文章了

- Going deeper with convolutions
  [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)

  GoogleNet，引入了 Inception module，有很好的 performance

  作为打败了 VGG 的同年模型，好像用的人没有 VGG 多，估计是太大了而且效果也没有好太多

- Deep residual learning for image recognition
  [[pdf]](https://arxiv.org/pdf/1512.03385.pdf)

  不好意思，我还没看，但是用的人已经很多了

#### Object Detection

有效的特征对于 Detection 是极为重要，所以说 Classification 是绕不开的

Detection 是很多方向的基石

- Rich feature hierarchies for accurate object detection and semantic segmentation
  [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf)

  Region-based CNN，打开了 CNN 用于 Detection 的大门
  也有很多人写过论文笔记，比如
  [1](http://zhangliliang.com/2014/07/23/paper-note-rcnn/) &
  [2](https://zhuanlan.zhihu.com/p/22287237?refer=startdl)

- Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition
  [[pdf]](https://arxiv.org/pdf/1406.4729v2.pdf)

  突破了传统 CNN 输入大小固定的限制

- Fast R-CNN (2015)
  [[pdf]](https://arxiv.org/pdf/1504.08083.pdf)

  除此之外，还有
  [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf),
  [ION](https://arxiv.org/pdf/1512.04143v1.pdf),
  [YOLO](https://arxiv.org/pdf/1506.02640v5.pdf),
  [R-FCN](https://arxiv.org/pdf/1605.06409v2.pdf)

#### Understanding CNN

有助于理解 CNN 里面发生了什么事情

- Visualizing and Understanding Convolutional Networks
  [[pdf]](https://www.cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf)

  用反卷积可视化了 feature maps，有助于理解 CNN

- Understanding image representations by measuring their equivariance and equivalence
  [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Lenc_Understanding_Image_Representations_2015_CVPR_paper.pdf)

#### More Papers

在 cv 里面还有很多的方向，比如 Object Tracking, Image Caption 等

- [Awesome - Most Cited Deep Learning Papers](https://github.com/terryum/awesome-deep-learning-papers)

- [Awesome Deep Vision](https://github.com/kjw0612/awesome-deep-vision)

## Computer Science

### Python

#### Learning Python

- 如果没有编程基础，推荐在线学习 [Python | Codecademy](https://www.codecademy.com/learn/python)

- 如果有不错的编程基础，则推荐随便找个简单的教程看一下基本的语法，比如[廖雪峰的 Python 教程](http://www.liaoxuefeng.com/wiki/001374738125095c955c1e6d8bb493182103fac9270762a000)

- 推荐书籍
  [《Python核心编程2》](https://www.gitbook.com/book/wizardforcel/core-python-2e/details)
  或者 《Python高级编程》

#### Deep Learning Framework

基于 Python 的或者提供了 Python 接口的 DL 框架主要有 TensorFlow, MXNet, Caffe(自带PyCaffe)等

这些框架都有官方文档或者他人写的教学文档，学习上手都非常方便

## Deep Reinforcement Learning

### Reinforcement Learning

- [Reinforcement Learning: An Introduction](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html)
  by Richard S. Sutton and Andrew G. Barto

  一本非常好的 RL 教材，可能是最好的

### DRL Papers

- Playing Atari with Deep Reinforcement Learning
  [[pdf]](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

  NIPS 2013版的 DQN，开创性的一篇文章

- Human-level control through deep reinforcement learning
  [[pdf]](http://www.nature.com/nature/journal/v518/n7540/pdf/nature14236.pdf)

  Nature 2015版的 DQN，在之前版本的基础上，用固定的参数计算目标 Q 值

#### More Papers

- [Deep Reinforcement Learning Papers](https://github.com/muupan/deep-reinforcement-learning-papers)

- [Deep Reinforcement Learning Papers](https://github.com/junhyukoh/deep-reinforcement-learning-papers)
