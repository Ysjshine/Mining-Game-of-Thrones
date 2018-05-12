# Mining-Game-of-Thrones
Mining the human's relationship of Game of Throne

> 最近在学习自然语言处理，看完了《Python 自然语言处理》这本书，想做一点实践性的练习。之前我在知乎上看见有人用机器学习来做过红楼梦人物关系梳理，于是我便想尝试着做了下《权力的游戏》中人物关系的梳理

## 背景知识

* 词向量

    在自然语言处理（natural language processing，NLP）领域，首要任务便是将单词转换为由数字组成的向量，这样计算机才可以进行更进一步处理。下面我们简要介绍一下词向量的表示方法

1. One-Hot representation
    
    这是最简单的一种词向量的表示方式。其基本思想便是用一个很长很长的向量来表示一个单词，向量的长度便是字典的长度，该向量由0和1组成，有且仅有一个1，其他全为0，1的位置便是该词在词典中的顺序。

    例如，在某个词典中，'like'的词向量可表示为[0,0,0,1,0,0,...]，'hate'的词向量可表示为[0,1,0,0,0,0,...]。可以根据以上向量可知，like在词典中处于第四个位置，hate在词典中处于第二个位置。

    这种向量表示法的优点是直观，易于理解。缺点也是很明显的。首先，会出现维数灾难，词典的长度通常是上万的，一篇简单的文档，例如新闻，也都是上千上万的单词，如果用这种方式来表达一篇文档很容易都达到极高的维度，同时也浪费了很多存储空间。再者就是这种表示方法假设了任意两个词之间都是孤立的存在，无法表达语义相近的两个词之间的关系，比如hate与dislike。

2. Distributed Representation

    这种表示方法最早是由Hinton提出来的，它克服了one-hot表示方法的缺点。其基本思想是，通过对某种语言中的词语进行训练，获得一个普通向量（例如[0.0268,-0.234,0.5263,...]这种形式，分布较为均匀），其维数通常为50-100（相比于one-hot可是大大减少了）。

    利用这种表示方法，就可以计算不同词之间的相似度，可利用常用的距离计算公式或者相似度计算公式来进行计算。

    训练生成词向量通常采用的是神经网络算法，例如RNN（循环神经网络）。

    在本项目中我们将采用gensim库中的word2vec来训练并获得所需要的词向量。

* 相似度计算

    在得到每个词的向量过后，计算两个词之间的相似度就很方便了。常见的计算相似度的方法有Euclidean距离，向量余弦相似度，Jaccard方法。这个比较简单，不在此赘述了

* 名字识别

    对《权力的游戏》小说中人物的识别我采用的策略是：这个词不是处在一句话的首位；正则表达式“[A-Z][a-z]+”,即首字母大写，后面的字母小写；利用nltk中的词性标注，如果是人名应该是属于名词。

    利用这种方法可以正确识别大部分的名字，也会将一些地名认为是人物名字，当然由于我找到的这个权力的游戏的txt文本中存在一些排版错误，也可能会影响到实体识别。

### 实验过程
* 完整代码

    项目的github地址：[可查看完整代码](https://github.com/Ysjshine/Mining-Game-of-Thrones)

* 实验材料

    《Game of Thrones》txt文本资源的下载。

* 利用gensim获取词向量

    前面提到，这里我利用Python的gensim库来训练得到相关的词向量。

1. 安装gensim库

    Python库的安装都比较简单,在终端中输入下面的命令行语句即可
    ```
    $ pip3 install gensim
    ```
2. 获取词向量

    导入gensim中的word2vec
    ```
    from gensim.models import word2vec
    ```

    读取game_of_thrones.txt,对它进行分词，并且识别出相关的人物名称。在我的代码中创建了一个类BuildDict来完成这项工作,以下是该类的的部分代码
    ```
    class BuildDict(object):
    def __init__(self,filename):
        self.filename = filename


    def get_tokens_and_names(self):
        '''

        :return:tokens and names from the file
        '''
        file = open(self.filename)
        tokens = [];names = []
        for line in file:
            token,name = self.get_tokens_and_names_per_line(line)
            tokens.append(token)
            names.extend(name)
        return tokens,names
    ```
    get_tokens_and_names()这个方法就是处理文本，将文本切割成一个一个的单词，返回tokens和names。

    如果文档很多，gensim库提供了一次读一个文档的方法，由于这里只有一个文档，并且文档并不大，所以未使用该方法。如果有兴趣可以看一看gensim的官方文档。

    在得到tokens和names过后，下面便进行训练了,利用之前导入的word2vec里的Word2Vec类。将训练的结果保存在一个文件中，方便后面使用。
    ```
    sentences,names= BuildDict("./game_of_thrones.txt").get_tokens_and_names()

    #train and get word vector
    model = word2vec.Word2Vec(sentences,min_count=1)
    model.save("./model.txt")

    #compute names' frequency and save the list
    freq = nltk.FreqDist(names)
    name_list = [w for w in set(names) if freq[w]>10]
    f = open('names.txt','w')
    for i in name_list:
        f.write(i+'\n')
    ```

    model文件有乱码，在这就不展示了。得到的names.txt部分内容如下图所示：

![name_list](http://upload-images.jianshu.io/upload_images/6297527-b88bc76d72d559a7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
* 人物关系分析

    经过前面一系列的操作，我们现在获得到了《权力的游戏》里大部分人的名字（频数大于10的人名）以及这些名字的词向量，接下来我们就可以来分析这些人物之间关系的紧密程度了。

1. 导入词典向量文件和人名

    将之前训练好的模型文件以及保存人名的txt文件导入到代码中。
    ```
    model = word2vec.Word2Vec.load("./model.txt")
    names = open('names.txt')
    ```

2. 相似度矩阵

    计算人物两两之间的相似度，可以选择前面所提到的方法来进行计算，在这里我嫌麻烦直接就用了model对象的similarity方法得到相似度。
    ```
    sim = model.similarity(name_row,name_column)
    ```
    最终得到了一个相似度的对称矩阵，并且对角线上的数字都是1，由于所有人物来自同一个文档，毫无疑问他们之间的相似度都比较大，粗略观察了一下，我所得到的相似度最小的也有0.65左右，这样不利于我们分析他们的亲疏关系。于是我对这个进行了归一化处理
    ```
    from sklearn import preprocessing
    name_mat = preprocessing.MinMaxScaler().fit_transform(np.mat(name_similarity_array))
    ```

    好，这就是我们所需要的相似度矩阵了。

3. 数据处理以及可视化

    光得到相似度矩阵可不行哇，无法给人直观的感受，下面我们来对这个矩阵进行一些分析。这个矩阵大概是196*196的规模，还是比较大的，下面我们对它进行降维处理。目前我采用的降维方法有主成分分析(PCA)以及矩阵的奇异值分解(SVD)这两种，两种方法得到的结果差不太多.
    
    * PCA
    ```
    from sklearn import decomposition
    
    pca = decomposition.PCA(n_components=3)
    pca.fit(name_mat)
    U = pca.transform(name_mat)
    ```

    得到的三维图像如下图：
![PCA](http://upload-images.jianshu.io/upload_images/6297527-83f5fb71fc46fb8f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

    * SVD
    ```
    import numpy as np

    U,S,V = np.linalg.svd(name_mat)
    ```
    得到三维图像如下图：
![SVD](http://upload-images.jianshu.io/upload_images/6297527-7ee78fd6fb0574f2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
    得到的三维图像效果并不是很好，全部都聚集在一堆去了，也难怪，这都是一个文档的人物，无论怎样关系还是很大的。人物关系图嘛，用网络拓扑图来表示肯定比这个三维图像更加直观。正好，Python就有这么一个库，可以绘制网络拓扑图，这个库便是networkx。
     安装networkx
      ```
      $pip3 install networkx
      ```
    用network绘制拓扑图的代码如下
    ```
    def visulize2(name_mat,name_list,name,layer = 2,accurate = 0.9):
    '''
    
    :param name_mat: 
    :param name_list: 
    :param name: the center of the graph
    :param layer: decide the layer number of the graph
    :param accurate: the similarity between the center with next layer
    :return: 
    '''
    graph = nx.Graph()
    add_element(graph,name_mat,name_list,name,layer,accurate)
    nx.draw(graph, node_size=100, node_color='g', with_labels=True,
            font_size =8,alpha = 0.5,font_color='b',edge_color = 'gray')
    plt.show()

    def add_element(graph,name_mat,name_list,name,layer,accurate):
        '''
        add element to the graph,including node and edge
        
        :param graph: 
        :param name_mat: 
        :param name_list: 
        :param name: 
        :param layer: 
        :param accurate: 
        :return: 
        '''
        graph.add_node(name)
        if layer == 1:return
        pos = name_list.index(name)
        for i in range(len(name_list)):
            if name_mat[pos, i] >= accurate:
                graph.add_node(name_list[i])
                graph.add_edge(name, name_list[i])
                add_element(graph,name_mat,name_list,name_list[i],layer-1,accurate)
    ```

    你只需要向visualize2方法传入相似度矩阵，名字列表，以及作为中心点的人物的名字，即可绘制出该人物与和他相似度大于0.9的人物之间的关系图。你也可以自定义layer和accurate的值，绘制更复杂的拓扑图。

     下面是layer = 2，accurate = 0.9,name = 'Snow'的关系图:
![Snow's Relationship](http://upload-images.jianshu.io/upload_images/6297527-3470947c910ffce6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

    下面是layer = 3，accurate = 0.95,name = 'Daenerys'的关系图
![Daenerys's Relationship](http://upload-images.jianshu.io/upload_images/6297527-7c4b3f688ebd88cc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

    虽然关系还是很复杂，但是不是很直观了呢！

### 总结

本项目中还是存在很多的不足，比如不能完全正确识别人物名字，人物间区分度不高（相似度都比较大），不过作为才学两周nlp的我来说已经很满意了，刚刚画出拓扑图的时候还是踌躇满志呢。由于笔者水平较低，是一枚刚入门的菜鸟，如有错误之处望各位读者不吝赐教。
