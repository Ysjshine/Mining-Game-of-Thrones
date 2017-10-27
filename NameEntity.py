import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from gensim.models import word2vec
from sklearn import preprocessing,decomposition
from mpl_toolkits.mplot3d import Axes3D

def get_similarity_mat():
    '''

    :return:the matrix of similarity
    '''
    #load the model that saved before
    model = word2vec.Word2Vec.load("./model.txt")
    #read name_list
    names = open('names.txt')
    name_list = []
    for name in names:
        name_list.append(name[:-1])

    #compute similarity matrix
    name_similarity_array = []
    for name_row in name_list:
        tmp = []
        for name_column in name_list:
             sim = model.similarity(name_row,name_column)
             tmp.append(sim)
        name_similarity_array.append(tmp)
    #normalize
    name_mat = preprocessing.MinMaxScaler().fit_transform(np.mat(name_similarity_array))
    return name_mat,name_list

def save_similar_mat(name_mat,name_list):
    '''
    save matrix
    :param name_mat:
    :param name_list:
    :return:
    '''
    f = open('./human_similar_vector.csv','w')
    for i in range(len(name_list)+1):
        for j in range(len(name_list)+1):
            if i==0 and j==0:
                f.write(' ')
            elif i == 0 and j > 0:
                f.write(name_list[j-1])
            elif i>0 and j == 0:
                f.write(name_list[i-1])
            elif i>0 and j >0:
                f.write(str(name_mat[i-1,j-1]))
            if j == len(name_list):
                f.write('\n')
            else:f.write(',')

def visulize(name_mat,name_list):
    #pcva
    # pca = decomposition.PCA(n_components=3)
    # pca.fit(name_mat)
    # U = pca.transform(name_mat)
    #svd
    U,S,V = np.linalg.svd(name_mat)

    #draw 3D picture
    ax = Axes3D(plt.figure())
    for i in range(len(name_list)):
        ax.scatter(U[i, 0], U[i, 1],U[i,2])
        ax.text(U[i, 0], U[i, 1],U[i,2],name_list[i])
    plt.show()



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

name_mat,name_list = get_similarity_mat()
#save_similar_mat(name_mat,name_list)
#visulize(name_mat,name_list)
visulize2(name_mat,name_list,'Snow')
