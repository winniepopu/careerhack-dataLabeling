import glob
import json
import gensim
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
#import pdb

####### load word2vec model #######
model = gensim.models.KeyedVectors.load("fasttext.model", mmap='r')#embedding dim=300
#print(len(model['apple']))

####### variable #######
json_list = []
name_list = []
embedding_list = []
total_embedding_list = []

####### read json file #######
for filename in glob.glob('./train/train/Input/*.json'):###set input dir #./train/train/Input/*.json
    name_list.append(filename[-18:])
    json_dict = json.load(open(filename, 'r'))
    json_list.append(json_dict)

####### word2vec #######
i = 0
for afile in json_list:    
    for line in afile['recognitionResults'][0]['lines']:
        line_width = abs(line['boundingBox'][0]+line['boundingBox'][4])/2.0
        line_width = line_width/afile['recognitionResults'][0]['width']
        line_height = abs(line['boundingBox'][1]+line['boundingBox'][5])/2.0
        line_height = line_height/afile['recognitionResults'][0]['height']
        for word in line['words']:
            deldot_word = word['text'].replace('.','')
            deldot_word = deldot_word.replace(',','')
            if deldot_word.lower() not in model.vocab:
                continue
            if len(deldot_word) == 1 or deldot_word.isdigit():
                continue             
            #print(deldot_word)
            embedding = model[deldot_word.lower()].tolist()
            embedding.append(line_width)
            embedding.append(line_height)
            embedding_list.append(embedding)
    ####### save .npy #######
    output_dir = './newinput/' + name_list[i][:-4] + 'npy'### set output dir #
    with open(output_dir, 'w') as outfile:
        np.save(output_dir, np.array(embedding_list))
    #########################
    #print(embedding_list)       
    total_embedding_list += embedding_list
    embedding_list.clear()

    i += 1         

np_embedding_list = np.array(total_embedding_list)   
print(np_embedding_list.shape)

#np_embedding_list = preprocessing.scale(np_embedding_list)

####### kmean #######
estimator = KMeans(n_clusters=10)
estimator.fit(np_embedding_list)
print('Kmean finish!')
result = estimator.labels_

with open('./kmean.pickle', 'wb') as f:
    pickle.dump(estimator, f)
####### plot #######
tsne = TSNE(n_components=2)
reduced_data_tsne = tsne.fit_transform(np_embedding_list)

colors = ['black', 'blue', 'purple', 'yellow', 'gray', 'cyan', 'darkred', 'gold', 'green', 'pink']
for i in range(len(colors)):
    x = reduced_data_tsne[:, 0][result == i]
    y = reduced_data_tsne[:, 1][result == i]
    plt.scatter(x, y, c=colors[i])

plt.show()



























