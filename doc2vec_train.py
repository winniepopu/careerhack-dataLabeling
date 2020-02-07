import glob
import json
import gensim
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

###### variable
json_list = []
name_list = []
my_line_list = []
dict_list = []

###### read file
for filename in glob.glob('./train/train/Input/*.json'):###set input dir #./train/train/Input/*.json
    name_list.append(filename[-18:])
    json_dict = json.load(open(filename, 'r'))
    json_list.append(json_dict)

###### find line
count = 0
for afile in json_list:
    json_rotate = afile['recognitionResults'][0]['clockwiseOrientation']
    my_line_str = ''
    last_x = -1
    lu_b_x = 0
    lu_b_y = 0
    rd_b_x = 0
    rd_b_y = 0
    for line in afile['recognitionResults'][0]['lines']:
        if json_rotate < 180:
            if last_x == -1:   
                last_x = line['boundingBox'][0]
                lu_b_x = line['boundingBox'][0]
                lu_b_y = line['boundingBox'][1]
                rd_b_x = line['boundingBox'][4]
                rd_b_y = line['boundingBox'][5]
                my_line_str += line['text'] + ' '
                continue

            now_x = line['boundingBox'][0]
            if now_x < last_x:
                my_line_list.append(my_line_str[:-1])
                line_dict = {'boundingBox':[lu_b_x,lu_b_y,rd_b_x,lu_b_y,rd_b_x,rd_b_y,lu_b_x,rd_b_y], 'text':my_line_str}
                dict_list.append(line_dict)

                my_line_str = line['text'] + ' '
                last_x = now_x
                lu_b_x = line['boundingBox'][0]
                lu_b_y = line['boundingBox'][1]
                rd_b_x = line['boundingBox'][4]
                rd_b_y = line['boundingBox'][5]

            else:
                my_line_str += line['text'] + ' '
                last_x = now_x
                rd_b_x = line['boundingBox'][4]
                rd_b_y = line['boundingBox'][5]
        else:
            if last_x == -1:   
                last_x = line['boundingBox'][0]
                lu_b_x = line['boundingBox'][6]
                lu_b_y = line['boundingBox'][7]
                rd_b_x = line['boundingBox'][2]
                rd_b_y = line['boundingBox'][3]
                my_line_str += line['text'] + ' '
                continue

            now_x = line['boundingBox'][0]
            if now_x < last_x:
                my_line_list.append(my_line_str[:-1])
                line_dict = {'boundingBox':[lu_b_x,rd_b_y,rd_b_x,rd_b_y,rd_b_x,lu_b_y,lu_b_x,lu_b_y], 'text':my_line_str}
                dict_list.append(line_dict)

                my_line_str = line['text'] + ' '
                last_x = now_x
                lu_b_x = line['boundingBox'][0]
                lu_b_y = line['boundingBox'][1]
                rd_b_x = line['boundingBox'][4]
                rd_b_y = line['boundingBox'][5]

            else:
                my_line_str += line['text'] + ' '
                last_x = now_x
                rd_b_x = line['boundingBox'][4]
                rd_b_y = line['boundingBox'][5]
    my_line_list.append(my_line_str[:-1])
    if json_rotate > 180:
        line_dict = {'boundingBox':[lu_b_x,rd_b_y,rd_b_x,rd_b_y,rd_b_x,lu_b_y,lu_b_x,lu_b_y], 'text':my_line_str}
    else:
        line_dict = {'boundingBox':[lu_b_x,lu_b_y,rd_b_x,lu_b_y,rd_b_x,rd_b_y,lu_b_x,rd_b_y], 'text':my_line_str}
    dict_list.append(line_dict)
    ###### save new line as json file
    with open('./my_line/' + name_list[count], 'w') as f:
        json.dump(dict_list, f)
    ###### 
    dict_list.clear()
    count += 1


print(len(my_line_list))

###### train doc2vec
train_input = []

tokenized_sents = [word_tokenize(i) for i in my_line_list]
for i,text in enumerate(tokenized_sents):   
    train_input.append(TaggedDocument(text,tags=[i]))
#print(train_input)

model = Doc2Vec(train_input,min_count=1,vector_size=200,workers=4)
model.train(train_input,total_examples=model.corpus_count,epochs=100)
print('train finish!')
model.save("d2v.model")
#model= Doc2Vec.load("d2v.model")

test_str = 'Total $45.0'
test_data = test_str.split(' ')
infer_test = model.infer_vector(doc_words=test_data,alpha=0.025,steps=500)

sims = model.docvecs.most_similar([infer_test],topn=10)
for i,sim in sims:
    print(i,sim)
    sentence = my_line_list[i]

    print(sentence)
























            


