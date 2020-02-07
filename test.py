import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image

json_path = './train/train/Input/labeling_0000.json'
json_dict = json.load(open(json_path, 'r'))
#print(len(json_dict['recognitionResults'][0]['lines']))

json_line = json_dict['recognitionResults'][0]['lines']

words = []

for i in json_line:
    for j in i['words']:
        words += [j['text']]

print(words)

################# json structure: #######################
#status,recognitionResults
#            |->page,clockwiseOrientation,width,height,unit,lines
#                                                               |->boundingBox,text,words
#                                                                                    |-> boundingBox,text  

################# qt drawLine #######################
'''qp.drawLine(line_position[0]*self.resize_ratio, line_position[1]*self.resize_ratio, line_position[2]*self.resize_ratio, line_position[3]*self.resize_ratio)
                    qp.drawLine(line_position[2]*self.resize_ratio, line_position[3]*self.resize_ratio, line_position[4]*self.resize_ratio, line_position[5]*self.resize_ratio)
                    qp.drawLine(line_position[4]*self.resize_ratio, line_position[5]*self.resize_ratio, line_position[6]*self.resize_ratio, line_position[7]*self.resize_ratio)
                    qp.drawLine(line_position[6]*self.resize_ratio, line_position[7]*self.resize_ratio, line_position[0]*self.resize_ratio, line_position[1]*self.resize_ratio)'''

################# trasfer fasttect .vec file to gensim model(can read model faster) #######################
#model = gensim.models.KeyedVectors.load_word2vec_format('./crawl-300d-2M.vec', binary=False)
#model.save("fasttext.model")
