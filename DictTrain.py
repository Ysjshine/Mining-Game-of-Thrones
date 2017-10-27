from gensim.models import word2vec
import nltk
from nltk.corpus import stopwords
import re


stopPunctuation = [',','.','/','\\','\'','\"',':',';','?','&','*','$',
                   '#','@','!','-','--','the','The','’','”','“','‘：',"？"]

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


    def get_tokens_and_names_per_line(self,line):
        '''

        :param line: a line from file when reading file
        :return: token and names from this line
        '''
        token = [];name = []

        #split the line to get sentences
        sentences = nltk.sent_tokenize(line)
        for one_sentence in sentences:
            #split the sentences
            words = nltk.word_tokenize(one_sentence)
            for one_token in words:
                token.append(one_token)

                #the token is a name?
                is_name = self.get_name(token=one_token)
                if is_name and words.index(one_token) > 0 \
                        and one_token.lower() not in stopwords.words('english'):
                    name.append(one_token)

        return token,name

    def get_name(self,token):
        tag = nltk.pos_tag([token])[0]
        if re.match('[A-Z][a-z]+',token) and tag[1] == 'NN' :
            return True
        else:return False


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

#the finish signal
print("cool")



