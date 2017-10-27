import gensim
from gensim.models import word2vec
import nltk
from nltk.corpus import stopwords
import re


stopPunctuation = [',','.','/','\\','\'','\"',':',';','?','&','*','$',
                   '#','@','!','-','--','the','The','’','”','“','‘：',"？"]

class BuildVovabulary(object):
    def __init__(self,filename):
        self.filename = filename


    def gets(self):
        file = open(self.filename)
        tokens = []
        names = []
        for line in file:
            token,name = self.get_tokens(line)
            tokens.append(token)
            names.extend(name)
            self.names = names
        return tokens,names


    def get_tokens(self,line):
        ans = []
        name = []
        sentences = nltk.sent_tokenize(line)
        for one_sentence in sentences:
            words = nltk.word_tokenize(one_sentence)
            for one_token in words:
                ans.append(one_token)
                is_name = self.get_name(token=one_token)
                if is_name and words.index(one_token) > 0 \
                        and one_token.lower() not in stopwords.words('english'):
                    name.append(one_token)

        return ans,name

    def get_name(self,token):
        tag = nltk.pos_tag([token])[0]
        if re.match('[A-Z][a-z]+',token) and tag[1] == 'NN' and token[-1] != '.':
            return True
        else:return False

sentences,names= BuildVovabulary("./game.txt").gets()
model = gensim.models.word2vec.Word2Vec(sentences,min_count=1)
model.save("./model.txt")
freq = nltk.FreqDist(names)
name_list = [w for w in set(names) if freq[w]>10]
f = open('names.txt','w')
for i in name_list:
    f.write(i+'\n')
print("cool")



