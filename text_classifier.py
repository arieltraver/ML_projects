#text classification
#tutorial: https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html


import torch
from torchtext.datasets import AG_NEWS #dataset to use
from torchtext.data.utils import get_tokenizer #links text -> words
from torchtext.vocab import build_vocab_from_iterator #develops a vocabulary from the data
from torch.utils.data import DataLoader 
from torch import nn


tokenizer = get_tokenizer('basic_english') #recognize english words
train_iter = iter(AG_NEWS(split='train'))

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text) #return a generator which will turn words into ints


vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"]) #use iterator to get vocab
vocab.set_default_index(vocab["<unk>"]) #unk = unknown word


print(vocab(['example', 'idea', 'try', 'this', 'ok']))

text_pipeline = lambda x : vocab(tokenizer(x)) #data pipe so we don't load whole file into ram
label_pipeline = lambda x : int(x) - 1 #assign label to raw text

ex = text_pipeline('example example ok')
print(ex)
print(label_pipeline(ex[0]))

#now we need a collate function because tokenized text is not rectangular
#take the data processed according to our tokenizers and turn it into a usable tensor
#for our model

#use the GPU if it's available for rapid processing of small tasks, otherwise just use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for _label, _text in batch:
        label_list.append(label_pipeline(_label)) #put the label - 1 in the list
        #make a same-type array (tensor) from the converted text
        processed_text = torch.tensor(text_pipeline(_text), dtype = torch.int64) 
        text_list.append(processed_text)
    label_list = torch.tensor(offsets[:-1]).cumsum(dim=0) #flatten labels into 1d tensor
    text_list = torch.cat(text_list) #concatenate everything into a single tensor
    return label_list.to(device), text_list.to(device), offsets.to(device) #convert to gpu/cpu and return

class TextClassificationModel(nn.Model): #the model to be trained

#instance variables: vocab_size, embed_dim (dimensions of embedding vectors), num_class (how many categories)
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True) #turn words into informational vectors
        self.fc = nn.Linear(embed_dim, num_class) #applies a linear transformation to the oncoming data
        self.init_weights()
    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, +initrange) #random weights
        self.fc.weight.data.uniform_(-initrange, +initrange)
        self.fc.bias.data.zero_() #init bias to 0
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded) #apply the linear transformation

#AG news has 4 classes