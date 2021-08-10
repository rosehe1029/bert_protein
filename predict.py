#1 导入必要的库
import pandas as pd
import numpy as np
import json, time
from tqdm import tqdm
#from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
#from transformers import BertModel, BertConfig, BertTokenizer, AdamW, get_cosine_schedule_with_warmup
from tape import ProteinBertModel, TAPETokenizer,ProteinBertAbstractModel,ProteinBertForMaskedLM
from tape.optimization import AdamW,WarmupCosineSchedule
import warnings
import  collections
warnings.filterwarnings('ignore')
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--IN', type=str)
parser.add_argument('--OUT',type=str)
args = parser.parse_args()

DATA=str(args.IN)
out=str(args.OUT)

# 该文件夹下存放三个文件（'vocab.txt', 'pytorch_model.bin', 'config.json'）
tokenizer = TAPETokenizer(vocab='iupac')   # 初始化分词器
#print(tokenizer)
# 加载词典
BATCH_SIZE = 32  # 如果会出现OOM问题，减小它
class ProteinBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = ProteinBertForMaskedLM.from_pretrained('bert-base')

    def forward(self,input_ids,input_mask=None,targets=None):
        outputs = self.bert(input_ids, input_mask=input_mask,targets=targets)
        #sequence_output, pooled_output = outputs[:2]
        # add hidden states and attention if they are here
        #outputs = self.mlm(sequence_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs
IUPAC_VOCAB = collections.OrderedDict([
    ("<mask>", 1),
    ("<cls>", 2),
    ("<sep>", 3),
    ("<unk>", 4),
    ("A", 5),
    ("B", 6),
    ("C", 7),
    ("D", 8),
    ("E", 9),
    ("F", 10),
    ("G", 11),
    ("H", 12),
    ("I", 13),
    ("K", 14),
    ("L", 15),
    ("M", 16),
    ("N", 17),
    ("O", 18),
    ("P", 19),
    ("Q", 20),
    ("R", 21),
    ("S", 22),
    ("T", 23),
    ("U", 24),
    ("V", 25),
    ("W", 26),
    ("X", 27),
    ("Y", 28),
    ("Z", 29),
    ("J",0)])
id_token_dict = {v: k for k, v in IUPAC_VOCAB.items()}
maxlen=256
def read_fasta(input):  # 用def定义函数read_fasta()，并向函数传递参数用变量input接收
    i=0
    with open(input, 'r') as f:  # 打开文件
        fasta = {}  # 定义一个空的字典
        for line in f:
            i=i+1
            line = line.strip()  # 去除末尾换行符
            if line[0] == '>':
                header = line[1:]
            else:
                sequence = line
                if len(sequence)>maxlen-2:
                    #print("序列长度超过512,暂不处理")
                    #continue
                    sequence = line[:(maxlen-2)]
                fasta[header] = fasta.get(header, '') + sequence
            if i==20:
                break
    return fasta

input_ids, input_masks = [], []  # input char ids, segment type ids,  attention mask
fasta_info = read_fasta(DATA)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=ProteinBert().to(DEVICE)
model.load_state_dict(torch.load("best_bert_model0.pth"))

def predict(model, data_loader, device):
    model.eval()
    val_pred = []
    with torch.no_grad():
        for idx, (ids, mk) in enumerate(data_loader):
            #print("ids",ids)
            y_pred = model(ids.to(device), mk.to(device))[0]
            y_pred = y_pred.cuda().data.cpu().numpy().squeeze()
            y_pred=y_pred.argmax(axis=1)
            #print(y_pred.shape)
            #y_pred=y_pred
            val_pred.extend([y_pred])
    return np.array(val_pred)

in_dataa=[]
in_maskk=[]
for key, value in fasta_info.items():
    in_data=[]
    mask=[]
    for v in value:
        #print(v)
        if v=="_":
            v="<mask>"
        in_data.append(v)
    #print(in_data)
    pad_len = maxlen - len(value) - 2
    in_data=in_data+['<pad>']*pad_len
    token_ids = tokenizer.encode(in_data)
    #print(token_ids)
    in_dataa.append(token_ids)
    mid=[idx for idx ,t in enumerate(token_ids) if t==1]
    #print(mid)
    mask=[0]* maxlen
    for midd in mid:
        mask[midd]=1
    mask_ids=mask
    in_maskk.append(mask_ids)
    #print(mask_ids)
    #print(len(token_ids))
    #print(len(mask_ids))
input_ids, input_masks = np.array(in_dataa ), np.array(in_maskk)
#print(input_ids.shape,input_masks.shape)
test_data = TensorDataset(torch.LongTensor(input_ids.astype(float)),
                              torch.LongTensor(input_masks.astype(float)))
test_sampler = SequentialSampler(test_data)
test_loader = DataLoader(test_data, sampler=test_sampler)#, batch_size=BATCH_SIZE, drop_last=True)
pred_test = predict(model, test_loader, DEVICE)
#print(pred_test.shape[0])
print("共"+str(pred_test.shape[0])+"条序列")

h=-1
result=[]
data=[]
for key1, value1 in fasta_info.items():
    print(">"+key1)
    data.append(">"+key1)
    h=h+1
    y=-1
    #print(h)
    r = []
    for v in value1:
        y=y+1
        #print(y)
        if v=="_":
            v=id_token_dict[pred_test[h][y]]
        #print(v)
        r.append(v)
    str1=""
    for rr in r:
        str1=str1+str(rr)
    data.append(str1)
    print(str1)

with open(out, 'w') as f:
    for i in data:
        f.write(i + '\n')


