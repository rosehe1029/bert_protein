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
from tqdm  import tqdm
warnings.filterwarnings('ignore')
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--BATCH_SIZE', type=int)
parser.add_argument('--MAXLEN',type=int)
parser.add_argument('--EPOCHS', type=int)
parser.add_argument('--LR',type=float)


args = parser.parse_args()
batch_size=args.BATCH_SIZE
maxlen0=args.MAXLEN
epochs=args.EPOCHS #= 16
LR=args.LR

# 该文件夹下存放三个文件（'vocab.txt', 'pytorch_model.bin', 'config.json'）
tokenizer = TAPETokenizer(vocab='iupac')   # 初始化分词器
#print(tokenizer)
# 加载词典

BATCH_SIZE = batch_size#32  # 如果会出现OOM问题，减小它

IUPAC_VOCAB = collections.OrderedDict([
    ("<pad>", 0),
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
print("1.预训练模型获取完毕！！！")
print("$"*100)
###################################
#2 预处理数据集
maxlen = maxlen0

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
            if i>=2400000:
                break
    #print(i)
    return fasta

input_ids, input_masks = [], []  # input char ids, segment type ids,  attention mask
labels,targets = [],[]  # 标签
#
fasta_info = read_fasta('data.fasta')

mk_ids=[]
for key, value in fasta_info.items():
    #print(len(value))
    mask_num = 1
    if len(value)> maxlen-2:
        start_string = value[:maxlen//2]
        end_string=value[maxlen//2+1:]
        y=value[maxlen//2]
        mk_id=maxlen//2+1
    else:
        start_string=value[:len(value)//2]
        end_string = value[len(value)//2+1:]
        y = value[len(value)// 2]
        mk_id=len(value)//2+1
    pad_len=maxlen-len(value)-2
    #print(pad_len)
    if pad_len!=0:
        string = list(start_string) + ['<mask>'] * mask_num + list(end_string)+['<pad>']*pad_len
        target = list(value) + ['<pad>'] * pad_len
    else:
        string = list(start_string) + ['<mask>'] * mask_num + list(end_string)
        target = list(value)
    #print(string)
    mk_id=mk_id%BATCH_SIZE
    masks = [0] * maxlen
    #print(target)
    for i in range(mask_num):
        masks[len(start_string)+i+1] = 1
    target_ids=tokenizer.encode(target)
    token_ids = tokenizer.encode(string)
    #print("token_ids",type(token_ids))
    #print(token_ids.shape)
    #print(token_ids[8])
    mk_ids.append(mk_id)
    y_ids=tokenizer.encode(y)[1]
    input_ids.append(token_ids)
    input_masks.append(masks)
    targets.append(target_ids)
    labels.append(y_ids)
    #print(input_ids)
input_ids, input_masks,mk_ids = np.array(input_ids),np.array(input_masks),np.array(mk_ids)
labels,targets = np.array(labels), np.array(targets)
print(input_ids.shape,  input_masks.shape, labels.shape,targets.shape,mk_ids.shape)
print("2.数据集加载完毕！！！")
print("$"*100)

#3 切分训练集、验证集和测试集
# 随机打乱索引
idxes = np.arange(input_ids.shape[0])
np.random.seed(2019)   # 固定种子
np.random.shuffle(idxes)
print(idxes.shape, idxes[:10])

# 8:1:1 划分训练集、验证集、测试集
input_ids_train, input_ids_valid, input_ids_test = input_ids[idxes[:960000]], input_ids[idxes[960000:1080000]], input_ids[idxes[1080000:]]
input_masks_train, input_masks_valid, input_masks_test = input_masks[idxes[:960000]], input_masks[idxes[960000:1080000]], input_masks[idxes[1080000:]]
mk_ids_train, mk_ids_valid, mk_ids_test = mk_ids[idxes[:960000]], mk_ids[idxes[960000:1080000]], mk_ids[idxes[1080000:]]

t_train, t_valid, t_test = targets[idxes[:960000]],targets[idxes[960000:1080000]], labels[idxes[1080000:]]
y_train, y_valid, y_test = labels[idxes[:960000]], labels[idxes[960000:1080000]], labels[idxes[1080000:]]

print(input_ids_train.shape, y_train.shape,t_train.shape,mk_ids_train.shape, input_ids_valid.shape, y_valid.shape,t_valid.shape,mk_ids_valid.shape,
      input_ids_test.shape, y_test.shape,t_test.shape,mk_ids_test.shape)
print("3.划分训练集验证集测试集完毕！！！")
print("$"*100)

#4加载到pytorch的DataLoader


# 训练集
#print("y_train",y_train)
train_data = TensorDataset(torch.LongTensor(input_ids_train.astype(float)) ,
                           torch.LongTensor(input_masks_train.astype(float)),
                           torch.LongTensor(t_train.astype(float)),
                           torch.LongTensor(mk_ids_train.astype(int)),
                           torch.LongTensor(y_train.astype(float)))
train_sampler = RandomSampler(train_data)  
train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE,drop_last=True,num_workers=10)

# 验证集
valid_data = TensorDataset(torch.LongTensor(input_ids_valid.astype(float)),
                          torch.LongTensor(input_masks_valid.astype(float)),
                          torch.LongTensor(t_valid.astype(float)),
                          torch.LongTensor(mk_ids_valid.astype(int)),
                          torch.LongTensor(y_valid.astype(float)))
valid_sampler = SequentialSampler(valid_data)
valid_loader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE,drop_last=True,num_workers=10)

# 测试集（是没有标签的）
test_data = TensorDataset(torch.LongTensor(input_ids_test.astype(float)),
                          torch.LongTensor(input_masks_test.astype(float)),
                          torch.LongTensor(mk_ids_test.astype(int)))
test_sampler = SequentialSampler(test_data)
test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE,drop_last=True,num_workers=10)
print("4.加载到pytorch的DataLoader完毕！！！")
print("$"*100)


#5定义bert模型
# 定义model
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

print("5.定义bert模型完毕！！！")
print("$"*100)

#6 实例化bert模型
def get_parameter_number(model):
    #  打印模型参数量
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 'Total parameters: {}, Trainable parameters: {}'.format(total_num, trainable_num)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = epochs
model = ProteinBert().to(DEVICE)
#print(get_parameter_number(model))
print("6.实例化bert模型完毕！！！")
print("$"*100)


#7 定义优化器
#2e-4
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4) #AdamW优化器
scheduler = WarmupCosineSchedule(optimizer, warmup_steps=len(train_loader),
                                            t_total=EPOCHS*len(train_loader))
# 学习率先线性warmup一个epoch，然后cosine式下降。
# 这里给个小提示，一定要加warmup（学习率从0慢慢升上去），要不然你把它去掉试试，基本上收敛不了。
print("7.定义优化器完毕！！！")
print("$"*100)

#8定义训练函数和验证测试函数
# 评估模型性能，在验证集上
def accuracy_score(val_true, val_pred):
    correct = 0
    total = 0
    for label, score in zip(val_true, val_pred):
        label_array = np.asarray(label)
        pred_array = np.asarray(score).argmax(-1)
        mask = label_array != -1
        is_correct = label_array[mask] == pred_array[mask]
        correct += is_correct.sum()
        total += is_correct.size
    #print(correct)
    #print(total)
    return correct / total

def evaluate(model, data_loader, device):
    model.eval()
    val_true, val_pred = [], []
    with torch.no_grad():
        for idx, (ids, mk, ta,mid, y) in (enumerate(data_loader)):
            y_pred = model(ids.to(device), mk.to(device),ta.to(device))[1]
            #print(y_pred)
            y_pred = y_pred.cuda().data.cpu().numpy().squeeze()
            #print(y_pred.shape)
            #print(mid)
            y_pred = y_pred[mid].argmax(axis=1)

            val_pred.extend([y_pred])
            val_true.extend([y.squeeze().cpu().numpy().tolist()])
    return accuracy_score(val_true, val_pred)  # 返回accuracy


# 测试集没有标签，需要预测提交
def predict(model, data_loader, device):
    model.eval()
    val_pred = []
    with torch.no_grad():
        for idx, (ids, mk,mid) in tqdm(enumerate(data_loader)):
            y_pred = model(ids.to(device), mk.to(device))[0]
            #print(y_pred)
            y_pred = y_pred.cuda().data.cpu().numpy().squeeze()
            y_pred = y_pred[mid].argmax(axis=1)
            val_pred.extend(y_pred)
    return val_pred


def train_and_eval(model, train_loader, valid_loader,
                   optimizer, scheduler, device, epoch):
    best_acc = 0.0
    patience = 0
    #criterion = nn.CrossEntropyLoss()
    for i in range(epoch):
        """训练模型"""
        start = time.time()
        model.train()
        print("***** Running training epoch {} *****".format(i + 1))
        train_loss_sum = 0.0
        for idx, (ids, mk, ta,mid, y) in enumerate(tqdm(train_loader)):
            
            ids, mk, ta,mid,y = ids.to(device), mk.to(device), ta.to(device), mid.to(device) ,y.to(device)
            #y_pred = model(ids, att, tpe)[1]
            loss = model(ids, mk, ta)[0]
            #print("loss",loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  # 学习率变化

            train_loss_sum += loss.item()
            if (idx + 1) % (len(train_loader) // 5) == 0:  # 只打印五次结果
                print("Epoch {:04d} | Step {:04d}/{:04d} | Loss {:.4f} | Time {:.4f}".format(
                    i + 1, idx + 1, len(train_loader), train_loss_sum / (idx + 1), time.time() - start))
                # print("Learning rate = {}".format(optimizer.state_dict()['param_groups'][0]['lr']))

        """验证模型"""
        model.eval()
        acc = evaluate(model, valid_loader, device)  # 验证模型的性能
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_bert_model.pth")

        print("current acc is {:.4f}, best acc is {:.4f}".format(acc, best_acc))
        print("time costed = {}s \n".format(round(time.time() - start, 5)))
print("8.定义训练函数和验证测试函数完毕！！！")
print("$"*100)


#9 开始训练和验证模型
# 训练和验证评估
train_and_eval(model, train_loader, valid_loader, optimizer, scheduler, DEVICE, EPOCHS)
print("9.训练和验证模型完毕！！！")
print("$"*100)

#10 加载最优模型进行测试
# 加载最优权重对测试集测试
model.load_state_dict(torch.load("best_bert_model.pth"))
pred_test = predict(model, test_loader, DEVICE)
print("\n Test Accuracy = {} \n".format(accuracy_score(y_test, pred_test)))
print("10.加载最优模型进行测试完毕！！！")
print("$"*100)
#acc = evaluate(model, valid_loader,DEVICE)
#'''


