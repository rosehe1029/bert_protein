# bert_protein

Protein sequence prediction  for bert

problem 

Given a list of masked protein sequences, write a program to predict the masked amino acids (i.e., characters) as followings:

./run.sh MASKED_IN.fasta UNMASKED_OUT.fasta

python protein_bert_predict.py S1.fasta S1pred.fasta

The output format is as followings:

>ID0
SEQUENCE0
>ID1
SEQUENCE1
>ID2
SEQUENCE2
>ID3
SEQUENCE3


1.train

python train.py --BATCH_SIZE 32 --MAXLEN 256 --EPOCHS 16 --LR 2e-4

2.predict 

python predict.py --IN out2.fasta --OUT out3.fasta


参考： Evaluating Protein Transfer Learning with TAPE

Tasks Assessing Protein Embeddings (TAPE)
