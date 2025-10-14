import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_excel('D:/SP_designer/data/SP1.xlsx')
output_path='D:/SP_designer/prediction/generated_sp.xlsx'

singal_peptide=data['sp_seq'].tolist()

AA=['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']

enzyme='MDSNGNQEINGKEKLSVNDSKLKDFGKTVPVGIDEENGMIKVSFMLTAQFYEIKPTKENEQYIGMLRQAVKNESPVHIFLKPNSNEIGKVESASPEDVRYFKTILTKEVKGQTNKLASVIPDVATLNSLFNQIKNQSCGTSTASSPCITFRYPVDGCYARAHKMRQILMNNGYDCEKQFVYGNLKASTGTCCVAWSYHVAILVSYKNASGVTEKRIIDPSLFSSGPVTDTAWRNACVNTSCGSASVSSYANTAGNVYYRSPSNSYLYDNNLINTNCVLTKFSLLSGCSPSPAPDVSSCGF'

freq=[]

for i in range(49): #max length of sp_seq
    AA_dict={'A':0,'R':0,'N':0,'D':0,'C':0,'Q':0,'E':0,'G':0,'H':0,'I':0,'L':0,'K':0,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0}
    for seq in singal_peptide:
        if i<len(seq):
            AA_dict[seq[i]]+=1
    whole=0
    for key in AA_dict.keys():
        if AA_dict[key]!=0:
            whole+=AA_dict[key]
    for key in AA_dict.keys():
        AA_dict[key]/=whole
    freq.append(AA_dict)


'''26'''
import random

peptide_list=[]
seq_list=[]
output_list=[]


# high_threshold=0.5
low_threshold=0.05
#
# for freq_dict in freq:
#     for key in freq_dict.keys():
#         if freq_dict[key]>high_threshold:
#             for key_1 in freq_dict.keys():
#                 if freq_dict[key_1]<=high_threshold:
#                     freq_dict[key_1]=0
#                 else:
#                     freq_dict[key_1]=1
#             break
#
for freq_dict in freq:
    for key in freq_dict.keys():
        if 0 < freq_dict[key] <= low_threshold:
            for key_1 in freq_dict.keys():
                if 0 < freq_dict[key_1] <= low_threshold:
                    freq_dict[key_1] = 0
            temp_sum=0
            for key_1 in freq_dict.keys():
                temp_sum+=freq_dict[key_1]
            for key_1 in freq_dict.keys():
                if freq_dict[key_1]>low_threshold:
                    freq_dict[key_1]=freq_dict[key_1]/temp_sum


matrix_data = [[each_pos_list[each_aa_key] for each_pos_list in freq] for each_aa_key in freq[0].keys()]


matrix_data = np.array(matrix_data)
plt.imshow(matrix_data, cmap='hot', interpolation='nearest')
plt.yticks(np.arange(len(freq[0].keys())), list(freq[0].keys()))
plt.colorbar()
plt.show()


for i in range(5000):
    peptide=''
    length=random.randint(15,35)
    for j in range(length):
        probabilities=freq[j]
        total_probability = sum(probabilities.values())
        weights = [probabilities[letter] / total_probability for letter in probabilities.keys()]
        letters = list(probabilities.keys())
        random_letter = random.choices(letters, weights=weights, k=1)[0]
        peptide+=random_letter
    peptide_list.append(peptide)

for peptide in peptide_list:
    seq_list.append(peptide+enzyme)

assert len(seq_list)==len(peptide_list)
for i in range(len(seq_list)):
    output_list.append([peptide_list[i],seq_list[i]])

import openpyxl
wb=openpyxl.Workbook()
ws=wb.active
headers = ['信号肽', '序列', '长度', '预测类别','预测酶活','SP(Sec/SPI)']
ws.append(headers)

for item in output_list:
    ws.append([item[0],item[1],str(len(item[1])),"",""])

wb.save(output_path)