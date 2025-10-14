import pandas as pd

class source_protein:
    def __init__(self):
        self.name=''
        self.data=[]
        self.mean=0.0

def calculate_source_mean(source_name:str,sp_data:list[list]):
    source_obj=source_protein()
    source_obj.name=source_name
    for row in sp_data:
        if row[-4]==source_name:
            source_obj.data.append(row)
    count=0
    total=0.0
    for row in source_obj.data:
        if row[-3]!=0:
            count+=1
            total+=row[-3]
    source_obj.mean=total/count
    return source_obj.mean

def allocate_label(sp_data:list[list]):
    for row in sp_data:
        mean=calculate_source_mean(row[-4],sp_data)
        if row[-3] == 0:
            row[-1] = "低"
        elif row[-3] <= mean:
            row[-1] = "低"
        else:
            row[-1] = "高"


def standardize(xlsx_path:str,out_path:str):
    df=pd.read_excel(xlsx_path)
    df['target'] = df.groupby('source_protein')['experimental yield'].transform(lambda x: (x - x.mean()) / x.std())
    df.to_excel(out_path,index=False)

standardize('SP1.xlsx','SP1_.xlsx')
standardize('SP2.xlsx','SP2_.xlsx')



