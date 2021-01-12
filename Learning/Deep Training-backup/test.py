import pandas as pd
from pathlib import Path
from datetime import datetime

path = Path(__file__).parent.absolute()/'Deep Training'

before = datetime.now()
s = pd.Series([0,1,85,45,45,7,9674,3,7468,76,54,894,5,74,687,5,7468,74,68])
d = pd.DataFrame({'S':s.tolist(),'A':['a']*len(s),'F':[0.1]*len(s),'B':[False]*len(s)})
print(d.loc[:,d.dtypes==object])
after = datetime.now()
print(before, after)
print(before<after,before>after,before==after,before!=after)
print((d.dtypes==bool).any())
print(d['S'].to_dict())
print(d.to_csv(header=False, index=False, mode='a', line_terminator='')[:-1])
print(d.to_csv(header=False, index=False, mode='a', line_terminator='')[:-1])
s.to_csv(path/'test.csv')
#print(s.index.tolist()+s.index.tolist())
#print(s.drop(s.sample(4).index))