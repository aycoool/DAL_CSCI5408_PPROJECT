import pandas as pd

data = pd.read_csv('loan_data.csv')

data.info()

print("\n\n")

info = data.describe()
print(info)

df = pd.DataFrame(info)

df.to_csv('loanstats.csv', sep=',', encoding='utf-8', index=False)

cat_feats = ['purpose']

done_data = pd.get_dummies(data,columns=cat_feats,drop_first=True)

mf = pd.DataFrame(done_data)

mf.to_csv('final_loan_data.csv', sep=',',encoding='utf-8',index=False)