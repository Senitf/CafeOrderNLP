import pandas as pd

df = pd.read_excel('data_cafe.xlsx')
df_list = df.values.tolist()
df_col = list([col for col in df])

fixed_df_list = []

for li in df_list:
    nxtidx = df_list.index(li) + 1
    if nxtidx == len(df_list):
        break
    if li[0] == '고객' and df_list[nxtidx][0] == '점원' and li[2] < df_list[nxtidx][2]:
        li[4] = 'Q'
        df_list[nxtidx][4] = 'A'
        fixed_df_list.append(li)
        #fixed_df_list.append(df_list[nxtidx])

df = pd.DataFrame(fixed_df_list, columns=df_col)

dic = []
idx = 0

f = open('data_train.txt', 'w')
#f.write('Q,A,label\n')
f.write('title,label\n')
for li in fixed_df_list:
    if li[4] == 'Q':
        #data_line = str(li[1].replace(',', '') + ',')
        data_line = str(li[1].replace(',', '') + ',' + str(idx) + '\n')
        f.write(data_line)
        if li[3] in dic:
            idx = dic.index(li[3])
        else:
            dic.append(li[3])
            idx = dic.index(li[3])
    if li[4] == 'A':
        data_line = str(li[1].replace(',', '')) + ',' + str(idx) + '\n'
        f.write(data_line)
f.close()

print(len(dic))
'''
df.to_excel('data_cafe_fixed.xlsx', index=False)
df.to_string('data_cafe_fixed.txt', justify='center', col_space=45, index=False)
'''