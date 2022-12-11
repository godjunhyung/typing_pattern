import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import random
import shutil

def assemble(data_path, file_name, count):
    df = pd.read_csv(os.path.join(data_path, file_name), sep='\n', names=["pw"])
    df = df[~df.pw.str.contains(' ')]
    
    df[['id', 'pw']] = df['pw'].str.split('@', 1, expand=True)
    df['pw'] = df['pw'].str.split(':').str[1]
    df = df.reset_index(drop=True)
    
    df['count'] = df.groupby(by=['id']).transform('count')

    df = df.loc[df['count'] >= count].reset_index(drop=True)
    df = df.loc[df['count'] < 1000].reset_index(drop=True)
    
    return df

def get_strings(data):
    li = []
    for i in range(len(data)):
        pw = data.iloc[i]['pw']
        for j in pw:
            if j not in li:
                li.append(j)
    return li

def check_str_dict():    
    chk_li = []
    
    """Upper"""
    for i in range(65, 91):
        chk_li.append(chr(i))
        
    """Number"""
    for i in range(10):
        chk_li.append(str(i))    
        
    """Lower"""
    for i in range(97, 123):
        chk_li.append(chr(i))
        
    special = [' ','!','"','#','$','%','&',"'",'(',
               ')','*','+','-','.','/', ',', ':',
               ';', '<','=','>','?','@','[', '\\', "]",'^',
                '_','`','{','|','}','~', 'ᆢ', '※', 'ㆍ',
               '…', '》']
    for i in special:
        chk_li.append(i)
    
    chk_dict = {}
    for i in range(len(chk_li)):
        chk_dict[chk_li[i]] = i

    return chk_dict

def visualization(name, pw_li, chk_dict):
    chk_li = list(chk_dict.keys())
    path_value = np.zeros((len(chk_li), len(chk_li)))
    
    r_lst = []
    for pw in pw_li:
        for i in range(len(pw)-1):
            path_value[chk_dict[pw[i]]][chk_dict[pw[i+1]]] += 1

    plt.imsave(f'../data/{name}.png', path_value, dpi = 300)
    return path_value

def main():
    data_path = '../data/pwd'
    df = assemble(data_path, '0', 100)
    chk_dict = check_str_dict()
    te_data = df.groupby('id')
    for i in df['id'].unique():
        pw_li = []
        for pw in te_data.get_group(i)['pw']:
            match = re.search(r'[A-Za-z0-9-=+,#/\?ᆢ:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]+', pw)
            if match:
                pw_li.append(pw)
        
        if len(pw_li) >= 100:
            random.shuffle(pw_li)
            for k in range(30):
                sampled_li = random.sample(pw_li[:50], 30)
                try:
                    visualization(f"train/{i}_{str(k).zfill(2)}", sampled_li, chk_dict)
                except:
                    continue
            for k in range(30, 60):
                sampled_li = random.sample(pw_li[50:100], 30)
                try:
                    visualization(f"test/{i}_{str(k).zfill(2)}", sampled_li, chk_dict)
                except:
                    continue

if __name__=="__main__":
    print("Delete pre-generated visualized images to create new images")
    print("The result can be changed.")
    shutil.rmtree('../data/train')
    shutil.rmtree('../data/test')
    print("start visualize user password pattern")
    main()
    print('finish')