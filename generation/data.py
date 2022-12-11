import torch
import random
import os
from torch.utils.data import Dataset
import cv2
import pandas as pd

import torch
import random
import os
from torch.utils.data import Dataset
import cv2
import pandas as pd

def get_labels(tr_img_list, te_img_list, num_classes):
    temp = {}
    for i in tr_img_list:
        img = i[:-7]
        if img not in temp:
            temp[img] = 1
        elif img in temp:
            temp[img] += 1
    for i in te_img_list:
        img = i[:-7]
        if img not in temp:
            temp[img] = 1
        elif img in temp:
            temp[img] += 1

    li = []
    for i in temp:
        if i not in li and temp[i] >= 60:
            li.append(i)

    lbl_dict = {}
    for i in range(len(num_classes)):
        # lbl_dict[li[num_classes[i]]] = i for reset the label
        lbl_dict[li[num_classes[i]]] = num_classes[i]

    tr_data_li = []
    for i in tr_img_list:
        img = i[:-7]
        if img in list(lbl_dict.keys()):
            tr_data_li.append(i)
    
    te_data_li = []
    for i in te_img_list:
        img = i[:-7]
        if img in list(lbl_dict.keys()):
            te_data_li.append(i)

    return lbl_dict, tr_data_li, te_data_li

def check_str_dict():    
    chk_li = ['unk']
    
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

def assemble(
    data_path = '../data/pwd',
    file_name = '0'):

    """delete useless data"""
    df = pd.read_csv(os.path.join(data_path, file_name), sep='\n', names=["pw"])
    df = df[~df.pw.str.contains(' ')]

    """split id and pw"""
    df[['id', 'pw']] = df['pw'].str.split('@', 1, expand=True)
    df['pw'] = df['pw'].str.split(':').str[1]
    df = df.reset_index(drop=True)

    strings = '[A-Za-z0-9-=+,#/\?ᆢ:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]+'
    filter = df['pw'].str.fullmatch(strings)
    df = df[filter]

    return df

class Password_1(Dataset):
    def __init__(self, num_classes, tr_chk, p_len, transforms=None):
        tr_root_path = '../data/train'
        tr_img_list = os.listdir(tr_root_path)
        te_root_path = '../data/test'
        te_img_list = os.listdir(te_root_path)

        lbl_dict, tr_data_li, te_data_li = get_labels(tr_img_list, te_img_list, num_classes)

        if tr_chk:
            self.data_li = tr_data_li
            self.img_path = tr_root_path
        else:
            self.img_path = te_root_path
            self.data_li = []

            chk_li = []
            for i in te_data_li:
                if i[:-7] not in chk_li:
                    self.data_li.append(i)
                    chk_li.append(i[:-7])

        self.lbl_dict = lbl_dict
        self.str_dict = check_str_dict()
        self.df = assemble()
        self.p_len = p_len
        self.transforms = transforms

    def __len__(self):
        return len(self.data_li)

    def __getitem__(self, index):
        img_root = os.path.join(self.img_path, self.data_li[index])
        img = cv2.imread(img_root)
        pwd = random.choice(self.df.loc[self.df['id'] == self.data_li[index][:-7]]['pw'].values.tolist())
        pwd = self.make_pwd(pwd).long()

        if self.transforms:
            img = self.transforms(img)

        return img, pwd

    def make_pwd(self, pwd):
        pwd_torch = torch.zeros((self.p_len))
        if len(pwd) > self.p_len:
            for i in range(self.p_len):
                pwd_torch[i] = self.str_dict[pwd[i]]
        else:
            for i in range(len(pwd)):
                pwd_torch[i] = self.str_dict[pwd[i]]
        return pwd_torch

class Password_10(Dataset):
    def __init__(self, num_classes, tr_chk, p_len, transforms=None):
        tr_root_path = '../data/train'
        tr_img_list = os.listdir(tr_root_path)
        te_root_path = '../data/test'
        te_img_list = os.listdir(te_root_path)

        lbl_dict, tr_data_li, te_data_li = get_labels(tr_img_list, te_img_list, num_classes)

        if tr_chk:
            self.data_li = tr_data_li
            self.img_path = tr_root_path
        else:
            self.img_path = te_root_path
            self.data_li = []

            chk_dic = {}
            for i in te_data_li:
                if i[:-7] not in chk_dic:
                    self.data_li.append(i)
                    chk_dic[i[:-7]] = 1
                elif chk_dic[i[:-7]] < 10:
                    self.data_li.append(i)
                    chk_dic[i[:-7]] += 1

        self.lbl_dict = lbl_dict
        self.str_dict = check_str_dict()
        self.df = assemble()
        self.p_len = p_len
        self.transforms = transforms

    def __len__(self):
        return len(self.data_li)

    def __getitem__(self, index):
        img_root = os.path.join(self.img_path, self.data_li[index])
        img = cv2.imread(img_root)
        pwd = random.choice(self.df.loc[self.df['id'] == self.data_li[index][:-7]]['pw'].values.tolist())
        pwd = self.make_pwd(pwd).long()

        if self.transforms:
            img = self.transforms(img)

        return img, pwd

    def make_pwd(self, pwd):
        pwd_torch = torch.zeros((self.p_len))
        if len(pwd) > self.p_len:
            for i in range(self.p_len):
                pwd_torch[i] = self.str_dict[pwd[i]]
        else:
            for i in range(len(pwd)):
                pwd_torch[i] = self.str_dict[pwd[i]]
        return pwd_torch

class Password_20(Dataset):
    def __init__(self, num_classes, tr_chk, p_len, transforms=None):
        tr_root_path = '../data/train'
        tr_img_list = os.listdir(tr_root_path)
        te_root_path = '../data/test'
        te_img_list = os.listdir(te_root_path)

        lbl_dict, tr_data_li, te_data_li = get_labels(tr_img_list, te_img_list, num_classes)

        if tr_chk:
            self.data_li = tr_data_li
            self.img_path = tr_root_path
        else:
            self.img_path = te_root_path
            self.data_li = []

            chk_dic = {}
            for i in te_data_li:
                if i[:-7] not in chk_dic:
                    self.data_li.append(i)
                    chk_dic[i[:-7]] = 1
                elif chk_dic[i[:-7]] < 20:
                    self.data_li.append(i)
                    chk_dic[i[:-7]] += 1

            # 30
            # self.data_li = te_data_li

        self.lbl_dict = lbl_dict
        self.str_dict = check_str_dict()
        self.df = assemble()
        self.p_len = p_len
        self.transforms = transforms

    def __len__(self):
        return len(self.data_li)

    def __getitem__(self, index):
        img_root = os.path.join(self.img_path, self.data_li[index])
        img = cv2.imread(img_root)
        pwd = random.choice(self.df.loc[self.df['id'] == self.data_li[index][:-7]]['pw'].values.tolist())
        pwd = self.make_pwd(pwd).long()

        if self.transforms:
            img = self.transforms(img)

        return img, pwd

    def make_pwd(self, pwd):
        pwd_torch = torch.zeros((self.p_len))
        if len(pwd) > self.p_len:
            for i in range(self.p_len):
                pwd_torch[i] = self.str_dict[pwd[i]]
        else:
            for i in range(len(pwd)):
                pwd_torch[i] = self.str_dict[pwd[i]]
        return pwd_torch

class Password_30(Dataset):
    def __init__(self, num_classes, tr_chk, p_len, transforms=None):
        tr_root_path = '../data/train'
        tr_img_list = os.listdir(tr_root_path)
        te_root_path = '../data/test'
        te_img_list = os.listdir(te_root_path)

        lbl_dict, tr_data_li, te_data_li = get_labels(tr_img_list, te_img_list, num_classes)

        if tr_chk:
            self.data_li = tr_data_li
            self.img_path = tr_root_path
        else:
            self.img_path = te_root_path
            self.data_li = []

            self.data_li = te_data_li

        self.lbl_dict = lbl_dict
        self.str_dict = check_str_dict()
        self.df = assemble()
        self.p_len = p_len
        self.transforms = transforms

    def __len__(self):
        return len(self.data_li)

    def __getitem__(self, index):
        img_root = os.path.join(self.img_path, self.data_li[index])
        img = cv2.imread(img_root)
        pwd = random.choice(self.df.loc[self.df['id'] == self.data_li[index][:-7]]['pw'].values.tolist())
        pwd = self.make_pwd(pwd).long()

        if self.transforms:
            img = self.transforms(img)

        return img, pwd

    def make_pwd(self, pwd):
        pwd_torch = torch.zeros((self.p_len))
        if len(pwd) > self.p_len:
            for i in range(self.p_len):
                pwd_torch[i] = self.str_dict[pwd[i]]
        else:
            for i in range(len(pwd)):
                pwd_torch[i] = self.str_dict[pwd[i]]
        return pwd_torch

class Password(Dataset):
    def __init__(self, num_classes, tr_chk, p_len, transforms=None):
        tr_root_path = '../data/train'
        tr_img_list = os.listdir(tr_root_path)
        te_root_path = '../data/train'
        te_img_list = os.listdir(te_root_path)
        lbl_dict, tr_data_li, te_data_li = get_labels(tr_img_list, te_img_list, num_classes)

        if tr_chk:
            self.data_li = tr_data_li
            self.img_path = tr_root_path
        else:
            self.img_path = te_root_path
            self.data_li = []

            self.data_li = te_data_li

        self.lbl_dict = lbl_dict
        self.str_dict = check_str_dict()
        self.df = assemble()
        self.p_len = p_len
        self.transforms = transforms

    def __len__(self):
        return len(self.data_li)

    def __getitem__(self, index):
        img_root = os.path.join(self.img_path, self.data_li[index])
        img = cv2.imread(img_root)
        pwd = self.df.loc[self.df['id'] == self.data_li[index][:-7]]['pw'].values.tolist()[0]
        dis_input = self.make_pwd_dis(pwd).float()
        pwd = self.make_pwd(pwd).long()

        if self.transforms:
            img = self.transforms(img)

        return img, pwd, dis_input

    def make_pwd(self, pwd):
        pwd_torch = torch.zeros((self.p_len))
        if len(pwd) > self.p_len:
            for i in range(self.p_len):
                pwd_torch[i] = self.str_dict[pwd[i]]
        else:
            for i in range(len(pwd)):
                pwd_torch[i] = self.str_dict[pwd[i]]
        return pwd_torch

    def make_pwd_dis(self, pwd):
        pwd_torch = torch.zeros((self.p_len, 101))
        if len(pwd) > self.p_len:
            for i in range(self.p_len):
                pwd_torch[i][self.str_dict[pwd[i]]] = 1
        else:
            for i in range(len(pwd)):
                pwd_torch[i][self.str_dict[pwd[i]]] = 1
            for i in range(len(pwd), self.p_len):
                pwd_torch[i][0] = 1
        return pwd_torch