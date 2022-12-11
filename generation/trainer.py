import torch
from torch import nn
import numpy as np

def val_epoch(generator, te_loader, tr_chk=False, device="cuda"):
    generator.eval()
    acc = 0
    cnt = 0
    our_total_acc = 0
    with torch.no_grad():
        if tr_chk:
            for i, (img, pwd, _) in enumerate(te_loader):
                img, pwd = img.to(device), pwd.to(device)

                out = generator(img)

                _, pred = torch.max(out, 1)
                wer_acc, our_acc = compute_acc(pred, pwd)
                acc += wer_acc
                our_total_acc += our_acc
                cnt += 1
        else:
            for i, (img, pwd) in enumerate(te_loader):
                img, pwd = img.to(device), pwd.to(device)

                out = generator(img)

                _, pred = torch.max(out, 1)
                wer_acc, our_acc = compute_acc(pred, pwd)
                acc += wer_acc
                our_total_acc += our_acc
                cnt += 1

    val_acc = our_total_acc / cnt
    wer_acc = acc / cnt

    return val_acc

def compute_acc(preds, labels, costs=(7, 7, 10)):
    # cost according to HTK: http://www.ee.columbia.edu/~dpwe/LabROSA/doc/HTKBook21/node142.html
    if not len(preds) == len(labels):
        raise ValueError('# predictions not equal to # labels')

    correct = 0
    total = len(preds)
    Ns, Ds, Ss, Is = 0, 0, 0, 0
    for i, _ in enumerate(preds):
        H, D, S, I = iterative_levenshtein(preds[i], labels[i], costs)
        Ns += len(labels[i])
        Ds += D
        Ss += S
        Is += I
        if D == 0 and S == 0 and I == 0:
            correct += 1
    try:
        acc = 100 * (Ns - Ds - Ss - Is) / Ns
        our_acc = 100 * correct / total
    except ZeroDivisionError:
        raise ZeroDivisionError('Empty labels')
    return acc, our_acc

def iterative_levenshtein(s, t, costs=(1, 1, 1)):
    """
    Computes Levenshtein distance between the strings s and t.
    For all i and j, dist[i,j] will contain the Levenshtein
    distance between the first i characters of s and the
    first j characters of t

    s: source, t: target
    costs: a tuple or a list with three integers (d, i, s)
           where d defines the costs for a deletion
                 i defines the costs for an insertion and
                 s defines the costs for a substitution
    return:
    H, S, D, I: correct chars, number of substitutions, number of deletions, number of insertions
    """

    rows = len(s) + 1
    cols = len(t) + 1
    deletes, inserts, substitutes = costs

    dist = [[0 for x in range(cols)] for x in range(rows)]
    H, D, S, I = 0, 0, 0, 0
    for row in range(1, rows):
        dist[row][0] = row * deletes
    for col in range(1, cols):
        dist[0][col] = col * inserts

    for col in range(1, cols):
        for row in range(1, rows):
            if s[row - 1] == t[col - 1]:
                cost = 0
            else:
                cost = substitutes
            dist[row][col] = min(dist[row - 1][col] + deletes,
                                 dist[row][col - 1] + inserts,
                                 dist[row - 1][col - 1] + cost)
    row, col = rows - 1, cols - 1
    while row != 0 or col != 0:
        if row == 0:
            I += col
            col = 0
        elif col == 0:
            D += row
            row = 0
        elif dist[row][col] == dist[row - 1][col] + deletes:
            D += 1
            row = row - 1
        elif dist[row][col] == dist[row][col - 1] + inserts:
            I += 1
            col = col - 1
        elif dist[row][col] == dist[row - 1][col - 1] + substitutes:
            S += 1
            row, col = row - 1, col - 1
        else:
            H += 1
            row, col = row - 1, col - 1
    D, I = I, D
    return H, D, S, I

def run_epoch(generator, discriminator, opt_gen, opt_dis, tr_loader, device="cuda"):
    generator.train()
    discriminator.train()
    bce_loss = nn.BCEWithLogitsLoss()
    crs_loss = nn.CrossEntropyLoss()
    g_loss_li = []
    d_loss_li = []
    for i, (img, pwd, dis_input) in enumerate(tr_loader):
        img, pwd = img.to(device), pwd.to(device)
        dis_input = dis_input.to(device)
        y_real = torch.Tensor(img.shape[0], 1).fill_(1.0).to(device)
        y_fake = torch.Tensor(img.shape[0], 1).fill_(0.0).to(device)

        discriminator.zero_grad()
        real_output = discriminator(dis_input)
        d_loss_real = bce_loss(real_output, y_real)
        d_loss_real.backward()

        fake = generator(img)
        fake_output = discriminator(fake.detach())
        d_loss_fake = bce_loss(fake_output, y_fake)
        d_loss_fake.backward()

        d_loss = d_loss_real.item() + d_loss_fake.item()
        opt_dis.step()
        
        opt_gen.zero_grad()

        g_loss_2 = crs_loss(fake, pwd)
        g_loss_2.backward()
        opt_gen.step()

        fake_output = discriminator(fake)
        g_loss_3 = bce_loss(fake_output, y_real)
        
        g_loss = g_loss_2 + g_loss_3

        g_loss = g_loss_2.item() + g_loss_3.item()

        g_loss_li.append(g_loss)
        d_loss_li.append(d_loss)

    return np.mean(g_loss_li), np.mean(d_loss_li)

def decode(seq, reverse=False):
    chk = check_str_dict()
    inv_map = {v: k for k, v in chk.items()}
    li = []
    for j in seq:
        li.append(inv_map[int(j)])
    return li

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