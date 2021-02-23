#encoding=utf-8
#author: liang xinyan
#email: liangxinyan48@163.com
import os
import copy
import utils
import random
from config import get_configs
paras = get_configs()
nb_fusion_way = len(paras['fusion_ways'])
nb_view = paras['nb_view']
is_remove = paras['is_remove']

def quchong(p):
    views_p = utils.get_nb_view_by_individal_code(p)
    view_code = p[:views_p]

    view_code1, fusion_code1 = [], []
    for k, v in enumerate(view_code):
        if v not in view_code1:
            view_code1.append(v)
            if k != 0:
                fusion_code1.append(p[k+views_p-1])
    pp = view_code1+fusion_code1
    return pp


def crossover(p_1, p_2, crossover_rate, is_remove=is_remove, max_len=40):
    p1 = copy.deepcopy(p_1)
    p2 = copy.deepcopy(p_2)
    views_p1 = utils.get_nb_view_by_individal_code(p1)
    views_p2 = utils.get_nb_view_by_individal_code(p2)
    # print(views_p1, views_p2)
    r = random.random()
    if r < crossover_rate:
        co_i = random.randint(0, views_p1-1)
        co_j = random.randint(0, views_p2-1)
        o1 = p1[:co_i+1] + p2[co_j+1:views_p2] + p1[views_p1: views_p1+co_i] + p2[views_p2+co_j:]
        o2 = p2[:co_j+1] + p1[co_i+1:views_p1] + p2[views_p2: views_p2+co_j] + p1[views_p1+co_i:]
        if is_remove:# remove出现重复视图
            o1 = quchong(o1)
            o2 = quchong(o2)
        else:
            if len(o1) > max_len:
                o1 = quchong(o1)  # remove出现重复视图
            if len(o2) > max_len:
                o2 = quchong(o2)  # remove出现重复视图
        return o1, o2
    else:
        return p1, p2


def mutation(p1, mutation_rate, is_remove=is_remove, max_len=40):
    p = copy.deepcopy(p1)
    views_p = utils.get_nb_view_by_individal_code(p)
    len_code = len(p)
    r = random.random()
    if r < mutation_rate:
        i = random.randint(0, len_code-1)
        if i < views_p:
            # 变异点位于视图编码，进行视图变异
            mutation_view = list(range(nb_view))
            # mutation_view.remove(p[i])   #会出现重复视图
            mutation_view.remove(p[i])
            p[i] = random.choice(mutation_view)
        else:
            mutation_view = list(range(nb_fusion_way))
            mutation_view.remove(p[i])
            p[i] = random.choice(mutation_view)
    if is_remove:
        p = quchong(p)  # remove出现重复视图
    else:
        if len(p) > max_len:
            p = quchong(p)  # remove出现重复视图
    return p


def selection(P_t, Q_t):
    shared_code_acc = utils.load_result()
    # print(f'P_t: {P_t}')
    # print(f'Q_t: {Q_t}')
    # print(f'f: {shared_code_acc}')

    def select_p1(select_pool):
        two = random.sample(range(len(select_pool)), 2)
        a1 = '-'.join([str(i) for i in select_pool[two[0]]])
        a2 = '-'.join([str(i) for i in select_pool[two[1]]])
        p1 = select_pool[two[0]] if shared_code_acc[a1] > shared_code_acc[a2] else select_pool[two[1]]
        return p1
    P_t1 = []
    Pt_Qt = P_t+Q_t
    while len(P_t1) < len(P_t):
        p = select_p1(Pt_Qt)
        P_t1.append(p)

    # 如果最好的个体不在P_t1，用最好的替换最差的
    max_code = []
    for k, v in shared_code_acc.items():
        if v == max(shared_code_acc.values()):
            max_code_str = k
            max_code = k.strip().split('-')
            max_code = [int(i) for i in max_code]
        if v == min(shared_code_acc.values()):
            min_code_str = k

    is_max = False
    for i, v in enumerate(P_t1):
        v_str = utils.list2str(v)
        if v_str == max_code_str:
            is_max = True
            break
    if not is_max:
        min_i = 0
        for i, v in enumerate(P_t1):
            v_str = utils.list2str(v)
            if v_str == min_code_str:
                min_i = i
                break
        P_t1[min_i] = max_code
    return P_t1


def gen_offspring(P_t):
    shared_code_acc = utils.load_result()
    # print(shared_code_acc)
    # 1. Crossover
    def select_p():
        two = random.sample(range(len(P_t)), 2)
        a1 = '-'.join([str(i) for i in P_t[two[0]]])
        a2 = '-'.join([str(i) for i in P_t[two[1]]])
        p1 = P_t[two[0]] if shared_code_acc[a1] > shared_code_acc[a2] else P_t[two[1]]
        return p1
    Q_t = []
    while len(Q_t) < len(P_t):
        p1 = select_p()
        p2 = select_p()
        while '-'.join(str(i) for i in p1) == '-'.join(str(i) for i in p2):
            p2 = select_p()
        o1, o2 = crossover(p_1=p1, p_2=p2, crossover_rate=paras['crossover_rate'])
        Q_t.append(o1)
        Q_t.append(o2)
    # 2. Mutation
    Q_tt = []
    for p in Q_t:
        p1 = mutation(p1=p, mutation_rate=paras['mutation_rate'])
        Q_tt.append(p1)
    Q_t = Q_tt
    return Q_t


# crossover(p1=[2,3,6,1,1], p2=[2,5,4,2,1], crossover_rate=1.0)
# mutation(p=[2, 3, 4, 1, 1], mutation_rate=1.0)
# p = quchong(p=[2, 3, 3, 4, 4, 1, 2, 3, 4])
# print(p)