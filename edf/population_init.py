#encoding=utf-8
#author: liang xinyan
#email: liangxinyan48@163.com
#encoding=utf-8
#author: liang xinyan
#email: liangxinyan48@163.com
import numpy as np
import random
import config
import utils


def generate_population(views=10, pop_size=10, verbose=0):
    '''
    种群初始化
    :param views: 视图个数
    :param pop_size: 种群大小
    :return:
    '''
    fusion_ways = config.get_configs()['fusion_ways']
    population = []
    population_set = set()
    while len(population) < pop_size:
    # for i in range(pop_size):
        # view_code at least contains two elements
        view_code = random.sample(range(0, views), k=random.randint(2, views))
        fusion_code = random.choices(range(0, len(fusion_ways)), k=len(view_code)-1)
        pop = view_code+fusion_code
        if verbose == 1:
            print(f'view_code:{view_code}')
            print(f'fusion_code:{fusion_code}')
            print(f'pop:{pop}')
            print('='*30)
        if utils.list2str(pop) not in population_set:
            population.append(pop)
            population_set.add(utils.list2str(pop))
    return population


if __name__ == '__main__':
    population = generate_population()
    # for i in population:
    #     print(i)
