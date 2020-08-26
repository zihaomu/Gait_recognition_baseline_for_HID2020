import random

def randint(num_a, num_b):
    return random.randint(num_a,num_b)

def random_clip(list,k = 1):
    # input continous sequence and return random subsequences

    len_list = len(list)

    if len_list>=k:
        can_use_num = len_list - k
    else:
        s = [i for i in range(len_list)]
        for i in range(k - len_list):
            s.append(len_list-1)
        return s

    random_num = random.randint(0,can_use_num)
    s = []

    for i in range(k):
        s.append(random_num+i)
    # print(len(s))
    return s

def random_select(list,k = 1):
    # input continous sequence and return random subset.
    s = []
    for i in range(k):
        s.append(random.choice(list))
    # print(len(s))
    return s