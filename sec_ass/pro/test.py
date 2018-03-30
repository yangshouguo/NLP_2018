import numpy as np

def split_dataset():
    x = np.random.rand(100,5)
    np.random.shuffle(x)

    training, test = x[:80,:] , x[80:,:]

    print(test)
    pass
def dict_test():
    dict = {}

    if 'N' not in dict:
        dict['N']['S'] = 1
    else:
        dict['N'] += 1

    print(dict)

def set_test():

    ll = 'asdasd'
    print(ll[0])

def logtest():
    import math

    print(math.log(math.e))

    print(math.pow(math.e, -5.0998))

def list_reverse():
    li = ['1','2',3,4,5,6]
    li.reverse()
    print(li)


def split_from_end(str, spliter='/'):
    pos = str.rfind(spliter)
    return str[:pos], str[pos + 1:]

if __name__ == '__main__':
    x = [1,2,3]
    print(x[0:0])