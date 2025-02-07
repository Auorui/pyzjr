import random

def rand(a=0.0, b=1.0):
    """生成在指定范围内的随机浮点数,进行缩放和偏移来映射到[a, b)的范围。"""
    random_value = random.random()
    scaled_value = a + (b - a) * random_value
    return scaled_value

def randint(a=0, b=1):
    """生成一个指定范围内的随机整数，在 [a, b] 范围内的随机整数。"""
    return random.randint(a, b)

def randlist(size, a=0.0, b=1.0):
    """生成指定数量的随机浮点数列表，范围在 [a, b) 内。"""
    return [rand(a, b) for _ in range(size)]

def randintlist(size, a=0, b=1):
    """生成指定数量的随机整数列表，范围在 [a, b] 内。"""
    return [randint(a, b) for _ in range(size)]

def randbool():
    """生成一个随机的布尔值。"""
    return random.choice([True, False])

def randshuffle(lst):
    """随机打乱列表中的元素顺序。"""
    return random.shuffle(lst)

def randnormal(mu=0.0, sigma=1.0):
    """生成符合正态分布的随机数。"""
    return random.gauss(mu, sigma)

def randstring(length=8):
    """生成指定长度的随机字符串。"""
    characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return ''.join(random.choice(characters) for _ in range(length))

def randfluctuate(a=0.0, b=1.0, fluctuation=0.1):
    """生成指定区间内，带有波动的随机数。"""
    return rand(a - fluctuation, b + fluctuation)

def rand_choice_weighted(choices, weights):
    """根据给定的权重，从选项中随机选择一个。"""
    return random.choices(choices, weights=weights, k=1)[0]

def randstep(a=0.0, b=1.0, step=0.1):
    """生成指定步长范围内的随机浮点数。"""
    num_steps = int((b - a) // step)
    random_step = random.randint(0, num_steps)
    return a + random_step * step

if __name__ == "__main__":
    # 测试 rand() 函数
    print("rand() test:")
    for _ in range(5):
        print(rand(1.0, 5.0))

    # 测试 randint() 函数
    print("\nrandint() test:")
    for _ in range(5):
        print(randint(10, 20))

    # 测试 randlist() 函数
    print("\nrandlist() test:")
    print(randlist(5, 1.0, 10.0))

    # 测试 randintlist() 函数
    print("\nrandintlist() test:")
    print(randintlist(5, 1, 100))

    # 测试 randbool() 函数
    print("\nrandbool() test:")
    for _ in range(5):
        print(randbool())

    # 测试 randshuffle() 函数
    print("\nrandshuffle() test:")
    my_list = [1, 2, 3, 4, 5]
    randshuffle(my_list)
    print(my_list)

    # 测试 randnormal() 函数
    print("\nrandnormal() test:")
    for _ in range(5):
        print(randnormal(0.0, 1.0))

    # 测试 randstring() 函数
    print("\nrandstring() test:")
    for _ in range(5):
        print(randstring(8))

    # 测试 randfluctuate() 函数
    print("\nrandfluctuate() test:")
    for _ in range(5):
        print(randfluctuate(10.0, 20.0, 2.0))

    # 测试 rand_choice_weighted() 函数
    print("\nrand_choice_weighted() test:")
    choices = ['apple', 'banana', 'cherry']
    weights = [0.1, 0.3, 0.6]
    for _ in range(5):
        print(rand_choice_weighted(choices, weights))

    # 测试 randstep() 函数
    print("\nrandstep() test:")
    for _ in range(5):
        print(randstep(1.0, 10.0, 0.5))



