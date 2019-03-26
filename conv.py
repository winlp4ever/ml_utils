def convolution(x: list, k: list) -> list:
    n = int(len(x) ** 0.5)
    l = int(len(k) ** 0.5)
    ans = []
    # nb of zeros padding at each side
    padding = l // 2
    # new size of a padded matrix
    new_size = n + 2 * padding
    # generate a zeros matrix (for zeros padding) of this size
    arr = [0] * new_size ** 2

    # copy x to this new matrix
    for id in range(len(x)):
        i, j = id // n, id % n
        arr[(i + padding) * (n + 2 * padding) + j + padding] = x[id]

    # compute conv products over this new matrix and store results in ans
    for id in range(len(arr)):
        i, j = id //  new_size, id % new_size
        if i >= n or j >= n:
            continue
        aij = 0
        for k_id in range(len(k)):
            w, s = k_id // l, k_id % l
            aij += arr[(i + w) * new_size + (j + s)] * k[k_id]
        ans.append(aij)
    return ans

if __name__ == '__main__':
    x = [1, 2, 2, 3]
    k = [3]
    print(convolution(x, k))
    x = [1, 2, 3, 4, 5] * 5
    k = [0, 0, 0, 0, 1, 0, 0, 0, 0]
    print(convolution(x, k))
