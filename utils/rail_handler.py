import numpy as np
import queue


def euclidean_distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


def get_rail_from_mask(mask):
    print(mask.shape)

    camera = np.zeros([2])
    camera[0] = mask.shape[0]
    camera[1] = mask.shape[1] // 2
    start = np.zeros([2])
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if np.sum(mask[i][j]) > 0:
                if np.sum(mask[int(start[0])][int(start[1])]) == 0 or euclidean_distance(camera,
                                                                                         [i, j]) < euclidean_distance(
                        camera, start):
                    start = [i, j]
    print('start =', start)

    new_mask = np.zeros(mask.shape)
    new_mask[int(start[0])][int(start[1])] = mask[int(start[0])][int(start[1])]
    q = queue.Queue()
    q.put([int(start[0]), int(start[1])])
    while not q.empty():
        front = q.get()
        neighbors = []
        if front[0] > 0:
            neighbors.append([front[0] - 1, front[1]])
        if front[1] > 0:
            neighbors.append([front[0], front[1] - 1])
        if front[1] < mask.shape[1]:
            neighbors.append([front[0], front[1] + 1])
        for neighbor in neighbors:
            if np.sum(mask[neighbor[0]][neighbor[1]]) > 0 and np.sum(new_mask[neighbor[0]][neighbor[1]]) == 0:
                new_mask[neighbor[0]][neighbor[1]] = mask[neighbor[0]][neighbor[1]]
                q.put(neighbor)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.title('display')
    plt.subplot(211)
    plt.imshow(mask)
    plt.subplot(212)
    plt.imshow(new_mask)
    plt.show()
