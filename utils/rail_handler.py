import numpy as np
import queue


def euclidean_distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


def get_new_mask(mask):
    camera = np.zeros([2])
    camera[0] = mask.shape[0]
    camera[1] = mask.shape[1] // 2
    start = np.zeros([2])
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if np.sum(mask[i][j]) > 0:
                if np.sum(mask[int(start[0])][int(start[1])]) == 0 or \
                        euclidean_distance(camera, [i, j]) < euclidean_distance(camera, start):
                    start = [i, j]

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
        if front[1] + 1 < mask.shape[1]:
            neighbors.append([front[0], front[1] + 1])
        for neighbor in neighbors:
            if np.sum(mask[neighbor[0]][neighbor[1]]) > 0 and \
                    np.sum(new_mask[neighbor[0]][neighbor[1]]) == 0:
                new_mask[neighbor[0]][neighbor[1]] = mask[neighbor[0]][neighbor[1]]
                q.put(neighbor)

    return new_mask


def check_curve(curve):
    if len(curve) < 20:
        return 'unknown'
    start_direction = curve[4] - curve[0] + curve[9] - curve[5]
    lmt = np.cos(np.pi / 12)
    for i in range(10, len(curve) - 10, 10):
        direction = curve[i + 4] - curve[i] + curve[i + 9] - curve[i + 5]
        cos_alpha = np.dot(direction, start_direction) / (np.linalg.norm(direction) * np.linalg.norm(start_direction))
        if cos_alpha < lmt:
            if np.cross(direction, start_direction) > 0:
                return 'left'
            else:
                return 'right'
    return 'straight'


def get_rail_from_mask(mask):
    new_mask = get_new_mask(mask)

    curve = []
    for i in reversed(range(mask.shape[0])):
        min_x = -1
        for j in range(mask.shape[1]):
            if np.sum(new_mask[i][j]) > 0:
                min_x = j
                break
        if min_x == -1:
            continue
        max_x = -1
        for j in reversed(range(mask.shape[1])):
            if np.sum(new_mask[i][j]) > 0:
                max_x = j
                break
        mid_x = (min_x + max_x) // 2
        new_mask[i][mid_x] = [255, 255, 255]
        curve.append(np.array([(min_x + max_x) / 2, i]))

    return new_mask, check_curve(curve)
