import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from math import sqrt
from icecream import ic
import numpy as np
import random

limiter_x = 70
limiter_y = 40


def visualize(coords):
    global limiter_x, limiter_y
    fig, ax = plt.subplots()

    ax.axhline(y=0, color='lightgrey', zorder=1)
    ax.axvline(x=0, color='lightgrey', zorder=1)

    ax.set_xlim(-limiter_x, limiter_x)
    ax.set_ylim(-limiter_y, limiter_y)

    ax.annotate('', xy=(limiter_x, 0), xytext=(limiter_x - 0.2, 0),
                arrowprops=dict(arrowstyle='->', color='grey'))
    ax.annotate('', xy=(0, limiter_y), xytext=(0, limiter_y - 0.2),
                arrowprops=dict(arrowstyle='->', color='grey'))
    ax.set_aspect('equal')  # потому что изначально масштабность х и у не одинакова

    for pair in coords:
        obj = Polygon(pair, closed=True, facecolor='none', edgecolor='black', zorder=2)
        ax.add_patch(obj)

    ax.set_axis_off()
    plt.tight_layout()

    plt.show()


def tr_translate_2(delta_x, delta_y):
    def decorator(func):
        def wrapper(*args, **kwargs):
            polygons = list(func(*args, **kwargs))

            return tuple(map(lambda p: tr_translate(p, (delta_x, delta_y)), polygons))

        return wrapper

    return decorator


def tr_rotate_2(angle):
    def decorator(func):
        def wrapper(*args, **kwargs):
            polygons = list(func(*args, **kwargs))

            return tuple(map(lambda p: tr_rotate(p, angle), polygons))

        return wrapper

    return decorator


def tr_symmetry_2(from_x=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            polygons = list(func(*args, **kwargs))

            return tuple(map(lambda p: tr_symmetry(p, from_x), polygons))

        return wrapper

    return decorator


def tr_homothety_2(k):
    def decorator(func):
        def wrapper(*args, **kwargs):
            polygons = list(func(*args, **kwargs))

            return tuple(map(lambda p: tr_homothety(p, k), polygons))

        return wrapper

    return decorator


def flt_short_side_2(side):
    def decorator(func):
        def wrapper(*args, **kwargs):
            polygons = list(func(*args, **kwargs))

            return tuple(filter(lambda p: flt_short_side(p, side), polygons))

        return wrapper

    return decorator


# @tr_rotate_2(45)
# @tr_homothety_2(4)
def gen_rectangle(a=3, b=5, st=-limiter_x, end=limiter_x):
    global limiter_x

    if st < -limiter_x or end > limiter_x:
        raise ValueError('Границы генерации больше графика')

    return (((c, 0), (c, a), (c + b, a), (c + b, 0)) for c in range(st, end - b, b + 1))


# @tr_translate_2(1, 6)
def gen_triangle(a=6, st=-limiter_x, end=limiter_x):
    global limiter_x

    if st < -limiter_x or end > limiter_x:
        raise ValueError('Границы генерации больше графика')

    return (((c, 0), (a / 2 + c, a * (3 ** 0.5) / 2), (c + a, 0))
            for c in range(st + 1, end - a, a + 1))


# @tr_symmetry_2(from_x=False)
def gen_hexagon(a=3, st=-limiter_x, end=limiter_x):
    global limiter_x

    if st < -limiter_x or end > limiter_x:
        raise ValueError('Границы генерации больше графика')

    return (((c, 0), (-a / 2 + c, a * (3 ** 0.5) / 2), (c, a * sqrt(3)),
             (c + a, a * sqrt(3)), (1.5 * a + c, a * (3 ** 0.5) / 2),
             (a + c, 0)) for c in range(st + a, end - a, a * 2 + 1))


def gen_trapezoid(a=2, b=5, h=5, st=-limiter_x, end=limiter_x):
    global limiter_x

    if st < -limiter_x or end > limiter_x:
        raise ValueError('Границы генерации больше графика')

    return (((c, 0), (c + abs(b - a) / 2, h), (c + abs(b - a) / 2 + a, h),
             (c + b, 0)) for c in range(st + b, end - b, b * 2 - 2))


@flt_short_side_2(4)
def random_figures(tr_a=3, rec_a=5, hex_a=7):
    trs = list(gen_triangle(a=tr_a))
    recs = list(gen_rectangle(a=rec_a))
    hexs = list(gen_hexagon(a=hex_a))

    # shuffled = itertools.permutations(trs + recs + hexs, 7)
    # random_pair = random.randint(0, 100000)
    #
    # for i, val in enumerate(shuffled):
    #
    #     if i == random_pair:
    #         return val

    mixed = random.sample(trs + recs + hexs, 7)
    mixed = sorted(mixed)
    closer = []
    fig_space = {3: tr_a / 2, 4: rec_a / 2 + 1, 6: hex_a + 1}

    for i, pair in enumerate(mixed):

        if i == 0:
            pair_end = pair[-1]
            closer.append(mixed[0])

        else:
            delta = pair[0][0] - pair_end[0]

            pair = list(pair)
            temp = []
            for j in range(len(pair)):
                x, y = pair[j]
                x -= delta
                temp.append((x + fig_space[len(pair)], y))

            pair_end = (x + fig_space[len(pair)], y)

            closer.append(tuple(temp))

    return closer


def tr_translate(figure, vect):
    return tuple([(coord[0] + vect[0], coord[1] + vect[1]) for coord in figure])


def tr_rotate(figure, angle, around_x=True):
    a_rad = np.radians(angle)
    rotation_matrix = np.array([[np.cos(a_rad), -np.sin(a_rad)],
                                [np.sin(a_rad), np.cos(a_rad)]])

    if around_x:
        rotated = []

        for polygon in figure:
            rotated_vertex = np.dot(rotation_matrix, polygon)  # перемножаем полигон и матрицу вращения
            rotated.append(tuple(rotated_vertex))

        return tuple(rotated)

    else:
        figure = np.array(figure)
        centre = np.mean(figure)
        tr_polygon = figure - centre

        rotated = np.dot(tr_polygon, rotation_matrix)
        rotated += centre

        return rotated.tolist()


def tr_symmetry(coords, from_x=True):
    symmetric = []
    if from_x:

        for coord in coords:
            x, y = coord
            symmetric.append((x, -y))

    else:

        for coord in coords:
            x, y = coord
            symmetric.append((-x, -y))

    return tuple(symmetric)


def tr_homothety(coords, k):
    return [(coord[0] * k, coord[1] * k) for coord in coords]


def gen_three():
    center = list(gen_triangle())
    center = list(map(lambda p: tr_rotate(p, 30), center))

    up = center.copy()
    down = center.copy()

    up = list(map(lambda p: tr_translate(p, (-1, 5)), up))
    down = list(map(lambda p: tr_translate(p, (2, -5)), down))

    visualize(up + down + center)


def gen_two():
    l1 = list(gen_rectangle())
    l2 = list(gen_rectangle())

    l1 = list(map(lambda p: tr_rotate(p, 35), l1))
    l2 = list(map(lambda p: tr_rotate(p, 150), l2))

    l1 = list(map(lambda p: tr_translate(p, (-3, 2)), l1))
    l2 = list(map(lambda p: tr_translate(p, (1, 0)), l2))

    visualize(l1 + l2)


def two_symmetric():
    t = list(gen_triangle())
    ts = t.copy()

    ts = list(map(lambda p: tr_symmetry(p), ts))
    ts = list(map(lambda p: tr_translate(p, (0, 12)), ts))

    visualize(t + ts)


def tr_scale():
    t = list(gen_trapezoid(a=3, b=5, h=6))
    t = t[len(t) // 2 + 1:]

    t = list(map(lambda p: tr_rotate(p, 45), t))
    t = list(map(lambda p: tr_rotate(p, -90, around_x=False), t))
    t = list(map(lambda p: tr_translate(p, (-3.3, -3.3)), t))
    t2 = [t[0]]
    scale = 5 / 3

    for i in range(len(t)):
        temp = np.array(tr_homothety(t2[-1], scale)) + 4
        t2.append(temp.tolist())

    t2 = [pair for pair in t2 if all(coord[0] <= 50 and coord[1] <= 50 for coord in pair)]

    t2_left = list(map(lambda p: tr_symmetry(p, from_x=False), t2.copy()))
    visualize(t2 + t2_left)


def flt_convex_polygon(coords):
    n = len(coords)

    if n < 3:
        return False

    signs = []
    for i in range(n):  # знаки векторных произведений говорят о выпуклости

        dx1 = coords[(i + 1) % n][0] - coords[i][0]
        dy1 = coords[(i + 1) % n][1] - coords[i][1]
        dx2 = coords[(i + 2) % n][0] - coords[(i + 1) % n][0]
        dy2 = coords[(i + 2) % n][1] - coords[(i + 1) % n][1]

        vect_mul = dx1 * dy2 - dy1 * dx2
        signs.append(-1 if vect_mul < 0 else 1)

    if len(set(signs)) == 1:
        return True

    return False


def flt_square(coords, s):
    n = len(coords)

    if n < 3:
        return False

    area = 0.0

    for i in range(n):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]
        trapez = 0.5 * (x2 - x1) * (y1 + y2)

        area += trapez

    return abs(area) < s


def flt_short_side(coords, side):
    n = len(coords)
    sides = []

    for i in range(n):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]

        sides.append(sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

    return min(sides) < side


def flt_angle_point(coords, angle):
    if len(coords) < 3:
        return False

    for pair in coords:
        if pair == angle:
            return True

    return False


def flt_polygon_angles_inside(coords, angle):
    is_convex = flt_convex_polygon(coords)
    is_angle = flt_angle_point(coords, angle)

    return is_convex and is_angle


def flt_point_inside(coords, point):
    if not flt_convex_polygon(coords):
        return False

    x, y = point
    is_inside = False

    p1x, p1y = coords[0]

    for i in range(len(coords)):
        p2x, p2y = coords[i % len(coords)]

        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            x_inter = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x

            if p1x == p2x or x <= x_inter:
                is_inside = not is_inside

        p1x, p1y = p2x, p2y

    return is_inside


def filter_4():
    random_15 = []
    gens = [gen_rectangle, gen_trapezoid, gen_triangle, gen_hexagon]
    count = 0

    for limit in range(-60, 60, 10):
        if count < 15:

            a = random.randint(4, 7)
            fig_r = list(random.choice(gens)(a=a, st=limit, end=limit + a * 2 + a // 2))

            if fig_r:
                random_15.extend(fig_r)
                count += 1

        elif count == 15:
            break

    visualize(random_15)
    visualize(tuple(filter(lambda p: flt_short_side(p, 5), random_15)))


# 2nd part -------------------------------
# visualize(gen_triangle(a = 5))
# visualize(gen_rectangle(a = 3))
# visualize(gen_hexagon())
# visualize(random_figures())

# 3rd part -------------------------------
# test = list(gen_triangle())
# test2 = list(gen_hexagon())

# print(test)
# visualize(test)
# visualize(test2)

# par = tuple(map(lambda p: tr_translate(list(p), (0, 5)), test))
# visualize(par)

# rot = tuple(map(lambda p: tr_rotate(p, 90), test))
# visualize(rot)

# sym = tuple(map(lambda p: tr_symmetry(p), test2))
# visualize(sym)

# homot = tuple(map(lambda p: tr_homothety(p, 2), test))
# visualize(homot)

# 4th part -------------------------------
# gen_three()
# gen_two()
# two_symmetric()
# tr_scale()

# 5th part ------------------------------- 2 extra points
test3 = (((10, 0), (12, 6), (0, 0)), ((-10, 0), (-10, 10), (-6, 8), (-4, 10), (-4, 0)),
         ((15, 15), (17, 15), (16, 17), (18, 17)), ((4, 0), (4, 2), (7, 2), (7, 0)))

test4 = (((10, 0), (12, 6), (0, 0)), ((-10, 0), (-10, 10), (-6, 8), (-4, 10), (-4, 0)),
         ((0, 0), (0, 20), (20, 20), (20, 0)))

test5 = (((1, 0), (1, 10), (3, 15), (3, 0)), ((-10, 0), (-10, 10), (-3, 6), (1, 0)),
         ((10, 0), (12, 6), (0, 0)), ((-10, 0), (-10, 10), (-6, 8), (-4, 10), (-4, 0)),
         ((0, 0), (0, 20), (20, 20), (20, 0)))

test6 = (((-10, 0), (-10, 10), (-6, 8), (-4, 10), (-4, 0)), ((-4, 0), (-4, 6), (10, 6), (10, 0)),
         ((10, 0), (12, 6), (0, 0)), ((-45, 0), (-45, 15), (-30, 15), (-30, 0)))
# visualize(test3)
# visualize(test4)
# visualize(test5)
# visualize(test6)

# visualize(list(filter(lambda p: flt_convex_polygon(p), test3)))
# visualize(list(filter(lambda p: flt_square(p, 15), test3)))
# visualize(list(filter(lambda p: flt_short_side(p, 10), test4)))
# visualize(list(filter(lambda p: flt_angle_point(p, (1, 0)), test5)))
# visualize(list(filter(lambda p: flt_polygon_angles_inside(p, (-4, 0)), test6)))
# visualize(list(filter(lambda p: flt_point_inside(p, (3, 1)), test6)))


# 6th part -------------------------------
# filter_4()

# 7th part ------------------------------- 1 extra, sum = 3
# visualize(gen_triangle())
# visualize(gen_rectangle())
# visualize(gen_hexagon())
# visualize(random_figures())

# 8th part ------------------------------- 2 extra, sum = 5