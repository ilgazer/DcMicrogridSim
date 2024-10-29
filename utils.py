def limit(x, min: float, max: float):
    if x < min:
        return min
    elif x > max:
        return max
    return x


def linear(x1: float, y1: float, x2: float, y2: float):
    return lambda x: (x - x1) * (y2 - y1) / (x2 - x1) + y1
