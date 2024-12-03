import math

import cv2


def calculate_dist(p1, p2):
    """Két pont távolságának kiszámítása."""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def find_slope(p1, p2):
    return (p1[1] - p2[1]) / (p1[0] - p2[0])


def ns_to_s(ns, to_unit='s'):
    """
        Convert nanoseconds to seconds or milliseconds.

        Args:
            ns (int): The time in nanoseconds.
            to_unit (str): Target unit ("s" for seconds, "ms" for milliseconds).

        Returns:
            float: The converted time.
        """
    if to_unit == "s":
        return ns / 1_000_000_000  # Nanoseconds to seconds
    elif to_unit == "ms":
        return ns / 1_000_000  # Nanoseconds to milliseconds
    else:
        raise ValueError("Invalid target unit. Use 's' for seconds or 'ms' for milliseconds.")


def find_line_coordinate(p1x, p1y, p2x, p2y, distance):

    dx = p2x - p1x
    dy = p2y - p1y
    dist_between_points = math.sqrt(dx ** 2 + dy ** 2)

    if dist_between_points == 0:
        print("two matching point")
        return p2x, p2y

    unit_dx = dx / dist_between_points
    unit_dy = dy / dist_between_points

    p3x = p1x + unit_dx * distance
    p3y = p1y + unit_dy * distance

    return p3x, p3y


def check_monotonity(li):
    if li[0] < li[1]:
        for i in range(1, len(li)):
            if li[i-1] > li[i]:
                return False
        return True

    for i in range(1, len(li)):
        if li[i-1] < li[i]:
            return False
    return True


def get_first_not_monoton_el(li):
    if li[0] < li[1]:
        for i in range(1, len(li)):
            if li[i - 1] > li[i]:
                return i
        # Csökkenő lista keresése
    elif li[0] > li[1]:
        for i in range(1, len(li)):
            if li[i - 1] < li[i]:
                return i
        # Ha az első két elem egyenlő, akkor visszaadjuk -1-et
    return -1

