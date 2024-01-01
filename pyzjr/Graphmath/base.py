import math
import numpy as np

class GraphPoint:
    def __init__(self, x, y, z=None):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        if self.z is not None:
            return f"({self.x}, {self.y}, {self.z})"
        else:
            return f"({self.x}, {self.y})"

    def __sub__(self, other: 'GraphPoint') -> 'GraphPoint':
        return GraphPoint(self.x - other.x, self.y - other.y, self.z - other.z if self.z is not None and other.z is not None else None)

    def __mul__(self, scalar: float):
        return GraphPoint(self.x * scalar, self.y * scalar, self.z * scalar if self.z is not None else None)

    def __rmul__(self, scalar: float):
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float):
        return GraphPoint(self.x / scalar, self.y / scalar, self.z / scalar if self.z is not None else None)

    def as_tuple(self):
        """
        GraphPoint转为tuple类型
        >>> p3d = GraphPoint(2, 3, 4)
        >>> p2d = GraphPoint(5, 6)
        >>> tuple_3d = p3d.as_tuple()
        >>> # (2, 3, 4)
        >>> tuple_2d = p2d.as_tuple()
        >>> # (5, 6)
        """
        if self.z is not None:
            return (self.x, self.y, self.z)
        else:
            return (self.x, self.y)

    def as_numpy(self):
        """
        GraphPoint转为numpy类型
        >>> p3d = GraphPoint(2, 3, 4)
        >>> np_array = p3d.as_numpy()
        >>> print(np_array)
        >>> print(type(np_array))
        >>> # [2 3 4]
        >>> # <class 'numpy.ndarray'>
        """
        if self.z is not None:
            return np.array([self.x, self.y, self.z])
        else:
            return np.array([self.x, self.y])

    @classmethod
    def from_numpy(cls, np_array):
        """
        numpy转为GraphPoint类型
        >>> new_p3d = GraphPoint.from_numpy(np_array)
        >>> print(new_p3d)
        >>> print(type(new_p3d))
        >>> # Point(2, 3, 4)
        >>> # <class '__main__.GraphPoint'>
        """
        if len(np_array) == 2:
            return cls(np_array[0], np_array[1])
        elif len(np_array) == 3:
            return cls(np_array[0], np_array[1], np_array[2])
        else:
            raise ValueError("Invalid array shape for conversion.")

    def rotate(self, angle_degrees: float):
        """
        将点逆时针旋转45度
        >>> p = GraphPoint(1, 0)  # 创建一个点 (1, 0)
        >>> rotated_point = p.rotate(90)
        >>> # 输出: (6.123233995736766e-17, 1.0)
        将点逆时针旋转45度
        >>> rotated_point = p.rotate(45)
        >>> # 输出: (0.7071067811865476, 0.7071067811865476)
        """
        angle_radians = math.radians(angle_degrees)
        cos_angle = math.cos(angle_radians)
        sin_angle = math.sin(angle_radians)
        x_rotated = self.x * cos_angle - self.y * sin_angle
        y_rotated = self.x * sin_angle + self.y * cos_angle
        return GraphPoint(x_rotated, y_rotated, self.z)

    def distance(self, other_point2: "GraphPoint"):
        """
        计算两点间间距
        >>> p = GraphPoint(1, 0)
        >>> p2 = GraphPoint(2, 2)
        >>> distance = p.distance(p2)
        >>> # 2.23606797749979
        """
        dx = other_point2.x - self.x
        dy = other_point2.y - self.y
        if self.z is not None and other_point2.z is not None:
            dz = other_point2.z - self.z
            return math.sqrt(dx**2 + dy**2 + dz**2)
        else:
            return math.sqrt(dx**2 + dy**2)

    def move(self, dx, dy, dz=None):
        """
        在x轴上移动dx个单位，y轴上移动dy个单位，z轴上移动dz个单位
        >>> p = GraphPoint(1, 0, 2)
        >>> p.move(2, 3, 1)  #
        >>> # Point(3, 3, 3)
        """
        self.x += dx
        self.y += dy
        if dz is not None and self.z is not None:
            self.z += dz

    def rotate_by(self, angle_degrees, origin=None):
        """
        绕着某点进行旋转
        >>> p = GraphPoint(1, 0)
        >>> origin = GraphPoint(0, 0)
        >>> p.rotate_by(90, origin)
        >>> # Point(6.123233995736766e-17, 1.0)
        """
        if origin is None:
            origin = GraphPoint(0, 0)

        angle_radians = math.radians(angle_degrees)
        cos_angle = math.cos(angle_radians)
        sin_angle = math.sin(angle_radians)

        translated_x = self.x - origin.x
        translated_y = self.y - origin.y

        rotated_x = translated_x * cos_angle - translated_y * sin_angle
        rotated_y = translated_x * sin_angle + translated_y * cos_angle

        self.x = rotated_x + origin.x
        self.y = rotated_y + origin.y


    def scale(self, factor):
        """
        缩放二维或三维点的坐标
        >>> p3d = GraphPoint(2, 3, 4)
        >>> p3d.scale(2)
        >>> # (4, 6, 8)
        """
        self.x *= factor
        self.y *= factor
        if self.z is not None:
            self.z *= factor

    def midpoint_to(self, other_point:"GraphPoint"):
        """
        中点计算
        >>> p1 = GraphPoint(1, 2, 3)
        >>> p2 = GraphPoint(4, 5, 6)
        >>> mid_point = p1.midpoint_to(p2)
        >>> # (2.5, 3.5, 4.5)
        """
        mx = (self.x + other_point.x) / 2
        my = (self.y + other_point.y) / 2
        if self.z is not None and other_point.z is not None:
            mz = (self.z + other_point.z) / 2
            return GraphPoint(mx, my, mz)
        else:
            return GraphPoint(mx, my)

    def reflect(self, origin=None):
        """
        反射对称点,默认是原点对称
        >>> p = GraphPoint(2, 3, 4)
        >>> reflected_point = p.reflect()
        >>> # (-2, -3, -4)
        """
        if origin is None:
            origin = GraphPoint(0, 0, z=0)

        rx = 2 * origin.x - self.x
        ry = 2 * origin.y - self.y
        if self.z is not None and origin.z is not None:
            rz = 2 * origin.z - self.z
            return GraphPoint(rx, ry, rz)
        else:
            return GraphPoint(rx, ry)

    def angle_between(self, other_point:"GraphPoint"):
        """
        以度为单位,两点间的角度
        >>> p1 = GraphPoint(1, 0)
        >>> p2 = GraphPoint(0, 1)
        >>> angle_degrees = p1.angle_between(p2)
        >>> # 135.0
        """
        if self.z is not None or other_point.z is not None:
            raise ValueError("Angle between is not supported for 3D points.")
        dx = other_point.x - self.x
        dy = other_point.y - self.y
        angle_rad = math.atan2(dy, dx)
        return math.degrees(angle_rad)

    def polar_coordinates(self):
        """
        笛卡尔转为极坐标
        >>> p2d = GraphPoint(3, 4)  # 创建一个二维点 (3, 4)
        >>> r, theta = p2d.polar_coordinates()
        >>> print(f"r = {r}, theta = {theta} degrees")
        >>> # r = 5.0, theta = 53.13010235415598 degrees
        """
        if self.z is not None:
            raise ValueError("Polar coordinates are not supported for 3D points.")

        r = math.sqrt(self.x ** 2 + self.y ** 2)
        theta = math.degrees(math.atan2(self.y, self.x))
        return r, theta

    def set_polar_coordinates(self, r, theta, origin=None):
        """
        将极坐标系转为笛卡尔坐标系
        >>> p2d = GraphPoint(0, 0)
        >>> p2d.set_polar_coordinates(5, 45)
        >>> # (3.5355339059327378, 3.5355339059327378)
        """
        if origin is None:
            origin = GraphPoint(0, 0)

        theta_radians = math.radians(theta)
        self.x = origin.x + r * math.cos(theta_radians)
        self.y = origin.y + r * math.sin(theta_radians)

    def slope(self, other_point: "GraphPoint"):
        """
        斜率
        >>> p1 = GraphPoint(1, 2)
        >>> p2 = GraphPoint(3, 4)

        >>> slope = p1.slope(p2)
        >>> # 1.0
        """
        dy = other_point.y - self.y
        dx = other_point.x - self.x
        if dx == 0:
            return float('inf')
        return dy / dx

    def is_collinear(self, a:"GraphPoint", b:"GraphPoint"):
        """
        两条线是否共线
        >>> p1 = GraphPoint(1, 1)
        >>> p2 = GraphPoint(2, 2)
        >>> p3 = GraphPoint(3, 3)
        >>> collinear = p1.is_collinear(p2, p3)
        >>> # True
        """
        return (b.y - a.y) * (b.x - self.x) == (b.y - self.y) * (b.x - a.x)

    def distance_to_line(self, line_point1:"GraphPoint", line_point2:"GraphPoint"):
        """
        点到线段的最短距离
        1,计算线段的方向向量 (dx, dy):
            dx = x2 - x1
            dy = y2 - y1
        2,计算点 (x0, y0) 到线段起点 (x1, y1) 的向量 (dpx, dpy)：
            dpx = x0 - x1
            dpy = y0 - y1
        3,计算点 (x0, y0) 到线段的垂直距离 dist，使用向量叉积：
            dist = |dpx * dy - dpy * dx| / sqrt(dx^2 + dy^2)
        """
        num = abs((line_point2.y - line_point1.y) * self.x - (line_point2.x - line_point1.x) * self.y + line_point2.x * line_point1.y - line_point2.y * line_point1.x)
        den = ((line_point2.y - line_point1.y)**2 + (line_point2.x - line_point1.x)**2)**0.5
        return num / den

    def point_to(self, other_point: "GraphPoint"):
        """
        点指向目标点的方向
        >>> point1 = GraphPoint(1, 2)
        >>> point2 = GraphPoint(4, 6)
        >>> angle = point1.point_to(point2)
        >>> # 53.13010235415598
        """
        angle = math.atan2(other_point.y - self.y, other_point.x - self.x)
        bearing = math.degrees(angle) % 360
        return bearing

    def distance_to_circle(self, center:"GraphPoint", radius):
        """
        计算点到圆的距离，如果点在圆内，则返回0表示在圆内
        >>> p = GraphPoint(5, 4)
        >>> center = GraphPoint(0, 0)
        >>> radius = 5
        >>> distance = p.distance_to_circle(center, radius)
        >>> # 1.4031242374328485
        """
        distance_to_center = self.distance(center)
        return max(0, distance_to_center - radius)

    def distance_to_pline(self, polyline):
        """
        计算点到多段线的最短距离
        >>> p = GraphPoint(3, 3)
        >>> polyline = [GraphPoint(3, 2), GraphPoint(2, 7), GraphPoint(1, 5)]
        >>> distance = p.distance_to_pline(polyline)
        >>> # 0.19611613513818404
        """
        min_distance = float("inf")
        for i in range(len(polyline) - 1):
            distance = self.distance_to_line(polyline[i], polyline[i + 1])
            min_distance = min(min_distance, distance)

        return min_distance

    def interpolate(self, point1, point2, ratio):
        """
        在两个点之间按照给定的比例插值出一个新的点
        >>> point = GraphPoint(0, 0, 0)
        >>> point1 = GraphPoint(1, 2, 3)
        >>> point2 = GraphPoint(4, 5, 6)
        >>> ratio = 0.5
        >>> interpolated_point = point.interpolate(point1, point2, ratio)
        >>> # (2.5, 3.5, 4.5)
        """
        x = point1.x + ratio * (point2.x - point1.x)
        y = point1.y + ratio * (point2.y - point1.y)
        if self.z is not None and point1.z is not None and point2.z is not None:
            z = point1.z + ratio * (point2.z - point1.z)
            return GraphPoint(x, y, z)
        return GraphPoint(x, y)

    def triangle_area(self, point1, point2):
        """
        >>> point = GraphPoint(0, 0)
        >>> point1 = GraphPoint(1, 2)
        >>> point2 = GraphPoint(4, 5)
        >>> area = point.triangle_area(point1, point2)
        >>> # 1.5
        """
        area = 0.5 * abs((point1.x - self.x) * (point2.y - self.y) - (point1.y - self.y) * (point2.x - self.x))
        return area

    def is_inside_polygon(self, polygon):
        """
        射线法检查点是否在多边形内部
        >>> polygon = [GraphPoint(1, 1),GraphPoint(2, 2),GraphPoint(4, 3),GraphPoint(4, 1)]
        >>> test_point = GraphPoint(3, 2)

        >>> is_inside = test_point.is_inside_polygon(polygon)
        >>> # True
        """
        num_vertices = len(polygon)
        j = num_vertices - 1
        inside = False

        for i in range(num_vertices):
            if ((polygon[i].y > self.y) != (polygon[j].y > self.y)) and \
                    (self.x < (polygon[j].x - polygon[i].x) * (self.y - polygon[i].y) / (polygon[j].y - polygon[i].y) + polygon[i].x):
                inside = not inside
            j = i
        return inside

class GraphLine():
    def __init__(self, point1: GraphPoint, point2: GraphPoint):
        if point1 == point2:
            raise ValueError("The two points must be distinct.")
        self.point1 = point1
        self.point2 = point2

    def __repr__(self):
        return f"({self.point1}, {self.point2})"

    def length(self):
        """计算线段长度"""
        return self.point1.distance(self.point2)

    def slope(self):
        """计算线段斜率"""
        return (self.point2.y - self.point1.y) / (self.point2.x - self.point1.x)

    def intercept(self):
        """计算线段于y轴的截距"""
        return self.point1.y - self.slope() * self.point1.x

    def midpoint(self):
        """计算线段的中点"""
        return self.point1.midpoint_to(self.point2)

    def contains_point(self, point):
        """判断点是否在线段上"""
        return point.is_collinear(self.point1, self.point2) and self.point1.distance(point) + self.point2.distance(point) == self.length()

    def perpendicular_bisector(self):
        """构造线段的垂线"""
        midpoint = self.midpoint()
        if self.point2.x == self.point1.x:
            return GraphLine(midpoint, GraphPoint(midpoint.x + 1, midpoint.y))
        slope = -1 / self.slope()
        intercept = midpoint.y - slope * midpoint.x
        return GraphLine(midpoint, GraphPoint(midpoint.x + 1, slope * (midpoint.x + 1) + intercept))

    def angle_between(self, other_line):
        """计算两线段之间的角度"""
        tan_angle = abs((self.slope() - other_line.slope()) / (1 + self.slope() * other_line.slope()))
        return math.degrees(math.atan(tan_angle))

    def point_at_distance(self, point, distance):
        """返回线段上距离给定点指定距离的点"""
        if not self.contains_point(point):
            raise ValueError("The given point must be on the line.")

        line_length = self.length()
        ratio = distance / line_length

        x = point.x + ratio * (self.point2.x - self.point1.x)
        y = point.y + ratio * (self.point2.y - self.point1.y)

        return GraphPoint(x, y)

    def intersection(self, other_line):
        """返回两条直线的交点坐标"""
        if self.slope() == other_line.slope():
            return None
        x = (other_line.intercept() - self.intercept()) / (self.slope() - other_line.slope())
        y = self.slope() * x + self.intercept()
        return GraphPoint(x, y)

    def parallel_to(self, other_line)-> bool:
        """判断当前直线是否与另一条直线平行"""
        return self.slope() == other_line.slope()

    def perpendicular_to(self, other_line) -> bool:
        """判断当前直线是否与另一条直线垂直（正交）"""
        return self.slope() * other_line.slope() == -1

    def is_perpendicular(self, other_line) -> bool:
        """判断当前直线是否与另一条直线近似垂直（正交）"""
        return abs(self.slope() * other_line.slope() + 1) < 1e-9

    def is_parallel(self, other_line) -> bool:
        """判断当前直线是否与另一条直线近似平行"""
        return abs(self.slope() - other_line.slope()) < 1e-9

    def angle_with_x_axis(self) -> float:
        """计算当前直线与 x 轴的夹角"""
        return math.degrees(math.atan(self.slope()))

    def angle_with_another_line(self, other_line)->float:
        """计算当前直线与另一条直线的夹角"""
        if self.is_parallel(other_line):
            return 0.0
        angle = abs(math.atan((other_line.slope() - self.slope()) / (1 + self.slope() * other_line.slope())))
        return math.degrees(angle)

    def as_tuple(self):
        """将当前直线表示为一个元组，包含两个点的坐标元组"""
        return (self.point1.as_tuple(), self.point2.as_tuple())

    def as_numpy(self):
        """
        GraphLine类型转为numpy类型
        >>> point1 = GraphPoint(1, 2)
        >>> point2 = GraphPoint(3, 4)
        >>> line = GraphLine(point1, point2)
        >>> point_numpy = line.as_numpy()
        >>> # (array([1, 2]), array([3, 4]))
        """
        if self.point1.z is not None and self.point2.z is not None:
            point1 = np.array([self.point1.x, self.point1.y, self.point1.z])
            point2 = np.array([self.point2.x, self.point2.y, self.point2.z])
        else:
            point1 = np.array([self.point1.x, self.point1.y])
            point2 = np.array([self.point2.x, self.point2.y])
        return point1, point2

    @classmethod
    def from_numpy(cls, np_array):
        """
        numpy转为GraphLine类型
        >>> point1_numpy = np.array([1, 2])
        >>> point2_numpy = np.array([3, 4])
        >>> line = GraphLine.from_numpy((point1_numpy, point2_numpy))
        >>> ((1, 2), (3, 4))
        """
        point1_numpy = np.array(np_array[0])
        point2_numpy = np.array(np_array[1])

        if point1_numpy.shape == (2,):
            return cls(GraphPoint(point1_numpy[0], point1_numpy[1]), GraphPoint(point2_numpy[0], point2_numpy[1]))
        elif point1_numpy.shape == (3,):
            return cls(
                GraphPoint(point1_numpy[0], point1_numpy[1], point1_numpy[2]),
                GraphPoint(point2_numpy[0], point2_numpy[1], point2_numpy[2])
            )
        else:
            raise ValueError("Invalid array shape for conversion.")


if __name__ == "__main__":
    point1_numpy = np.array([1, 2])
    point2_numpy = np.array([3, 4])
    line = GraphLine.from_numpy((point1_numpy, point2_numpy))

    # 打印结果
    print(line)