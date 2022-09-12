import sys
import time
import random

from math import pi, sin, cos, sqrt


class Vector3(object):

    def __init__(self, x=0., y=0., z=0.):
        self.x = x
        self.y = y
        self.z = z
        self._content = [x, y, z]

    def __getitem__(self, item):
        return self._content[item]

    def __setitem__(self, key, value):
        self._content[key] = value

    def __str__(self):
        return '{x}, {y}, {z}'.format(x=self.x, y=self.y, z=self.z)

    def __repr__(self):
        return 'Vector3({x}, {y}, {z})'.format(x=self.x, y=self.y, z=self.z)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __ne__(self, other):
        return self.x != other.x and self.y != other.y and self.z != other.z

    def __neg__(self):
        return Vector3(- self.x, - self.y, - self.z)

    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        if isinstance(other, Vector3):
            return Vector3(self.x * other.x, self.y * other.y, self.z * other.z)
        else:
            return Vector3(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other):
        return Vector3(self.x / other, self.y / other, self.z / other)

    def __floordiv__(self, other):
        return Vector3(self.x // other, self.y // other, self.z // other)

    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        self.z += other.z
        return self

    def __isub__(self, other):
        self.x -= other.x
        self.y -= other.y
        self.z -= other.z
        return self

    def __imul__(self, other):
        self.x *= other.x
        self.y *= other.y
        self.z *= other.z
        return self

    def __idiv__(self, other):
        self.x /= other.x
        self.y /= other.y
        self.z /= other.z
        return self

    def length(self):
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self):
        return self / self.length()

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        x = self.y * other.z - self.z * other.y
        y = self.z * other.x - self.x * other.z
        z = self.x * other.y - self.y * other.x
        return Vector3(x, y, z)


def length(vector):
    return sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z)


def normalize(vector):
    vector_length = length(vector)
    return vector / vector_length


def dot(vector_a, vector_b):
    return vector_a.x * vector_b.x + vector_a.y + vector_b.y + vector_a.z * vector_b.z


def cross(vector_a, vector_b):
    x = vector_a.y * vector_b.z - vector_a.z * vector_b.y
    y = vector_a.z * vector_b.x - vector_a.x * vector_b.z
    z = vector_a.x * vector_b.y - vector_a.y * vector_b.x
    return Vector3(x, y, z)


class Ray(object):

    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction


class Sphere(object):

    def __init__(self, radius, center, emission, color, material_type):
        self.radius = radius
        self.center = center
        self.color = color
        self.emission = emission
        self.material_type = material_type

    def intersect(self, ray):
        oc = self.center - ray.origin
        neg_b = oc.dot(ray.direction)
        det = neg_b * neg_b - oc.dot(oc) + self.radius * self.radius
        if det < 0:
            return 0
        else:
            det = sqrt(det)

        epsilon = 1e-4
        if neg_b - det > epsilon:
            return neg_b - det
        elif neg_b + det > epsilon:
            return neg_b + det
        return 0


scene = [
    Sphere(1e5, Vector3(1e5 - 50, 0, 0), Vector3(), Vector3(0.6, 0.05, 0.05), 'DIFFUSE'),
    Sphere(1e5, Vector3(-1e5 + 50, 0, 0), Vector3(), Vector3(0.15, 0.45, 0.1), 'DIFFUSE'),
    Sphere(1e5, Vector3(0, 0, 1e5 - 81.6), Vector3(), Vector3(0.75, 0.75, 0.75), 'DIFFUSE'),
    Sphere(1e5, Vector3(0, 0, -1e5 + 88.4), Vector3(), Vector3(0.75, 0.75, 0.75), 'DIFFUSE'),
    Sphere(1e5, Vector3(0, 1e5 - 50, 0), Vector3(), Vector3(0.75, 0.75, 0.75), 'DIFFUSE'),
    Sphere(1e5, Vector3(50, -1e5 + 49, 0), Vector3(), Vector3(0.75, 0.75, 0.75), 'DIFFUSE'),

    Sphere(20, Vector3(-20, -30, -40), Vector3(), Vector3(1, 1, 1), 'SPECULAR'),
    Sphere(20, Vector3(20, -30, 0), Vector3(), Vector3(1, 1, 1), 'REFRACT'),
    Sphere(600, Vector3(0, 649 - 0.4, -20), Vector3(20, 20, 20), Vector3(), 'DIFFUSE'),
]


def clamp(num):
    if num < 0:
        return 0
    elif num > 1:
        return 1
    else:
        return num


def gamma(num):
    clamp_num = clamp(num)
    gamma_num = pow(clamp_num, 1 / 2.2)
    return int(gamma_num * 255)


def get_intersection(ray):
    idx = None
    infinity = 1e20
    min_distance = infinity
    sphere_count = len(scene)

    for index in range(sphere_count):
        distance = scene[index].intersect(ray)
        if distance and distance < min_distance:
            min_distance = distance
            idx = index

    return min_distance < infinity, min_distance, idx


def radiance(ray, depth):
    intersect_status, distance, idx = get_intersection(ray)

    if not intersect_status:
        return Vector3()

    obj = scene[idx]

    if depth > 10:
        return obj.emission

    position = ray.origin + ray.direction * distance
    normal = (position - obj.center).normalize()
    if normal.dot(ray.direction) < 0:
        shading_normal = normal
    else:
        shading_normal = - normal

    albedo = obj.color
    max_component = max(albedo.x, albedo.y, albedo.z)

    depth += 1
    if depth > 5:
        if random.random() < max_component:
            albedo = albedo * (1 / max_component)
        else:

            return obj.emission

    if obj.material_type == 'DIFFUSE':
        random1 = 2 * pi * random.random()
        random2 = random.random()
        random2_sqrt = sqrt(random2)

        w = shading_normal
        if abs(w.x > 1):
            u = Vector3(0, 1, 0).cross(w).normalize()
        else:
            u = Vector3(1, 0, 0).cross(w).normalize()
        v = w.cross(u)

        direction = normalize((u * cos(random1) + v * sin(random1)) * random2_sqrt + w * sqrt(1 - random2))

        # albedo = albedo / pi
        # abs_cos_theta = abs(shading_normal.dot(direction))
        # pdf = abs_cos_theta / pi
        return obj.emission + albedo * radiance(Ray(position, direction), depth)

    elif obj.material_type == 'SPECULAR':
        direction = ray.direction - normal * 2 * normal.dot(ray.direction)
        return obj.emission + albedo * radiance(Ray(position, direction), depth)

    else:
        into = normal.dot(shading_normal) > 0

        eta_i = 1
        eta_t = 1.5

        if into:
            eta = eta_i / eta_t
        else:
            eta = eta_t / eta_i

        reflect_ray = Ray(position, ray.direction - normal * 2 * normal.dot(ray.direction))
        cos_theta_i = ray.direction.dot(shading_normal)
        cos_theta_t2 = 1 - eta * eta * (1 - cos_theta_i * cos_theta_i)
        if cos_theta_t2 < 0:

            return obj.emission + albedo * radiance(reflect_ray, depth)

        cos_theta_t = sqrt(cos_theta_t2)
        if into:
            reflect_direction = normalize((ray.direction * eta - normal * (cos_theta_i * eta + cos_theta_t)))
        else:
            reflect_direction = normalize((ray.direction * eta + normal * (cos_theta_i * eta + cos_theta_t)))

        a = eta_t - eta_i
        b = eta_t + eta_i
        r0 = a * a / (b * b)
        if into:
            c = 1 + cos_theta_i
        else:
            c = 1 - reflect_direction.dot(normal)

        re = r0 + (1 - r0) * pow(c, 5)
        tr = 1 - re

        p = 0.25 + 0.5 * re
        rp = re / p
        tp = tr / (1 - p)

        if depth > 2:
            if random.random() < p:
                li = radiance(reflect_ray, depth) * rp
            else:
                li = radiance(Ray(position, reflect_direction), depth) * tp
        else:
            direct = radiance(reflect_ray, depth) * re
            indirect = radiance(Ray(position, reflect_direction), depth) * tr
            li = direct + indirect
        return obj.emission + albedo * li


def write(width, height, frame_buffer, path):
    if width * height != len(frame_buffer):
        raise ValueError('[ERROR] please check input params.')
    print('\n[INFO] start write image..\n')
    with open(path, 'w') as fp:
        fp.write('P3\n{width} {height}\n{color}\n'.format(width=width, height=height, color=255))
        for index in range(width * height):
            pixel = frame_buffer[index]
            r = gamma(pixel.x)
            g = gamma(pixel.y)
            b = gamma(pixel.z)
            sys.stdout.write('\rWriting %.2f %%' % (100 * index / (width * height - 1)))
            fp.write('{r} {g} {b}\n'.format(r=r, g=g, b=b))
    print('\n[INFO] write image done, output to: {}'.format(path))


def run():
    width, height, spp = 512, 512, 64

    camera = Ray(Vector3(0, 0, 220), Vector3(0, 0, -1))
    cx = Vector3(width * 0.5 / height, 0, 0)
    cy = cx.cross(camera.direction).normalize() * 0.5

    frame_buffer = [Vector3()] * (width * height)

    print('[INFO] start ray tracing...\n')
    start = time.time()
    for y in range(0, height):
        end = time.time()
        sys.stdout.write('\rRendering %.6f %%' % (100 * y / (height - 1)))
        sys.stdout.write('      using %.2f second.' % (end - start))
        for x in range(0, width):
            li = Vector3()
            idx = (height - y - 1) * width + x
            for s in range(0, spp):
                direction = cx * ((x + s / spp) / width - 0.5) +\
                    cy * ((y + s / spp) / height - 0.5) + camera.direction

                li = li + radiance(Ray(camera.origin + direction * 140, direction.normalize()), 0) / spp

            frame_buffer[idx] = frame_buffer[idx] + Vector3(clamp(li.x), clamp(li.y), clamp(li.z))
    write(width, height, frame_buffer, path='C:/Users/Cronix/Documents/cronix_dev/raytracing/image_py.ppm')
