import math
import random
import glm
from PIL import Image

# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================
WIDTH = 480
HEIGHT = 360

FOV_DEG = 45.0
FOCAL_DISTANCE = 1.0
ASPECT_RATIO = WIDTH / HEIGHT

EPSILON = 1e-4
GAMMA = 2.2

# Quantidade de amostras na luz de área (5 x 5 = 25)
LIGHT_SAMPLES_U = 5
LIGHT_SAMPLES_V = 5

# Semente fixa para o resultado ficar reproduzível
random.seed(42)

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
def clamp(x, a=0.0, b=1.0):
    return max(a, min(x, b))


def hadamard(a, b):
    return glm.vec3(a.x * b.x, a.y * b.y, a.z * b.z)


def vec3_from_vec4(v):
    return glm.vec3(v.x, v.y, v.z)


def point_transform(M, p):
    return vec3_from_vec4(M * glm.vec4(p.x, p.y, p.z, 1.0))


def dir_transform(M, d):
    return vec3_from_vec4(M * glm.vec4(d.x, d.y, d.z, 0.0))


def to_rgb(color):
    # clamp + correção gama
    r = clamp(color.r)
    g = clamp(color.g)
    b = clamp(color.b)

    r = int((r ** (1.0 / GAMMA)) * 255.0)
    g = int((g ** (1.0 / GAMMA)) * 255.0)
    b = int((b ** (1.0 / GAMMA)) * 255.0)

    return (r, g, b)


# ============================================================
# CÂMERA
# ============================================================
EYE = glm.vec3(0.0, 0.0, 0.0)
CENTER = glm.vec3(0.0, 0.0, -1.0)
UP = glm.vec3(0.0, 1.0, 0.0)

VIEW = glm.lookAt(EYE, CENTER, UP)
VIEW_INV = glm.inverse(VIEW)


def generate_ray(nx, ny):
    dv = FOCAL_DISTANCE * math.tan(math.radians(FOV_DEG) * 0.5)
    du = dv * ASPECT_RATIO

    p_camera = glm.vec4(
        -du + 2.0 * du * nx,
        -dv + 2.0 * dv * ny,
        -FOCAL_DISTANCE,
        1.0
    )

    o4 = VIEW_INV * glm.vec4(0.0, 0.0, 0.0, 1.0)
    t4 = VIEW_INV * p_camera

    origin = vec3_from_vec4(o4)
    target = vec3_from_vec4(t4)
    direction = glm.normalize(target - origin)

    return origin, direction


# ============================================================
# MATERIAIS
# ============================================================
AMBIENT_LIGHT = glm.vec3(0.12, 0.12, 0.12)

ELLIPSOID_MATERIAL = {
    "ambient": glm.vec3(0.18, 0.00, 0.00),
    "diffuse": glm.vec3(0.85, 0.05, 0.05),
    "specular": glm.vec3(1.00, 1.00, 1.00),
    "shininess": 100.0
}

PLANE_MATERIAL = {
    "ambient": glm.vec3(0.22, 0.22, 0.22),
    "diffuse": glm.vec3(0.75, 0.75, 0.75),
    "specular": glm.vec3(0.00, 0.00, 0.00),
    "shininess": 8.0
}

# ============================================================
# PLANO
# ============================================================
PLANE_POINT = glm.vec3(0.0, -1.0, 0.0)
PLANE_NORMAL = glm.normalize(glm.vec3(0.0, 1.0, 0.0))

# ============================================================
# ELIPSÓIDE = ESFERA UNITÁRIA + TRANSFORMAÇÃO
# ============================================================
ELLIPSOID_SCALE = glm.vec3(0.55, 1.15, 0.55)
ELLIPSOID_CENTER = glm.vec3(0.0, PLANE_POINT.y + ELLIPSOID_SCALE.y, -4.8)

ELLIPSOID_M = glm.translate(glm.mat4(1.0), ELLIPSOID_CENTER)
ELLIPSOID_M = glm.scale(ELLIPSOID_M, ELLIPSOID_SCALE)

ELLIPSOID_MINV = glm.inverse(ELLIPSOID_M)
ELLIPSOID_NORMAL_M = glm.transpose(ELLIPSOID_MINV)

# ============================================================
# LUZ DE ÁREA RETANGULAR
# p = origem (um canto)
# ei, ej = arestas do retângulo
# ============================================================
LIGHT_P = glm.vec3(-0.55, 2.6, -4.1)
LIGHT_EI = glm.vec3(1.10, 0.0, 0.0)   # largura
LIGHT_EJ = glm.vec3(0.0, 0.0, 0.80)   # profundidade

LIGHT_NORMAL = glm.normalize(glm.cross(LIGHT_EI, LIGHT_EJ))  # aponta para baixo
LIGHT_AREA = glm.length(glm.cross(LIGHT_EI, LIGHT_EJ))

LIGHT_POWER = glm.vec3(60.0, 60.0, 60.0)
LIGHT_IRRADIANCE = LIGHT_POWER / LIGHT_AREA

LIGHT_VISIBLE_COLOR = glm.vec3(1.0, 1.0, 1.0)


def light_sample_count():
    return LIGHT_SAMPLES_U * LIGHT_SAMPLES_V


def generate_area_light_samples():
    """
    Amostragem estratificada (jitter dentro de cada célula).
    """
    for j in range(LIGHT_SAMPLES_V):
        for i in range(LIGHT_SAMPLES_U):
            u = (i + random.random()) / LIGHT_SAMPLES_U
            v = (j + random.random()) / LIGHT_SAMPLES_V
            sample_point = LIGHT_P + u * LIGHT_EI + v * LIGHT_EJ
            yield sample_point, LIGHT_NORMAL


# ============================================================
# INTERSEÇÕES
# ============================================================
def intersect_plane(ray_origin, ray_dir):
    denom = glm.dot(ray_dir, PLANE_NORMAL)

    if abs(denom) < EPSILON:
        return None

    t = glm.dot(PLANE_POINT - ray_origin, PLANE_NORMAL) / denom
    if t <= EPSILON:
        return None

    hit_point = ray_origin + t * ray_dir
    normal = PLANE_NORMAL

    if glm.dot(normal, ray_dir) > 0.0:
        normal = -normal

    return {
        "t": t,
        "point": hit_point,
        "normal": normal,
        "material": PLANE_MATERIAL,
        "is_light": False,
        "name": "plane"
    }


def intersect_unit_sphere_local(ray_origin_local, ray_dir_local):
    a = glm.dot(ray_dir_local, ray_dir_local)
    b = 2.0 * glm.dot(ray_dir_local, ray_origin_local)
    c = glm.dot(ray_origin_local, ray_origin_local) - 1.0

    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None

    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    t_local = None
    if t1 > EPSILON:
        t_local = t1
    elif t2 > EPSILON:
        t_local = t2

    if t_local is None:
        return None

    p_local = ray_origin_local + t_local * ray_dir_local
    n_local = glm.normalize(p_local)

    return p_local, n_local


def intersect_ellipsoid(ray_origin, ray_dir):
    # Transforma o raio para o espaço local do objeto
    o_local = point_transform(ELLIPSOID_MINV, ray_origin)
    d_local = dir_transform(ELLIPSOID_MINV, ray_dir)

    result = intersect_unit_sphere_local(o_local, d_local)
    if result is None:
        return None

    p_local, n_local = result

    # Volta para o espaço global
    p_world = point_transform(ELLIPSOID_M, p_local)
    n_world = glm.normalize(dir_transform(ELLIPSOID_NORMAL_M, n_local))

    if glm.dot(n_world, ray_dir) > 0.0:
        n_world = -n_world

    # Como ray_dir está normalizado, esse t_world é a distância real
    t_world = glm.length(p_world - ray_origin)

    return {
        "t": t_world,
        "point": p_world,
        "normal": n_world,
        "material": ELLIPSOID_MATERIAL,
        "is_light": False,
        "name": "ellipsoid"
    }


def intersect_area_light(ray_origin, ray_dir):
    """
    Interseção do raio com o retângulo emissivo.
    Usado para poder VER a fonte de luz na imagem.
    """
    denom = glm.dot(ray_dir, LIGHT_NORMAL)

    if abs(denom) < EPSILON:
        return None

    t = glm.dot(LIGHT_P - ray_origin, LIGHT_NORMAL) / denom
    if t <= EPSILON:
        return None

    hit_point = ray_origin + t * ray_dir
    rel = hit_point - LIGHT_P

    # Como ei e ej são ortogonais, podemos projetar neles
    u = glm.dot(rel, LIGHT_EI) / glm.dot(LIGHT_EI, LIGHT_EI)
    v = glm.dot(rel, LIGHT_EJ) / glm.dot(LIGHT_EJ, LIGHT_EJ)

    if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0:
        normal = LIGHT_NORMAL
        if glm.dot(normal, ray_dir) > 0.0:
            normal = -normal

        return {
            "t": t,
            "point": hit_point,
            "normal": normal,
            "material": None,
            "is_light": True,
            "name": "area_light"
        }

    return None


def intersect_scene(ray_origin, ray_dir, include_light=True, max_t=float("inf")):
    nearest = None

    def try_hit(hit):
        nonlocal nearest
        if hit is None:
            return
        if hit["t"] >= max_t:
            return
        if nearest is None or hit["t"] < nearest["t"]:
            nearest = hit

    try_hit(intersect_ellipsoid(ray_origin, ray_dir))
    try_hit(intersect_plane(ray_origin, ray_dir))

    if include_light:
        try_hit(intersect_area_light(ray_origin, ray_dir))

    return nearest


# ============================================================
# SOMBRA + SHADING COM LUZ DE ÁREA
# ============================================================
def shade(hit, ray_origin):
    material = hit["material"]
    p = hit["point"]
    n = glm.normalize(hit["normal"])
    v = glm.normalize(ray_origin - p)

    # ambiente
    color = hadamard(material["ambient"], AMBIENT_LIGHT)

    sample_area = LIGHT_AREA / light_sample_count()

    for sample_point, light_normal in generate_area_light_samples():
        to_light = sample_point - p
        distance2 = glm.dot(to_light, to_light)
        if distance2 <= EPSILON:
            continue

        l = glm.normalize(to_light)

        ndotl = max(glm.dot(n, l), 0.0)
        cos_light = max(glm.dot(light_normal, -l), 0.0)

        if ndotl <= 0.0 or cos_light <= 0.0:
            continue

        # raio de sombra
        shadow_origin = p + n * (EPSILON * 10.0)
        light_distance = math.sqrt(distance2)

        blocker = intersect_scene(
            shadow_origin,
            l,
            include_light=False,
            max_t=light_distance - EPSILON
        )

        if blocker is not None:
            continue

        # contribuição dessa amostra da luz de área
        Li = LIGHT_IRRADIANCE * (cos_light * sample_area / distance2)

        # difusa
        color += hadamard(material["diffuse"], Li) * ndotl

        # especular
        r = glm.reflect(-l, n)
        spec_angle = max(glm.dot(r, v), 0.0)
        if spec_angle > 0.0:
            color += hadamard(material["specular"], Li) * (spec_angle ** material["shininess"])

    return color


def trace_ray(ray_origin, ray_dir):
    hit = intersect_scene(ray_origin, ray_dir, include_light=True)

    if hit is None:
        return glm.vec3(0.0, 0.0, 0.0)

    if hit["is_light"]:
        return LIGHT_VISIBLE_COLOR

    return shade(hit, ray_origin)


# ============================================================
# RENDER
# ============================================================
def render():
    image = Image.new("RGB", (WIDTH, HEIGHT))
    pixels = image.load()

    for j in range(HEIGHT):
        for i in range(WIDTH):
            nx = (i + 0.5) / WIDTH
            ny = 1.0 - ((j + 0.5) / HEIGHT)

            ray_origin, ray_dir = generate_ray(nx, ny)
            color = trace_ray(ray_origin, ray_dir)

            pixels[i, j] = to_rgb(color)

    image.save("elipsoide_area_light.png")
    print("Imagem gerada com sucesso: elipsoide_area_light.png")


if __name__ == "__main__":
    render()