import math
import glm
from PIL import Image

# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================
WIDTH = 800
HEIGHT = 600

FOV_DEG = 45.0
FOCAL_DISTANCE = 1.0
ASPECT_RATIO = WIDTH / HEIGHT

EPSILON = 1e-4

# ============================================================
# CÂMERA
# ============================================================
EYE = glm.vec3(0.0, 0.0, 0.0)
CENTER = glm.vec3(0.0, 0.0, -1.0)
UP = glm.vec3(0.0, 1.0, 0.0)

VIEW = glm.lookAt(EYE, CENTER, UP)
VIEW_INV = glm.inverse(VIEW)

# ============================================================
# CENA
# ============================================================
SPHERE_CENTER = glm.vec3(0.0, 0.0, -4.0)
SPHERE_RADIUS = 0.7

PLANE_POINT = glm.vec3(0.0, -1.0, 0.0)     # plano horizontal
PLANE_NORMAL = glm.vec3(0.0, 1.0, 0.0)     # normal apontando para cima

LIGHT_POS = glm.vec3(0.0, 3.5, -2.0)
LIGHT_POWER = glm.vec3(40.0, 40.0, 40.0)

AMBIENT_LIGHT = glm.vec3(0.35, 0.35, 0.35)

SPHERE_MATERIAL = {
    "ambient": glm.vec3(0.20, 0.00, 0.00),
    "diffuse": glm.vec3(0.75, 0.00, 0.00),
    "specular": glm.vec3(1.00, 1.00, 1.00),
    "shininess": 120.0
}

PLANE_MATERIAL = {
    "ambient": glm.vec3(0.25, 0.25, 0.25),
    "diffuse": glm.vec3(0.70, 0.70, 0.70),
    "specular": glm.vec3(0.00, 0.00, 0.00),
    "shininess": 8.0
}

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
def clamp(x, a=0.0, b=1.0):
    return max(a, min(x, b))


def hadamard(a, b):
    """Multiplicação componente a componente de cores/vetores."""
    return glm.vec3(a.x * b.x, a.y * b.y, a.z * b.z)


def to_rgb(color):
    r = int(clamp(color.r) * 255)
    g = int(clamp(color.g) * 255)
    b = int(clamp(color.b) * 255)
    return (r, g, b)


# ============================================================
# GERAÇÃO DO RAIO DA CÂMERA
# Baseado na fórmula dos slides:
# p = (-du + 2du*nx, -dv + 2dv*ny, -f, 1)
# ============================================================
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

    origin = glm.vec3(o4.x, o4.y, o4.z)
    target = glm.vec3(t4.x, t4.y, t4.z)

    direction = glm.normalize(target - origin)
    return origin, direction


# ============================================================
# INTERSEÇÃO RAIO-ESFERA
# ============================================================
def intersect_sphere(ray_origin, ray_dir):
    oc = ray_origin - SPHERE_CENTER

    a = glm.dot(ray_dir, ray_dir)
    b = 2.0 * glm.dot(ray_dir, oc)
    c = glm.dot(oc, oc) - SPHERE_RADIUS * SPHERE_RADIUS

    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None

    sqrt_disc = math.sqrt(discriminant)

    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    t_hit = None
    if t1 > EPSILON:
        t_hit = t1
    elif t2 > EPSILON:
        t_hit = t2

    if t_hit is None:
        return None

    hit_point = ray_origin + t_hit * ray_dir
    normal = glm.normalize(hit_point - SPHERE_CENTER)

    return {
        "t": t_hit,
        "point": hit_point,
        "normal": normal,
        "material": SPHERE_MATERIAL,
        "name": "sphere"
    }


# ============================================================
# INTERSEÇÃO RAIO-PLANO
# ============================================================
def intersect_plane(ray_origin, ray_dir):
    denom = glm.dot(ray_dir, PLANE_NORMAL)

    if abs(denom) < EPSILON:
        return None  # raio paralelo ao plano

    t = glm.dot(PLANE_POINT - ray_origin, PLANE_NORMAL) / denom

    if t <= EPSILON:
        return None

    hit_point = ray_origin + t * ray_dir
    normal = PLANE_NORMAL

    return {
        "t": t,
        "point": hit_point,
        "normal": normal,
        "material": PLANE_MATERIAL,
        "name": "plane"
    }


# ============================================================
# ACHA A INTERSEÇÃO MAIS PRÓXIMA
# ============================================================
def intersect_scene(ray_origin, ray_dir, max_t=float("inf")):
    nearest = None

    sphere_hit = intersect_sphere(ray_origin, ray_dir)
    if sphere_hit is not None and sphere_hit["t"] < max_t:
        nearest = sphere_hit

    plane_hit = intersect_plane(ray_origin, ray_dir)
    if plane_hit is not None and plane_hit["t"] < max_t:
        if nearest is None or plane_hit["t"] < nearest["t"]:
            nearest = plane_hit

    return nearest


# ============================================================
# TESTE DE SOMBRA
# ============================================================
def is_in_shadow(point, normal):
    shadow_origin = point + normal * (EPSILON * 10.0)

    to_light = LIGHT_POS - shadow_origin
    light_distance = glm.length(to_light)
    shadow_dir = glm.normalize(to_light)

    shadow_hit = intersect_scene(shadow_origin, shadow_dir, light_distance - EPSILON)
    return shadow_hit is not None


# ============================================================
# SHADING PHONG + AMBIENTE + SOMBRA
# ============================================================
def shade(hit, ray_origin):
    material = hit["material"]
    p = hit["point"]
    n = glm.normalize(hit["normal"])

    # Componente ambiente
    color = hadamard(material["ambient"], AMBIENT_LIGHT)

    # Se estiver em sombra, fica só com a ambiente
    if is_in_shadow(p, n):
        return color

    # Vetor para luz
    to_light = LIGHT_POS - p
    distance2 = glm.dot(to_light, to_light)
    l = glm.normalize(to_light)

    # Radiância simplificada da luz pontual: P / r²
    Li = LIGHT_POWER * (1.0 / distance2)

    # Difusa
    ndotl = max(glm.dot(n, l), 0.0)
    diffuse = hadamard(material["diffuse"], Li) * ndotl
    color += diffuse

    # Especular
    if ndotl > 0.0:
        v = glm.normalize(ray_origin - p)
        r = glm.reflect(-l, n)
        spec_angle = max(glm.dot(r, v), 0.0)
        specular = hadamard(material["specular"], Li) * (spec_angle ** material["shininess"])
        color += specular

    return color


# ============================================================
# TRAÇA UM RAIO
# ============================================================
def trace_ray(ray_origin, ray_dir):
    hit = intersect_scene(ray_origin, ray_dir)

    if hit is None:
        # fundo preto, igual ao topo da imagem dos slides
        return glm.vec3(0.0, 0.0, 0.0)

    return shade(hit, ray_origin)


# ============================================================
# RENDER
# ============================================================
def render():
    image = Image.new("RGB", (WIDTH, HEIGHT))
    pixels = image.load()

    for j in range(HEIGHT):
        for i in range(WIDTH):
            # amostra no centro do pixel
            nx = (i + 0.5) / WIDTH
            ny = 1.0 - ((j + 0.5) / HEIGHT)

            ray_origin, ray_dir = generate_ray(nx, ny)
            color = trace_ray(ray_origin, ray_dir)

            pixels[i, j] = to_rgb(color)

    image.save("esfera_phong.png")
    print("Imagem gerada com sucesso: esfera_phong.png")


if __name__ == "__main__":
    render()