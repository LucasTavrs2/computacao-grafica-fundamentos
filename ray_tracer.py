import math
import glm
from PIL import Image

# --------------------------------------------------
# Configurações da imagem
# --------------------------------------------------
WIDTH = 800
HEIGHT = 600
FOV = 60.0  # campo de visão em graus

# --------------------------------------------------
# Configurações da cena
# --------------------------------------------------
camera_origin = glm.vec3(0.0, 0.0, 0.0)

sphere_center = glm.vec3(0.0, 0.0, -3.0)
sphere_radius = 1.0

# --------------------------------------------------
# Funções auxiliares
# --------------------------------------------------
def clamp(value, min_value=0.0, max_value=1.0):
    return max(min_value, min(value, max_value))


def ray_sphere_intersection(ray_origin, ray_dir, center, radius):
    """
    Resolve a interseção entre um raio e uma esfera.

    Equação do raio:
        r(t) = o + t*d

    Equação da esfera:
        (x - c)·(x - c) - r² = 0

    Substituindo o raio na esfera, obtemos uma equação do 2º grau em t.
    """
    oc = ray_origin - center

    a = glm.dot(ray_dir, ray_dir)
    b = 2.0 * glm.dot(oc, ray_dir)
    c = glm.dot(oc, oc) - radius * radius

    discriminant = b * b - 4.0 * a * c

    if discriminant < 0.0:
        return None  # não houve interseção

    sqrt_discriminant = math.sqrt(discriminant)

    t1 = (-b - sqrt_discriminant) / (2.0 * a)
    t2 = (-b + sqrt_discriminant) / (2.0 * a)

    epsilon = 1e-4

    # Queremos a menor raiz positiva
    if t1 > epsilon:
        return t1
    if t2 > epsilon:
        return t2

    return None


def background_color(ray_dir):
    """
    Fundo em degradê simples para não ficar tudo preto.
    """
    t = 0.5 * (ray_dir.y + 1.0)
    white = glm.vec3(1.0, 1.0, 1.0)
    blue = glm.vec3(0.5, 0.7, 1.0)
    return (1.0 - t) * white + t * blue


def normal_to_color(normal):
    """
    Converte normal de [-1, 1] para [0, 1].
    Isso não é iluminação física ainda.
    É só uma forma de visualizar a esfera em 3D.
    """
    return 0.5 * (normal + glm.vec3(1.0, 1.0, 1.0))


def to_rgb(color):
    r = int(clamp(color.r) * 255)
    g = int(clamp(color.g) * 255)
    b = int(clamp(color.b) * 255)
    return (r, g, b)


# --------------------------------------------------
# Renderização
# --------------------------------------------------
image = Image.new("RGB", (WIDTH, HEIGHT))
pixels = image.load()

aspect_ratio = WIDTH / HEIGHT
scale = math.tan(math.radians(FOV * 0.5))

for j in range(HEIGHT):
    for i in range(WIDTH):
        # Amostra no centro do pixel
        x_ndc = (i + 0.5) / WIDTH
        y_ndc = (j + 0.5) / HEIGHT

        # Converte para espaço da câmera
        x = (2.0 * x_ndc - 1.0) * aspect_ratio * scale
        y = (1.0 - 2.0 * y_ndc) * scale

        # Direção do raio saindo da câmera
        ray_dir = glm.normalize(glm.vec3(x, y, -1.0))

        # Testa interseção com a esfera
        t_hit = ray_sphere_intersection(
            camera_origin,
            ray_dir,
            sphere_center,
            sphere_radius
        )

        if t_hit is None:
            color = background_color(ray_dir)
        else:
            hit_point = camera_origin + t_hit * ray_dir
            normal = glm.normalize(hit_point - sphere_center)
            color = normal_to_color(normal)

        pixels[i, j] = to_rgb(color)

image.save("esfera.png")
print("Imagem gerada com sucesso: esfera.png")