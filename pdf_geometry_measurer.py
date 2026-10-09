import math
from typing import Dict, List, Tuple

import cv2
import fitz
import numpy as np


class GeometryMeasurer:
    """Classe para extrair imagens de PDFs e medir figuras geométricas."""

    def __init__(self, scale_factor: float = 1.0):
        self.scale_factor = scale_factor

    def pdf_to_images(self, pdf_path: str, dpi: int = 200) -> List[np.ndarray]:
        """Converte cada página do PDF em uma imagem em RGB."""
        doc = fitz.open(pdf_path)
        images: List[np.ndarray] = []

        for page in doc:
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            images.append(img)

        return images

    def detect_shapes(self, image: np.ndarray) -> Dict[str, List[Dict]]:
        """Detecta formas geométricas em uma imagem."""
        if image is None or image.size == 0:
            raise ValueError('Imagem vazia ou inválida.')

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        shapes = {
            'circles': [],
            'rectangles': [],
            'triangles': [],
            'polygons': [],
        }

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 3:
                shapes['triangles'].append({
                    'contour': contour,
                    'approx': approx,
                    'area': area,
                    'perimeter': perimeter,
                })
            elif len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / float(h) if h else 0
                if 0.8 <= aspect_ratio <= 1.2:
                    shapes['circles'].append({
                        'contour': contour,
                        'approx': approx,
                        'area': area,
                        'perimeter': perimeter,
                    })
                else:
                    shapes['rectangles'].append({
                        'contour': contour,
                        'approx': approx,
                        'area': area,
                        'perimeter': perimeter,
                    })
            elif len(approx) >= 5:
                shapes['polygons'].append({
                    'contour': contour,
                    'approx': approx,
                    'vertices': len(approx),
                    'area': area,
                    'perimeter': perimeter,
                })

        return shapes

    def calculate_area(self, shape_type: str, **kwargs) -> float:
        """Calcula a área em unidades geométricas convertidas pelo scale_factor."""
        if shape_type == 'circle':
            radius = float(kwargs.get('radius', 0.0))
            return math.pi * (radius * self.scale_factor) ** 2

        if shape_type == 'rectangle':
            width = float(kwargs.get('width', 0.0))
            height = float(kwargs.get('height', 0.0))
            return (width * self.scale_factor) * (height * self.scale_factor)

        if shape_type == 'triangle':
            base = float(kwargs.get('base', 0.0))
            height = float(kwargs.get('height', 0.0))
            return (base * self.scale_factor) * (height * self.scale_factor) / 2

        if shape_type == 'polygon':
            vertices = kwargs.get('vertices', [])
            if len(vertices) < 3:
                return 0.0
            area = 0.0
            for i in range(len(vertices)):
                x1, y1 = vertices[i]
                x2, y2 = vertices[(i + 1) % len(vertices)]
                area += x1 * y2 - x2 * y1
            return abs(area / 2.0) * (self.scale_factor ** 2)

        raise ValueError(f'Tipo de forma não suportado: {shape_type}')

    def calculate_volume(self, shape_type: str, **kwargs) -> float:
        """Calcula volume para sólidos 3D."""
        if shape_type == 'cube':
            side = float(kwargs.get('side', 0.0))
            return (side * self.scale_factor) ** 3

        if shape_type == 'sphere':
            radius = float(kwargs.get('radius', 0.0))
            return (4 / 3) * math.pi * (radius * self.scale_factor) ** 3

        if shape_type == 'cylinder':
            radius = float(kwargs.get('radius', 0.0))
            height = float(kwargs.get('height', 0.0))
            return math.pi * (radius * self.scale_factor) ** 2 * (height * self.scale_factor)

        if shape_type == 'cone':
            radius = float(kwargs.get('radius', 0.0))
            height = float(kwargs.get('height', 0.0))
            return (1 / 3) * math.pi * (radius * self.scale_factor) ** 2 * (height * self.scale_factor)

        if shape_type == 'pyramid':
            base_area = float(kwargs.get('base_area', 0.0))
            height = float(kwargs.get('height', 0.0))
            return (1 / 3) * (base_area * self.scale_factor ** 2) * (height * self.scale_factor)

        raise ValueError(f'Tipo de volume não suportado: {shape_type}')

    def calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calcula a distância euclidiana entre dois pontos."""
        x1, y1 = point1
        x2, y2 = point2
        d = math.hypot(x2 - x1, y2 - y1)
        return d * self.scale_factor

    def annotate_image(self, image: np.ndarray, shapes: Dict[str, List[Dict]]) -> np.ndarray:
        """Desenha os contornos detectados na imagem original."""
        output = image.copy()

        for circle in shapes.get('circles', []):
            contour = circle['contour']
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius_int = int(radius)
            cv2.circle(output, center, radius_int, (0, 255, 0), 2)

        for rectangle in shapes.get('rectangles', []):
            contour = rectangle['contour']
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)

        for triangle in shapes.get('triangles', []):
            cv2.drawContours(output, [triangle['approx']], -1, (0, 0, 255), 2)

        for polygon in shapes.get('polygons', []):
            cv2.drawContours(output, [polygon['approx']], -1, (255, 255, 0), 2)

        return output


if __name__ == '__main__':
    measurer = GeometryMeasurer(scale_factor=0.1)
    images = measurer.pdf_to_images('exemplo.pdf')
    print(f'Páginas lidas: {len(images)}')

    if images:
        shapes = measurer.detect_shapes(images[0])
        print('Formas detectadas:')
        for key, value in shapes.items():
            print(f'  {key}: {len(value)}')

        print('Exemplo área circulo:', measurer.calculate_area('circle', radius=10))
        print('Exemplo volume cilindro:', measurer.calculate_volume('cylinder', radius=10, height=20))
        print('Exemplo distância:', measurer.calculate_distance((0, 0), (100, 100)))
