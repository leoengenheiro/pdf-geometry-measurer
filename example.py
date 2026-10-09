from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import fitz
import numpy as np


@dataclass
class ShapeResult:
    name: str
    kind: str
    area: float = 0.0
    perimeter: float = 0.0
    width: float = 0.0
    height: float = 0.0
    radius: float = 0.0
    volume: float = 0.0
    center: Tuple[float, float] | None = None
    contour: Any = None
    approx: Any = None


class GeometryMeasurer:
    """Lê PDFs, detecta figuras geométricas e mede área, volume e distância em unidades reais."""

    def __init__(self, scale_factor: float = 1.0, unit: str = 'mm'):
        self.scale_factor = float(scale_factor)
        self.unit = unit

    def pdf_to_images(self, pdf_path: str | Path, dpi: int = 220) -> List[np.ndarray]:
        """Converte cada página do PDF em uma imagem BGR."""
        doc = fitz.open(str(pdf_path))
        images: List[np.ndarray] = []

        for page in doc:
            matrix = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            images.append(img)

        return images

    def _convert_pixels_to_units(self, value_pixels: float) -> float:
        return value_pixels * self.scale_factor

    def _area_from_contour(self, contour: np.ndarray) -> float:
        return float(cv2.contourArea(contour))

    def detect_shapes(self, image: np.ndarray) -> List[ShapeResult]:
        """Detecta círculos, retângulos, triângulos, polígonos e retorna resultados medidos."""
        if image is None or image.size == 0:
            raise ValueError('Imagem vazia ou inválida.')

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results: List[ShapeResult] = []

        for contour in contours:
            area_px = cv2.contourArea(contour)
            if area_px < 200:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue

            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            vertices = len(approx)

            if vertices == 3:
                result = ShapeResult(
                    name='triangulo',
                    kind='triangle',
                    area=self._convert_pixels_to_units(area_px),
                    perimeter=self._convert_pixels_to_units(perimeter),
                    contour=contour,
                    approx=approx,
                )
                results.append(result)
                continue

            if vertices == 4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect = float(w) / float(h) if h else 0.0
                if 0.8 <= aspect <= 1.2:
                    area = self._convert_pixels_to_units(area_px)
                    result = ShapeResult(
                        name='circulo',
                        kind='circle',
                        area=area,
                        perimeter=self._convert_pixels_to_units(perimeter),
                        radius=self._convert_pixels_to_units(math.sqrt(area_px / math.pi)),
                        center=(float(x + w / 2), float(y + h / 2)),
                        contour=contour,
                        approx=approx,
                    )
                    results.append(result)
                else:
                    result = ShapeResult(
                        name='retangulo',
                        kind='rectangle',
                        area=self._convert_pixels_to_units(area_px),
                        perimeter=self._convert_pixels_to_units(perimeter),
                        width=self._convert_pixels_to_units(w),
                        height=self._convert_pixels_to_units(h),
                        contour=contour,
                        approx=approx,
                    )
                    results.append(result)
                continue

            if vertices >= 5:
                area = self._convert_pixels_to_units(area_px)
                result = ShapeResult(
                    name='poligono',
                    kind='polygon',
                    area=area,
                    perimeter=self._convert_pixels_to_units(perimeter),
                    contour=contour,
                    approx=approx,
                )
                results.append(result)

        return results

    def calculate_area(self, shape_type: str, **kwargs) -> float:
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
            return (base * self.scale_factor) * (height * self.scale_factor) / 2.0

        if shape_type == 'polygon':
            vertices = kwargs.get('vertices', [])
            if len(vertices) < 3:
                return 0.0
            s = 0.0
            for i in range(len(vertices)):
                x1, y1 = vertices[i]
                x2, y2 = vertices[(i + 1) % len(vertices)]
                s += x1 * y2 - x2 * y1
            return abs(s / 2.0) * (self.scale_factor ** 2)

        raise ValueError(f'Tipo de área não suportado: {shape_type}')

    def calculate_volume(self, shape_type: str, **kwargs) -> float:
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
            return (1 / 3) * base_area * (self.scale_factor ** 2) * (height * self.scale_factor)

        raise ValueError(f'Tipo de volume não suportado: {shape_type}')

    def calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        x1, y1 = point1
        x2, y2 = point2
        return math.hypot(x2 - x1, y2 - y1) * self.scale_factor

    def annotate_image(self, image: np.ndarray, shapes: List[ShapeResult], output_path: str | None = None) -> np.ndarray:
        """Desenha contornos e rótulos na imagem original."""
        output = image.copy()

        for idx, shape in enumerate(shapes):
            if shape.approx is not None:
                color = (0, 255, 0)
                if shape.kind == 'rectangle':
                    color = (255, 0, 0)
                elif shape.kind == 'triangle':
                    color = (0, 0, 255)
                elif shape.kind == 'polygon':
                    color = (255, 255, 0)
                elif shape.kind == 'circle':
                    color = (0, 255, 255)
                cv2.drawContours(output, [shape.approx], -1, color, 2)

            if shape.center is not None:
                x, y = map(int, shape.center)
                label = f'{shape.name} {shape.area:.2f} {self.unit}²'
                cv2.putText(output, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if output_path:
            cv2.imwrite(output_path, output)

        return output

    def analyze_pdf(self, pdf_path: str | Path, output_dir: str | Path | None = None) -> List[Dict[str, Any]]:
        """Analisa todas as páginas do PDF e retorna resultados por página."""
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir) if output_dir else pdf_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        pages = self.pdf_to_images(pdf_path)
        report: List[Dict[str, Any]] = []

        for idx, image in enumerate(pages, start=1):
            shapes = self.detect_shapes(image)
            annotated = self.annotate_image(image, shapes, str(output_dir / f'page_{idx}_annotated.png'))

            page_items = []
            for s in shapes:
                page_items.append({
                    'name': s.name,
                    'kind': s.kind,
                    'area': round(s.area, 4),
                    'perimeter': round(s.perimeter, 4),
                    'width': round(s.width, 4) if s.width else None,
                    'height': round(s.height, 4) if s.height else None,
                    'radius': round(s.radius, 4) if s.radius else None,
                    'volume': round(s.volume, 4) if s.volume else None,
                    'unit': self.unit,
                })

            report.append({
                'page': idx,
                'num_shapes': len(page_items),
                'shapes': page_items,
                'output_image': str(output_dir / f'page_{idx}_annotated.png'),
            })

        return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Lê PDF e mede figuras geométricas.')
    parser.add_argument('--pdf', type=str, required=True, help='Caminho do PDF para analisar.')
    parser.add_argument('--scale', type=float, default=0.1, help='Fator de escala para converter pixels em unidades reais.')
    parser.add_argument('--unit', type=str, default='mm', help='Unidade de medida utilizada no desenho.')
    parser.add_argument('--output-dir', type=str, default='output', help='Diretório para salvar imagens processadas.')
    args = parser.parse_args()

    measurer = GeometryMeasurer(scale_factor=args.scale, unit=args.unit)
    report = measurer.analyze_pdf(args.pdf, args.output_dir)

    for page in report:
        print(f"Página {page['page']}: {page['num_shapes']} forma(s) detectada(s)")
        for shape in page['shapes']:
            print(f"  - {shape['name']}: área={shape['area']} {shape['unit']}²")
            if shape.get('width') is not None:
                print(f"    largura={shape['width']} {shape['unit']}, altura={shape['height']} {shape['unit']}")
            if shape.get('radius') is not None:
                print(f"    raio={shape['radius']} {shape['unit']}")


if __name__ == '__main__':
    main()


