# PDF Geometry Measurer

Ferramenta para ler páginas de PDF, converter para imagens, identificar figuras geométricas e calcular:
- área
- volume
- distância

## Requisitos

- Python 3.10+
- pip

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# ou .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Executar exemplo

```bash
python example.py
```

## Estrutura do projeto

- `pdf_geometry_measurer.py`: lógica principal
- `example.py`: demonstração prática
- `requirements.txt`: dependências

## Exemplo de uso em código

```python
from pdf_geometry_measurer import GeometryMeasurer

measurer = GeometryMeasurer(scale_factor=0.1)
images = measurer.pdf_to_images('exemplo.pdf')
shapes = measurer.detect_shapes(images[0])

print('Círculos:', len(shapes['circles']))
print('Retângulos:', len(shapes['rectangles']))
print('Triângulos:', len(shapes['triangles']))

area = measurer.calculate_area('circle', radius=50)
volume = measurer.calculate_volume('cylinder', radius=10, height=20)
distance = measurer.calculate_distance((0, 0), (100, 100))

print('Área:', area)
print('Volume:', volume)
print('Distância:', distance)
```

## Observações

- `scale_factor` converte pixels em unidades reais, por exemplo: `0.1` significa 1 pixel = 0,1 unidade.
- A detecção de formas funciona melhor quando a figura está bem contrastada e com bordas claras.
- Para PDFs com texto e gráficos misturados, pode ser necessário pré-processar a imagem antes da medição.

## Licença

MIT
