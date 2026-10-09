from pathlib import Path

from pdf_geometry_measurer import GeometryMeasurer


def main():
    pdf_path = Path('exemplo.pdf')
    measurer = GeometryMeasurer(scale_factor=0.1, unit='mm')

    if not pdf_path.exists():
        print('Arquivo exemplo.pdf não encontrado.')
        print('Use o comando abaixo para analisar um PDF real:')
        print('python pdf_geometry_measurer.py --pdf seu_arquivo.pdf --scale 0.1 --unit mm')
        return

    report = measurer.analyze_pdf(pdf_path, 'output')

    for page in report:
        print(f'Página {page["page"]}: {page["num_shapes"]} formas detectadas')
        for shape in page['shapes']:
            print(f"  - {shape['name']}: área={shape['area']} mm²")


if __name__ == '__main__':
    main()


