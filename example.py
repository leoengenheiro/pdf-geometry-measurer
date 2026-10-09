from pdf_geometry_measurer import GeometryMeasurer


def main():
    measurer = GeometryMeasurer(scale_factor=0.1)

    pdf_path = 'exemplo.pdf'
    images = measurer.pdf_to_images(pdf_path)

    print(f'PDF processado com {len(images)} página(s).')

    for idx, image in enumerate(images):
        shapes = measurer.detect_shapes(image)
        annotated = measurer.annotate_image(image, shapes)

        print(f'Página {idx + 1}:')
        print(f'  círculos: {len(shapes["circles"])}')
        print(f'  retângulos: {len(shapes["rectangles"])}')
        print(f'  triângulos: {len(shapes["triangles"])}')
        print(f'  polígonos: {len(shapes["polygons"])}')

        area = measurer.calculate_area('circle', radius=25)
        volume = measurer.calculate_volume('cylinder', radius=10, height=30)
        distancia = measurer.calculate_distance((0, 0), (100, 100))

        print(f'  exemplo área: {area}')
        print(f'  exemplo volume: {volume}')
        print(f'  exemplo distância: {distancia}')

        # Salvar imagem anotada para visualização
        import cv2
        cv2.imwrite(f'pagina_{idx + 1}_medida.png', annotated)


if __name__ == '__main__':
    main()
