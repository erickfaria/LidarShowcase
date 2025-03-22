# LidarShowcase

Ferramenta simples para download e processamento de arquivos LiDAR em formato LAZ para LAS.

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/LidarShowcase.git
cd LidarShowcase
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

### Interface de linha de comando

Para baixar e processar arquivos em uma única operação:

```bash
python main.py --output ./data
```

Para especificar URLs em um arquivo de texto:
```bash
python main.py --input urls.txt --output ./data
```

### Como módulo Python

```python
from lidar.downloader import download_files_parallel
from lidar.processor import LazProcessor

# Download de arquivos
urls = ["https://example.com/file.laz"]
output_dir = "./data"
downloaded = download_files_parallel(urls, output_dir)

# Processamento LAZ para LAS
processor = LazProcessor(output_dir)
processor.process_all_files(output_dir=output_dir)
```

## Exemplos

Veja a pasta `examples/` para exemplos de uso:

- `download_sample.py`: Exemplo simples de download de arquivo
- `download_and_process.py`: Exemplo completo de fluxo de trabalho

## Licença

Este projeto está licenciado sob a licença MIT.
