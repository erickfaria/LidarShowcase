# LiDAR Showcase

Este repositório contém exemplos e tutoriais para processamento de dados LiDAR (Light Detection and Ranging) utilizando Python.

## Estrutura do Projeto

- `data/`: Diretório para armazenar arquivos LAS/LAZ
- `processed_data/`: Diretório para armazenar resultados processados
- `intro_lidar_notebook.ipynb`: Notebook introdutório com operações básicas em dados LiDAR

## Configuração do Ambiente

### Requisitos

- Python 3.7+
- Dependências listadas em `requirements.txt`

### Instalação

1. Clone este repositório:
```bash
git clone https://github.com/seu-usuario/LidarShowcase.git
cd LidarShowcase
```

2. Crie e ative um ambiente virtual:
```bash
python -m venv venv
# No Windows
venv\Scripts\activate
# No Linux/Mac
source venv/bin/activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

1. Coloque seus arquivos LAS/LAZ na pasta `data/`
2. Execute o Jupyter Lab:
```bash
jupyter lab
```
3. Abra o notebook `intro_lidar_notebook.ipynb` para começar

## Funcionalidades Demonstradas

- Leitura de arquivos LAS/LAZ
- Visualização 2D e 3D de nuvens de pontos
- Análise de atributos (elevação, intensidade, classificação)
- Filtragem e segmentação de pontos
- Criação de modelos digitais de terreno simplificados
- Exportação de resultados processados

## Recursos Adicionais

- [Documentação do laspy](https://laspy.readthedocs.io/)
- [Documentação do Open3D](http://www.open3d.org/docs/)
- [Padrão LAS](https://www.asprs.org/divisions-committees/lidar-division/laser-las-file-format-exchange-activities)

## Exemplos

Veja a pasta `examples/` para exemplos de uso:

- `download_sample.py`: Exemplo simples de download de arquivo
- `download_and_process.py`: Exemplo completo de fluxo de trabalho

## Licença

Este projeto está licenciado sob a licença MIT.
