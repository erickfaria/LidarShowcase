# LidarShowcase

## Um toolkit completo para processamento, visualização e análise de dados LiDAR

![Banner LiDAR](https://github.com/erickfaria/LidarShowcase/raw/main/assets/banner_lidar.png)

## Visão Geral

**LidarShowcase** é uma suíte de ferramentas Python para o processamento abrangente de dados LiDAR (Light Detection and Ranging). Este projeto oferece uma pipeline completa para trabalhar com nuvens de pontos 3D, desde a leitura inicial até a geração de modelos digitais de terreno e análises avançadas de superfície.

Desenvolvido para cientistas de dados espaciais, geólogos, engenheiros civis e pesquisadores ambientais, o LidarShowcase facilita a extração de informações valiosas de conjuntos de dados LiDAR com uma interface simples e intuitiva implementada em Jupyter Notebooks.

## Características Principais

- **Carregamento e Exploração de Dados**: Interface intuitiva para seleção e carregamento de arquivos LAS/LAZ
- **Visualização Avançada**:
  - Renderização 2D com controle de densidade
  - Visualização 3D interativa via Plotly e Open3D
  - Codificação por cores baseada em atributos (elevação, intensidade, classificação)

![Visualização 3D](https://github.com/erickfaria/LidarShowcase/raw/main/assets/3d_visualization.png)

- **Geração de MDT (Modelo Digital de Terreno)**:
  - Múltiplos métodos de interpolação
  - Resolução personalizável
  - Pós-processamento (preenchimento de lacunas, suavização)

![MDT Exemplo](https://github.com/erickfaria/LidarShowcase/raw/main/assets/dtm_example.png)

- **Análise de Terreno**:
  - Cálculo de inclinação (slope)
  - Aspecto (aspect)
  - Hillshade (sombreamento)
  - Curvatura
  - Geração de curvas de nível com intervalos personalizáveis

![Análise de Terreno](https://github.com/erickfaria/LidarShowcase/raw/main/assets/terrain_analysis.png)

- **Exportação de Dados**:
  - Suporte a múltiplos formatos (GeoTIFF, ASC, PNG, XYZ)
  - Metadados e georreferenciamento preservados

## Estrutura do Projeto

```
LidarShowcase/
├── lidar/                           # Pacote principal
│   ├── lidarReadAndViewer.py        # Leitura e visualização de dados
│   ├── lidarDTM.py                  # Geração e análise de MDT
│   └── ...
├── data/                            # Pasta para armazenar arquivos LiDAR
├── processed_data/                  # Resultados exportados
├── examples/                        # Jupyter notebooks de demonstração
│   ├── lidarReadAndViewer.ipynb     # Tutorial de visualização
│   ├── createLidarMDT.ipynb         # Tutorial de criação de MDT
│   ├── download_sample.py           # Exemplo simples de download de arquivo
│   └── download_and_process.py      # Exemplo completo de fluxo de trabalho
├── assets/                          # Imagens e recursos
└── README.md                        # Este documento
```

## Requisitos e Instalação

### Pré-requisitos

- Python 3.7+
- Jupyter Notebook/Lab

### Instalação

```bash
# Clonar o repositório
git clone https://github.com/erickfaria/LidarShowcase.git
cd LidarShowcase

# Configurar ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

## Como Usar

1. **Preparação de Dados**
   - Coloque seus arquivos LAS/LAZ na pasta `data/`

2. **Exploração e Visualização**
   - Abra o notebook `examples/lidarReadAndViewer.ipynb`
   - Siga as instruções para explorar visualizações 2D/3D e atributos

   ![Exemplo de Exploração](https://github.com/erickfaria/LidarShowcase/raw/main/assets/exploration_example.png)

3. **Geração de Modelo Digital de Terreno**
   - Abra o notebook `examples/createLidarMDT.ipynb`
   - Ajuste parâmetros para criar MDTs otimizados
   - Explore análises derivadas como slope, aspect e hillshade

   ![Exemplo de MDT](https://github.com/erickfaria/LidarShowcase/raw/main/assets/dtm_workflow.png)

## Exemplos de Aplicação

### Mapeamento Topográfico

O LidarShowcase permite criar mapas topográficos detalhados a partir de dados LiDAR. Através da geração de MDT e curvas de nível, é possível obter representações precisas do terreno.

![Mapa Topográfico](https://github.com/erickfaria/LidarShowcase/raw/main/assets/topographic_map.png)

### Análise de Relevo

Utilizando as funcionalidades de análise de terreno, é possível identificar características geomorfológicas importantes como declives acentuados, planícies e padrões de drenagem.

![Análise de Relevo](https://github.com/erickfaria/LidarShowcase/raw/main/assets/relief_analysis.png)

### Visualização Avançada

As ferramentas de visualização 3D permitem explorar interativamente a nuvem de pontos, facilitando a identificação de estruturas e padrões nos dados LiDAR.

![Visualização Avançada](https://github.com/erickfaria/LidarShowcase/raw/main/assets/advanced_visualization.png)

## Tecnologias Utilizadas

- **Processamento LiDAR**: laspy, pylas
- **Análise Espacial**: numpy, scipy
- **Visualização**: matplotlib, plotly, Open3D
- **Interface Interativa**: IPython, ipywidgets
- **Processamento Geoespacial**: rasterio, gdal

## Recursos Adicionais

- [Documentação do laspy](https://laspy.readthedocs.io/)
- [Documentação do Open3D](http://www.open3d.org/docs/)
- [Padrão LAS](https://www.asprs.org/divisions-committees/lidar-division/laser-las-file-format-exchange-activities)

## Roadmap

Funcionalidades planejadas para futuras versões:

- Classificação automática de pontos LiDAR
- Detecção e extração de objetos (árvores, edificações)
- Comparação de múltiplos conjuntos de dados LiDAR
- Interface web para visualização e processamento remoto
- Suporte para processar datasets muito grandes (> 1 bilhão de pontos)

## Contribuições

Contribuições são bem-vindas! Se você deseja contribuir para o LidarShowcase:

1. Faça um fork do repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -m 'Adicionando nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

Para problemas, sugestões ou dúvidas, por favor abra uma issue no GitHub.

## Licença

Este projeto está licenciado sob a licença MIT.

## Contato

Erick Faria - [GitHub](https://github.com/erickfaria)

Link do projeto: [https://github.com/erickfaria/LidarShowcase](https://github.com/erickfaria/LidarShowcase)

---

<p align="center">
  <i>Desenvolvido com ❤️ para a comunidade geoespacial</i>
</p>
