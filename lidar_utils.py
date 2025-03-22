"""
Módulo de utilitários para processamento de dados LiDAR.
Este módulo contém funções e classes para carregamento, visualização,
análise e exportação de dados LiDAR.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import laspy
import open3d as o3d
import plotly.graph_objects as go
from IPython.display import display
import ipywidgets as widgets


class LidarFileHandler:
    """Classe para gerenciar arquivos LiDAR e fornecer widgets para seleção."""
    
    @staticmethod
    def list_las_files(data_dir="data"):
        """Lista arquivos LAS/LAZ em um diretório."""
        las_files = [f for f in os.listdir(data_dir) if f.endswith(('.las', '.laz'))]
        return las_files, data_dir
    
    @staticmethod
    def create_file_selector(las_files, data_dir):
        """Cria um widget para seleção de arquivos."""
        selected_file_path = os.path.join(data_dir, las_files[0]) if las_files else None
        
        def select_file(change):
            nonlocal selected_file_path
            selected_file_path = os.path.join(data_dir, change['new'])
            print(f"Arquivo selecionado: {selected_file_path}")
        
        file_dropdown = widgets.Dropdown(
            options=las_files,
            description='Selecione o arquivo:',
            style={'description_width': 'initial'}
        )
        
        file_dropdown.observe(select_file, names='value')
        return file_dropdown, selected_file_path
    
    @staticmethod
    def read_las_file(file_path):
        """Lê um arquivo LAS/LAZ."""
        try:
            las_data = laspy.read(file_path)
            print("\nArquivo carregado com sucesso!")
            return las_data
        except Exception as e:
            print(f"Erro ao carregar o arquivo: {e}")
            return None
    
    @staticmethod
    def export_subset(las_data, output_path, mask=None, header_changes=None):
        """Exporta um subconjunto de pontos para um novo arquivo LAS."""
        try:
            # Criar uma cópia dos pontos
            if mask is None:
                # Exportar todos os pontos
                subset = las_data.copy()
            else:
                # Exportar apenas os pontos que correspondem à máscara
                subset = las_data.points[mask].copy()
            
            # Aplicar alterações ao cabeçalho, se houver
            if header_changes:
                for key, value in header_changes.items():
                    if hasattr(subset.header, key):
                        setattr(subset.header, key, value)
            
            # Salvar o arquivo
            subset.write(output_path)
            print(f"Arquivo salvo com sucesso em: {output_path}")
            print(f"Total de pontos exportados: {subset.header.point_count:,}")
            return True
        except Exception as e:
            print(f"Erro ao exportar arquivo: {e}")
            return False


class LidarInfo:
    """Classe para obter e exibir informações sobre dados LiDAR."""
    
    @staticmethod
    def print_basic_info(las_data):
        """Exibe informações básicas sobre o arquivo LAS/LAZ."""
        if las_data is None:
            return
        
        print(f"Versão do formato LAS: {las_data.header.version}")
        print(f"Número total de pontos: {las_data.header.point_count}")
        print(f"Extensão dos dados:")
        print(f"  X: {las_data.header.mins[0]:.2f} a {las_data.header.maxs[0]:.2f}")
        print(f"  Y: {las_data.header.mins[1]:.2f} a {las_data.header.maxs[1]:.2f}")
        print(f"  Z: {las_data.header.mins[2]:.2f} a {las_data.header.maxs[2]:.2f}")
        
        # Mostrar as dimensões disponíveis (atributos) nos dados
        print("\nAtributos disponíveis:")
        for dimension in las_data.point_format.dimensions:
            print(f"  - {dimension.name}")
    
    @staticmethod
    def get_class_names():
        """Retorna um dicionário com os nomes das classes padrão LAS."""
        return {
            0: "Criado, nunca classificado",
            1: "Não classificado",
            2: "Solo",
            3: "Vegetação baixa",
            4: "Vegetação média",
            5: "Vegetação alta",
            6: "Edificação",
            7: "Ponto baixo (ruído)",
            8: "Ponto-chave",
            9: "Água",
            10: "Trilho ferrovia",
            11: "Superfície de rodovia",
            12: "Ponto sobreposto",
            13: "Fio - proteção",
            14: "Fio - condutor (fase)",
            15: "Torre de transmissão",
            16: "Conector de fio",
            17: "Cobertura da ponte",
            18: "Ruído alto"
        }


class LidarVisualization:
    """Classe para visualização de dados LiDAR."""
    
    @staticmethod
    def get_points(las_data, max_points=None):
        """Extrai coordenadas e cria array de pontos."""
        # Extrair coordenadas
        x = las_data.x
        y = las_data.y
        z = las_data.z
        
        # Limitar o número de pontos para visualização mais rápida (se necessário)
        if max_points and max_points < len(x):
            idx = np.random.choice(len(x), max_points, replace=False)
            x = x[idx]
            y = y[idx]
            z = z[idx]
        
        return np.vstack((x, y, z)).transpose()
    
    @staticmethod
    def plot_2d_view(las_data, max_points=100000, point_size=0.1, point_alpha=0.5, color_map='viridis', use_hexbin=False):
        """Cria uma visualização 2D (vista superior) da nuvem de pontos."""
        if las_data is None:
            return
        
        plt.figure(figsize=(10, 10))
        # Amostragem para tornar a visualização mais rápida
        sample_idx = np.random.choice(len(las_data.x), min(max_points, len(las_data.x)), replace=False)
        
        if use_hexbin:
            # Visualização usando hexbin - cria uma representação contínua
            hb = plt.hexbin(las_data.x[sample_idx], las_data.y[sample_idx], 
                           C=las_data.z[sample_idx], 
                           gridsize=100,    # Ajuste para controlar a resolução
                           cmap=color_map,
                           mincnt=1,        # Mostra células com pelo menos 1 ponto
                           alpha=point_alpha)
            plt.colorbar(hb, label='Elevação média (Z)')
        else:
            # Visualização usando scatter
            plt.scatter(las_data.x[sample_idx], las_data.y[sample_idx], 
                        c=las_data.z[sample_idx], cmap=color_map, 
                        s=point_size, alpha=point_alpha)
            plt.colorbar(label='Elevação (Z)')
        
        plt.title('Vista Superior da Nuvem de Pontos LiDAR')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_3d_plotly(las_data, max_points=200000, point_size=1, opacity=0.8, color_map='Viridis'):
        """Cria uma visualização 3D interativa usando Plotly."""
        if las_data is None:
            return
        
        # Amostragem para tornar a visualização mais rápida
        sample_size = min(max_points, len(las_data.x))
        sample_idx = np.random.choice(len(las_data.x), sample_size, replace=False)
        
        # Criar figura 3D
        fig = go.Figure(data=[go.Scatter3d(
            x=las_data.x[sample_idx],
            y=las_data.y[sample_idx],
            z=las_data.z[sample_idx],
            mode='markers',
            marker=dict(
                size=point_size,
                color=las_data.z[sample_idx],
                colorscale=color_map,
                opacity=opacity,
                colorbar=dict(title="Elevação (Z)")
            )
        )])
        
        fig.update_layout(
            title="Visualização 3D da Nuvem de Pontos LiDAR",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            width=900,
            height=700,
        )
        
        return fig
    
    @staticmethod
    def visualize_open3d(las_data, max_points=500000):
        """Visualiza a nuvem de pontos usando Open3D."""
        if las_data is None:
            return
        
        # Obter pontos (com limite para melhor desempenho)
        points = LidarVisualization.get_points(las_data, max_points=max_points)
        
        # Criar nuvem de pontos Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Normalizar alturas para coloração
        heights = points[:, 2]  # valores Z
        colors = plt.cm.viridis((heights - np.min(heights)) / (np.max(heights) - np.min(heights)))[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Instruções
        print("\nAbrindo visualizador Open3D (janela separada)...")
        print("Dicas de navegação:")
        print("- Rotação: clique e arraste com o botão esquerdo do mouse")
        print("- Zoom: roda do mouse ou botão direito + arrastar")
        print("- Movimento: Shift + clique e arraste")
        
        # Visualizar
        o3d.visualization.draw_geometries([pcd])
    
    @staticmethod
    def plot_elevation_histogram(las_data):
        """Plota um histograma da distribuição de alturas (Z)."""
        if las_data is None:
            return
        
        plt.figure(figsize=(12, 6))
        plt.hist(las_data.z, bins=100, alpha=0.7, color='green')
        plt.title('Distribuição das Alturas (Z)')
        plt.xlabel('Altura (Z)')
        plt.ylabel('Frequência')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Estatísticas básicas
        print(f"Estatísticas de altura (Z):")
        print(f"  Mínimo: {np.min(las_data.z):.2f}")
        print(f"  Máximo: {np.max(las_data.z):.2f}")
        print(f"  Média: {np.mean(las_data.z):.2f}")
        print(f"  Mediana: {np.median(las_data.z):.2f}")
        print(f"  Desvio padrão: {np.std(las_data.z):.2f}")
    
    @staticmethod
    def plot_intensity(las_data, max_points=100000):
        """Plota a distribuição de intensidade e visualização 2D colorida por intensidade."""
        if las_data is None or not hasattr(las_data, 'intensity'):
            return
        
        plt.figure(figsize=(12, 10))
        
        # Subplot 1: Histograma de intensidade
        plt.subplot(2, 1, 1)
        plt.hist(las_data.intensity, bins=100, alpha=0.7, color='purple')
        plt.title('Distribuição dos Valores de Intensidade')
        plt.xlabel('Intensidade')
        plt.ylabel('Frequência')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Visualização da intensidade na nuvem de pontos
        plt.subplot(2, 1, 2)
        sample_idx = np.random.choice(len(las_data.x), min(max_points, len(las_data.x)), replace=False)
        plt.scatter(las_data.x[sample_idx], las_data.y[sample_idx], 
                    c=las_data.intensity[sample_idx], cmap='inferno', s=0.1, alpha=0.7)
        plt.colorbar(label='Intensidade')
        plt.title('Vista Superior com Valores de Intensidade')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_classification_stats(las_data):
        """Plota estatísticas de classificação dos pontos."""
        if las_data is None or not hasattr(las_data, 'classification'):
            return
        
        class_names = LidarInfo.get_class_names()
        
        # Contagem de pontos por classe
        unique_classes, counts = np.unique(las_data.classification, return_counts=True)
        class_counts = dict(zip(unique_classes, counts))
        
        # Criar labels para o gráfico
        labels = [f"{c} - {class_names.get(c, 'Desconhecido')} ({class_counts[c]})" for c in unique_classes]
        
        # Plotar distribuição de classes
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(unique_classes)), counts, alpha=0.7)
        plt.xticks(range(len(unique_classes)), [str(c) for c in unique_classes], rotation=45)
        plt.title('Distribuição das Classes de Pontos')
        plt.xlabel('Código de Classificação')
        plt.ylabel('Número de Pontos')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Adicionar labels no topo das barras
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:,}', ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        plt.show()
        
        # Mostrar descrição das classes encontradas
        print("\nClasses identificadas no conjunto de dados:")
        for c in unique_classes:
            percentage = (class_counts[c] / len(las_data.classification)) * 100
            print(f"  Classe {c} - {class_names.get(c, 'Desconhecido')}: {class_counts[c]:,} pontos ({percentage:.2f}%)")
    
    @staticmethod
    def plot_points_by_class(las_data, class_value, max_points=None, point_size=0.5, title=None):
        """
        Visualiza pontos com uma classe específica ou lista de classes.
        
        Args:
            las_data: Dados LAS carregados
            class_value: Valor inteiro da classe ou lista de valores de classe a serem visualizados
            max_points: Número máximo de pontos a serem plotados (para melhor desempenho)
            point_size: Tamanho dos pontos no gráfico
            title: Título personalizado para o gráfico
        """
        # Verificar se class_value é uma lista ou um valor único
        if not isinstance(class_value, list):
            class_value = [class_value]
        
        # Criar máscara para pontos das classes desejadas
        class_mask = np.isin(las_data.classification, class_value)
        
        # Filtrar pontos
        x = las_data.x[class_mask]
        y = las_data.y[class_mask]
        z = las_data.z[class_mask]
        
        # Limitar o número de pontos se necessário
        if max_points and len(x) > max_points:
            idx = np.random.choice(len(x), max_points, replace=False)
            x = x[idx]
            y = y[idx]
            z = z[idx]
        
        # Criar visualização
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Definir título do gráfico
        if title:
            plot_title = title
        elif len(class_value) == 1:
            plot_title = f"Pontos da Classe {class_value[0]}"
        else:
            plot_title = f"Pontos das Classes {', '.join(map(str, class_value))}"
        
        # Criar scatter plot com cores baseadas na elevação
        scatter = ax.scatter(x, y, c=z, s=point_size, cmap='viridis', alpha=0.7)
        
        # Adicionar barra de cores e legendas
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Elevação (m)')
        
        ax.set_title(plot_title)
        ax.set_xlabel('Coordenada X')
        ax.set_ylabel('Coordenada Y')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
        
        # Retornar os pontos filtrados
        return x, y, z


class LidarProcessing:
    """Classe para processamento de dados LiDAR."""
    
    @staticmethod
    def filter_ground_points(las_data, visualize=True, max_points=100000):
        """Filtra e opcionalmente visualiza pontos de solo (classe 2)."""
        if las_data is None or not hasattr(las_data, 'classification'):
            return None
        
        # Verificar se existem pontos de solo (classe 2)
        if 2 in np.unique(las_data.classification):
            # Filtrar pontos de solo
            ground_mask = las_data.classification == 2
            ground_points = {
                'x': las_data.x[ground_mask],
                'y': las_data.y[ground_mask],
                'z': las_data.z[ground_mask]
            }
            
            print(f"Pontos de solo extraídos: {len(ground_points['x']):,} de {len(las_data.x):,} ({len(ground_points['x'])/len(las_data.x)*100:.2f}%)")
            
            if visualize and len(ground_points['x']) > 0:
                # Visualizar pontos de solo
                plt.figure(figsize=(10, 10))
                # Amostragem para visualização mais rápida
                sample_size = min(max_points, len(ground_points['x']))
                sample_idx = np.random.choice(len(ground_points['x']), sample_size, replace=False)
                
                plt.scatter(ground_points['x'][sample_idx], ground_points['y'][sample_idx], 
                            c=ground_points['z'][sample_idx], cmap='terrain', s=0.5, alpha=0.7)
                plt.colorbar(label='Elevação (Z)')
                plt.title('Pontos Classificados como Solo (Classe 2)')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.axis('equal')
                plt.grid(True, alpha=0.3)
                plt.show()
            
            return ground_mask, ground_points
        else:
            print("Não foram encontrados pontos classificados como solo (classe 2) no arquivo.")
            return None, None
    
    @staticmethod
    def create_simple_dtm(las_data, resolution=1.0):
        """Cria um modelo digital de terreno (MDT) simples."""
        if las_data is None:
            return
        
        # Usar pontos de solo se disponíveis, caso contrário usar todos os pontos
        if hasattr(las_data, 'classification') and 2 in np.unique(las_data.classification):
            mask = las_data.classification == 2
            x = las_data.x[mask]
            y = las_data.y[mask]
            z = las_data.z[mask]
            print("Criando MDT a partir dos pontos classificados como solo.")
        else:
            x = las_data.x
            y = las_data.y
            z = las_data.z
            print("Criando MDT a partir de todos os pontos (sem filtragem por classe).")
        
        # Calcular os limites da grade
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        
        # Criar grade
        x_grid = np.arange(x_min, x_max, resolution)
        y_grid = np.arange(y_min, y_max, resolution)
        xx, yy = np.meshgrid(x_grid, y_grid)
        
        # Inicializar grade de elevação com NaN
        elevation_grid = np.full(xx.shape, np.nan)
        
        # Preencher grade - abordagem simplificada usando bins 2D
        x_bins = np.searchsorted(x_grid, x) - 1
        y_bins = np.searchsorted(y_grid, y) - 1
        
        # Filtrar índices fora dos limites
        valid_idx = (x_bins >= 0) & (y_bins >= 0) & (x_bins < len(x_grid)) & (y_bins < len(y_grid))
        x_bins = x_bins[valid_idx]
        y_bins = y_bins[valid_idx]
        z_vals = z[valid_idx]
        
        # Para cada célula da grade, encontrar o ponto com menor Z (simplificação do MDT)
        for i in range(len(x_grid)):
            for j in range(len(y_grid)):
                cell_points = (x_bins == i) & (y_bins == j)
                if np.any(cell_points):
                    elevation_grid[j, i] = np.min(z_vals[cell_points])  # Use mínimo para MDT
        
        # Retornar os dados do MDT para visualização
        return {
            'elevation_grid': elevation_grid,
            'x_grid': x_grid,
            'y_grid': y_grid,
            'xx': xx,
            'yy': yy,
            'extent': [x_min, x_max, y_min, y_max]
        }
    
    @staticmethod
    def create_interpolated_dtm(las_data, resolution=1.0, method='linear'):
        """
        Cria um Modelo Digital de Terreno (MDT) interpolado a partir dos pontos de solo.
        
        Parâmetros:
            las_data (laspy.LasData): Dados LiDAR
            resolution (float): Resolução do MDT em metros
            method (str): Método de interpolação ('linear', 'cubic', 'nearest')
        
        Retorna:
            dict: Dicionário contendo os dados do MDT, incluindo:
                - grid_x, grid_y: Coordenadas da grade
                - dem: Matriz de elevação
                - resolution: Resolução usada
                - bounds: Limites da área
        """
        import numpy as np
        from scipy.interpolate import griddata
        
        # Filtrar apenas pontos classificados como solo (classe 2)
        if hasattr(las_data, 'classification'):
            ground_mask = las_data.classification == 2
            if np.sum(ground_mask) == 0:
                print("Aviso: Nenhum ponto classificado como solo encontrado. Usando todos os pontos.")
                ground_mask = np.ones_like(las_data.classification, dtype=bool)
        else:
            print("Aviso: Classificação não disponível. Usando todos os pontos.")
            ground_mask = np.ones(len(las_data.x), dtype=bool)
        
        # Extrair pontos de solo
        x = las_data.x[ground_mask]
        y = las_data.y[ground_mask]
        z = las_data.z[ground_mask]
        
        # Verificar se há pontos suficientes
        if len(x) < 3:
            raise ValueError("Não há pontos suficientes para criar um MDT.")
        
        # Calcular limites da área
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        
        # Criar grade regular
        grid_x = np.arange(x_min, x_max + resolution, resolution)
        grid_y = np.arange(y_min, y_max + resolution, resolution)
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
        
        # Realizar interpolação
        points = np.column_stack((x, y))
        grid_z = griddata(points, z, (mesh_x, mesh_y), method=method)
        
        # Preencher valores nulos (pode ocorrer com cubic)
        if method == 'cubic':
            mask = np.isnan(grid_z)
            if np.any(mask):
                grid_z[mask] = griddata(points, z, (mesh_x[mask], mesh_y[mask]), method='nearest')
        
        # Criar dicionário de saída com os dados processados
        dtm_data = {
            'grid_x': grid_x,
            'grid_y': grid_y,
            'dem': grid_z,
            'resolution': resolution,
            'bounds': (x_min, x_max, y_min, y_max)
        }
        
        return dtm_data
    
    @staticmethod
    def visualize_enhanced_dtm(dtm_data, cmap='terrain', alpha=0.8):
        """
        Visualiza um MDT com realce de relevo (hillshade).
        
        Parâmetros:
            dtm_data (dict): Dicionário contendo os dados do MDT
            cmap (str): Mapa de cores para visualização
            alpha (float): Transparência do MDT sobre o hillshade
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import LightSource
        
        # Extrair dados do dicionário
        grid_x = dtm_data['grid_x']
        grid_y = dtm_data['grid_y']
        dem = dtm_data['dem']
        
        # Criar fonte de luz para hillshade
        ls = LightSource(azdeg=315, altdeg=45)
        
        # Calcular hillshade
        hillshade = ls.hillshade(dem, vert_exag=3)
        
        # Configurar figura
        plt.figure(figsize=(12, 10))
        
        # Renderizar hillshade com MDT colorido
        rgb = ls.blend_overlay(hillshade, dem, cmap=cmap)
        plt.imshow(rgb, extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()], 
                   origin='lower', aspect='equal')
        
        # Adicionar barra de cores
        cbar = plt.colorbar(label='Elevação (m)')
        cbar.ax.tick_params(labelsize=8)
        
        plt.title('Modelo Digital de Terreno com Realce de Relevo')
        plt.xlabel('Coordenada X (m)')
        plt.ylabel('Coordenada Y (m)')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_slope(dtm_data, cmap='viridis'):
        """
        Visualiza o mapa de declividade derivado do MDT.
        
        Parâmetros:
            dtm_data (dict): Dicionário contendo os dados do MDT
            cmap (str): Mapa de cores para visualização
        """
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Extrair dados do dicionário
        grid_x = dtm_data['grid_x']
        grid_y = dtm_data['grid_y']
        dem = dtm_data['dem']
        resolution = dtm_data['resolution']
        
        # Calcular gradientes
        dy, dx = np.gradient(dem, resolution, resolution)
        
        # Calcular declividade em graus
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        
        # Plotar mapa de declividade
        plt.figure(figsize=(12, 10))
        slope_img = plt.imshow(slope, cmap=cmap, extent=[grid_x.min(), grid_x.max(), 
                                                        grid_y.min(), grid_y.max()], 
                              origin='lower', aspect='equal')
        
        # Adicionar barra de cores
        cbar = plt.colorbar(slope_img, label='Declividade (graus)')
        cbar.ax.tick_params(labelsize=8)
        
        plt.title('Mapa de Declividade')
        plt.xlabel('Coordenada X (m)')
        plt.ylabel('Coordenada Y (m)')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
