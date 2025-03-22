"""
Exemplo: Download de arquivo LAZ individual
"""

import os
import sys

# Adicionar o diretório pai ao path para importar o pacote lidar
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lidar.downloader import download_file

def main():
    # URL de um arquivo LAZ de exemplo
    url = "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/NE_Statewide_D23/NE_Statewide_3_D23/LAZ/USGS_LPC_NE_Statewide_D23_13T_GH_2362.laz"
    
    # Diretório de saída
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Baixando arquivo de exemplo para {output_dir}...")
    file_path = download_file(url, output_dir)
    
    if file_path:
        print(f"Download concluído! Arquivo salvo em: {file_path}")
    else:
        print("Falha no download do arquivo.")

if __name__ == "__main__":
    main()
