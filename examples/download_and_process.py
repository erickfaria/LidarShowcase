"""
Exemplo: Download e processamento de arquivos LAZ
"""

import os
import sys

# Adicionar o diretório pai ao path para importar o pacote lidar
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lidar.downloader import download_files_parallel
from lidar.processor import LazProcessor, ensure_lazrs_installed

def main():
    # Garantir que as dependências estão instaladas
    if not ensure_lazrs_installed():
        print("Erro ao instalar dependências necessárias.")
        return
    
    # Diretório de saída
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(output_dir, exist_ok=True)
    
    # URLs de exemplo
    urls = [
        "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/NE_Statewide_D23/NE_Statewide_3_D23/LAZ/USGS_LPC_NE_Statewide_D23_13T_GH_2362.laz",
    ]
    
    # 1. Download dos arquivos
    print("=== Etapa 1: Baixando arquivos LAZ ===")
    downloaded = download_files_parallel(urls, output_dir, max_workers=2)
    print(f"Baixados {len(downloaded)} arquivos para {output_dir}")
    
    # 2. Processamento dos arquivos LAZ para LAS
    print("\n=== Etapa 2: Convertendo arquivos LAZ para LAS ===")
    processor = LazProcessor(output_dir)
    processor.process_all_files(output_dir=output_dir)
    
    print("\nProcessamento completo! Os arquivos LAS estão disponíveis em:", output_dir)

if __name__ == "__main__":
    main()
