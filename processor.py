import os
import laspy
import concurrent.futures
from tqdm import tqdm
import argparse

class LazProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
    def get_laz_files(self):
        """Return all LAZ files in the data directory"""
        return [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) 
                if f.lower().endswith('.laz')]
    
    def process_file(self, file_path, output_dir=None):
        """Convert a single LAZ file to LAS format"""
        try:
            # Set output directory to same as input if not specified
            if output_dir is None:
                output_dir = os.path.dirname(file_path)
            
            # Make sure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Get filename without extension
            base_name = os.path.basename(file_path)
            name_no_ext = os.path.splitext(base_name)[0]
            
            # Read the LAZ file and convert to LAS
            with laspy.open(file_path) as f:
                las = f.read()
                
                # Save as LAS
                las_path = os.path.join(output_dir, f"{name_no_ext}.las")
                las.write(las_path)
            
            return {
                "file": base_name,
                "status": "success",
                "output": las_path
            }
        
        except Exception as e:
            return {
                "file": os.path.basename(file_path),
                "status": "error",
                "error": str(e)
            }
    
    def process_all_files(self, output_dir=None, workers=4):
        """Process all LAZ files using multiple workers"""
        files = self.get_laz_files()
        results = []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self.process_file, f, output_dir) for f in files]
            
            # Process with progress bar
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(files), desc="Convertendo LAZ para LAS"):
                results.append(future.result())
        
        # Summarize results
        successful = sum(1 for r in results if r["status"] == "success")
        print(f"Convertidos com sucesso {successful} de {len(files)} arquivos")
        
        # Log errors
        errors = [r for r in results if r["status"] == "error"]
        if errors:
            print(f"Erros ocorreram em {len(errors)} arquivos:")
            for e in errors:
                print(f"  {e['file']}: {e['error']}")
        
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converter arquivos LAZ para LAS")
    parser.add_argument("--input", "-i", default="./data", help="Diretório contendo arquivos LAZ")
    parser.add_argument("--output", "-o", default="./data", help="Diretório de saída para arquivos LAS")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Número de processos trabalhadores")
    args = parser.parse_args()
    
    processor = LazProcessor(args.input)
    processor.process_all_files(output_dir=args.output, workers=args.workers)
