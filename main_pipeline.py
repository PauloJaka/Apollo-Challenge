from data_processing import DataProcessor
from data_visualization import analyze_and_create_collage
from classification_process import knn_classification
import config

pickle_path = config.PICKLE_PATH
processor = DataProcessor(pickle_path)

# load
df = processor.load_and_flatten_data()

if df is not None:
    # pipeline run
    output_path = 'syndrome_collage.png'
    analyze_and_create_collage(df, output_path)
    
    processor.calculate_statistics()  
    processor.print_statistics()  
    processor.generate_profile_report() 
    knn_classification(df)
else:
    print("[ERROR] DataFrame is empty or failed to load.")
