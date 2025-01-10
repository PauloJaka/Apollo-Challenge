from data_processing import DataProcessor
from data_visualization import analyze_and_create_collage

pickle_path = "/media/paulo-jaka/Extras/DesafiosML/mini_gm_public_v0.1.p"
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
else:
    print("[ERROR] DataFrame is empty or failed to load.")
