"""
Main entry point for TC Forecast
Demonstrates complete workflow from data loading to prediction
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings


import sys
from pathlib import Path


# Add tc_forecast to path
sys.path.insert(0, str(Path(__file__).parent))


from tc_forecast import TCDataLoader, TCForecastModel, TCTrainer, load_config




def main():
   """
   Complete workflow example
  
   Steps:
   1. Load configuration
   2. Load and prepare data
   3. Build model
   4. Train model
   5. Evaluate on test set
   """
  
   # =========================================================================
   # 1. Load Configuration
   # =========================================================================
   print("Step 1: Loading configuration...")
   config_path = Path(__file__).parent / "tc_forecast" / "config.yaml"
   config = load_config(config_path)
   print(f"Config loaded from {config_path}")
   print(f"  - Model type: {config.model.type}")
   print(f"  - Sequence length: {config.data.sequence_length}")
   print(f"  - Forecast steps: {config.data.forecast_steps}")
  
   # =========================================================================
   # 2. Load and Prepare Data
   # =========================================================================
   print("\nStep 2: Loading and preparing data...")
   data_loader = TCDataLoader(config)
  
   try:
       X_train, y_train, X_val, y_val, X_test, y_test = data_loader.load_and_prepare()
       print("Data loading complete")
   except Exception as e:
       print(f"Error loading data: {e}")
       return
  
   # =========================================================================
   # 3. Build Model
   # =========================================================================
   print("\nStep 3: Building model...")
   model = TCForecastModel(config)
   keras_model = model.build()
   print("Model built successfully")
   print(f"  - Total parameters: {model.count_parameters():,}")
  
   # =========================================================================
   # 4. Train Model
   # =========================================================================
   print("\nStep 4: Training model...")
   trainer = TCTrainer(config, model)
  
   try:
       history = trainer.train(X_train, y_train, X_val, y_val)
       print("Training complete")
   except Exception as e:
       print(f"Error during training: {e}")
       import traceback
       traceback.print_exc()
       return
  
   # =========================================================================
   # 5. Evaluate Model
   # =========================================================================
   print("\nStep 5: Evaluating model...")
  
   try:
       # Use test metadata that was saved during data loading (NO DATA LEAKAGE)
       test_metadata = data_loader.metadata_test
      
       results = trainer.evaluate(X_test, y_test, test_metadata, data_loader)
       print("Evaluation complete")
      
       if 'distance_errors' in results:
           mean_72h = results['distance_errors']['mean'][-1]
           print(f"\n72-hour forecast error: {mean_72h:.2f} km")
  
   except Exception as e:
       print(f"Evaluation completed with warnings: {e}")
       import traceback
       traceback.print_exc()
  
   # =========================================================================
   # Summary
   # =========================================================================
   print(f"\nResults saved to: {trainer.output_path}")





if __name__ == "__main__":
   main()


