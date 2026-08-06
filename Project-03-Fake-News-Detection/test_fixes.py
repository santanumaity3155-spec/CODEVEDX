"""
Test script to verify Options 2, 3, and 6 work correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.predictor import predictor
from src.data_handler import data_handler
from src.utils import format_percentage

print("=" * 70)
print("TESTING FAKE NEWS DETECTION - OPTIONS 2, 3, AND 6")
print("=" * 70)

# Test 1: Single prediction (Option 2)
print("\n[TEST 1] Single Prediction - Fake News Article")
print("-" * 70)

fake_news_text = """
Scientists have confirmed that drinking two cups of coffee every hour 
can make people live up to 200 years old. This groundbreaking discovery 
was made by a team of researchers who found that coffee contains a 
mysterious compound that can reverse aging at the cellular level.
"""

result = predictor.predict(fake_news_text)
print(f"Success: {result['success']}")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {format_percentage(result['confidence'])}")
print(f"Processing Time: {result['processing_time']:.4f} seconds")

if result['success']:
    # Add to history
    history_data = {
        'timestamp': result['timestamp'],
        'input_length': result['input_length'],
        'prediction': result['prediction'],
        'confidence': f"{result['confidence']:.4f}",
        'probability_fake': f"{result['probability_fake']:.4f}",
        'probability_real': f"{result['probability_real']:.4f}"
    }
    added = data_handler.add_prediction_to_history(history_data)
    print(f"Added to history: {added}")
else:
    print(f"ERROR: {result['error']}")

# Test 2: Single prediction - Real news (Option 2)
print("\n[TEST 2] Single Prediction - Real News Article")
print("-" * 70)

real_news_text = """
The Reserve Bank today announced that it will keep the benchmark 
interest rate unchanged following its latest monetary policy meeting. 
The decision was widely expected by economists and market analysts.
"""

result2 = predictor.predict(real_news_text)
print(f"Success: {result2['success']}")
print(f"Prediction: {result2['prediction']}")
print(f"Confidence: {format_percentage(result2['confidence'])}")
print(f"Processing Time: {result2['processing_time']:.4f} seconds")

if result2['success']:
    history_data2 = {
        'timestamp': result2['timestamp'],
        'input_length': result2['input_length'],
        'prediction': result2['prediction'],
        'confidence': f"{result2['confidence']:.4f}",
        'probability_fake': f"{result2['probability_fake']:.4f}",
        'probability_real': f"{result2['probability_real']:.4f}"
    }
    added2 = data_handler.add_prediction_to_history(history_data2)
    print(f"Added to history: {added2}")
else:
    print(f"ERROR: {result2['error']}")

# Test 3: Batch prediction (Option 3)
print("\n[TEST 3] Batch Prediction from CSV")
print("-" * 70)

csv_path = project_root / "sample_news.csv"
print(f"CSV file: {csv_path}")

if csv_path.exists():
    import pandas as pd
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from CSV")
    
    texts = df['text'].fillna("").astype(str).tolist()
    results = predictor.predict_batch(texts)
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"Total processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    # Add all to history
    print("\nAdding batch predictions to history...")
    for result in results:
        if result['success']:
            history_data = {
                'timestamp': result['timestamp'],
                'input_length': result.get('input_length', 0),
                'prediction': result['prediction'],
                'confidence': f"{result['confidence']:.4f}",
                'probability_fake': f"{result['probability_fake']:.4f}",
                'probability_real': f"{result['probability_real']:.4f}"
            }
            data_handler.add_prediction_to_history(history_data)
    
    print(f"Added {successful} predictions to history")
else:
    print(f"ERROR: CSV file not found at {csv_path}")

# Test 4: View prediction history (Option 6)
print("\n[TEST 4] View Prediction History (Option 6)")
print("-" * 70)

history_df = data_handler.get_prediction_history(10000)
print(f"Total records in history: {len(history_df)}")

if len(history_df) > 0:
    print("\nLast 5 predictions:")
    for idx, row in history_df.tail(5).iterrows():
        print(f"\n[{idx + 1}] {row.get('timestamp', 'N/A')}")
        print(f"    Prediction: {row.get('prediction', 'N/A')}")
        print(f"    Confidence: {format_percentage(float(row.get('confidence', 0)))}")
        print(f"    Input Length: {int(row.get('input_length', 0))} chars")
else:
    print("ERROR: No history found!")

# Test 5: Export history (Option 4)
print("\n[TEST 5] Export Predictions (Option 4)")
print("-" * 70)

if len(history_df) > 0:
    success, message = data_handler.export_predictions(history_df, 'csv')
    print(f"Export success: {success}")
    print(f"Message: {message}")
else:
    print("No history to export")

# Test 6: Clear history (Option 7)
print("\n[TEST 6] Clear Prediction History (Option 7)")
print("-" * 70)

confirm = input("Clear all history? (y/n): ").strip().lower()
if confirm == 'y':
    success = data_handler.clear_prediction_history()
    print(f"Clear success: {success}")
    
    # Verify
    history_after = data_handler.load_prediction_history()
    print(f"Records after clear: {len(history_after)}")
else:
    print("Clear cancelled")

print("\n" + "=" * 70)
print("ALL TESTS COMPLETED")
print("=" * 70)