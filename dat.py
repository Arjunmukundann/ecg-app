import wfdb
import pandas as pd
import os

# Folder where .dat and .hea files are stored
data_path = r"C:\Users\Arjun\OneDrive\Desktop\ECG\ecg-app\dataset"   # change if needed

# Get all record names automatically
records = [f.split('.')[0] for f in os.listdir(data_path) if f.endswith('.dat')]

print(f"Found {len(records)} records")

for record_name in records:
    try:
        record_path = os.path.join(data_path, record_name)
        
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal[:, 0]  # first channel
        
        df = pd.DataFrame({
            "time": range(len(signal)),
            "voltage": signal
        })
        
        output_file = os.path.join(data_path, f"{record_name}.csv")
        df.to_csv(output_file, index=False)
        
        print(f"✅ Converted: {record_name}")
    
    except Exception as e:
        print(f"❌ Error in {record_name}: {e}")