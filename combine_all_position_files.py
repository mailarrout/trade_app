import os
import glob
import pandas as pd

# folder path
folder = r"C:\Users\HP\OneDrive\Desktop\Shoonya\trade_app\modules\logs"

# get only *_positions.csv files
csv_files = glob.glob(os.path.join(folder, "*_positions.csv"))

df_list = []
for f in csv_files:
    try:
        df = pd.read_csv(f, on_bad_lines="skip")   # skip broken/misaligned rows
        df["source_file"] = os.path.basename(f)    # optional: track which file it came from
        df_list.append(df)
    except Exception as e:
        print(f"⚠️ Skipping {f} due to error: {e}")

# combine all into one DataFrame
combined_df = pd.concat(df_list, ignore_index=True)

# save to a single output file
output_file = os.path.join(folder, "all_positions_combined.csv")
combined_df.to_csv(output_file, index=False)

print(f"✅ Combined {len(df_list)} files into {output_file}")
