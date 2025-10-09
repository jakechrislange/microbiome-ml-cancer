import pandas as pd

# Part 1: Creating the CSVs from txt
# Read the text file (try changing sep depending on the format)
df = pd.read_csv("sample_metadata.txt", sep='\t')

# Save it as a CSV
df.to_csv("sample_metadata.csv", index=False)


# Read the text file (try changing sep depending on the format)
df = pd.read_csv("bacteria.sample.relabund.genus.txt", sep='\t')

# Save it as a CSV
df.to_csv("bacteria.sample.relabund.genus.csv", index=False)





# PART 2: Merging the CSVs
# Load both CSVs
meta = pd.read_csv("sample_metadata.csv")         # The one with columns like Sample, case_id, etc.
data = pd.read_csv("bacteria.sample.relabund.genus.csv")      # The one with 'name, TCGA-AG-3898-01A, ...'

# Transpose data so that sample IDs become rows
data_t = data.set_index("name").T.reset_index()
data_t.rename(columns={"index": "Unnamed: 0"}, inplace=True)

# Merge metadata and expression table on sample ID
merged = pd.merge(meta, data_t, on="Unnamed: 0", how="inner")

# Save to CSV
merged.to_csv("merged.csv", index=False)





# PART 3: Keep only rows where Sample == "PT"
merged_PT = merged[merged["Sample"] == "PT"]

# Save to a new CSV
merged_PT.to_csv("merged_PT.csv", index=False)





# PART 4: Drop columns that are entirely empty
cleaned = merged_PT.dropna(axis=1, how="all")

# Save the cleaned data
cleaned.to_csv("merged_PT_cleaned.csv", index=False)

# Verify Sample Distribution is: 
#HNSC (155 samples), STAD (127 samples), COAD (125 samples), ESCA (60 samples), and READ (45 samples).
if "project" in cleaned.columns:
    print("\n📊 Project sample counts:")
    print(cleaned["project"].value_counts())
else:
    print("\n⚠️ Column 'project' not found in the cleaned data.")





# PART 5: Remove genera with no abundance across all samples
cols = cleaned.columns  # or cleaned.columns

# find the index of the column 'HistologicalType'
idx = cols.get_loc("HistologicalType")

# count how many columns are to the right of it
num_cols_right = len(cols) - (idx + 1)

# Verify 221 genera before removing genera with no abundance in all samples
print(f"There are {num_cols_right} columns to the right of 'HistologicalType'.")

cols = cleaned.columns
start_idx = cols.get_loc("HistologicalType")

# Split into metadata and genus data
meta = cleaned.iloc[:, :start_idx + 1]     # everything up to and including HistologicalType
genera = cleaned.iloc[:, start_idx + 1:]   # everything to the right (the genus columns)

# Keep only genus columns that are NOT all zeros
nonzero_genera = genera.loc[:, (genera != 0.0).any()]

# Combine metadata + filtered genus columns
final = pd.concat([meta, nonzero_genera], axis=1)

# Save to CSV
final.to_csv("final.csv", index=False)

cols = final.columns

# find the index of the column 'HistologicalType'
idx = cols.get_loc("HistologicalType")

# count how many columns are to the right of it
num_cols_right = len(cols) - (idx + 1)

# Verify 131 genera after removing genera with no abundance in all samples
print(f"There are {num_cols_right} columns to the right of 'HistologicalType'.")