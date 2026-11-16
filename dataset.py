import pandas as pd
import re
 
TATOEBA_FILES = [
    (r"C:\Users\MINT\Downloads\Sentence pairs in French-English - 2026-11-14.tsv", "French", "English"),
    (r"C:\Users\MINT\Downloads\Sentence pairs in Portuguese-English - 2026-11-14.tsv", "Portuguese", "English"),
    (r"C:\Users\MINT\Downloads\Sentence pairs in Shona-English - 2026-11-14.tsv", "Shona", "English"),
    (r"C:\Users\MINT\Downloads\Sentence pairs in Spanish-English - 2026-11-14.tsv", "Spanish", "English"),
    (r"C:\Users\MINT\Downloads\Sentence pairs in Mandarin Chinese-English - 2026-11-15.tsv", "Mandarin Chinese", "English")
    ]

OUTPUT_FILE = r"C:\Users\MINT\Downloads\clean_balanced_data.csv"
 
def clean_text(text):
    text = str(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
 
all_sentences =[]
for file_path, lang1, lang2 in TATOEBA_FILES:
    print(f"Reading {lang1} and {lang2}...")
    df = pd.read_csv(file_path, sep="\t", header=None, encoding="utf-8")
 
    df1 = pd.DataFrame({
        "Text": df[1].dropna().apply(clean_text),
        "Language": lang1
    })
 
    df2 = pd.DataFrame({
        "Text": df[5].dropna().apply(clean_text),
        "Language": lang2
    })
 
    all_sentences.extend([df1, df2])
 
combined_df = pd.concat(all_sentences, ignore_index=True)
combined_df = combined_df[combined_df["Text"].str.strip() != ""]
combined_df = combined_df.drop_duplicates(
    subset=["Text", "Language"]
)
 
print("\nSentences found per language:")
print(combined_df["Language"].value_counts())
 
min_count = combined_df["Language"].value_counts().min()
print(f"\nBalancing all languages to {min_count} sentences each...")
 
balanced_df = (
    combined_df.groupby("Language", group_keys=False)
    .sample(n=min_count, random_state=42)
    .reset_index(drop=True)
)
 
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
balanced_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"Done! Saved {len(balanced_df)} clean, balanced rows to '{OUTPUT_FILE}'.")
    
print("\nDataset successfully created!")
print("Total sentences:", len(balanced_df))
print("\nFinal language counts:")
print(balanced_df["Language"].value_counts())
print("\nSaved to:", OUTPUT_FILE)
