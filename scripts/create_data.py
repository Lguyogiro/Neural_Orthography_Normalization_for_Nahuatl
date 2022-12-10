import pandas as pd

path_to_master = "../texts_with_normalizations.csv"
master_data = pd.read_csv(path_to_master)

orthographies = {"inali", "ilv", "sep", "ack"}

#
# Data for word-level training
#
word_rows = []
for i, row in master_data.iterrows():
    if i > 0 and i % 500 == 0:
        print(f"{i} rows complete...")
    original = row.orig
    is_spa = row.is_spa
    lang = "spa" if is_spa == 1 else "nhi"
    transforms = {
        "original": row.orig, 
        "inali": row.inali_keep_spanish, 
        "ilv": row.ilv_keep_spanish, 
        "sep": row.sep_keep_spanish, 
        "ack": row.ack_keep_spanish
    }
    for orth1 in orthographies:
        if orth1 == "original":
            continue
        word_rows.append({"input": f"<{orth1}> " + " ".join(transforms["original"]), 
                          "target": " ".join(transforms[orth1]), 
                          "lang": lang})
        for orth2 in orthographies:
            if orth2 == "original" or orth2 == orth1:
                continue
            word_rows.append({"input": f"<{orth2}> " + " ".join(transforms[orth1]), 
                              "target": " ".join(transforms[orth2]), 
                              "lang": lang})
    
    output_df = pd.DataFrame(word_rows, columns=["input", "target", "lang"])
    output_df.to_csv("../data/all_normalization_data.csv", index=False)
        
#
# Data for multiword/sentence level training
#
# TODO