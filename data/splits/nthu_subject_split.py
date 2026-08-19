"""
Subject-independent 5-fold cross validation split generator for NTHU-DDD.
"""

NTHU_SUBJECTS = [
    "001", "002", "004", "005", "006", "008", "009", "010",
    "011", "012", "013", "014", "015", "017", "018", "019",
    "020", "021", "022", "023", "024", "026", "030"
]

def get_nthu_subject_splits():
    # 5-fold partition
    folds = []
    chunk_size = len(NTHU_SUBJECTS) // 5
    for i in range(5):
        val_subjs = NTHU_SUBJECTS[i*chunk_size : (i+1)*chunk_size] if i < 4 else NTHU_SUBJECTS[i*chunk_size:]
        train_subjs = [s for s in NTHU_SUBJECTS if s not in val_subjs]
        folds.append({"train": train_subjs, "val": val_subjs})
    return folds
