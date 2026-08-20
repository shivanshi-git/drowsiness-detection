"""
Subject-independent split for MRL Eye Dataset (37 subjects).
"""

def get_mrl_subject_splits():
    all_subjects = [f"s{i:04d}" for i in range(1, 38)]
    train = all_subjects[:28]
    val = all_subjects[28:33]
    test = all_subjects[33:]
    return {"train": train, "val": val, "test": test}
