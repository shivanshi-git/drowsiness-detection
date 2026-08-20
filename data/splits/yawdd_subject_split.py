"""
Subject split for YawDD dataset (Male & Female driver categories).
"""

def get_yawdd_subject_splits():
    return {
        "train_scenarios": ["Male_Drivers_Normal", "Female_Drivers_Normal", "Male_Drivers_Yawn"],
        "val_scenarios": ["Female_Drivers_Yawn", "Male_Drivers_Talking"],
        "test_scenarios": ["Female_Drivers_Talking", "Male_Drivers_Glasses"]
    }
