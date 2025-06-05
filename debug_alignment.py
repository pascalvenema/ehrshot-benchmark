#!/usr/bin/env python3

import pickle
import numpy as np
import os
from femr.labelers import load_labeled_patients
from ehrshot.utils import compute_feature_label_alignment

# Load ICU labels
print("Loading ICU labels...")
labeled_patients = load_labeled_patients('./EHRSHOT_ASSETS/benchmark/guo_icu/labeled_patients.csv')
label_patient_ids, label_values, label_times = labeled_patients.as_numpy_arrays()
label_times = label_times.astype("datetime64[us]")

# Find the problematic patient
target_pid = 115967096
label_indices = np.where(label_patient_ids == target_pid)[0]
print(f"\nPatient {target_pid} in ICU labels:")
for idx in label_indices:
    print(f"  Index {idx}: {label_times[idx]} (as int64: {label_times[idx].astype(np.int64)})")

target_label_time = label_times[label_indices[0]].astype(np.int64)

# Test all available feature types
feature_files = [
    ('count', './EHRSHOT_ASSETS/features/count_features.pkl'),
    ('clmbr', './EHRSHOT_ASSETS/features/clmbr_features.pkl'),
    ('clinicalbert_type3_max_pool', './EHRSHOT_ASSETS/features/clinicalbert_type3_max_pool_features.pkl'),
    ('clinicalbert_type3_mean_pool', './EHRSHOT_ASSETS/features/clinicalbert_type3_mean_pool_features.pkl'),
    ('clinicalbert_type3_clinicalbert_pool', './EHRSHOT_ASSETS/features/clinicalbert_type3_clinicalbert_pool_features.pkl'),
]

for feature_name, feature_path in feature_files:
    if not os.path.exists(feature_path):
        print(f"\n{feature_name.upper()} FEATURES: File not found at {feature_path}")
        continue
        
    print(f"\n{feature_name.upper()} FEATURES:")
    print(f"Loading from {feature_path}...")
    
    with open(feature_path, 'rb') as f:
        feats = pickle.load(f)
        
        # Handle different data structures
        if isinstance(feats, dict):
            feature_patient_ids = feats['patient_ids']
            feature_times = feats['labeling_time']
        else:
            feature_patient_ids = feats[1]
            feature_times = feats[3]

    feature_times = feature_times.astype("datetime64[us]")

    # Find the same patient in features
    feature_indices = np.where(feature_patient_ids == target_pid)[0]
    print(f"  Patient {target_pid} found in {len(feature_indices)} records")
    
    if len(feature_indices) == 0:
        print(f"  ERROR: Patient {target_pid} not found in {feature_name} features!")
        continue
        
    print(f"  Timestamps for patient {target_pid}:")
    target_feature_time_candidates = feature_times[feature_indices].astype(np.int64)
    
    for i, idx in enumerate(feature_indices[:10]):  # Show first 10
        print(f"    Index {idx}: {feature_times[idx]} (as int64: {feature_times[idx].astype(np.int64)})")
    if len(feature_indices) > 10:
        print(f"    ... and {len(feature_indices) - 10} more")

    # Test for exact match
    matches = target_feature_time_candidates == target_label_time
    print(f"  Looking for exact match with {target_label_time}:")
    print(f"    Match found: {np.any(matches)}")
    
    if np.any(matches):
        match_idx = np.where(matches)[0][0]
        print(f"    SUCCESS: Exact match at feature index {feature_indices[match_idx]}")
    else:
        print(f"    FAILURE: No exact match found")
        # Show closest match
        differences = np.abs(target_feature_time_candidates - target_label_time)
        min_diff_idx = np.argmin(differences)
        closest_time = target_feature_time_candidates[min_diff_idx]
        print(f"    Closest match: {closest_time} (diff: {differences[min_diff_idx]} microseconds)")
        
        # Convert to human readable time difference
        diff_seconds = differences[min_diff_idx] / 1000000
        print(f"    Time difference: {diff_seconds:.2f} seconds")

print(f"\nSUMMARY: Testing alignment for patient {target_pid} with timestamp {target_label_time}")
print("This shows which feature files have the correct timestamps vs. mismatched ones.") 