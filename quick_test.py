#!/usr/bin/env python3

def original_sigmoid(x):
    if abs(1-x) > abs(x):
        return True
    else:
        return False

# Test with some values
test_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
for val in test_vals:
    result = original_sigmoid(val)
    print(f"Input: {val:.1f} -> Fracture: {result}")