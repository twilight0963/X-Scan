#!/usr/bin/env python3
"""
Test the sigmoid logic to understand the prediction behavior
"""

def original_sigmoid(x):
    """Original sigmoid logic from notebook"""
    if abs(1-x) > abs(x):
        return True
    else:
        return False

def simple_sigmoid(x):
    """Simple threshold logic"""
    return x > 0.5

# Test cases
test_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

print("Testing sigmoid logic:")
print("Value | Original | Simple | Interpretation")
print("-" * 45)

for val in test_values:
    orig = original_sigmoid(val)
    simple = simple_sigmoid(val)
    print(f"{val:5.1f} | {orig:8} | {simple:6} | {'Fracture' if orig else 'Normal'}")

print("\nOriginal logic analysis:")
print("abs(1-x) > abs(x) means:")
print("- When x < 0.5: abs(1-x) > abs(x) → True (Fracture)")
print("- When x > 0.5: abs(1-x) < abs(x) → False (Normal)")
print("\nThis means the original logic treats LOW probability as fracture!")
print("This suggests the model might output probability of NORMAL, not fracture.")