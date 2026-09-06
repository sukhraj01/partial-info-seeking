"""Create a disjoint 80/20 word split for out-of-vocabulary evaluation.

Since the word list is unique, a random word-level split is inherently
disjoint -- no word can land in both sets -- so the holdout measures
generalisation to strings the model has never seen rather than recall.
"""
import random

random.seed(42)

with open("data/train.txt") as f:
    words = [w.strip() for w in f if w.strip()]

random.shuffle(words)
split_idx = int(len(words) * 0.8)
train_split = words[:split_idx]
holdout_split = words[split_idx:]

assert set(train_split).isdisjoint(set(holdout_split))

with open("data/train_split.txt", "w") as f:
    f.write("\n".join(train_split) + "\n")

with open("data/holdout_split.txt", "w") as f:
    f.write("\n".join(holdout_split) + "\n")

print(f"train_split: {len(train_split)} words -> data/train_split.txt")
print(f"holdout_split: {len(holdout_split)} words -> data/holdout_split.txt")
print(f"disjoint: {set(train_split).isdisjoint(set(holdout_split))}")
