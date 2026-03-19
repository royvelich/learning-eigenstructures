import pickle

# Load the pickle file
with open('./assets/stl10_train_mapping.pkl', 'rb') as f:
    mapping = pickle.load(f)

# Print first 20 entries
for i, (filename, info) in enumerate(mapping.items()):
    if i >= 1000:
        break
    print(f"{filename}: {info}")