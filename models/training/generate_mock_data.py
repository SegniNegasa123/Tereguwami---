import os
import numpy as np

data_dir = "data/raw_ceslr/preprocess/CESLR"
os.makedirs(data_dir, exist_ok=True)

# 1. Gloss Dictionary
glosses = ["greetings", "medicine_after_food", "explain_again", "court_dismissed", "hello", "thank_you", "doctor", "pain", "fever", "yes", "no"]
gloss_dict = {gloss: [i + 1] for i, gloss in enumerate(glosses)}
np.save(os.path.join(data_dir, "gloss_dict.npy"), gloss_dict)

# 2. Dataset Info (train, dev, test)
def generate_info(num_samples):
    info = {}
    for i in range(num_samples):
        # randomly pick 1-3 glosses
        words = np.random.choice(glosses, size=np.random.randint(1, 4))
        label = " ".join(words)
        info[f"video_{i}"] = {
            "num_frames": np.random.randint(20, 60),
            "label": label,
            "signer_id": np.random.randint(1, 10)
        }
    return info

train_info = generate_info(200)
dev_info = generate_info(50)
test_info = generate_info(50)

np.save(os.path.join(data_dir, "train_info.npy"), train_info)
np.save(os.path.join(data_dir, "dev_info.npy"), dev_info)
np.save(os.path.join(data_dir, "test_info.npy"), test_info)

print("Successfully generated mock dataset!")
