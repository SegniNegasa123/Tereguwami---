# Tereguwami (ተርጓሚ) — Dataset Directory

This directory is the ingestion root for training datasets.

---

## 📁 Recommended Folder Structure

```
data/
├── metadata.csv             # Primary mapping file (Video -> Transcriptions)
├── videos/                  # Raw video recordings (.mp4, .mov, .webm)
│   ├── sample_001.mp4
│   ├── sample_002.mp4
│   └── ...
└── landmarks/               # (Optional) Pre-extracted 543 MediaPipe landmarks (.npy / .json)
    ├── sample_001.npy
    └── ...
```

---

## 📄 `metadata.csv` Format

Place a CSV file named `metadata.csv` in this folder with the following columns:

```csv
video_id,amharic_text,english_text,domain,split
sample_001.mp4,ጤና ይስጥልኝ እንደምን ነዎት?,Hello how are you?,general,train
sample_002.mp4,ዶክተር ላለፉት ሦስት ቀናት ብርቱ የራስ ምታት አለኝ,Doctor I have had a severe headache for three days,healthcare,train
sample_003.mp4,መድኃኒቱን ከምግብ በኋላ ይውሰዱ,Take the medication after meals,healthcare,val
sample_004.mp4,ክሱ ሀሰት ነው፤ እኔ አልሰረቅኩም,The accusation is false; I did not steal,legal,test
```

### Column Descriptions:
- **`video_id`**: The filename of the video clip located in `data/videos/`.
- **`amharic_text`**: Target Amharic transcription (in Ge'ez script).
- **`english_text`**: (Optional) English translation.
- **`domain`**: `healthcare` | `legal` | `education` | `general`.
- **`split`**: `train` (80%) | `val` (10%) | `test` (10%).

---

## 🚀 Preprocessing

Once you place your video files and `metadata.csv` here, the feature extractor will automatically parse the **543 3D landmarks** (Pose, Hands, Face Mesh) and feed them into the Spatial-Temporal Graph Convolutional Transformer.
