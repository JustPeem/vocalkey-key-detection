# VocalKey — key detection for a cappella singing

Finds the musical key (one of 24: 12 major + 12 minor) of a sung clip from its
12-dimensional **chroma vector**, using four linear-algebra / data techniques:

| technique | used for |
|---|---|
| **Cosine similarity** | compare a clip with 24 key templates (Krumhansl–Kessler) and with other clips |
| **TF-IDF** | re-weight the 12 pitch classes before comparing |
| **Linear regression** | learn a linear map chroma → key, solved with the normal equations |
| **PCA** | look at the data in 2-D and compress 12 → *m* features |

Everything is in one notebook: [`VocalKey_project.ipynb`](VocalKey_project.ipynb).

## Data and train / validation / test split

| set | data | used for |
|---|---|---|
| train + validation | **Acapella**: 132 clips (6 songs, 22 singers), chroma from `data/raw/key_chroma_dataset_172.csv` | fit the models; leave-one-song-out validation to choose the model |
| test | **vocadito**: 40 real recordings (29 other singers, 7 languages), WAV audio in `vocadito/Audio/` | used once, at the end |

The test chroma is computed from the audio with `librosa.feature.chroma_stft` (22 050 Hz, averaged
over time); it matches the dataset values exactly.

## Results

| model | validation MIREX | test MIREX |
|---|---|---|
| **PCA (10 comp.) + linear regression** (chosen on validation) | 0.400 | **0.325** (accuracy 0.275) |
| Linear regression | 0.381 | 0.325 |
| Cosine templates (centred) | 0.214 | 0.383 |
| TF-IDF + cosine templates | 0.123 | 0.447 |
| Baseline: always the most common key | 0.280 | 0.163 |

MIREX weighted score: correct 1.0 · fifth 0.5 · relative 0.3 · parallel 0.2 · other 0.
The test set is small (one clip = 2.5 % accuracy).

## Repository layout

```
VocalKey_project.ipynb        the whole project, step by step
data/raw/                     key_chroma_dataset_172.csv (main), chroma_features_12_columns.csv, key_labels_172.csv
data/processed/               vocalkey_clean.csv, cleaning_log.csv
vocadito/                     test set: Audio/*.wav, vocadito_metadata.csv, Annotations/
requirements.txt
```

## How to run

```bash
pip install -r requirements.txt
jupyter lab VocalKey_project.ipynb
```

Run the cells from top to bottom. To classify your own recording, use
`classify_wav("my_singing.wav")` at the end of the notebook.

## Data sources

* **Acapella** — ccmusic-database/acapella (Hugging Face)
* **vocadito** — Bittner et al., *vocadito: A dataset of solo vocals with f0, note, and lyric
  annotations*, Zenodo, licensed **CC BY 4.0**. The audio and annotations in `vocadito/` are
  redistributed unchanged under that licence.
* Key profiles — Krumhansl, C. L. (1990). *Cognitive Foundations of Musical Pitch*.
