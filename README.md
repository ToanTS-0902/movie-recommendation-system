# Movie Recommendation System

An item-based collaborative-filtering project that recommends movies from user-rating patterns and compares multiple KNN similarity strategies.

## Highlights

- Sparse movie-user matrix built from MovieLens ratings.
- Item-based K-nearest neighbors with cosine distance.
- Fuzzy title matching for a friendlier recommendation interface.
- Experiments with cosine, Pearson, Pearson-baseline, MSD, L1, and L2 similarity/distance methods.
- Evaluation with held-out ratings and RMSE; the original experiment recorded an RMSE of **1.0621** for the documented Surprise KNN run.

## Tech stack

- Python, pandas, NumPy, SciPy
- scikit-learn `NearestNeighbors`
- Surprise collaborative-filtering toolkit
- Jupyter Notebook

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Download a MovieLens dataset as described in [`data/README.md`](data/README.md), then run:

```bash
python src/recommender.py "Toy Story (1995)" --count 10
```

Example output:

```text
 1. Toy Story 2 (1999) (similarity=0.812)
 2. A Bug's Life (1998) (similarity=0.784)
```

The exact ranking depends on the selected MovieLens release and the minimum-rating threshold.

## Repository structure

```text
src/recommender.py    reusable, local command-line recommender
*.ipynb               original research and comparison notebook
data/README.md        dataset acquisition instructions
```

## Approach

For each movie, ratings across users form a sparse feature vector. The recommender finds the nearest movie vectors using cosine distance, removes the query movie, and returns the closest titles. The notebook also explores model-based collaborative filtering and alternative similarity functions.

## Reproducibility notes

- Raw MovieLens files are not committed because they are large and are maintained by GroupLens.
- The original notebook was cleaned of execution output, Colab account metadata, and machine-specific Google Drive paths before publication.
- Reported metrics belong to the specific train/test split and dataset version documented in the notebook.

## My contribution

My primary focus was implementing and comparing the recommendation algorithms, evaluating KNN variants, and integrating the recommendation workflow into an interactive prototype.

## Author

**Trần Song Toàn** — AI Engineer  
FPT University, Can Tho Campus  
[transtoanvovankiet@gmail.com](mailto:transtoanvovankiet@gmail.com)
