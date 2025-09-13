
# Music Genre Classification

A deep learning project for automatic music genre classification using CNN and CRNN.

## Quick Start

1. Setup environment:
   conda env create -f environment.yml
   conda activate music-genre-classification

2. Download data:
   python scripts/download_data.py

3. Train model:
   python scripts/train_model.py

4. Run web demo:
   streamlit run app/streamlit_app.py

## Project Structure

- data/: Dataset and processed data
- src/: Source code modules  
- notebooks/: Jupyter notebooks
- scripts/: Training scripts
- app/: Streamlit web application
- models/: Saved model checkpoints
- results/: Experiment results
- configs/: Configuration files

## Performance

Target: >70% classification accuracy on test set
Supported genres: rock, pop, jazz, classical, metal, blues, reggae, country, hip-hop, disco
