# Movie Recommender System

An interactive demonstration of recommendation algorithms using the MovieLens dataset. This project showcases how different recommendation techniques like Collaborative Filtering and Matrix Factorization work in practice and helps understand the personalized content suggestions you see on Amazon Shopping, Netflix Movie Recommendations etc.

![Movie Recommender Demo](assets/demo.png)

## Live Demo

Try the Streamlit application here: [Movie Recommender System App](https://recommender-systems-demo.streamlit.app/)

## Project Components

This repository contains two main components:

### 1. Interactive Jupyter Notebook

The [Jupyter Notebook](movielens.ipynb) ([viewable as a Jupyter Book](https://gaurav-shah05.github.io/Recommender-Systems-Demo/movielens.html)) provides:

- Detailed walkthrough of recommendation algorithms.
- Step-by-step model training process for both recommendation approaches.
- Exploratory data analysis of the MovieLens dataset.
- Performance evaluation and comparison of recommendation techniques.
- Code for saving trained models that power the interactive app.

The notebook serves as both a tutorial and the foundation for the interactive app, making it an essential component for understanding how recommendation systems work.

### 2. Streamlit Web Application

The [Streamlit app](app.py) provides:

- Movie rating interface with posters and star ratings.
- Personalized movie recommendations using two different algorithms mentioned above.

## Dataset

This project uses the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/), which contains 100,000 ratings from 943 users on 1,682 movies. The dataset is a standard benchmark in recommendation system research.

Note that the Movielens-100k dataset used here contains the movies till 1998, so it's outdated now.You can explore newer & larger dataset called [MovieLens-32M](https://grouplens.org/datasets/movielens/), which contains 32 million ratings and two million tag applications applied to 87,585 movies by 200,948 users. It was released on 05/2024.


## Getting Started

### Prerequisites
- Python 3.8+
- Git

### Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Gaurav-Shah05/Recommender-Systems-Demo.git
   cd Recommender-Systems-Demo
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Jupyter Notebook

The notebook contains all the code used to train the recommendation models:

```bash
jupyter notebook movielens.ipynb
```

If you want to retrain the models from scratch, run all cells in the notebook. The trained models will be saved to the `models/` directory.

### Running the Streamlit App

The app uses pre-trained models (included in the repository):

```bash
streamlit run app.py
```

Then open your browser to http://localhost:8501

## Project Structure

```
Recommender-Systems-Demo/
├── app.py                  # Streamlit application
├── movielens.ipynb         # Jupyter notebook for model training & explanation
├── requirements.txt        # Python dependencies
├── models/                 # Pre-trained recommendation models
│   ├── movies.pkl          # MovieLens movie data 
│   ├── ratings.pkl         # MovieLens ratings data
│   ├── svd_model.pkl       # Trained SVD model
│   └── user_item_matrix.pkl # User-item matrix for collaborative filtering
├── movie_poster.csv        # Movie ID to poster URL mapping
├── recommender-system-book/# Jupyter Book version of the notebook
└── extras/                 # Utility scripts and additional files
```

## Academic Context

This project is part of the "Challenges of Digital Society" course at IIT Gandhinagar. It accompanies the research paper "Recommender Systems and the Making of Influence," which explores how recommendation algorithms shape user choices and behavior in digital spaces.

## License

This project is available under the MIT License. The MovieLens dataset has its own license terms specified by GroupLens Research.

## Acknowledgements

- [MovieLens](https://grouplens.org/datasets/movielens/) for the dataset
- [GroupLens Research](https://grouplens.org/) at the University of Minnesota
- [Streamlit](https://streamlit.io/) for the web app framework
- [MovieLens-Posters](https://github.com/babu-thomas/movielens-posters) for the movie poster database
