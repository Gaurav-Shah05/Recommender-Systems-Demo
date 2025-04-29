import time
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from PIL import Image
from io import BytesIO
import requests
import plotly.express as px

# ─── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Movie Recommender Demo", page_icon="🎬", layout="wide")

# ─── GLOBAL CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.movie-card { text-align: center; margin-bottom: 2rem; }
.movie-title {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin: 0.5rem 0;
  font-weight: bold;
}
.star-row { display: inline-block; }
</style>
""", unsafe_allow_html=True)

# ─── DATA LOADING ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_data():
    movies  = pd.read_pickle('models/movies.pkl')
    ratings = pd.read_pickle('models/ratings.pkl')
    uim     = pd.read_pickle('models/user_item_matrix.pkl')

    # Perform a lightweight truncated SVD on the user-item matrix
    # to get movie latent factors without scikit-surprise.
    # Keep only k latent dimensions:
    k = 20
    U, S, Vt = np.linalg.svd(uim.values, full_matrices=False)
    V = Vt.T[:, :k]              # movie × factors
    rec_matrix = V @ V.T         # item-item affinities

    return movies, ratings, uim, rec_matrix, V

@st.cache_data
def load_poster_map():
    df = pd.read_csv('movie_poster.csv', header=None,
                     names=['movie_id','poster_url'])
    return df.set_index('movie_id')['poster_url'].to_dict()

poster_map = load_poster_map()

def fetch_poster(mid):
    """Fetch a poster from the web (slow)."""
    url = poster_map.get(mid)
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=2)
        if resp.status_code == 200:
            return Image.open(BytesIO(resp.content))
    except:
        pass
    return None

# ─── STAR COMPONENT ──────────────────────────────────────────────────────────────
def star_rating_horizontal(key: str, default: int = 0) -> int:
    """
    Display 5 inline star‐buttons.  
    Stores rating in session_state[f"rating_{key}"].  
    Returns the current rating.
    """
    skey = f"rating_{key}"
    if skey not in st.session_state:
        st.session_state[skey] = default

    cols = st.columns(5, gap="small")
    for i, col in enumerate(cols, start=1):
        filled = i <= st.session_state[skey]
        label = "★" if filled else "☆"

        def set_rating(k=key, val=i):
            st.session_state[f"rating_{k}"] = val

        col.button(label, key=f"{key}_{i}", on_click=set_rating, help=f"Rate {i}")

    return st.session_state[skey]

# ─── RECOMMENDATION LOGIC ────────────────────────────────────────────────────────
def user_cf(user_ratings, uim, movies_df, k=10, n=6):
    vec = np.zeros(uim.shape[1])
    idx = {m:i for i,m in enumerate(uim.columns)}
    for m,r in user_ratings.items():
        if m in idx: vec[idx[m]] = r

    knn = NearestNeighbors(metric='cosine', algorithm='brute',
                           n_neighbors=k).fit(uim.values)
    _, neighs = knn.kneighbors([vec], n_neighbors=k)
    seen = set(user_ratings)
    candidates = []
    for m in uim.columns:
        if m in seen: continue
        col = idx[m]
        vals = [uim.values[u,col] for u in neighs.flatten() if uim.values[u,col]>0]
        if len(vals)>=2:
            candidates.append((m, np.mean(vals)))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:n]

def svd_recs(user_ratings, rec_matrix, movies_df, n=6):
    """
    Pure-NumPy SVD recommendations.
    rec_matrix: precomputed item-item affinity (n_movies x n_movies)
    uim: user-item DataFrame with movies_df.movie_id as columns.
    """
    # Build a temporary user rating vector aligned with rec_matrix rows/cols:
    tmp = np.zeros(rec_matrix.shape[0])
    col_index = {m:i for i,m in enumerate(uim.columns)}
    for mid, r in user_ratings.items():
        if mid in col_index:
            tmp[col_index[mid]] = r

    # Predict ratings = weighted sum over item affinities:
    scores = tmp @ rec_matrix    # shape = (n_movies,)

    # Rank movies not yet rated:
    ranked_idxs = np.argsort(scores)[::-1]
    recs = []
    for idx in ranked_idxs:
        mid = uim.columns[idx]
        if mid not in user_ratings:
            recs.append((mid, float(scores[idx])))
        if len(recs)>=n:
            break
    return recs

# ─── CHARTS FOR INSIGHTS ─────────────────────────────────────────────────────────
def plot_user_similarity(user_ratings, uim, k=10):
    vec = np.zeros(uim.shape[1])
    idx = {m:i for i,m in enumerate(uim.columns)}
    for m,r in user_ratings.items():
        if m in idx: vec[idx[m]] = r

    knn = NearestNeighbors(metric='cosine', algorithm='brute',
                           n_neighbors=k+1).fit(uim.values)
    dists, idxs = knn.kneighbors([vec], n_neighbors=k+1)
    df = pd.DataFrame({
        'Other User': uim.index[idxs.flatten()],
        'Similarity': 1 - dists.flatten()
    }).iloc[1:]  # drop self
    fig = px.bar(df, x='Other User', y='Similarity',
                 title="Users Most Similar to You")
    fig.update_layout(height=300)
    return fig

def plot_svd_latent_factors(V, n_factors=5):
      """
      V: numpy array of shape (n_movies, k)
      Displays the first 10 movies' loadings on n_factors latent dimensions.
      """
      df = pd.DataFrame(
          V[:10, :n_factors],
          index=[f"Movie {i+1}" for i in range(10)],
          columns=[f"Factor {i+1}" for i in range(n_factors)]
      )
      fig = px.imshow(
          df,
          labels={'x':'Latent Factor','y':'Sample Movie Index','color':'Value'},
          title="Sample of SVD Latent Factors"
      )
      fig.update_layout(height=300)
      return fig

def plot_rating_distribution(ratings):
    cnt = ratings['rating'].value_counts().sort_index()
    fig = px.bar(x=cnt.index, y=cnt.values,
                 labels={'x':'Rating','y':'Count'},
                 title="Rating Distribution in Dataset")
    fig.update_layout(height=300)
    return fig

# ─── NAVIGATION & STATE ─────────────────────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = 'landing'
if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = {}

def to_setup():
    st.session_state.page = 'setup'

def to_buffer():
    st.session_state.page = 'buffer'

def to_landing():
    st.session_state.page = 'landing'
    st.session_state.user_ratings.clear()
    for k in ['posters_fetched','posters',
              'recs','rec_posters','buffer_shown']:
        st.session_state.pop(k, None)

def to_insights():
    st.session_state.page = 'insights'

def to_recommend_from_insights():
    st.session_state.page = 'recommend'

# ─── PAGE 1: LANDING (RATE) ─────────────────────────────────────────────────────
def landing_page(movies, ratings):
    st.title("🎬 Rate Some Movies")
    st.write("Please rate **at least five** to continue.")

    # Top 15 by popularity
    top15 = (ratings.groupby('movie_id').size()
             .sort_values(ascending=False)
             .head(15).index.tolist())

    # Fetch once & cache
    if 'posters_fetched' not in st.session_state:
        st.session_state.posters = {}
        with st.spinner("Fetching movie posters…"):
            prog = st.progress(0)
            for i, mid in enumerate(top15):
                st.session_state.posters[mid] = fetch_poster(mid)
                prog.progress((i + 1) / len(top15))
        st.session_state.posters_fetched = True

    # Display 3×5 grid
    for row in [top15[i : i5] for i in range(0, 15, 5)]:
        cols = st.columns(5, gap="large")
        for mid, col in zip(row, cols):
            with col:
                st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                img = st.session_state.posters[mid]
                if img:
                    st.image(img, width=120)
                else:
                    st.image("https://via.placeholder.com/120x180?text=NoImage",
                             width=120)
                full = movies.loc[movies['movie_id']==mid,'title'].iloc[0]
                short = full if len(full)<=15 else full[:15] + "…"
                st.markdown(
                    f'<p class="movie-title" title="{full}">{short}</p>',
                    unsafe_allow_html=True
                )
                r = star_rating_horizontal(str(mid))
                if r>0:
                    st.session_state.user_ratings[mid] = r
                elif mid in st.session_state.user_ratings:
                    del st.session_state.user_ratings[mid]
                st.markdown('</div>', unsafe_allow_html=True)

    # See more via expander
    with st.expander("🔍 See more / search & rate"):
        q = st.text_input("Movie title contains:")
        if q:
            res = movies[movies['title'].str.contains(q, case=False)]
            if not res.empty:
                for _, r in res.head(5).iterrows():
                    st.write(r.title)
                    rr = star_rating_horizontal(f"search_{r.movie_id}")
                    if rr>0:
                        st.session_state.user_ratings[r.movie_id] = rr
            else:
                st.write("No match found.")

    # Continue button
    cols = st.columns(5)
    if len(st.session_state.user_ratings) >= 5:
        cols[2].button("Continue ▶️", on_click=to_setup)
    else:
        cols[2].button("Continue ▶️", disabled=True)
        cols[2].caption(f"Rate {5-len(st.session_state.user_ratings)} more")

# ─── PAGE 2: SETUP (CHOOSE ALGORITHM) ────────────────────────────────────────────
def setup_page():
    st.title("⚙️ Set Up Your Recommendations")
    st.write("Choose which algorithm you’d like to use:")

    choice = st.radio("", ["User-Based Collaborative Filtering",
                          "Matrix Factorization (SVD)"],
                      key="chosen_method")

    st.write("When you’re ready, click below:")
    st.button("🔄 Generate Recommendations", on_click=to_buffer)

# ─── PAGE 3: BUFFER (SPINNER  COMPUTE) ─────────────────────────────────────────
def buffer_page(movies, ratings, uim, rec_matrix):
    # compute & cache once
    if 'recs' not in st.session_state:
        ur     = st.session_state.user_ratings
        method = st.session_state.chosen_method
        if method.startswith("User"):
            st.session_state.recs = user_cf(ur, uim, movies)
        else:
            st.session_state.recs = svd_recs(ur, rec_matrix, movies, uim)
 
        # fetch posters for recommendations
        st.session_state.rec_posters = {
             mid: fetch_poster(mid) for mid,_ in st.session_state.recs
        }

    # spinner messages once
    if 'buffer_shown' not in st.session_state:
        msgs_cf = [
            "Tuning in to similar users…",
            "Gathering crowd favorites…",
            "Double‐checking taste profiles…"
        ]
        msgs_svd = [
            "Factorizing your preferences…",
            "Aligning latent flavors…",
            "Finalizing hidden patterns…"
        ]
        msgs = msgs_cf if st.session_state.chosen_method.startswith("User") else msgs_svd

        placeholder = st.empty()
        prog = st.progress(0)
        for i, msg in enumerate(msgs):
            placeholder.info(msg)
            prog.progress((i+1)/len(msgs))
            time.sleep(3)
        placeholder.empty()
        prog.empty()
        st.session_state.buffer_shown = True

    # now show recommendations in the same run
    recommend_page(movies, ratings, uim, rec_matrix)

# ─── PAGE 4: RECOMMENDATIONS ─────────────────────────────────────────────────────
def recommend_page(movies, ratings, uim, svd_model):
    st.title("🎉 Your Personalized Picks")
    st.write(f"You rated **{len(st.session_state.user_ratings)}** movies.")

    recs = st.session_state.recs
    cols = st.columns(3, gap="large")
    for (mid, score), col in zip(recs, cols*2):
        with col:
            st.markdown('<div class="movie-card">', unsafe_allow_html=True)
            img = st.session_state.rec_posters[mid]
            if img:
                st.image(img, width=150)
            else:
                st.image("https://via.placeholder.com/150x225?text=NoImage",
                         width=150)
            title = movies.loc[movies['movie_id']==mid,'title'].iloc[0]
            st.markdown(f"### {title}")
            stars = "★" * int(round(score)) + "☆" * (5 - int(round(score)))
            st.markdown(f"<div style='font-size:24px;color:#FFD700;'>{stars}</div>",
                        unsafe_allow_html=True)
            st.caption(f"{score:.2f}/5 predicted")
            st.markdown('</div>', unsafe_allow_html=True)


    # ─── Buttons next to each other ──────────────────────────
    col1, col2, _ = st.columns([1, 1, 4], gap="small")
    with col1:
        st.button(
            "🔄 Start Over",
            key="btn_start_over",
            on_click=to_landing
        )
    with col2:
        st.button(
            "❔ How It Works",
            key="btn_how_it_works",
            on_click=to_insights
        )

# ─── PAGE 5: INSIGHTS ───────────────────────────────────────────────────────────
def insights_page(movies, ratings, uim, svd_model, V):
    st.title("🔍 How It Works")
    st.markdown("""
**Recommender systems** turn your star ratings into suggestions by discovering patterns:

1. **User‐Based Collaborative Filtering**  
   We look for *other users* whose ratings closely match yours and recommend the films *they* loved but you haven’t rated yet.  
   - Think of it as asking a friend with similar tastes for suggestions.

2. **Matrix Factorization (SVD)**  
   We decompose the huge user×movie ratings matrix into “latent factors” that capture hidden attributes (e.g., *action level*, *romantic content*, *dialogue intensity*).  
   - Imagine each film and each viewer sitting somewhere in a multi-dimensional “taste space,” and we predict where you’d land on new movies.
""")

    st.subheader("1. Your Similarity Network")
    st.markdown("""
This bar chart shows the **top 10 users** whose rating patterns are most similar to yours (cosine similarity).  
- A bar near **1.0** means their taste **almost exactly** matches yours.  
- We base your recommendations on what these top-matches enjoyed.
""")
    sim_fig = plot_user_similarity(st.session_state.user_ratings, uim)
    st.plotly_chart(sim_fig, use_container_width=True)

    st.subheader("2. SVD Latent Factors Heatmap")
    st.markdown("""
This heatmap displays the first few **latent factors** uncovered by the SVD algorithm.  
- **Rows** = sample users, **Columns** = hidden factors (e.g., “Factor 1” might correlate with *action intensity*).  
- **Brighter colors** mean a stronger affiliation with that factor.  
- We predict your rating by seeing how strongly you and each unseen movie align across all these factors.
""")
    svd_fig = plot_svd_latent_factors(V)
    st.plotly_chart(svd_fig, use_container_width=True)

    st.subheader("3. Dataset Rating Distribution")
    st.markdown("""
Here’s how the **MovieLens 100K** ratings are distributed:  
- Notice most ratings cluster at **4** and **5** stars.  
- Algorithms must account for this bias—otherwise they’ll over‐recommend very popular films.
""")
    dist_fig = plot_rating_distribution(ratings)
    st.plotly_chart(dist_fig, use_container_width=True)

    st.markdown("""
---

📚 **Learn more:**  
- **Full Tutorial & Walk-Through of model training**  
  https://gaurav-shah05.github.io/Recommender-Systems-Demo/movielens.html  
- **Source Code on GitHub**  
  https://github.com/Gaurav-Shah05/Recommender-Systems-Demo  
""")

    st.button(
        "← Back to Recommendations",
        key="btn_back_insights",
        on_click=lambda: st.session_state.__setitem__('page','recommend')
    )

# ─── APP ENTRYPOINT ─────────────────────────────────────────────────────────────
def main():
    movies, ratings, uim, svd_model, V = load_data()
    page = st.session_state.page

    if page == 'landing':
        landing_page(movies, ratings)
    elif page == 'setup':
        setup_page()
    elif page == 'buffer':
        buffer_page(movies, ratings, uim, svd_model)
    elif page == 'insights':
        insights_page(movies, ratings, uim, svd_model)
    else:  # recommendations
        recommend_page(movies, ratings, uim, svd_model)

if __name__ == "__main__":
    main()
