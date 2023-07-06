import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

st.title('Data and Model Info')
left_col, right_col = st.columns([1,1])
with left_col:
    st.subheader('Data: MovieLens 1M ratings')
    st.markdown(
        """
        - URL: https://grouplens.org/datasets/movielens/1m/
        - Post data clean: ~1M ratings. 6040 users. 3706 movies. Movies cutoff after year 2000
        - Features: user, movie, rating, time, title, genres, gender, age, occupation, zipcode
        - Min # ratings per user: 20
        - Min/Max rating: 1 - 5
        - Val. data: Per user, most recent 5 movie ratings
        - Rating distribution: 
        """
    )
    im = Image.open('imgs/movielens_1m_ratings.jpg')
    st.image(im)
#     st.image('movielens_1m_ratings.jpg')

with right_col:
    st.subheader('Model: Factorization Machine')
    st.markdown(
        """
        - URL: https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
        - PyTorch Factorization Machine. Added feature biases and global offset
        - Input X features: user, movie name, gender, age
        - Output Y prediction: movie rating
        - Loss func: Mean squared error
        - Optimizer: AdamW with weight decay
        - Original & predicted rating distribution:
        """
    )
    st.image('imgs/ratings_post_train.jpg')
    st.markdown(
        """
        - TSNE showing model learned to separate movie genres:
        """
    )
    st.image('imgs/tsne_genre.jpg')