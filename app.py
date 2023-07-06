import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
device = torch.device('cpu')

# LOAD EVERYTHING
# @st.cache_resource
def load_model(fp):
    mdl = torch.jit.load(fp, map_location=device)
    mdl.eval()
    return mdl

@st.cache_data
def load_utils(fp):
    utils = pd.read_pickle(fp)
    return utils

def name2itemId(names):
    if not isinstance(names, list):
        names = [names]
    return d['movieId'].transform([d_name2le[name] for name in names])

mdl = load_model('mdls/fm_pt.pkl')
utils = load_utils('data/d_utils.pkl')

d = utils['label_encoder']
feature_offsets = utils['feature_offsets']
movie_offset = feature_offsets['movieId_index']
movie_embs = utils['movie_embs']
movies = utils['movies']
d_name2le = utils['d_name2le']

movie_list = sorted(movies.title.to_list())
idxs_movies = torch.tensor(movies['movieId_index'].values, device=device)
movie_biases = mdl.x_bias[idxs_movies]

d_gender = dict(zip(d['gender'].classes_, range(len(d['gender'].classes_))))
d_age = dict(zip(d['age'].classes_, range(len(d['age'].classes_))))
d_age_meta = {'Under 18': 1, '18-24': 18, '25-34': 25, '35-44': 35,
              '45-49': 45, '50-55': 50, '56+': 56}

st.title('Movie Recs - Factorization Machines')
left_col, right_col = st.columns([5,1])
with right_col:
    gender_meta = st.radio('Gender', d_gender)
    gender = d_gender[gender_meta]
    age_meta = st.radio('Age', d_age_meta)
    age = d_age[d_age_meta[age_meta]] # Meta data str -> labelEncoder label
    
with left_col:
    left_col2, right_col2 = st.columns([2,2])
    with right_col2:
        # COLD START
        st.subheader('Cold Start')
        st.write(f'Features: {gender_meta} & {age_meta}')
        gender_emb = mdl.x_emb(
            torch.tensor(feature_offsets['gender_index']+gender, device=device)
        )
        age_emb = mdl.x_emb(
            torch.tensor(feature_offsets['age_index']+age, device=device)
        )
        metadata_emb = gender_emb + age_emb
        rankings = movie_biases + (metadata_emb*movie_embs).sum(1) # dot product
        rankings = rankings.detach()
        cold_start_recs = movies.iloc[rankings.argsort(descending=True)]['title'].values[:5]
        for i,rec in enumerate(cold_start_recs):
            st.write(f'{i} - {rec}')
    
    with left_col2:
        st.subheader('Movie (cosine sim.)')
        fn = movie_list.index('Toy Story 2 (1999)')
        movie = st.selectbox('Movie pick', movie_list, index=fn)
        if st.button('Feeling lucky'):
            movie = movies.title.sample(1).values[0]
        st.write(movie)

        # MOVIE (cosine sim.)
        IDX = name2itemId(movie)[0]
        IDX = IDX + movie_offset # Add offset to get input movie emb
        emb_toy2 = mdl.x_emb(torch.tensor(IDX, device=device))
        cosine_sim = torch.tensor(
            [F.cosine_similarity(emb_toy2, emb, dim=0) for emb in movie_embs]
        )
        top8 = cosine_sim.argsort(descending=True)[1:6]
        movie_sims = cosine_sim[top8]
        movie_recs = movies.iloc[top8.detach().numpy()]['title'].values

        for rec, sim in zip(movie_recs, movie_sims):
            st.write(f'{sim.tolist():0.3f} - {rec}')