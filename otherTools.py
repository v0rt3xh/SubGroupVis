import pandas as pd 
import numpy as np
import itertools
from sklearn import preprocessing
import random
import streamlit as st

def random_categories(categories, num = 3):
  N = len(categories)
  index_array = np.random.choice(N, replace = False, size = num)
  default_list = []
  for i in index_array:
    default_list.append(categories[i])
  return default_list

@st.cache(suppress_st_warning = True)
def standardizer(df, numerics):
  numeric_segment = df[numerics]
  numeric_segment_copy = df[numerics].to_numpy()
  normalizer = preprocessing.StandardScaler().fit(numeric_segment_copy)
  normalized_numeric_segment = pd.DataFrame(normalizer.transform(numeric_segment_copy),
                                          columns = numerics)
  return normalized_numeric_segment

@st.cache(suppress_st_warning = True)
def get_subgroups(df: pd.DataFrame, categories: list):
  '''
  args:
    df: dataset dataframe
    categories: a list of names of selected categories
  returns: a list subgroups
  '''
  cat_list = []
  for c in categories:
    cat_list.append(set(df[c]))
  subgroups = sorted(list(itertools.product(*cat_list)))
  subgroups_dict = {}
  for sg in subgroups:
    idx = pd.Series(np.ones([len(df)], dtype=bool))
    for i, feat in enumerate(sg):
      idx = idx & (df[categories[i]]==feat)
    subgroups_dict[sg] = pd.Index(idx)
  return subgroups_dict

@st.cache(suppress_st_warning = True)
def retrieve_levels(data, categories):
  '''
  Retrieve the levels of categorical variables
  A rather silly way though
  args:
    data: the dataset
    categories: the full list of categorical variables
  return:
    a dictionary containing levels.
  '''
  level_dict = {}
  for c in categories:
    level_dict[c] = data[c].unique()
  return level_dict

def plot_switcher(figure_dict, mode):
  '''
  Help readeres switch to different plots
  arg:
    figure_dict: A dictionary for figures
    mode: either "Similar" or "Different"
  '''
  cur_list = figure_dict[mode]
  figure_list_length = len(cur_list)
  if figure_list_length == 1:
    with st.container():
      st.pyplot(cur_list[0])
  elif figure_list_length > 1:
    with st.container():
      figure_index = st.slider(label = "Select" + mode + "Groups",min_value = 0,
                     max_value = figure_list_length, step = 1, value = 1)
      st.pyplot(cur_list[figure_index - 1])

