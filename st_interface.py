# Loading packages

import warnings
# import pandas as pd 
import streamlit as st
apptitle = "CS765 DC2"
st.set_page_config(page_title=apptitle, page_icon=":heavy_check_mark:")
from otherTools import *
from backEnd import * 


# Load a sample dataset
# Here, we first go with heart failure prediction

dataset = pd.read_csv("Data/heart.csv")

# if we want to add a component for switching datasets,
# the following numerics list should be in a dictionary
numerics = ['RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

category_full_list = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 
					  'ExerciseAngina', 'ST_Slope', 'HeartDisease']

level_full_options = retrieve_levels(dataset, category_full_list)

# Preprocess the numeric data
# consider smart caching for the numerical segments.
normalized_numerics = standardizer(dataset, numerics)


# streamlit interface

st.sidebar.markdown("## Similarity Subgrouper")
cluster_k_selectbox = st.sidebar.slider(
    label = "Number of Clusters",
    min_value = 2,
    max_value = 10,
    step = 1,
    value = 5 # Default set as five / TODO: come up with an algorithm?
)

cluster_model = clusterModel(normalized_numerics, mode="kmeans")
cluster_labels = cluster_model.train(num_clusters = cluster_k_selectbox)

# TODO:
# How would determine the default display?
# Here we use a random approach

# default_categories = random_categories(category_full_list, num = 3)

categorical_scope = st.sidebar.multiselect(
	label = "Select Categorical Variables",
	options = category_full_list,
	default = ['Sex', 'ChestPainType', 'RestingECG']
	# TODO need a default setup?
)
level_selections = []
for cat in categorical_scope:
	current_choices = st.sidebar.selectbox(cat, level_full_options[cat])
	level_selections.append(current_choices)

# Display warning messages
if len(level_selections) == 0:
	warnings.warn("No categorical variables are selected!")
	st.markdown("<p style='color:red;'>WARNING: </p>Please select categorical variables to begin!",
				unsafe_allow_html=True)


# Create Subgroups

subgroups = get_subgroups(dataset, categorical_scope)

# Fit the model and find

backEnd_result = backEndModel(subgroups=subgroups, labels=cluster_labels, 
				 num_clusters=cluster_k_selectbox, data=dataset)
# try and catch when no levels are selected.
s_indices, d_indices = backEnd_result.retrieve_group_indices(tuple(level_selections))
subgroup_index = list(subgroups.keys()).index(tuple(level_selections))
visualizer = ourVisualizer(subgroup_index, s_indices, d_indices, subgroups, dataset[numerics])

# display the plot 
# add some button to shift plot type

radio_column1, radio_column2 = st.columns([1.5, 1])
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

# For radio, the default option is the first option (we can change it though)
radio1 = radio_column1.radio(label="Plot type", options=["Scatter plot", "Histograms"])
radio2 = radio_column2.radio(label="Presented Group", options=["Similar", "Different"])

# Quite complicated process ...
if radio1 == "Scatter plot":
	scatter_figure_dict = visualizer.scatter2D()
	plot_switcher(scatter_figure_dict, radio2)
else:
	st.text("PlaceHolder")

