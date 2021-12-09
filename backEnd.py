import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt


class clusterModel:
    def __init__(self, normalized_data, mode="kmeans"):
        self.mode = mode
        self.data = normalized_data

    def train(self, num_clusters, random_state=0):
        if self.mode == "kmeans":
            # Try normalized version
            kmeans_model = KMeans(
                n_clusters=num_clusters, random_state=random_state
            ).fit(self.data)
            cluster_labels = kmeans_model.labels_
            # Transform into Pandas dataframe,
            cluster_labels = pd.DataFrame(cluster_labels)
            return cluster_labels


class backEndModel:
    """
    input args in the constructor:
      described in the text section above
    """

    def __init__(self, subgroups, labels, num_clusters, data):
        self.subgroups = subgroups
        self.labels = labels
        self.num_clusters = num_clusters
        self.data = data
        self.subgroups_names = list(subgroups.keys())

    def get_feature_vectors(self, index: object):
        """
        args:
          index: the index of a selected subgroup
        return the feature vector of a subgroup in the form of np array
        """
        labels, num_clusters = self.labels, self.num_clusters
        cluster_conditions = labels[index]

        # Not all clusters would have data points, so we use a dictionary
        # Complexity O(k), the for loop should not be a big deal
        number_of_samples = len(cluster_conditions)
        feature_vector = [0.0] * self.num_clusters
        if number_of_samples == 0:
            return np.array(feature_vector)
        label_dict = cluster_conditions[0].value_counts().to_dict()
        for i in range(self.num_clusters):
            if i in label_dict:
                feature_vector[i] = label_dict[i] * 1.0 / number_of_samples
        return np.array(feature_vector)

    def subgroup_feature_matrix(self):
        """
        args:
          subgroup: the resulf of function get_subgroups -> a list subgroups with length L
          labels: the cluster labels of the dataset
          num_cluster: number of cluster
        return: a numpy array with dimension (L, num_cluster)
        """
        subgroups_names = self.subgroups_names
        L = len(subgroups_names)
        feature_array = np.empty([L, self.num_clusters])
        for i in range(L):
            subgroup_index = self.subgroups[subgroups_names[i]]
            feature_array[i] = self.get_feature_vectors(subgroup_index)
        self.feature_array = pd.DataFrame(feature_array)
        return feature_array

    def derive_underRep(self):
        """
        arg:
          feature_matrix: the matirx consists of feature vectors
        return: indices of underrepresented groups if exists
        """
        # Notice the sum should equal to 1 if not underrepresented
        underRep_indices = self.feature_array.index[self.feature_array.sum(axis=1) == 0]
        self.underRep_indices = underRep_indices
        return underRep_indices

    def similarity_measure(self, input_key):
        """
        args:
          self.feature_array: a matrix consists of feature vectors of the subgroup list
          self.input_key: the input of a user, would be a tuple, e.g ('F', 'TA', 'ST', 0)
          self.subgroups: the subgroup list
          self.underRep_indices: indices of underrepresented groups if specified
        return: the similarity vectors for a subgroup, we remove the vector itself
              and underrepresented groups.
        """
        feature_matrix = self.feature_array
        current_index = self.subgroups_names.index(input_key)
        current_vector = feature_matrix.loc[current_index]
        current_v_norm = np.linalg.norm(current_vector)
        if current_v_norm == 0:
            # Deal with underrepresented groups
            # Output WARNING and directly return
            warnings.warn("The selected group is underrepresented")
            return
        if len(self.underRep_indices):
            feature_matrix_meta = feature_matrix.drop(self.underRep_indices)
            feature_matrix_meta = feature_matrix_meta.drop(current_index)
        else:
            feature_matrix_meta = feature_matrix.drop(current_index)
        matrix_norm = np.linalg.norm(feature_matrix_meta, axis=1)
        inner_product = feature_matrix_meta.dot(current_vector) / (matrix_norm)
        inner_product /= current_v_norm
        return inner_product

    def retrieve_group_indices(self, input_key):
        """
        arg:
          similarity: the similarity of selected subgroups and others,
        return: the indices of the most similar subgroup and the most different subgroup
              in tuples. (similar_indices, different_indices)
        NOTICE: The most similar/different subgroup may not be unique
        """
        # We did not use df.idxMax since it returns the first one
        self.subgroup_feature_matrix()
        self.derive_underRep()
        similarity = self.similarity_measure(input_key)
        max_value = max(similarity)
        min_value = min(similarity)
        similarity_data = pd.DataFrame(similarity, columns=["score"])
        similar_indices = similarity_data.index[similarity_data["score"] == max_value]
        different_indices = similarity_data.index[similarity_data["score"] == min_value]
        similar_indices = list(similar_indices)
        different_indices = list(different_indices)
        # draft notification for the results
        print("Similar Subgroups for Current Selection:")
        for s in similar_indices:
            print(self.subgroups_names[s], end=";\n")
        # draft notification for the results
        print("")
        print("============================")
        print("Different Subgroups for Current Selection:")
        for d in different_indices:
            print(self.subgroups_names[d], end=";\n")
        return (similar_indices, different_indices)


# Visualizer class
# We add different designs into this class


class ourVisualizer:
    def __init__(self, cur_index, similar_indices, different_indices, subgroups, data):
        """
        args:
          cur_index: index of the selected subgroup, has to be an int
          similar_indices: the indices of similar_indices
          different_indices: the indices of different subgroups
          numeric_columns: current selection of numerical columns
          subgroups: the subgroups we have
          data: original data, but only with numeric values
        """
        self.indices_Dict = {"Similar": similar_indices, "Different": different_indices}
        self.data = data
        self.subgroups = subgroups
        self.subgroups_names = list(subgroups.keys())
        self.cur_group = data[subgroups[self.subgroups_names[cur_index]]]
        self.column_names = data.columns.values
        self.cur_mean = self.cur_group.mean()

    def getSubset(self, group_index):
        """
        args:
          group_index
        return:
          the data subset (corresponds to that group)
        """
        # First retrieve the 2 data subset.
        # We can keep the full dataset,
        # Though here, we choose to get only the numeric ones
        subset_group = self.data[self.subgroups[self.subgroups_names[group_index]]]
        return subset_group

    def getDirections(self, related_group, mode="Similar"):
        """
        Greedy approach,
        compare the difference in sample mean
        """
        related_mean = related_group.mean()
        difference = ((related_mean - self.cur_mean) ** 2).values
        difference_sorted = np.sort(difference)
        if mode == "Similar":
            first_value = difference_sorted[0]
            second_value = difference_sorted[1]
        if mode == "Different":
            first_value = difference_sorted[-1]
            second_value = difference_sorted[-2]
        first_index = np.argwhere(difference == first_value).item()
        second_index = np.argwhere(difference == second_value).item()
        first_direction = self.column_names[first_index]
        second_direction = self.column_names[second_index]
        return first_direction, second_direction

    def scatter_plot(self, mode, colors):
        """
        helper method to plot the scatterplot
        return a list of figures (matplotlib form)
        # TODO: I think it should be a dictionary
        # in order to add more descriptions.
        args:
          v1: Dimension 1
          v2: Dimension 2
          mode: "Similar" or "Different", to help us make titles
        """
        figure_list = []
        for s in self.indices_Dict[mode]:
            current_group = self.getSubset(s)
            direction1, direction2 = self.getDirections(current_group, mode)
            fig, ax = plt.subplots()
            ax.scatter(
                current_group[direction1], current_group[direction2], color=colors[0]
            )
            ax.scatter(
                self.cur_group[direction1], self.cur_group[direction2], color=colors[2]
            )
            ax.set_xlabel(direction1)
            ax.set_ylabel(direction2)
            ax.set_title("Why are they " + mode + "?")
            figure_list.append(fig)
        return figure_list

    def scatter_altair(self, mode, colors):
        """
        helper method to make the scatterplot with altair
        return a list of figures (matplotlib form)
        args:
          v1: Dimension 1
          v2: Dimension 2
          mode: "Similar" or "Different", to help us make titles
        """
        figure_list = []
        for s in self.indices_Dict[mode]:
            current_group = self.getSubset(s)
            direction1, direction2 = self.getDirections(current_group, mode)
            # `other_cols` for showing other information in the interactive figure
            other_cols = [i for i in current_group.columns if i not in [direction1, direction2]]
            fig = alt.Chart(current_group).mark_circle().encode(
                x=direction1,
                y=direction2,
                color=alt.value(colors[0]),
                tooltip=other_cols
            ).interactive() + \
            alt.Chart(self.cur_group).mark_circle().encode(
                x=direction1,
                y=direction2,
                color=alt.value(colors[2]),
                tooltip=other_cols
            ).interactive()
            fig = fig.properties(title="Why are they " + mode + "?", width=960, height=640)
            figure_list.append(fig)
        return figure_list

    def scatter2D(self, colors=["green", "blue", "red"]):
        """
        intention:
          draw scatterplots along 2 directions
          Directions are selected in a greedy approach
          closest mean values
        args:
          colors: choice of color [similar, different, itself]
        """
        # Plot similar first

        # We can also write a auxilary function for the repeated scripts
        # 这里应该单独搞个plot子函数，代码有冗余
        similar_figures = self.scatter_altair(mode="Similar", colors=colors)
        different_figures = self.scatter_altair(mode="Different", colors=colors)
        figure_dict = {"Similar": similar_figures, "Different": different_figures}
        return figure_dict
