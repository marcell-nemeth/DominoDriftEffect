from typing import Dict

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


class Dataset:

    def __init__(self, full_df: pd.DataFrame, column_mapping: Dict, window_size: int, pca: bool = False,
                 pca_dim: int = 2, change_points: list = None, first_chgpnt_lookback: int = 1000,
                 reference_size: int = 1000):

        self.pca_dim = pca_dim
        self.column_mapping = column_mapping
        self.original_df = full_df
        self.full_df = full_df

        if change_points:
            self.change_points = change_points[1:]  # leaving out the initial 0

        self.reference_size = reference_size

        if reference_size:
            self.reference_df = self.full_df.loc[:self.reference_size, :]
            self.analysis_df = self.full_df.loc[self.reference_size:, :]

        else:
            if self.change_points:
                self.reference_df = self.full_df.loc[:self.change_points[0] - first_chgpnt_lookback, :]
                self.analysis_df = self.full_df.loc[self.change_points[0] - first_chgpnt_lookback:, :]

            else:
                self.reference_df = self.full_df.loc[:int(len(self.full_df) / 2), :]
                self.analysis_df = self.full_df.loc[int(len(self.full_df) / 2):, :]

        self.reference_df['period'] = 'reference'
        self.reference_df['sample_num'] = self.reference_df.index

        self.analysis_df['period'] = 'analysis'
        self.analysis_df['sample_num'] = self.analysis_df.index

        self.fitted_pca = None
        self.explained_variance_absolutes = None
        self.explained_variance_ratios = None
        self.components_features_all = None
        self.components_features = None

        if pca:
            self.full_df, self.fitted_pca = self.prepare_pca_set()
            self.explained_variance_absolutes, self.explained_variance_ratios = self.set_pca_properties()
            self.components_features, self.components_features_all = self.get_features_per_components()

        self.window_size = window_size
        self.set_splits()
        self.df_dim_reduced = {}

    def add_changepoints(self, _chgps):
        self.change_points = _chgps

    def prepare_pca_set(self):
        pca = PCA(n_components=self.pca_dim)

        trfd_df = pd.DataFrame(pca.fit_transform(self.full_df.drop(self.column_mapping["target"], axis=1)))
        trfd_df[self.column_mapping["target"]] = self.full_df[self.column_mapping["target"]]

        return trfd_df, pca

    def get_features_per_components(self):

        pca = self.fitted_pca
        # number of components
        n_pcs = pca.components_.shape[0]

        # get the index of the most important feature on EACH component
        # LIST COMPREHENSION HERE
        most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]

        initial_feature_names = self.original_df.columns


        ordered_featnames = [[x for _, x in sorted(zip(np.abs(pca.components_[f]), initial_feature_names), reverse=True)] for f in range(n_pcs)]

        # get the names
        most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

        # LIST COMPREHENSION HERE AGAIN
        feat_names = {'PC{}'.format(i): ordered_featnames[i] for i in range(n_pcs)}
        dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}

        # build the dataframe
        comp_df = pd.DataFrame(dic.items())
        comp_df_all = pd.DataFrame(feat_names.items())

        return comp_df, comp_df_all

    def set_splits(self):
        """Use the specified window_size to set an attribute that holds corresponding index splits"""
        idx = self.window_size

        splits = []
        while idx < len(self.full_df):
            splits.append(idx)
            idx += self.window_size

        self.splits = splits

    def get_split_idx(self, window_idx):
        """Given a window_idx from an experiment, lookup the split_idx"""
        return self.splits[window_idx]

    def get_sliding_window_data(self, start, end, split_labels=True):
        window_data = self.full_df[start:end]

        if split_labels:
            features, labels = self.split_df(window_data, self.column_mapping["target"])
            return features, labels
        else:
            return window_data

    def get_window_data(self, window_idx, split_labels=True):
        """
        Given a window_idx corresponding to a split_idx, return the data up to that
        split value starting from the split_idx - 1 value.

        Args:
            window_idx (int) - index corresponding to the end point of the desired data window
            split_labels (bool) - return features and labels separately vs. as one dataframe

        Returns:
            features (pd.DataFrame)
            labels (pd.Series)

        TO-DO: add test to make sure this function gets the expected window data
        """

        end_idx = self.splits[window_idx]

        if window_idx == 0:
            window_data = self.full_df[:end_idx]
        else:
            start_idx = self.splits[window_idx - 1]
            window_data = self.full_df[start_idx:end_idx]

        if split_labels:
            features, labels = self.split_df(window_data, self.column_mapping["target"])
            return features, labels
        else:
            return window_data

    def get_data_by_idx(self, start_idx, end_idx, split_labels=True):
        """
        Given an index into the full_df, return all records up to that observation.

        Args:
            start_idx (int) - index corresponding to the row in full_df
            end_idx (int) - index corresponding to the row in full_df
            split_labels (bool) - return features and labels separately vs. as one dataframe

        Returns:
            features (pd.DataFrame)
            labels (pd.Series)

        TO-DO: should this skip over the first full window that was trained on.. meaning eval data only?

        """

        window_data = self.full_df[start_idx:end_idx]

        if split_labels:
            features, labels = self.split_df(window_data, self.column_mapping["target"])
            return features, labels
        else:
            return window_data

    @staticmethod
    def split_df(df, label_col):
        """Splits the features from labels in a dataframe, returns both"""
        return df.drop(label_col, axis=1), df[label_col]

    def set_pca_properties(self):
        return self.fitted_pca.explained_variance_, self.fitted_pca.explained_variance_ratio_
