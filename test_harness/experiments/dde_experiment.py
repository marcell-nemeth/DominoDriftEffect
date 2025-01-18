import pandas as pd
from scipy.spatial import distance
import numpy as np
import nannyml as nml

class DDEffectExperiment:
    def __init__(self, _dataset):
        self.dataset = _dataset
        self.results = {}

        # drift scores
        self.results_df = {}
        self.results_all = pd.DataFrame()

        # correlations
        self.corr_scores = {}
        self.corr_scores_all = pd.DataFrame()

        self.univar_drift_calc = self.create_univar_calc(colnames=list(self.dataset.reference_df.columns),
                                                         cont_methods=['jensen_shannon'],
                                                         cat_methods=['jensen_shannon'],
                                                         chunk_number=1)

    def create_univar_calc(self, colnames, cont_methods, cat_methods, chunk_number):
        calc = nml.UnivariateDriftCalculator(
            column_names=colnames,
            continuous_methods=cont_methods,
            categorical_methods=cat_methods,
            chunk_number=chunk_number,
        )

        return calc


    def changepoint_radius_driftscore(self, radius_left, radius_right):
        for chp in self.dataset.change_points:
            print(chp-radius_left, chp, chp+radius_right)

            ref_df = self.dataset.full_df.loc[chp-radius_left:chp, :]
            ref_df['period'] = 'reference'
            ref_df['sample_num'] = ref_df.index

            analysis_df = self.dataset.full_df.loc[chp:chp+radius_right:, :]
            analysis_df['period'] = 'analysis'
            analysis_df['sample_num'] = analysis_df.index

            self.univar_drift_calc.fit(ref_df)
            self.results[chp] = self.univar_drift_calc.calculate(analysis_df)

            self.results_df[chp] = self.results[chp].filter(column_names=self.results[chp].continuous_column_names,
                                                            methods=['jensen_shannon']).to_df()
            self.results_df[chp].columns = ['_'.join(col) for col in self.results_df[chp].columns.values]
            self.results_df[chp] = self.results_df[chp].loc[:, [col for col in self.results_df[chp].columns if 'jensen_shannon_value' in col]]
            self.results_df[chp].loc[:, 'start'] = chp-radius_left
            self.results_df[chp].loc[:, 'change_point'] = chp
            self.results_df[chp].loc[:, 'end'] = chp + radius_left

        self.results_all = pd.concat([self.results_df[chp] for chp in self.dataset.change_points], ignore_index=True)

        cols_reordered = list(self.results_all.columns[-3:])+list(self.results_all.columns[:-3])
        self.results_all = self.results_all[cols_reordered]
        self.results_all.columns = [col.replace('_jensen_shannon_value', '') for col in self.results_all.columns]

    def changepoint_radius_corr(self, radius_left, radius_right):
        for chp in self.dataset.change_points:
            print(chp-radius_left, chp, chp+radius_right)

            ref_df = self.dataset.full_df.loc[chp - radius_left:chp, :]
            analysis_df = self.dataset.full_df.loc[chp:chp + radius_right:, :]

            self.corr_scores[chp] = {'reference': ref_df.corr(), 'analysis': analysis_df.corr(), 'corr_delta': np.abs(ref_df.corr()-analysis_df.corr())}






