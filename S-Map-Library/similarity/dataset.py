

from . import models as md

import os as os
import io as io
import math as m
import pandas as pd
import numpy as np
import copy as cp
import pickle as pk
import datetime as dt
import time as ti
import itertools as it
import sklearn as sk

import sklearn as sk
from sklearn import preprocessing as skp
from sklearn import dataset_selection as skms
from sklearn import pipeline as skpl
from sklearn import decomposition as skd
from sklearn import linear_dataset as sklm
from sklearn import ensemble as skle
from sklearn import neighbors as skln
from sklearn import dummy as sky
from sklearn import metrics as skm
from sklearn import calibration as skc
from sklearn.utils import validation as skuv

from ..utilities import imagehelpers as ih
from ..utilities import httphelpers as hh
from ..utilities import filehelpers as fh
from ..utilities import classificationhelpers as ch
from . import embedders as emb



class InputFeature(object):

    def __init__(self, feature_name, feature_type):
        self.feature_name = feature_name
        self.feature_type = feature_type
        self.classes = None
        self.range = None


class ProductDataset(object):

    def __init__(self, dataset_name, input_features, cache_path=None, max_nb_categories=128):
        self.embedders = {}
        self.dataset_name = dataset_name
        self.input_features = input_features
        self.cache_path = cache_path
        self.max_nb_categories = max_nb_categories

        self.register_default_embedders(
            cache_path=cache_path)

    def register_embedder(self, embedder):
        self.embedders[embedder.input_type] = embedder

    def register_default_embedders(self, cache_path):
        self.register_embedder(emb.CategoryEmbedder(cache_path=cache_path))
        self.register_embedder(emb.NumericEmbedder(cache_path=cache_path))
        self.register_embedder(emb.TextEmbedder(cache_path=cache_path, **md.predefined_models['text']))
        self.register_embedder(emb.TextEmbedder(cache_path=cache_path, **md.predefined_models['longtext']))
        self.register_embedder(emb.ImageUrlEmbedder(cache_path=cache_path, **md.predefined_models['image_url']))
        self.register_embedder(emb.ImageUrlEmbedder(cache_path=cache_path, **md.predefined_models['clothing_url']))

    def flush_embedders_cache(self):
        for _, value in self.embedders.items():
            value.flush_cache()

    def save_configuration(self, include_data=False):
        configuration = {
            'dataset_name': self.dataset_name,
            'input_features': self.input_features,
            'max_nb_categories': self.max_nb_categories
        }

        return configuration

    @staticmethod
    def load_configuration(configuration):
        dataset = ProductDataset(
            dataset_name=configuration['dataset_name'],
            input_features=configuration['input_features'],
            max_nb_categories=configuration['max_nb_categories'])

        return dataset

    def save_data(self, filename):
        self.hierarchy.save_data(filename)

    def save_(self, filepath, include_data=False):
        dataset_filepath = fh.save_to_pickle(
            filepath=filepath,
            data=self.save_configuration(
                include_data=include_data))

        print('Dataset saved in "{0}"'.format(dataset_filepath))

    @staticmethod
    def load_dataset(filepath):
        return ProductDataset.load_configuration(
            fh.load_from_pickle(
                filepath=filepath))

    def get_summary(self, show_parameters=False):
        if self.hierarchy is None:
            return None

        return pd.DataFrame(
            self.hierarchy.get_summary(
                show_parameters=show_parameters)).sort_values(
                    by=['depth', 'path'],
                    ascending=[True, True])

    def get_min_samples_per_class(self, min_samples_per_class=10):
        return pd.DataFrame.from_records(
            data=self.hierarchy.get_min_samples_per_class(
                min_samples_per_class=min_samples_per_class))

    def load_from_csv(self, input_file, sep=',', header=0):
        self.load_from_dataframe(
            data=pd.read_csv(
                filepath_or_buffer=input_file,
                sep=sep,
                header=header))

    @staticmethod
    def analyze_csv(input_file, sep=',', header=0):
        return ProductDataset.analyze_dataframe(
            data=pd.read_csv(
                filepath_or_buffer=input_file,
                sep=sep,
                header=header))

    @staticmethod
    def analyze_dataframe(data):
        return data.describe(include=['object']).transpose()

    def process_data(self, data):

        X = []
        y = []
        index = 0

        y_cols = []
        output_feature = self.output_feature_hierarchy

        while output_feature is not None:
            y_cols.append(output_feature.feature_name)
            output_feature = output_feature.child_feature

        # load the data
        for r in data.itertuples(index=True):

            r_dict = r._asdict()

            X.append(
                self._get_vector(r_dict))

            _y = []

            for y_col in y_cols:
                _y.append(
                   r_dict[y_col])

            y.append(
                np.array(_y))

            index = index + 1

            if index % 10000 == 0:
                print(' -> Processed {0} inputs so far.'.format(index))

        return np.array(X, dtype=float), np.array(y, dtype=str)

    def load_data(self, data):

        for input_feature in self.input_features:
            if input_feature.feature_type == 'category':
                input_feature.classes = data[input_feature.feature_name].unique().tolist()

                if len(input_feature.classes) > self.max_nb_categories:
                    raise OverflowError(
                        'Too many values ({0}) for category "{1}". Maximum number of categories allowed is {2}. Switch type to "text" instead.'.format(
                            len(input_feature.classes),
                            input_feature.feature_name,
                            self.max_nb_categories))

            elif input_feature.feature_type == 'numerical' and input_feature.range is None:
                input_feature.range = (data[input_feature.feature_name].min(), data[input_feature.feature_name].max())

        x = []
        y = []
        index = 0

        # load the data
        for r in _data.itertuples(index=True):

            r_dict = r._asdict()

            if not self.valid_input(r_dict):
                raise OverflowError(
                    'Invalid input (missing one or more features): "{0}".'.format(r_dict))

            if r_dict[element.output_feature.feature_name] in element.classes:
                x.append(self._get_vector(r_dict))
                y.append(element.classes.index(r_dict[element.output_feature.feature_name]))
            else:
                print('  -> Unknown class: {0}'.format(r_dict[element.output_feature.feature_name]))

            index = index + 1

            if index % 10000 == 0:
                print(' -> Processed {0} inputs so far.'.format(index))

        if len(x) > 0 and len(y) > 0:

            element.data[subset]['X'] = np.array(x, dtype=np.float64)
            element.data[subset]['y'] = np.array(y, dtype=np.int32)

            # element.set_min_samples_per_class(
            #     min_samples_per_class=25)

            print(' -> There are {0} classes and {1} samples.'.format(
                len(element.classes),
                element.data[subset]['y'].shape[0]))

        for child_element in element.children:

            self._dataload_from_dataframe(
                element=child_element,
                data=_data,
                subset=subset)

    def _load_from_dataframe(self, data, parent_output_feature, output_feature, filter_value, depth, path):

        element = HierarchyElement()
        element.parent_output_feature = parent_output_feature
        element.output_feature = output_feature
        element.filter_value = filter_value
        element.depth = depth
        element.path = path

        print('Processing output-feature "{0}" for path "{1}".'.format(
            element.output_feature.feature_name,
            element.path))

        if element.output_feature is None:
            return None

        element.prepare_data_subset(
            subset='training')
        
        _data = None

        if element.parent_output_feature is None:
            _data = data
        else:
            _data = data[(
                data[element.parent_output_feature.feature_name] == element.filter_value)]

        element.classes = _data[element.output_feature.feature_name].unique(
        ).tolist()

        print(' -> There are {0} classes and {1} samples.'.format(
            len(element.classes),
            _data.shape[0]))

        x = []
        y = []
        index = 0

        # load the data
        for r in _data.itertuples(index=True):

            r_dict = r._asdict()

            if not self.valid_input(r_dict):
                raise OverflowError(
                    'Invalid input (missing one or more features): "{0}".'.format(r_dict))

            x.append(self._get_vector(r_dict))
            y.append(element.classes.index(r_dict[element.output_feature.feature_name]))

            index = index + 1

            if index % 10000 == 0:
                print(' -> Processed {0} inputs so far.'.format(index))

        if len(x) > 0 and len(y) > 0:
            element.X = np.array(x, dtype=np.float64)
            element.y = np.array(y, dtype=np.int32)

            element.set_min_samples_per_class(
                min_samples_per_class=25)

            print(' -> There are {0} classes and {1} samples.'.format(
                len(element.classes),
                element.y.shape[0]))

        if element.output_feature.child_feature is not None:
            for class_value in element.classes:
                child_element = self._load_from_dataframe(
                    data=_data,
                    parent_output_feature=element.output_feature,
                    output_feature=element.output_feature.child_feature,
                    filter_value=class_value,
                    depth=depth+1,
                    path='{0} / {1}'.format(element.path, class_value))

                if child_element is not None:
                    element.children.append(child_element)

        return element

    def valid_input(self, input):
        valid = True

        for input_feature in self.input_features:
            if input_feature.feature_name not in input:
                valid = False
                break

        return valid

    def predict(self, input, append_proba=False, last_level_only=False):
        output = []

        if self.valid_input(input):
            self._predict(
                X=self._get_vector(input),
                hierarchy_element=None,
                output=output,
                append_proba=append_proba,
                last_level_only=last_level_only)

        return output

    def _predict(self, X, hierarchy_element=None, output=None, append_proba=False, last_level_only=False):
        if hierarchy_element is None:
            if self.hierarchy is None:
                return None
            else:
                hierarchy_element = self.hierarchy

        if hierarchy_element.estimator is not None:
            if last_level_only == False or len(hierarchy_element.children) == 0:
                y = hierarchy_element.predict([X])[0]
                y_name = hierarchy_element.get_class_name(y)

                result = {
                    'output_feature': hierarchy_element.output_feature.feature_name,
                    'predicted_class': y_name,
                    'predicted_class_index': int(y)
                }

                if append_proba:
                    proba_values = hierarchy_element.estimator.predict_proba([X])[
                        0]

                    proba_results = {}

                    for idx, proba_value in enumerate(proba_values):
                        proba_results[hierarchy_element.get_class_name(
                            idx)] = float(proba_value)

                    result['predicted_proba'] = proba_results

                output.append(result)

            if len(hierarchy_element.children) > 0:
                self._predict(
                    X=X,
                    hierarchy_element=hierarchy_element.get_child_by_filter_value(
                        y_name),
                    output=output,
                    append_proba=append_proba)

    def _get_vector(self, input):

        _x = None

        for input_feature in self.input_features:
            _vect = self.embedders[input_feature.feature_type].embed_data(
                input[input_feature.feature_name],
                input_feature)

            # append to the vector
            if _x is None:
                _x = _vect.copy()
            else:
                _x = np.append(_x, _vect)

        return _x


