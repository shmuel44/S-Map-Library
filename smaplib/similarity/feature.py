import os
import numpy as np
import tensorflow_hub as hu
import pandas as pd
import shelve as she
import dataiku as di
from typing import Any, Dict, List, Literal, Tuple, Union, Optional
from enum import Enum

from ..img import helpers as ih
from ..misc import http as hh
from ..misc import common as co
from ..fsio import dss as lio
from ..img import dss as lim


##########
# Config #
##########

class FEATURE_MODULE_CONFIG:
    VERBOSE: bool = False
    CACHE_DIR: str = '.'

#########
# Enums #
#########

class FeatureKind(Enum):
    """An enumeration of the kinds of features.
    """
    Embedding = 'embedding'
    Metadata = 'metadata'
    Output = 'output'
    Ignored = 'ignored'


class CaseConversion(Enum):
    """An enumeration of the kinds of case conversion.
    """
    NoChange = 'no-change'
    LowerCase = 'lower-case'
    UpperCase = 'upper-case'


#################
# Base Features #
#################

class BaseFeature:

    def __init__(
            self,
            feature_name:str,
            feature_type:str, 
            feature_kind:FeatureKind=FeatureKind.Embedding,
            is_key: bool=False,
            input_size:Optional[Union[int, Tuple[int, int]]]=None, 
            output_size:Optional[int]=None) -> None:

        self.feature_name = feature_name
        self.feature_type = feature_type
        self.feature_kind = feature_kind
        self.is_key = is_key
        self.input_size = input_size
        self.output_size = output_size

    ###################
    # Private methods #
    ###################

    def _printout_self_state(self):
        print('type: "{0}"'.format(type(self)))
        print('feature_name: "{0}"'.format(self.feature_name))
        print('feature_kind: "{0}"'.format(self.feature_kind))
        print('feature_type: "{0}"'.format(self.feature_type))
        print('is_key: "{0}"'.format(self.is_key))
        print('input_size: "{0}"'.format(self.input_size))
        print('output_size: "{0}"'.format(self.output_size))

    ##################
    # Public methods #
    ##################

    def process(self, data:Any) -> Any:
        if FEATURE_MODULE_CONFIG.VERBOSE:
            print('----------')
            print('class: "{0}", function: "process", data: "{1}"'.format(self.__class__.__name__, data))
            print('----------')
            print('')

        return self.process_data(
            data=self.preprocess_data(
                data=data))

    #####################################
    # Methods to override in subclasses #
    #####################################

    def sanity_check(self, dataset:pd.DataFrame) -> None:
        if self.feature_name not in dataset.columns:
            raise Exception('Feature "{0}" not in dataset'.format(self.feature_name))

        if self.feature_kind == FeatureKind.Embedding:
            if self.dimension() is None:
                raise Exception('Dimension is not set for feature "{0}"'.format(self.feature_name))
    
    def preprocess_data(self, data:Any) -> Any:
        return data

    def process_data(self, data:Any) -> Any:
        return data
    
    def dimension(self) -> Optional[int]:
        return self.output_size


class BaseHubFeature(BaseFeature):

    models = {}

    def __init__(
            self, 
            feature_name:str,
            feature_type:str, 
            model_url:str, 
            is_key: bool=False,
            input_size:Optional[Union[int, Tuple[int, int]]]=None, 
            output_size:Optional[int]=None) -> None:
        super().__init__(
            feature_name=feature_name,
            feature_type=feature_type,
            feature_kind=FeatureKind.Embedding,
            is_key=is_key,
            input_size=input_size,
            output_size=output_size)

        self.model_url = model_url
        self.model = BaseHubFeature._register_model(self.model_url)

        self._embeddings_cache = None
        self._load_cache()

    ###################
    # Private methods #
    ###################

    def _printout_self_state(self):
        super()._printout_self_state()
        print('model_url: "{0}"'.format(self.model_url))

    def __exit__(self, *exc_info):
        self._save_cache()

    def _load_cache(self):
        cache_path = self._get_cache_path()

        self._embeddings_cache = she.open(
            filename=cache_path)

    def _get_cache_path(self):
        cache_path = FEATURE_MODULE_CONFIG.CACHE_DIR

        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        return os.path.join(
            cache_path, 
            'CACHE__{0}'.format(self.feature_type))

    def _save_cache(self):
        if not self._embeddings_cache is None:
            self._embeddings_cache.close()

    def _flush_cache(self):
        self._save_cache()
        self._load_cache()

    def _empty_embedding(self) -> np.ndarray:
        return np.array(
            [0]*self.dimension(), 
            np.float32)

    def _get_cache_key(self, key:Any) -> Tuple[Any, Any]:
        
        cache_key = None
        cache_key_exists = False

        if isinstance(key, float) and np.isnan(key):
            cache_key = None
        elif isinstance(key, str) and key == '':
            cache_key = None
        else:
            cache_key = str(key)
        
        if self._embeddings_cache is not None:
            if cache_key is not None:
                cache_key_exists = cache_key in self._embeddings_cache

        return cache_key, cache_key_exists
        
    ##################
    # Public methods #
    ##################

    def process(self, data:Any) -> Any:
        if FEATURE_MODULE_CONFIG.VERBOSE:
            print('----------')
            print('class: "{0}", function: "process", data: "{1}"'.format(self.__class__.__name__, data))
            print('----------')
            print('')

        output = None

        try:
            cache_key, cache_key_exists = self._get_cache_key(
                key=data)

            if cache_key_exists:
                output = self._embeddings_cache[cache_key]
            else:
                output = super().process(data=data)

                if cache_key is not None:
                    self._embeddings_cache[cache_key] = output
        except Exception as e:  
            if FEATURE_MODULE_CONFIG.VERBOSE:
                self._printout_self_state()
                print('data: "{0}" (type: "{1}")'.format(data, type(data)))
                print('output: "{0}"'.format(output))
                print('Exception: "{0}" (type: "{1}")'.format(e, type(e)))
        
        return output

    ####################
    # Static / Private #
    ####################

    @staticmethod
    def _register_model(model_url):
        if not model_url in BaseHubFeature.models:
            BaseHubFeature.models[model_url] = hu.load(model_url)

        return BaseHubFeature.models[model_url]

    #############
    # Overrides #
    #############

    def sanity_check(self, dataset:pd.DataFrame) -> None:
        super().sanity_check(dataset=dataset)

        if self.model is None:
            raise Exception('Model not set for feature "{0}"'.format(self.feature_name))

        if self.input_size is None:
            raise Exception('Input size not set for feature "{0}"'.format(self.feature_name))
        
    def process_data(self, data:Any) -> Any:
        if FEATURE_MODULE_CONFIG.VERBOSE:
            print('----------')
            print('class: "{0}", function: "process", data: "{1}"'.format(self.__class__.__name__, data))
            print('----------')
            print('')

        if data is None:
            return self._empty_embedding()
        else:
            return self.model(data)[0]

    def preprocess_data(self, data:Any) -> Any:
        raise NotImplementedError()


class BaseTextFeature(BaseHubFeature):

    def __init__(
            self, 
            feature_name:str,
            feature_type:str, 
            model_url:str, 
            is_key: bool=False,
            input_size:Optional[Union[int, Tuple[int, int]]]=None, 
            output_size:Optional[int]=None) -> None:
        super().__init__(
            feature_name=feature_name,
            feature_type=feature_type,
            model_url=model_url,
            is_key=is_key,
            input_size=input_size,
            output_size=output_size)

    #############
    # Overrides #
    #############

    def preprocess_data(self, data:Any) -> Any:
        input = data
        if self.input_size is not None and len(data) > self.input_size:
            input = data[:self.input_size]
            
        return np.array([input])


class BaseImageFeature(BaseHubFeature):

    def __init__(
            self, 
            feature_name: str,
            feature_type: str, 
            model_url: str, 
            is_key: bool=False,
            folder: Optional[Union[str, di.Folder]]=None,
            input_size: Optional[Union[int, Tuple[int, int]]]=None, 
            output_size: Optional[int]=None) -> None:
        super().__init__(
            feature_name=feature_name,
            feature_type=feature_type,
            model_url=model_url,
            is_key=is_key,
            input_size=input_size,
            output_size=output_size)

        self._folder = None
        
        if folder is not None:
            self._folder = lio.get_folder(folder)

    #############
    # Overrides #
    #############

    def preprocess_data(self, data:Any) -> Any:
        if FEATURE_MODULE_CONFIG.VERBOSE:
            print('----------')
            print('class: "{0}", function: "process", data: "{1}"'.format(self.__class__.__name__, data))
            print('----------')
            print('')

        if data is None:
            return None
        elif isinstance(data, float) and np.isnan(data):
            return None
        elif isinstance(data, str) and data == '':
            return None
        
        string_kind = co.get_string_kind(data)
        pil_image = None

        try:
            if string_kind == co.StringKind.Url:
                pil_image = hh.download_image(url=data)
            elif string_kind == co.StringKind.Path:
                if self._folder is None:
                    pil_image = ih.open_image(
                        file_path=data)
                else:
                    pil_image = lim.open_valid_image(
                        folder=self._folder, 
                        file_path=data)
        except:
            pil_image = None

        if pil_image is not None:
            if pil_image.size != self.input_size:
                pil_image = ih.extract_square_portion(
                    image=pil_image,
                    horizontal_position='center', 
                    vertical_position='middle', 
                    output_size=self.input_size)

            return ih.image_to_batch_array(
                pil_image=pil_image,
                rescaled=True)
        else:
            return None


class BaseMetadataFeature(BaseFeature):

    def __init__(
            self, 
            feature_name:str,
            feature_type:str, 
            is_key: bool=False):
        super().__init__(
            feature_name=feature_name,
            feature_type=feature_type,
            feature_kind=FeatureKind.Metadata,
            is_key=is_key,
            input_size=None,
            output_size=None)


class BaseOutputFeature(BaseFeature):

    def __init__(
            self, 
            feature_name:str,
            feature_type:str, 
            is_key: bool=False):
        super().__init__(
            feature_name=feature_name,
            feature_type=feature_type,
            feature_kind=FeatureKind.Output,
            is_key=is_key,
            input_size=None,
            output_size=None)


###################
# Usable Features #
###################

class CategoryFeature(BaseFeature):

    def __init__(
            self, 
            feature_name:str,
            trim:bool=True, 
            case_conversion:CaseConversion=CaseConversion.NoChange,
            is_key: bool=False,
            output_size:Optional[int]=None) -> None:
        
        super().__init__(
            feature_name=feature_name,
            feature_type='category',
            feature_kind=FeatureKind.Embedding,
            is_key=is_key,
            input_size=None,
            output_size=output_size)
        
        self.trim = trim
        self.case_conversion = case_conversion
        self.reset_categories()

    ###################
    # Private methods #
    ###################

    def _apply_conversion(
            self, 
            value:str, 
            trim:bool=True, 
            case_conversion:CaseConversion=CaseConversion.NoChange) -> str:
        
        if value is not None:
            if trim:
                value = value.strip()
                    
            if case_conversion == CaseConversion.LowerCase:
                value = value.lower()
            elif case_conversion == CaseConversion.UpperCase:
                value = value.upper()
        
        return value

    def _get_output_size(self) -> int:
        if self.output_size is None:
            return len(self.categories)
        else:
            return self.output_size
        
    def _empty_embedding(self) -> np.ndarray:
        return np.array(
            [0] * self._get_output_size(),
            dtype=np.float32)
    
    #############
    # Overrides #
    #############

    def process_data(self, data:Any) -> Any:
        vect = self._empty_embedding()

        if data is not None:
            if data in self.categories:
                vect[self.categories_revert[data]] = 1

        return vect
    
    def dimension(self) -> Optional[int]:
        return self._get_output_size()

    #############
    # Additions #
    #############

    def reset_categories(self) -> None:
        self.categories:List[str] = []
        self.categories_revert:Dict[str, int] = {}

    def load_categories(self, categories: List[str]) -> None:
        if categories is not None:
            for category in categories:
                self.register_category(category)

    def register_category(self, category:str):
        category = self._apply_conversion(
            value=category,
            trim=self.trim,
            case_conversion=self.case_conversion)
        
        if category not in self.categories:
            self.assert_capacity()
            self.categories.append(category)
            self.categories_revert[category] = self.categories.index(category)

    def assert_capacity(self):
        if self.output_size is not None and len(self.categories) >= self.output_size:
            raise Exception('Categories capacity full')


class CsvCategoryFeature(CategoryFeature):

    def __init__(
            self, 
            feature_name:str,
            separator: str=',',
            trim: bool=True, 
            case_conversion: CaseConversion=CaseConversion.NoChange,
            is_key: bool=False,
            output_size: int=128) -> None:
        
        super().__init__(
            feature_name=feature_name,
            trim=trim,
            case_conversion=case_conversion,
            is_key=is_key,
            output_size=output_size)

        self.feature_type = 'csv_categoy'        
        self.separator = separator

    ###################
    # Private methods #
    ###################

    def _concatenate_csv_strings(
            self, 
            csv_strings:List[str], 
            separator:str=',', 
            trim:bool=True, 
            case_conversion:CaseConversion=CaseConversion.NoChange) -> List[str]:
        result = []
        for csv_string in csv_strings:
            values = csv_string.split(separator)
            
            # Trim and change case for each value as specified
            for i, value in enumerate(values):
                values[i] = self._apply_conversion(value, trim, case_conversion)
            
            result.extend(values)
        return result
    
    #############
    # Overrides #
    #############

    def load_categories(self, categories):
        if categories is not None:
            for category in self._concatenate_csv_strings(
                    csv_strings=categories,
                    separator=self.separator,
                    trim=self.trim,
                    case_conversion=self.case_conversion):
                self.register_category(category)

    def register_category(self, category:str):
        for c in self._concatenate_csv_strings(
                csv_strings=[category],
                separator=self.separator,
                trim=self.trim,
                case_conversion=self.case_conversion):
            if not c in self.categories:
                self.assert_capacity()
                self.categories.append(c)
                self.categories_revert[c] = self.categories.index(c)


class BooleanFeature(BaseFeature):

    def __init__(
            self, 
            feature_name:str,
            default_value:float=0.5,
            is_key: bool=False) -> None:
        super().__init__(
            feature_name=feature_name,
            feature_type='boolean',
            feature_kind=FeatureKind.Embedding,
            is_key=is_key,
            input_size=None,
            output_size=1)
    
        self.default_value = default_value

    #############
    # Overrides #
    #############

    def preprocess_data(self, data:Any) -> Any:
        value = self.default_value

        if data == True or data == 1:
            value = 1
        elif data == False or data == 0:
            value = 0
        return value

    def process_data(self, data:Any) -> Any:
        return np.array(
            object=[data], 
            dtype=np.float32)
        

class NumericFeature(BaseFeature):

    def __init__(
            self, 
            feature_name:str,
            default_value:float=0.0, 
            range:Optional[Tuple[float, float]]=None,
            is_key: bool=False):
        super().__init__(
            feature_name=feature_name,
            feature_type='numeric',
            feature_kind=FeatureKind.Embedding,
            is_key=is_key,
            input_size=None,
            output_size=1)

        self.default_value = default_value
        self.range = range

    ###################
    # Private methods #
    ###################

    def _empty_embedding(self) -> np.ndarray:
        return np.array(
            [self.default_value],
            dtype=np.float32)
    
    #############
    # Overrides #
    #############

    def preprocess_data(self, data:Any) -> Any:
        if data is not None:
            data = float(data)            

            if self.range is not None:
                data = (data - self.range[0]) / (self.range[1] - self.range[0])
        else:
            data = self.default_value

        return data

    def process_data(self, data:Any) -> Any:
        return np.array(
            [data],
            dtype=np.float32)


class ShortTextFeature(BaseTextFeature):

    def __init__(
            self, 
            feature_name:str,
            is_key: bool=False,
            input_size:Optional[Union[int, Tuple[int, int]]]=None) -> None:
        super().__init__(
            feature_name=feature_name,
            feature_type='tfhub_dev_google_nnlm_en_dim128_2',
            model_url='https://tfhub.dev/google/nnlm-en-dim128/2',
            is_key=is_key,
            input_size=input_size,
            output_size=128)


class LongTextFeature(BaseTextFeature):

    def __init__(
            self, 
            feature_name:str,
            is_key: bool=False,
            input_size:Optional[Union[int, Tuple[int, int]]]=None) -> None:
        super().__init__(
            feature_name=feature_name,
            feature_type='tfhub_dev_google_universal_sentence_encoder_large_5',
            model_url='https://tfhub.dev/google/universal-sentence-encoder-large/5',
            is_key=is_key,
            input_size=input_size,
            output_size=512)


class StandardImageFeature(BaseImageFeature):

    def __init__(
            self, 
            feature_name: str,
            is_key: bool=False,
            folder: Optional[Union[str, di.Folder]]=None) -> None:
        super().__init__(
            feature_name=feature_name,
            feature_type='tfhub_dev_google_imagenet_inception_v3_feature_vector_5',
            model_url='https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5',
            is_key=is_key,
            folder=folder,
            input_size=(500, 500),
            output_size=2048)


class ClothingImageFeature(BaseImageFeature):

    def __init__(
            self, 
            feature_name:str,
            is_key: bool=False,
            folder: Optional[Union[str, di.Folder]]=None) -> None:
        super().__init__(
            feature_name=feature_name,
            feature_type='tfhub_dev_google_experts_bit_r50x1_in21k_clothing_1',
            model_url='https://tfhub.dev/google/experts/bit/r50x1/in21k/clothing/1',
            is_key=is_key,
            folder=folder,
            input_size=(500, 500),
            output_size=2048)


#####################
# Multiple Features #
#####################

class FeatureSet:

    def __init__(
            self,
            features: Optional[List[BaseFeature]]=None) -> None:
        self.features = features

    ###################
    # Private methods #
    ###################

    def _get_row_processed_data_of_kind(
            self, 
            row_data: Any,
            kind: FeatureKind) -> List[Any]:
        if self.features is None:
            raise Exception('Features not set')
        
        return [
            f.process(data=row_data[f.feature_name])
            for f in self.features if f.feature_kind == kind
        ]

    ##################
    # Public methods #
    ##################

    def set_features(self, features: List[BaseFeature]) -> None:
        self.features = features

    def sanity_check(
            self,
            dataset: pd.DataFrame) -> None:
        if dataset is None:
            raise Exception('Data not set')

        if self.features is None:
            raise Exception('Features not set')
        
        for feature in self.features:
            feature.sanity_check(dataset)

        if len([f for f in self.features if f.is_key]) == 0:
            raise Exception('No key features found')

    def get_row_embedding_dimension(self) -> int:
        if self.features is None:
            raise Exception('Features not set')
        
        return sum([
            f.output_size 
            for f in self.features 
            if f.feature_kind == FeatureKind.Embedding]) # type: ignore
    
    def get_row_embedding(self, row_data:Any) -> np.ndarray:
        return np.concatenate(
            tuple(self._get_row_processed_data_of_kind(
                row_data=row_data, 
                kind=FeatureKind.Embedding))) # type: ignore
    
    def get_row_metadata(self, row_data:Any) -> List[Any]:
        return self._get_row_processed_data_of_kind(
            row_data=row_data, 
            kind=FeatureKind.Metadata)
    
    def get_row_output(self, row_data:Any) -> List[Any]:
        return self._get_row_processed_data_of_kind(
            row_data=row_data, 
            kind=FeatureKind.Output)
    
    def get_row_key(self, row_data:Any) -> Any:
        '''Returns the key for the given row data.'''
        if self.features is None:
            raise Exception('Features not set')
        
        key = tuple([
            row_data[f.feature_name]
            for f in self.features
            if f.is_key
        ])

        return key if len(key) > 1 else key[0]
    
