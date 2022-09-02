# =================================================================================
#                      Metadata Generator from Column (MGC)
# =================================================================================
#
# TODO: convert min/max values to hashvalues
#
import hashlib
import bz2
import pickle
import re
import copy
import fuzzy
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from collections import defaultdict
from pyroaring import BitMap

# --- version
version = '0.0.1'

class Utilities:

    def __init__(self):
        """
        """
        self.sample_url = 'https://data.cityofnewyork.us/resource/erm2-nwe9.json'

    def generate_chunks(self, values:list, chunk_size:int):
        """
        ref: https://stackoverflow.com/a/434328
        """
        return (values[pos:pos + chunk_size] for pos in range(0, len(values), chunk_size))

    def load_sample_table(self):
        """
        """
        # --- required packages
        import requests
        import json

        # --- API request
        res = requests.get(self.sample_url)

        # --- JSON from requested output
        res_json = json.loads( res.text )

        # --- dataframe
        df = pd.DataFrame.from_dict(res_json, orient = 'columns', dtype = str )

        # --- drop na
        df.dropna()

        return df

    def exclude_missing_values(self, values:list):
        """
        """
        _values = [v for v in values if pd.isnull(v) == False]
        num_null = len(values) - len(_values)
        return (_values, num_null)

class GenerateMetadataValues(Utilities):

    def __init__(self):
        self.num_samples = 20
        self.chunk_size = 10
        self.stats_per_dtype = True

    def classify_datatype_and_length(self, value:str):
        """
        string invovles datetime. input should not be bytes.
        """
        try:
            return ('integer', int(value), len(value))
        except:
            try:
                return ('float', float(value), len(value))
            except:
                return ('string', value, len(value))

    def aggregate_datatype_and_length(self, values:list):
        """
        """
        # --- initialization
        res = []
        dict_metadata = defaultdict(int)

        # --- aggregate values by datatype and length
        for value in values:
            dtype, value, length = self.classify_datatype_and_length(value)
            dict_metadata[f'{dtype}_{length}'] += 1

        # --- elaborate aggregated results 
        for key,v in dict_metadata.items():
            dtype, length_string = key.split('_')
            res.append(
                {
                    'datatype': dtype,
                    'length': int(length_string),
                    'count': v
                }
            )    
        return res

    def calculate_stats_per_datatype(self, values:list, chunk_size:int):
        """
        """
        # --- initialization
        res = defaultdict()
        res_chunks = defaultdict(list)

        # --- loop
        for chunk in self.generate_chunks(values, chunk_size):

            _res = defaultdict(list)

            # --- 
            for value in chunk:
                dtype, value, _ = self.classify_datatype_and_length(value)
                _res[dtype].append(value)

            # --- evaluate min/max/mean
            for dtype in _res.keys():
                res_chunks[f'{dtype}_min'].append( min(_res[dtype]) )
                res_chunks[f'{dtype}_max'].append( max(_res[dtype]) )
                res_chunks['datatypes'].append(dtype)            

                # --- string does not have mean
                if dtype != 'string':
                    res_chunks[f'{dtype}_num_samples'].append( len(_res[dtype]) )
                    res_chunks[f'{dtype}_mean_per_chunk'].append( np.mean(_res[dtype]) )

        # ---
        for dtype in res_chunks['datatypes']:
            _res = {
                'value_min': min(res_chunks[f'{dtype}_min']),
                'value_max': max(res_chunks[f'{dtype}_max'])
            }
            try:
                tmp = [mu*n for mu,n in zip(res_chunks[f'{dtype}_mean_per_chunk'], res_chunks[f'{dtype}_num_samples'])]          
                _res['value_mean'] = sum(tmp)/sum(res_chunks[f'{dtype}_num_samples'])
            except:
                pass

            res[dtype] = _res

        return res

    def generate_kde_distribution(self, values:list, num_samples:int=None, bw_method=None):
        """
        np.nanmax() and np.nanmin() ignores the missing values
        """
        # --- define sample range
        sample_range = np.nanmax(values) - np.nanmin(values)

        if num_samples is None:
            num_samples = self.num_samples

        # --- define x
        x = np.linspace(
                        np.nanmin(values) - 0.5 * sample_range,
                        np.nanmax(values) + 0.5 * sample_range,
                        num_samples,
        )

        # --- kde model
        kde = gaussian_kde(values)

        # --- bandwidth
        if bw_method in ['scott', 'silverman']:
            pass
        elif bw_method is None:
            bw_method = 'scott'    
        else:
            bw_method = kde.factor * float(bw_method)

        # --- set_bandwidth
        kde.set_bandwidth(bw_method = bw_method)

        # --- fitting
        y = kde.evaluate(x)
        
        return (x, y)

    def generate_metadata(self, values:list, stats_per_dtype:bool=None, chunk_size:int=None):
        """
        """
        # --- initialization
        res = []

        if chunk_size == None:
            chunk_size = self.chunk_size

        if stats_per_dtype == None:
            stats_per_dtype = self.stats_per_dtype

        # --- preprocess
        _values, num_null = self.preprocess_values(values)

        # --- aggregation
        _res = self.aggregate_datatype_and_length(_values)

        # --- stats using pandas
        df = pd.DataFrame.from_dict(_res, orient = 'columns')

        # ---
        if stats_per_dtype:
            _res_stats = self.calculate_stats_per_datatype(_values, chunk_size)

        for _dtype in df.datatype.unique():

            _df = df[df.datatype == _dtype].copy()

            # --- min & max legnths and counts (per datatype)
            length_min  = _df['length'].min()
            length_max  = _df['length'].max()
            count_dtype = _df['count'].sum()

            # --- mean length
            _df['count_length'] = _df['length'] * _df['count']
            length_mean = _df['count_length'].sum()/_df['count'].sum()

            # --- distribution in length
            _values_inflated = []
            for _,row in _df[['length', 'count']].iterrows():
                _values_inflated.extend( [row['length']]*row['count'] )

            length_median = np.median(_values_inflated)

            # --- output
            _res = {
                'datatype': _dtype,
                'count': count_dtype,
                'num_null': num_null,
                'num_distinct': len(set(_values)),
                'length_min': length_min,
                'length_max': length_max,
                'length_mean': length_mean,
                'length_median': length_median      
            }

            # --- add kde profile
            if len(set(_values_inflated)) != 1:
                x,y = self.generate_kde_distribution(_values_inflated)
                _res['length_kde_distribution'] = (x,y)
            else:
                _res['length_kde_distribution'] = None

            # --- add stats
            if stats_per_dtype:
                _res.update( _res_stats[_dtype])

            res.append(_res)
        return res

class GenerateHashvalues(Utilities):

    def __init__(self):
        """
        num_digits is 4 since roaringbitmap can handle only 32 bit integers.
        """
        self.num_digits_partition = 2
        self.num_digits = 4
        self.salt = '4649'
        self.digest_size = 9

    def retrieve_hashvalue_partition(self, hash_digest:bytes):
        """
        """
        end_digits = self.num_digits_partition + self.num_digits
        res = (
            int.from_bytes(hash_digest[:self.num_digits_partition], 'little'),
            int.from_bytes(hash_digest[self.num_digits_partition:end_digits], 'little')
        )
        return res

    def digest_value(self, value:str):
        """
        """
        # --- cast values to bytes type
        if isinstance(value, bytes) == False:
            value = value.encode()

        # --- digest 
        hash_digest = hashlib.blake2b(
            value,
            digest_size = self.digest_size,
            salt = self.salt.encode()
        ).digest()
        return hash_digest

    def generate_hashvalue_parition(self, value:bytes):
        """
        """
        return self.retrieve_hashvalue_partition( self.digest_value(value) )

    def generate_bitmaps(self, values:list, chunk_size:int=10):
        """
        """
        # --- initialization
        bitmaps = {}
        hashvalues_aggregated = []
        buckets = defaultdict(list)

        # --- loop
        for chunk in self.generate_chunks(values, chunk_size):

            _res = defaultdict(list)

            _values, _ = self.exclude_missing_values(chunk)

            # --- 
            for value in _values:
                bucket_id, value_hash = self.generate_hashvalue_parition(value.encode())
                buckets[bucket_id].append(value_hash)          

        # --- roaringbitmaps + aggregated hashvalues
        for key, values_h in buckets.items():
            bitmaps[key] = BitMap(values_h)
            hashvalues_aggregated.extend(values_h)

        # --- roaringbitmap
        bitmap_aggregated = BitMap(hashvalues_aggregated)

        return (bitmaps, bitmap_aggregated)

    def save_pickled_bz2_file(self, object, file_path:str):
        """
        """
        if file_path.lower().endswith('.pbz2') == False:
            file_path = f'{file_path}.pbz2'
        with bz2.BZ2File(file_path, 'wb') as f:
            pickle.dump(object, f)
        return file_path

    def load_pickled_bz2_file(self, file_path:str):
        """
        """
        if file_path.lower().endswith('.pbz2') == False:
            file_path = f'{file_path}.pbz2'
        return pickle.load( bz2.BZ2File(file_path, 'rb') )

class GenerateMetadataString:
    """
    input should be lower letter string
    """
    def __init__(self):
        self.n_min = 2
        self.n_max = 3
        self.pattern_excluded = r'[0-9-]+'

    def generate_ngram(self, word:str, n:int):
        """
        ref: https://stackoverflow.com/a/18658215
        """
        return [word[i:i+n] for i in range(len(word)-n+1)]

    def generate_ngrams(self, word:str):
        """
        """
        # --- initialization
        ngrams_aggregated = []

        # --- generate ngrams
        for n in range(self.n_min, self.n_max+1):
            ngrams_aggregated.extend(self.generate_ngram(word, n))
        return ngrams_aggregated

    def preprocess_string(self, input_string:str):
        """
        standardization, cleanup, etc.
        """
        # --- standardization
        input_string = input_string.lower()

        # --- excluded segments (defined by "pattern_excluded")
        excluded_segment = re.findall(self.pattern_excluded, input_string)

        # --- cleaned input
        _input_string = ''.join( re.split(self.pattern_excluded, input_string) )

        # --- phonetic
        _input_phonetic = fuzzy.nysiis(_input_string).lower()

        res = {
            'input_string': _input_string,
            'input_phonetic': _input_phonetic,
            'excluded_segment': excluded_segment,
            'input_original': input_string
        }
        return res

    def process_string(self, source:dict, remove_duplicated:bool):
        """
        """
        # --- generate ngrams
        if remove_duplicated:
            ngrams_input = list(set(self.generate_ngrams( source['input_string'] )))
            ngrams_phonetic = list(set(self.generate_ngrams( source['input_phonetic'] )))
        else:
            ngrams_input = self.generate_ngrams( source['input_string'] )
            ngrams_phonetic = self.generate_ngrams( source['input_phonetic'] )

        # --- prepare output
        res = copy.deepcopy(source)
        res['ngrams_input'] = ngrams_input
        res['ngrams_phonetic'] = ngrams_phonetic
        return res

    def generate_metadata_string(self, input_string:str, remove_duplicated:bool=True):
        """
        """
        return self.process_string( self.preprocess_string( input_string), remove_duplicated )

class GenerateMetadataColumn:
    """
    """
    def __init__(self):
        # --- True -> keep only aggregated hashvalue
        self.simple_bitmap = True
        self.directory_bitmap = './data'
        self.obj_string = GenerateMetadataString()
        self.obj_value  = GenerateMetadataValues()
        self.obj_bitmap = GenerateHashvalues()

    def process_column_name(self, column_name:str):
        """
        """
        # --- 
        return self.obj_string.generate_metadata_string(column_name)

    def process_values(self, values:list, simple_bitmap:bool=None, directory_bitmap:str=None, save_bitmap:bool=None):
        """
        values should be string
        """
        # --- initialization
        if simple_bitmap == None:
            simple_bitmap = self.simple_bitmap

        if directory_bitmap == None:
            directory_bitmap == self.directory_bitmap
            save_bitmap = False

        # --- process
        res_value  = self.obj_value.generate_metadata(values)
        res_bitmap = self.obj_bitmap.generate_bitmaps(values)

        # --- partitioned or aggregated hashvalues
        if simple_bitmap:
            hashvalues = res_bitmap[1]
        else:
            hashvalues = res_bitmap[0]

        if save_bitmap:
            self.obj_bitmap.save_pickled_bz2_file(hashvalues, directory_bitmap)

        return (res_value, hashvalues)

def generate_metadata_column(column_name:str, values:list, simple_bitmap:bool=None, directory_bitmap:str=None, save_bitmap:bool=None):
    """
    """
    # --- initialization
    obj = GenerateMetadataColumn()

    # --- process column name
    res_column_name = obj.process_column_name(column_name)

    # --- process values
    res_value, res_bitmap = obj.process_values(
        values = values,
        simple_bitmap = simple_bitmap,
        directory_bitmap = directory_bitmap,
        save_bitmap = save_bitmap
    )

    # --- store results
    res = {
        'column_name': res_column_name,
        'values': res_value,
        'bitmap': res_bitmap
    }

    return res
