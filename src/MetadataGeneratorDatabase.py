# =================================================================================
#                      Metadata Generator from Database (MGDB)
# =================================================================================
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

    def generate_chunks(self, values:list, chunk_size:int):
        """
        ref: https://stackoverflow.com/a/434328
        """
        return (values[pos:pos + chunk_size] for pos in range(0, len(values), chunk_size))

class GenerateMetadataValues(Utilities):

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
            num_samples = 20

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

    def generate_metadata(self, values:list, stats_per_dtype:bool=True, chunk_size:int=10):
        """
        """
        # --- initialization
        res = []

        # --- aggregation
        _res = self.aggregate_datatype_and_length(values)

        # --- stats using pandas
        df = pd.DataFrame.from_dict(_res, orient = 'columns')

        # ---
        if stats_per_dtype:
            _res_stats = self.calculate_stats_per_datatype(values, chunk_size)

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

            if len(set(_values_inflated)) == 1:
                x = _values_inflated[0]
                y = 1
            else:
                x,y = self.generate_kde_distribution(_values_inflated)

            # --- output
            _res = {
                'datatype': _dtype,
                'count': count_dtype,
                'length_min': length_min,
                'length_max': length_max,
                'length_mean': length_mean,
                'length_median': length_median,
                'kde_distribution': (x,y)            
            }

            # --- add stats
            if stats_per_dtype:
                _res.update( _res_stats[_dtype])

            res.append(_res)
        return res

class GenerateMetadataHashvalues(Utilities):

    def retrieve_hashvalue_partition(self, hash_digest:bytes, num_digits_partition:int=2, num_digits:int=4):
        """
        num_digits is 4 since roaringbitmap can handle only 32 bit integers.
        """
        end_digits = num_digits_partition + num_digits
        res = (
            int.from_bytes(hash_digest[:num_digits_partition], 'little'),
            int.from_bytes(hash_digest[num_digits_partition:end_digits], 'little')
        )
        return res

    def ingest_value(self, value:bytes, digest_dize:int=9, salt:str='4649'):
        """
        """
        hash_digest = hashlib.blake2b(
            value,
            digest_size=digest_dize,
            salt = salt.encode()
        ).digest()
        return hash_digest

    def generate_hashvalue_parition(self, value:bytes):
        """
        """
        return self.retrieve_hashvalue_partition( self.ingest_value(value) )

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

            # --- 
            for value in chunk:
                tmp = self.generate_hashvalue_parition(value.encode())
                buckets[tmp[0]].append(tmp[1])          

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

# class GenerateMetadataColumn:
#     """
#     """

# class GenerateMetadataTable:
#     """
#     """

# class GenerateMeatdataSchema:
#     """
#     """

# class GenerateMetadataDatabase:
#     """
#     """


if __name__ == '__main__':

    # --- initialization
    arg_parser = argparse.ArgumentParser()

    # # --- load parameters
    # arg_parser.add_argument('--file_path', type=str)
    # arg_parser.add_argument('--idx_query', type=int, default = 0)

    # # --- parser arguments
    # options = arg_parser.parse_args()

    # # --- single query
    # res = generate_metadata_from_hive_query(
    #     file_path = options.file_path,
    #     idx_query = options.idx_query
    # )

    # # # --- multple queries   
    # # res = generate_metadata_from_hive_queries(
    # #     file_path = file_path
    # # )

    # print(res[0]['query'])
    # print('---')
    # print(res[0]['metadata_query'])

