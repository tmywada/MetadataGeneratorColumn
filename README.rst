# MetadataGeneratorDatabase
======================================

.. |buildstatus|_
.. |coverage|_
.. |docs|_
.. |packageversion|_

.. docincludebegin

This module generates metadata of RDBMS databases. This generate metadata for each column in the datbase. The metadata comes from the following items:

* schema/table/column names
* values
* distinct values

A combination of the metadata above is unique and enables us to distiguish similar columns. For example, 

Same column names
Even the column name is the same, schema and table should be different. In addition, their values including their distinct values would be different.

Find similar colummns
Embedding



Quick Start (command line)
-----------

.. code-block:: sh

   $ python MetadataGeneratorHiveQueries.py --file_path ./data/sample.sql

`MetadataGeneratorHiveQueires` reads a SQL file and processes queries in the file with `--file_path` option. In this example, the following query is in the SQL file::

   create table schema.table as select col1,col2 from schema_source.table_source;

First, queries without indentation will be re-formatted. The example above is formtted as follows::

   CREATE TABLE schema.table AS
   SELECT col1,
          col2
   FROM schema_source.table_source;

Then, this module extract its metadata. This analyzes each keyword and returns dictionaries. The metadata of example above is::

   {'token': 'create', 'type': 'table', 'value': 'schema.table', 'metadta': {'schema_name': 'schema', 'table_name': 'table', 'table_alias': None}}
   {'token': 'select', 'type': 'column', 'value': 'col1', 'metadata': {'column_name': 'col1'}}
   {'token': 'select', 'type': 'column', 'value': 'col2', 'metadata': {'column_name': 'col2'}}
   {'token': 'FROM', 'type': 'table', 'value': 'schema_source.table_source', 'metadata': {'schema_name': 'schema_source', 'table_name': 'table_source', 'table_alias': None}}


Quick Start (non-command line)
-----------

.. code-block:: python

   >>> from MetadataGeneratorHiveQueries import generate_metadata_from_hive_query

   >>> # define parameters
   >>> file_path = './data/sample.sql'
   >>> idx_query = 0

   >>> # execute function
   >>> result = generate_metadata_from_hive_query(
   >>>     file_path = file_path,
   >>>     idx_query = idx_query
   >>> )

.. code-block:: python
   
   >>> print(result[idx_query]['query'])

This returns the formatted query::

   CREATE TABLE schema.table AS
   SELECT col1,
          col2
   FROM schema_source.table_source;

.. code-block:: python

   >>> for item in result[idx_query]['metadata_query']:
   >>>     print(item)

This returns metadata of the query::

   {'token': 'create', 'type': 'table', 'value': 'schema.table', 'metadta': {'schema_name': 'schema', 'table_name': 'table', 'table_alias': None}}
   {'token': 'select', 'type': 'column', 'value': 'col1', 'metadata': {'column_name': 'col1'}}
   {'token': 'select', 'type': 'column', 'value': 'col2', 'metadata': {'column_name': 'col2'}}
   {'token': 'FROM', 'type': 'table', 'value': 'schema_source.table_source', 'metadata': {'schema_name': 'schema_source', 'table_name': 'table_source', 'table_alias': None}}

More use cases can be found in ``test_MetadataGeneratorHiveQuery.ipynb``.


Technical Notes
-----------

## Schema/table/column names

split characters
phonetic
string-level ngram


## Values

data type in Python
length
length-based statistics including distribution
value-based statistics


## Distinct values

Blake2b hashfunction
Two level of hashvalues (partitioned and non-patitioned)
Compressed Roaringbitmap
