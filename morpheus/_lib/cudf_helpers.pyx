from cudf import DataFrame

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.table.table cimport table
from cudf._lib.table cimport Table, table_view_from_table
from cudf._lib.cpp.io.types cimport table_with_metadata, table_metadata
from cudf._lib.utils cimport data_from_table_view, data_from_unique_ptr

cdef extern from "morpheus/table_info.hpp" namespace "morpheus" nogil:
   cdef cppclass TableInfo:
      TableInfo()
      TableInfo(table_view view, table_metadata meta)

      table_view table_view
      table_metadata metadata

cdef public api:
   Column make_column_from_view(column_view view):
      return Column.from_column_view(view, None)

   column_view make_view_from_column(Column col):
      return col.view()

   Table make_table_from_view_and_meta(table_view view, table_metadata meta):

      column_names = [x.decode() for x in meta.column_names]

      data, index = data_from_table_view(view, None, column_names=column_names)

      return DataFrame._from_data(data, index)

   TableInfo make_table_info_from_table(Table table):

      cdef table_view input_table_view = table_view_from_table(
         table
      )
      cdef table_metadata metadata_ = table_metadata()

      all_names = table._column_names

      if len(all_names) > 0:
         metadata_.column_names.reserve(len(all_names))
         if len(all_names) == 1:
               if all_names[0] in (None, ''):
                  metadata_.column_names.push_back('""'.encode())
               else:
                  metadata_.column_names.push_back(
                     str(all_names[0]).encode()
                  )
         else:
               for idx, col_name in enumerate(all_names):
                  if col_name is None:
                     metadata_.column_names.push_back(''.encode())
                  else:
                     metadata_.column_names.push_back(
                           str(col_name).encode()
                     )
      return TableInfo(input_table_view, metadata_)
