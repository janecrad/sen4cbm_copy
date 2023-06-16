# Actual Steps
# - import CSV file into a pandas DF
# - clean the table name and remove all extra symbols, spaces and capital letters
# - clean the column headers and remove all extra symbols, spaces and capital letters
# - write the create table SQL statement and pass to DB
# - import the data into the DB

import os
import pandas as pd
from database.db import conn_cur, tables, execute_sql, close_conn
from shutil import copy

def find_files_in_dir(dir=None, extension='.csv'):
    # get names of files by type
    # if directory is given or use cwd as default

    if dir is None:
        folder = os.getcwd()
    else:
        folder = dir

    file_list = []

    for file in os.listdir(folder):
        if file.endswith(extension):
            file_list.append(file)

    return file_list


def configure_dataset_directory(file_list, dataset_dir, dir=None):
    # make dataset dir to process files
    if dir is None:
        dir = os.getcwd()
    else:
        dir = dir
    if not os.path.isdir(dataset_dir):
        mkdir = 'mkdir {}'.format(dataset_dir)
        os.system(mkdir)
    # move files to dataset dir
    out_list = list()
    for file in file_list:
        infile = os.path.join(dir, file)
        outfile = os.path.join(dataset_dir,file)
        if not os.path.isfile(outfile):
            copy(infile, dataset_dir)
        out_list.append(outfile)

    return out_list


def create_df(data_path, file_list):


    # loop through the files and create the dataframe
    df_dict = {}
    for file in file_list:

        try:
            df_dict[file] = pd.read_csv(os.path.join(data_path, file))
        except UnicodeDecodeError:
            df_dict[file] = pd.read_csv(os.path.join(data_path, file), encoding="ISO-8859-1")  # if utf-8 encoding error
        print(file)

    return df_dict


def clean_tbl_name(filename):
    # rename csv, force lower case, no spaces, no dashes
    clean_tbl_name = filename.lower().replace(" ", "").replace("-", "_").replace(r"/", "_").replace("\\", "_").replace(
        "$", "").replace("%", "")

    tbl_name = '{0}'.format(clean_tbl_name.split('.')[0])

    return tbl_name


def clean_colname(dataframe):
    # print(type(dataframe))
    # force column names to be lower case, no spaces, no dashes
    dataframe.columns = [
        x.lower().replace(" ", "_").replace("-", "_").replace(r"/", "_").replace("\\", "_").replace(".", "_").replace(
            "$", "").replace("%", "").replace(":", "")  for x in dataframe.columns]

    # processing data - TODO consider moving out of functions if needed elsewhere
    replacements = {
        'timedelta64[ns]': 'varchar',
        'object': 'varchar',
        'float64': 'float',
        'int64': 'int',
        'datetime64': 'timestamp'
    }

    col_str = ", ".join(
        "{} {}".format(n, d) for (n, d) in zip(dataframe.columns, dataframe.dtypes.replace(replacements)))

    return col_str, dataframe


def upload_to_db(file, tbl_name=None, dataframe_columns=None, db='main', del_input=False):
    """ Function to upload a csv into a Database using psycopg2.
        Inputs required:

        file: string - csv file string containing full path if not present in current working directory

        Optional Inputs:
        tbl_name:           string          - name of the table to upload the csv. Table is created if it does not exist
                                              default is input filename from variable file
        dataframe_columns:  Index or list   - columns names to upload. default is all columns of file in input
        db:                 string          - database name as reported in configuration file. default main
        del_input:          boolean         - define whether to delete input file upon completion.
    """
    if tbl_name is None:
        tbl_name = clean_tbl_name(os.path.basename(file))
    else:
        tbl_name = clean_tbl_name(tbl_name)
    dataframe = create_df(os.path.dirname(file), [os.path.basename(file)])[os.path.basename(file)]
    if dataframe_columns is None:
        dataframe_columns = dataframe.columns
    else:
        dataframe = dataframe[dataframe_columns]
    col_str, dataframe = clean_colname(dataframe)

    try:
        connection, cursor = conn_cur(db)
        tables_list = tables(db)

        print('opened database successfully')
        # create table if it does not exist
        if not (tbl_name in tables_list):
            execute_sql("create table %s (%s);" % (tbl_name, col_str))
            print('{0} was created successfully'.format(tbl_name))
        else:
            print('{0} exists already. Skipping'.format(tbl_name))

        # insert values to table

        # save df to csv with renamed columns
        dataframe.to_csv(file, header=dataframe_columns, index=False, encoding='utf-8')

        # open the csv file, save it as an object
        #my_file = open(file)
        with open(file) as my_file:
            print('file opened in memory')

            # upload to db
            SQL_STATEMENT = """
                COPY %s FROM STDIN WITH
                    CSV
                    HEADER
                    DELIMITER AS ','
                """
            cursor.copy_expert(sql=SQL_STATEMENT % tbl_name, file=my_file)
            print('file copied to db')

            cursor.execute("grant select on table %s to public" % tbl_name)
            connection.commit()
            cursor.close()
            connection.close()
            print('table {0} imported to db completed'.format(tbl_name))

        if del_input:
            os.remove(file)
    except Exception:
        close_conn(connection)
        return False
    return True


# main
if __name__ == '__main__':

    # from csv_import_functions import *

    # settings
    # changed to full path - so you can copy files wherever
    dataset_dir = '/mnt/data/output/datasets'

    # db settings - from config file
    #host = 'localhost'
    #dbname = 'postgres '
    #user = 'postgres'
    #password = 'mydiaspassword'
    db = 'main'
    # configure environment and create main df
    dir = '/mnt/data/output'
    file_list = find_files_in_dir(dir)
    configure_dataset_directory(file_list, dataset_dir, dir=dir)
    df = create_df(dataset_dir, file_list)

    for k in file_list:
        # call dataframe
        dataframe = df[k]

        # clean table name
        tbl_name = clean_tbl_name(k)

        # clean column names
        #col_str, dataframe = clean_colname(dataframe)
        #print(df[k])
        file = os.path.join(dataset_dir, k)

        # upload data to db
        upload_to_db(file=file,
                     tbl_name=tbl_name,
                     dataframe_columns=dataframe.columns,
                     db=db)
