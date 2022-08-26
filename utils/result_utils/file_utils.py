from .dotdict import DotDict
from .xml2csv import AttrFinder, CSVWriter
import xml.sax
import os
import glob

def xml2csv_file(file_path):
    '''
    This is a simplified version of the main function from SUMO's xml2dict that does not take arguments.
    '''
    options_dict = {'separator':';', 'quotechar':'', 'xsd':None, 'validation':False, 'split':False, 'output':None, 'source':file_path}
    options = DotDict(options_dict)
    attrFinder = AttrFinder(options.xsd, options.source, options.split)
    handler = CSVWriter(attrFinder, options)
    xml.sax.parse(options.source, handler)

def xml2csv_path(path):
    if os.path.isdir(path):
        files = glob.glob(f'{path}/*.xml')
        result_files = [file for file in files if 'log' not in file]
        for filename in result_files:
            if 'log' not in filename:
                xml2csv_file(filename)
    else:
        xml2csv_file(path)
