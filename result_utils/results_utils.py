from utils import dotdict
from utils import xml2csv
import xml.sax
import os
import glob

def xml2csv_file(file_path):
    '''
    This is a simplified version of the main function from SUMO's xml2dict that does not take arguments.
    '''
    options_dict = {'separator':';', 'quotechar':'', 'xsd':None, 'validation':False, 'split':False, 'output':None, 'source':file_path}
    options = dotdict.dotdict(options_dict)
    attrFinder = xml2csv.AttrFinder(options.xsd, options.source, options.split)
    handler = xml2csv.CSVWriter(attrFinder, options)
    xml.sax.parse(options.source, handler)

def xml2csv_path(path):
    if os.path.isdir(path):
        for filename in glob.glob(f'{path}/*.xml'):
            xml2csv_file(filename)
    else:
        xml2csv_file(path)