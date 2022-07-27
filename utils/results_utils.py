from utils import dotdict
from utils import xml2csv
import xml.sax

def xml2csv_file(path):
    '''
    This is a simplified version of the main function from SUMO's xml2dict that does not take arguments.
    '''
    options_dict = {'separator':';', 'quotechar':'', 'xsd':None, 'validation':False, 'split':False, 'output':None, 'source':path}
    options = dotdict.dotdict(options_dict)
    attrFinder = xml2csv.AttrFinder(options.xsd, options.source, options.split)
    handler = xml2csv.CSVWriter(attrFinder, options)
    xml.sax.parse(options.source, handler)