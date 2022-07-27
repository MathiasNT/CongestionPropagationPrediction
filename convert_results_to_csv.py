from utils.results_utils import xml2csv_file, xml2csv_path
import argparse
import os

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--path", type=str, help='Path to folder or xml file. If folder all files in folder will be converted')
    args = arg_parser.parse_args()
    xml2csv_path(args.path)
