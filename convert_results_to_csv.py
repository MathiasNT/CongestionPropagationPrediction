from utils.results_utils import xml2csv_file
import argparse
import os

def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--path", type=str, help='Path to folder or xml file. If folder all files in folder will be converted')
    args = arg_parser.parse_args()
    return args


def main():
    args = get_args()

    if os.path.isdir(args.path):
        for filename in os.listdir():
            xml2csv_file(f'{args.path}/filename')
    else:
        xml2csv_file(args.path)


if __name__ == "__main__":
    main()
