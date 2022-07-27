#!/bin/bash
read -rp 'Folder path: ' results_path

echo Converting results in $results_path


for filename in $results_path/*; do
    python /usr/share/sumo/tools/xml/xml2csv.py $filename
    echo $filename converted to csv
done

