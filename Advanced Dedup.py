import csv
import logging
import optparse
import os
import re

import dedupe
from unidecode import unidecode

def preProcess(column):
    column = unidecode(column)
    column = re.sub("  +", " ", column)
    column = re.sub("\n", " ", column)
    column = column.strip().strip('"').strip("'").lower().strip()
    
    if not column:
        column = None
    return column

def readData(filename):
    data_d = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_row = [(k, preProcess(v)) for (k, v) in row.items()]
            row_id = int(row["FUSION_SITE_USE_NUMBER"])
            data_d[row_id] = dict(clean_row)

    return data_d


if __name__ == "__main__":
    optp = optparse.OptionParser()
    optp.add_option(
        "-v",
        "--verbose",
        dest="verbose",
        action="count",
        help="Increase verbosity (specify multiple times for more)",
    )
    (opts, args) = optp.parse_args()
    log_level = logging.WARNING
    if opts.verbose:
        if opts.verbose == 1:
            log_level = logging.INFO
        elif opts.verbose >= 2:
            log_level = logging.DEBUG
    logging.basicConfig(level=log_level)
    
    input_file = "input1.csv"
    output_file = "csv_example_output.csv"
    settings_file = "csv_example_learned_settings"
    training_file = "csv_example_training.json"

    print("importing data ...")
    data_d = readData(input_file)
    
    if os.path.exists(settings_file):
        print("reading from", settings_file)
        with open(settings_file, "rb") as f:
            deduper = dedupe.StaticDedupe(f)
    else:
        fields = [
            dedupe.variables.String("REVISED_CUSTOMER_NAME"),
            dedupe.variables.String("REVISED_ADDRESS_LINE_1"),
            dedupe.variables.Exact("POSTAL_CODE", has_missing=True),
        ]
        
        deduper = dedupe.Dedupe(fields)
        
        if os.path.exists(training_file):
            print("reading labeled examples from ", training_file)
            with open(training_file, "rb") as f:
                deduper.prepare_training(data_d, f)
        else:
            deduper.prepare_training(data_d)
            
            print("starting active labeling...")
            dedupe.console_label(deduper)
            deduper.train()
            
            with open(training_file, "wb") as tf:
                deduper.write_settings(tf)
            
            with open(settings_file, "wb") as sf:
                deduper.write_settings(sf)
            
            print("clustering...")
            clustered_dupes = deduper.partition(data_d, 0.5)

            print("# duplicate sets", len(clustered_dupes))
            
            cluster_membership = {}
            for cluster_id, (records, scores) in enumerate(clustered_dupes):
                for record_id, score in zip(records, scores):
                    cluster_membership[record_id] = {
                        "Cluster ID": cluster_id,
                        "confidence_score": score,
                    }

    with open(output_file, "w", newline='') as f_output, open(input_file, "r") as f_input:
        reader = csv.DictReader(f_input)
        fieldnames = ["Cluster ID", "confidence_score"] + reader.fieldnames

        writer = csv.DictWriter(f_output, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            row_id = int(row["FUSION_SITE_USE_NUMBER"])
            if row_id in cluster_membership:
                row.update(cluster_membership[row_id])
            writer.writerow(row)
