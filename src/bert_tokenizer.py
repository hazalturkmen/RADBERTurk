import json
from tqdm import tqdm
import argparse
import pandas as pd
from transformers import (
    AutoTokenizer,
)


def get_report_from_xlsx(path):
    df = pd.read_excel(path)
    rep = df['rapor']
    rep = rep.str.strip()
    rep = rep.replace('\n', ' ', regex=True)
    rep = rep.replace('\s+', ' ', regex=True)
    rep = rep.str.strip()
    return rep


def tokenize(report, tokenizer):
    new_report = []
    print("\nTokenizing report text. All reports are cut off at 512 tokens.")
    for i in tqdm(range(report.shape[0])):
        tokenized_imp = tokenizer.tokenize(report.iloc[i])
        tokenized_imp_plus = ['[CLS]'] + tokenized_imp + ['[SEP]']
        if tokenized_imp_plus:
            res = tokenizer.convert_tokens_to_ids(tokenized_imp_plus)
            if len(res) > 512:
                res = res[:511] + [tokenizer.sep_token_id]
            new_report.append(res)
        else:
            new_report.append([tokenizer.cls_token_id, tokenizer.sep_token_id])
    return new_report


def load_list(path):
    with open(path, 'r') as filehandle:
        report = json.load(filehandle)
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tokenize radiology report and save as a list.')
    parser.add_argument('-d', '--data', type=str, nargs='?', required=True,
                            help='path to xlxs containing reports. The reports should be \
                            under the \"rapor\" column')
    parser.add_argument('-o', '--output_path', type=str, nargs='?', required=True,
                            help='path to intended output file')
    args = parser.parse_args()
    csv_path = args.data
    out_path = args.output_path

    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    reports = get_report_from_xlsx(csv_path)
    new_reports = tokenize(reports, tokenizer)
    with open(out_path, 'w') as filehandle:
        json.dump(new_reports, filehandle)
