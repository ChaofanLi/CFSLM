import csv
import itertools
import re
import json
import jsonlines
import psutil
import ujson
import numpy as np
import pandas as pd
from config.data_processing_config import pretrain_data_config
from transformers import AutoTokenizer
from datasets import load_dataset

bos_token = pretrain_data_config["bos_token"]
eos_token = pretrain_data_config["eos_token"]


def pretrain_data_processing(chunk_size=pretrain_data_config["chunk_size"]):
    chunk_idx = 0
    with jsonlines.open(pretrain_data_config["original_copus_path"]) as reader:
        with open(pretrain_data_config["pretrain_data_path"], 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['text'])
            while True:
                chunk = list(itertools.islice(reader, chunk_size))
                if not chunk:
                    break
                for idx, obj in enumerate(chunk):
                    try:
                        content = obj.get('text', '')
                        if len(content) > 512:
                            content=content[:512]
                        writer.writerow([content])
                    except UnicodeDecodeError as e:
                        print(f"跳过无效行{chunk_idx * chunk_size + idx + 1}: {e}")
                        continue
                chunk_idx += 1
                print('chunk:', ((chunk_idx - 1) * chunk_size, chunk_idx * chunk_size), '处理完毕！')


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('./cfslm_tokenizer', use_fast=False)
    print('tokenizer词表大小：', len(tokenizer))
    pretrain_data_processing()