# from tqdm import tqdm
# from transformers import AutoTokenizer

# # 加载训练好的tokenizer
# tokenizer = AutoTokenizer.from_pretrained('./cfslm_tokenizer')

# def count_tokens_in_text_dataset(data_path, tokenizer):
#     total_tokens = 0
#     with open(data_path, "r", encoding="utf-8") as f:
#         for line in tqdm(f):
#             # 每行就是一条文本
#             text = line.strip()  # 去除行首尾的空格
#             encoding = tokenizer.encode(text)
#             total_tokens += len(encoding)
#     return total_tokens

# if __name__=="__main__":
#     data_path = "./dataset/pretrain_data.csv"
#     total_tokens = count_tokens_in_text_dataset(data_path, tokenizer)
#     print(f"Total tokens in dataset: {total_tokens}")



import csv

# 文件路径
file_path = './dataset/sft_data_single.csv'

# 读取并打印前5行
num_lines = 5
with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile)
    
    # 打印前几行
    for i, row in enumerate(csvreader):
        if i < num_lines:
            print(row)
        else:
            break
