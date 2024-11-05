import random
from tqdm import tqdm
from transformers import AutoTokenizer
from Config import tokenizer_config
import json
from datasets import load_dataset
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
import os

random.seed(27)

class Train_CF_Tokenizer():
    def __init__(self,data_path,tokenizer_config):
        self.data_path=data_path
        self.config=tokenizer_config
        
    def get_data(self):
        with open(self.data_path,"r",encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                yield data['text']
                
    def train(self):
        # tokenizer初始化
        tokenizer=Tokenizer(models.BPE())
        tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False) # add_prefix_space:是否在每个文本开头添加空格
        
        trainer = trainers.BpeTrainer(
        vocab_size=self.config["vocab_size"],
        special_tokens=self.config["special_tokens"],  # 三个特殊token
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        
        # 加载训练数据
        corpus=self.get_data()
        
        # 训练tokenizer
        tokenizer.train_from_iterator(corpus, trainer=trainer)
        
        # 设置解码器
        tokenizer.decoder = decoders.ByteLevel()
        
        #检查三个特殊token索引
        assert tokenizer.token_to_id("<unk>") == 0
        assert tokenizer.token_to_id("<s>") == 1
        assert tokenizer.token_to_id("</s>") == 2
        
        # 保存tokenizer
        tokenizer_dir=self.config["tokenizer_dir"]
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
        tokenizer.model.save(tokenizer_dir)
        
        with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
            json.dump(self.config, config_file, ensure_ascii=False, indent=4)
        
        print("Tokenizer training completed and saved.")
        
def format_chat_messages(messages):  
    """  
    将对话消息格式化为一个字符串。  
    这里假设每个消息都是一个字典，包含'role'和'content'键。  
    """  
    formatted_messages = []  
    for message in messages:  
        role = message['role'].capitalize()  # 将角色首字母大写  
        content = message['content']  
        formatted_message = f"{role}: {content}"  
        formatted_messages.append(formatted_message)  
    return "\n".join(formatted_messages)  
  
def eval_tokenizer(tokenizer_dir):  
    # 加载预训练的tokenizer  
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)  
  
    messages = [  
        {"role": "system", "content": "你是一个问答小助手，请正确回答用户问题！"},  
        {"role": "user", "content": '○□△'},  
        {"role": "assistant", "content": '456'},  
        {"role": "user", "content": 'whats your name?'},  
        {"role": "assistant", "content": '阿里嘎多'}  
    ]  
    # 使用自定义的格式化函数  
    new_prompt = format_chat_messages(messages)  
    print(new_prompt)  
    
    # 获取词汇表大小
    print('tokenizer词表大小：', tokenizer.vocab_size)  
  
    new_prompt = '你好，我是超级烦人的AI助手，你可以叫我超凡小模型，我的英文名叫做cfslm'  
    print(new_prompt)  
    
    # 使用分词器对新的提示进行编码  
    model_inputs = tokenizer(new_prompt, return_tensors='pt')  # 'pt'表示PyTorch张量  
    print(model_inputs)  
    print('长度：', len(model_inputs['input_ids'][0]))
  
    input_ids_ = model_inputs['input_ids'][0]
    response = tokenizer.decode(input_ids_) 
     
    print(response)
        
        
        
if __name__=="__main__":
    # data_path="dataset/tokenizer_train.jsonl"
    # tcft=Train_CF_Tokenizer(data_path,tokenizer_config)
    # tcft.train()
    eval_tokenizer(tokenizer_config["tokenizer_dir"])