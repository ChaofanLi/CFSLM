import os
import json
import random
from config.tokenizer_config import tokenizer_config
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)

random.seed(tokenizer_config["random_seed"]) # 设置随机种子

class Train_Cfslm_Tokenizer():
    def __init__(self,data_path,tokenizer_config):
        self.data_path=data_path
        self.config=tokenizer_config # 加载配置参数
        
    def get_data(self):
        with open(self.data_path,"r",encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                yield data['text']
                
    def train(self):
        tokenizer=Tokenizer(models.BPE()) # 选择BPE分词器
        # 初始化词表
        tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(
            add_prefix_space=self.config["add_prefix_space"])
        #初始化训练器
        trainer = trainers.BpeTrainer(
        vocab_size=self.config["vocab_size"],
        special_tokens=self.config["special_tokens"],
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()) # 初始字符集
        #加载语料
        corpus=self.get_data()
        # 训练分词器
        tokenizer.train_from_iterator(corpus, trainer=trainer)
        # 解码器
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
        
        print("tokenizer训练保存完成！")
        
        
        
if __name__=="__main__":
    data_path="dataset/tokenizer_train.jsonl"
    tcft=Train_Cfslm_Tokenizer(data_path,tokenizer_config)
    tcft.train()