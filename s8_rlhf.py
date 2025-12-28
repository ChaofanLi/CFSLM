import os
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from datasets import load_dataset

warnings.filterwarnings('ignore')


def init_model():
    device = 'cuda'
    model_name_or_path = "./cfslm-v1-small-multi-sft"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,trust_remote_code=True)
    tokenizer_name_or_path = "./cfslm-v1-small-multi-sft"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path,trust_remote_code=True,use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device)
    return model, tokenizer


if __name__ == '__main__':
    model, tokenizer = init_model()
    training_config = DPOConfig(
        output_dir="./out/cfslm_rlhf",
        per_device_train_batch_size=1,
        remove_unused_columns=False,
        report_to="none",
        save_steps=9999999999999999999999999999,
        learning_rate=4e-5
    )
# 4.29
    dataset_path = './dataset/dpo_dataset/train_data.json'
    train_dataset = load_dataset('json', data_files=dataset_path)

    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_config,
        beta=0.1,
        train_dataset=train_dataset['train'],
        tokenizer=tokenizer,
        max_length=512,
        max_prompt_length=512
    )
    dpo_trainer.train()
    dpo_trainer.save_model("./out/cfslm_rlhf",safe_serialization=False)