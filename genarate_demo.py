import os
import torch

from transformers import T5Tokenizer, GPT2LMHeadModel
import deepspeed

seed = 42
torch.manual_seed(seed)

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

ds_config = {
    "bf16": {
        "enabled": True,
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "allgather_partitions": True,
		"allgather_bucket_size": 5e7,
	    "reduce_scatter": True,
	    "reduce_bucket_size": 5e7,
        "sub_group_size": 1e9,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "offload_optimizer": {
            "device": "cpu"
        },
    },
    "steps_per_print": 2000,
    "train_batch_size": world_size,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False
}


model_name = "Path-to-BBT-model"

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

#model.half()

print("\n\n", "model loading.", "\n\n")
deepspeed.init_distributed()
engine = deepspeed.initialize(
    model=model, 
    config_params=ds_config,
    optimizer=None
)[0]
print("\n\n", "model loaded.", "\n\n")

#engine.half()
engine.eval()

def test(input_text, top_p=0.9, top_k=50):
    text = input_text
    input_length = 256
    input_idss = []
    attention_masks = []

    results = []
    with torch.no_grad():
        outputs = engine.module.generate(
            tokenizer(text, add_special_tokens=False, return_tensors="pt",)["input_ids"].to("cuda"),
            num_beams=4,
            repetition_penalty=1.1,
            max_length=640,
            eos_token_id=tokenizer.eos_token_id,
            top_p=top_p,
            top_k=top_k,
            #temperature=0.9,
            #no_repeat_ngram_size=5
            )
        print(outputs)
        ret = tokenizer.decode(outputs[0], skip_special_tokens=True)
        ret = ret.replace('[tab]', '    ').replace('[newline]', '\n')
        return ret

