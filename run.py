import time
import json, os
import torch
from my_tokenizer import Tokenizer
from my_llama import MyLlamaForCausalLM, warmup

model_path = "../../llama/TheBloke-Llama-2-7B-Chat-fp16"
experiment_config = {
    "model": 'llama-2-7b',
    "batch_size": 10,
    "prompt_len": 64,
    "max_gen_len": 64,
    "weight_offload_layer_num": 24,
    "KV_cache_offload_layer_num": 24,
    "pin_memory": True,
    "async_offload": True,  #pin_memory and async_offload 需要同時為True才能prefetch next layer wieghts
}
prompt_template = ["""Emma loved to read books. Every Saturday, she went to the old library in town. The library was big, with wooden floors that creaked and tall shelves full of dusty books.
                    One rainy afternoon, Emma found a strange book. It was very old, with a golden key printed on the cover. When she pulled the book from the shelf, she heard a click. Suddenly, part of the wall moved — revealing a hidden door!
                    Emma looked around. No one else was in the library. She took a deep breath, opened the door, and stepped inside.
                    Inside, Emma found a glowing blue room. Floating books zipped through the air. A strange voice boomed from nowhere.
                    "You found us!" the voice said. "We need your help!"
                    Suddenly, the floor opened up beneath her. Emma screamed as she fell down, down into darkness. She landed on a soft cushion in a massive underground library, ten times bigger than the one above.
                    "The Shadow Eaters are coming!" A giant book flew toward her. Its pages glowed. "Take this key and run!"
                    Black smoke seeped through the walls. Red eyes glowed in the darkness. The shadows were eating everything they touched!
                    Emma grabbed the key and ran between tall shelves. Books screamed as the shadows ate them. At the end of the room stood a huge mirror.
                    "Jump through!" yelled the book. "Pick any world! Hurry!"
                    The shadows reached for her feet. Emma didn't think - she just jumped. The mirror's surface rippled like water as she crashed through it.
                    When she opened her eyes, she was somewhere else entirely...
                    …Emma blinked. She was standing in a sky made of clock faces. Gears the size of houses floated around her, ticking loudly. Time itself was alive here — some clocks ran backward, others melted like wax. A cat with a monocle and ten tails appeared beside her, balancing on a floating minute hand.
                    "You're late!" it hissed, then sneezed out a swarm of tiny alarm clocks that flew away, ringing madly.
                    Emma clutched the golden key. It pulsed in her hand, growing warm. The sky suddenly cracked open like glass, and a rain of letters — actual letters, like A, B, Z — poured down, sticking to her clothes and forming strange words on her arms: DOOR, DANGER, DECIDE.
                    Then, the clocks began to scream. Not ring — scream. One by one, they turned toward Emma, their hands spinning wildly, pointing at her. The cat turned upside-down and whispered, “They’ve found you. The Time Snatchers are never far behind.”
                    From behind a colossal cuckoo clock, shapes began to emerge — faceless, tall figures with hourglasses for heads. Sand spilled from their necks as they moved in jerks, reaching for Emma with fingers made of second hands.
                    Emma ran.
                    The floor flipped sideways, becoming a spiral staircase of pages from books she hadn’t written yet. As she ran, the words changed beneath her feet: “RUN,” “FASTER,” “ALMOST,” “NO.”
                    Suddenly, a giant eye opened in the sky. It blinked once, and everything paused — even the Time Snatchers froze. A booming voice whispered, “Turn the key backward, or lose everything.”
                    Heart pounding, Emma turned the key in reverse. Instantly, the world began to unwind like a broken film reel. Backward screams, reversed clocks, falling up…
                    Then silence.
                    She found herself in a new place. A forest made of giant books. The trees whispered her name. And the shadows? They were already here."""]

prompts = prompt_template * experiment_config["batch_size"]
tokenizer = Tokenizer(model_path=model_path)
prompt_tokens = [tokenizer.encode(x, bos=True, eos=False, truncation_len=experiment_config["prompt_len"]) for x in prompts]
print("input dimensions:", f"batch_size={len(prompt_tokens)}, prompt_len={len(prompt_tokens[0])}")

print(f"tokenizer vocab size: {tokenizer.n_words}")
print(f"tokenizer pad id: {tokenizer.pad_id}")
print(f"tokenizer bos id: {tokenizer.bos_id}")
print(f"tokenizer eos id: {tokenizer.eos_id}")

model: MyLlamaForCausalLM = MyLlamaForCausalLM.from_pretrained(
    model_path = model_path, 
    max_seq_len = experiment_config["prompt_len"] + experiment_config["max_gen_len"], 
    max_batch_size = experiment_config["batch_size"], 
    tokenizer=tokenizer, 
    weight_offload_layer_num = experiment_config["weight_offload_layer_num"], 
    KV_cache_offload_layer_num = experiment_config["KV_cache_offload_layer_num"],
    pin_memory = experiment_config["pin_memory"], 
    async_offload = experiment_config["async_offload"]
)
warmup(model, tokenizer, prompt_len = experiment_config["prompt_len"], decode_steps = 10, warm_iters = 2) 

start_time = time.time()
outputs, _  = model.generate(prompt_tokens, max_gen_len = experiment_config["max_gen_len"], temperature = 0.6, top_p = 0.9, force_max_len=True)
end_time = time.time()
elapsed_time = end_time - start_time

prompt = tokenizer.decode(prompt_tokens[0])
print(f"prompt:\n {prompt}\n\n")

outputs = [tokenizer.decode(x) for x in outputs]
for i in range(experiment_config["batch_size"]):
    print("ID : ", i)
    print(outputs[i])
    print("--------------------------------------------------------")
    
decode_latency = model.model.total_forward_time - model.model.prefill_latency #ms
prefill_throughput = (experiment_config["batch_size"] * experiment_config["prompt_len"])/model.model.prefill_latency*1000
decode_throughput = (experiment_config["batch_size"] * experiment_config["max_gen_len"])/decode_latency*1000

print("model: llama-2-7b")
print(f"max GPU memory: {torch.cuda.max_memory_allocated(0) / (1024**3):.2f} GB")
print(f"batch_size: {experiment_config['batch_size']}")
print(f"prompt len: {experiment_config['prompt_len']}")
print(f"gen len: {experiment_config['max_gen_len']}")
print(f"KV cache size: {model.model.n_layers*2*experiment_config['batch_size']*(experiment_config['prompt_len'] + experiment_config['max_gen_len'])*model.model.params.dim*2/1024/1024:.3f} MB")
print(f"prefill_throughput: {prefill_throughput:.3f} token/s")
print(f"decode_throughput: {decode_throughput:.3f} token/s")
print(f"generate time: {elapsed_time:.3f} seconds")
print(f"total_weight_transfer_time: {model.model.total_weight_transfer_time/1000:.3f} seconds")
print(f"total_forward_time: {model.model.total_forward_time/1000:.3f} seconds")
print(f"total_layer_time: {model.model.total_layer_time/1000:.3f} seconds")
print(f"total_WqWkWvWo_time: {model.model.total_WqWkWvWo_time/1000:.3f} seconds")
print(f"total_attention_score_time: {model.model.total_attention_score_time/1000:.3f} seconds")
print(f"total_Attention_time: {model.model.total_Attention_time/1000:.3f} seconds")
print(f"total_FeedForward_time: {model.model.total_FeedForward_time/1000:.3f} seconds")
print(f"total_TransformerBlock_time: {model.model.total_TransformerBlock_time/1000:.3f} seconds")


performance_summary = {
    "prefill_throughput": prefill_throughput,
    "decode_throughput": decode_throughput,
    "max_gpu_memory_gb": torch.cuda.max_memory_allocated(0) / (1024**3),
    "total_execution_time_sec": model.model.total_forward_time / 1000,
    "prefill_latency_ms": model.model.prefill_latency,
    "decode_latency_ms": model.model.total_forward_time - model.model.prefill_latency,
    "kv_cache_size_mb": model.model.n_layers*2*experiment_config["batch_size"]*(experiment_config["prompt_len"]+experiment_config["max_gen_len"])*model.model.params.dim*2/1024/1024,
}
model_timing_summary = {
    "weight_transfer_time_sec": model.model.total_weight_transfer_time / 1000,
    "forward_time_sec": model.model.total_forward_time / 1000,
    "layer_time_sec": model.model.total_layer_time / 1000,
    "wqkvwo_time_sec": model.model.total_WqWkWvWo_time / 1000,
    "attention_score_time_sec": model.model.total_attention_score_time / 1000,
    "attention_time_sec": model.model.total_Attention_time / 1000,
    "feedforward_time_sec": model.model.total_FeedForward_time / 1000,
    "transformer_block_time_sec": model.model.total_TransformerBlock_time / 1000,
}
per_layer_information = []
for i in range(len(model.model.avg_layer_time)):
    per_layer_information.append({
        "forward_index": i,
        "avg_layer_time": model.model.avg_layer_time[i],
        "avg_per_layer_weight_transfer_time": model.model.avg_per_layer_weight_transfer_time[i],
        "avg_per_layer_TransformerBlock_time": model.model.avg_per_layer_TransformerBlock_time[i],
        "avg_per_layer_Attention_time": model.model.avg_per_layer_Attention_time[i],
        "avg_per_layer_WqWkWvWo_time": model.model.avg_per_layer_WqWkWvWo_time[i],
        "avg_per_layer_attention_score_time": model.model.avg_per_layer_attention_score_time[i],
        "avg_per_layer_FeedForward_time": model.model.avg_per_layer_FeedForward_time[i],
    })
result = {
    "experiment_config": experiment_config,
    "performance_summary": performance_summary,
    "model_timing_summary": model_timing_summary,
    "per_layer_information": per_layer_information
}
with open("test_result.json", "w") as f:
    json.dump(result, f, indent=4)
    f.flush()
    os.fsync(f.fileno())