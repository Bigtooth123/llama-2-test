import math
import torch
import torch.nn.functional as F
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
from torch import nn
from accelerate import init_empty_weights

@dataclass
class ModelArgs:
    dim: int = -1
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    intermediate_size: int = 11008
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # torch.view_as_complex([x,y]) -> (x+yj)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(
        self, args: ModelArgs, 
        layer_id,
        KV_cache_offload_layer_ids,
        async_offload: bool,
        store_KV_cache_stream: torch.cuda.Stream = None
    ):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads = args.n_heads
        self.n_kv_heads = self.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        
        self.KV_cache_offload_layer_ids = KV_cache_offload_layer_ids
        self.async_offload = async_offload
        self.store_KV_cache_stream = store_KV_cache_stream

        self.layer_id = layer_id

        self.wq = torch.nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = torch.nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wv = torch.nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wo = torch.nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        cache_k,
        cache_v,
        time_dict,
        temp_key_value
    ):
        #record computation time
        computation_start_event = torch.cuda.Event(enable_timing=True)  
        computation_end_event = torch.cuda.Event(enable_timing=True)
        WqWkWv_start_event = torch.cuda.Event(enable_timing=True)  
        WqWkWv_end_event = torch.cuda.Event(enable_timing=True)
        load_KV_cache_start_event = torch.cuda.Event(enable_timing=True)  
        load_KV_cache_end_event = torch.cuda.Event(enable_timing=True)
        attention_score_start_event = torch.cuda.Event(enable_timing=True)  
        attention_score_end_event = torch.cuda.Event(enable_timing=True)
        Wo_start_event = torch.cuda.Event(enable_timing=True)  
        Wo_end_event = torch.cuda.Event(enable_timing=True)

        computation_start_event.record()        

        WqWkWv_start_event.record()
        
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        #record computation time
        WqWkWv_end_event.record()       

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = None
        values = None
        dst_indices = (slice(0, bsz), slice(start_pos, start_pos + seqlen), slice(0, self.n_kv_heads), slice(0, self.head_dim))

        load_KV_cache_start_event.record()

        #offload and compute attention in cpu
        if self.layer_id in self.KV_cache_offload_layer_ids:  
            if mask != None:  #prefill stage -> compute in GPU
                self.store_KV_cache_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.store_KV_cache_stream):
                    #store cache back to cpu
                    cache_k[dst_indices].copy_(xk, non_blocking=self.async_offload)
                    cache_v[dst_indices].copy_(xv, non_blocking=self.async_offload)
                keys = xk
                values = xv
            elif mask == None:   #decode stage -> compute in cpu
                #store cache back to cpu
                cache_k[dst_indices].copy_(xk)
                cache_v[dst_indices].copy_(xv)
                keys = cache_k[:bsz, : start_pos + seqlen]
                values = cache_v[:bsz, : start_pos + seqlen]
                xq = xq.to("cpu") 
        #no offload
        else:     
            cache_k[dst_indices].copy_(xk)
            cache_v[dst_indices].copy_(xv)
            keys = cache_k[:bsz, : start_pos + seqlen]
            values = cache_v[:bsz, : start_pos + seqlen]

        load_KV_cache_end_event.record()

        #record computation time
        attention_score_start_event.record()
        
        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2) # (bs, n_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2) # (bs, n_heads, cache_len + seqlen, head_dim)

        #if this layer is offloading KV cache, we need to copy to make key and value contiguous
        if self.layer_id in self.KV_cache_offload_layer_ids and mask == None:
            temp_key_value[0].copy_(keys)  #copy keys to key_value buffer
            temp_key_value[1].copy_(values)  #copy values to key_value buffer
            keys = temp_key_value[0]
            values = temp_key_value[1]

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        #record computation time
        attention_score_end_event.record()
        

        if self.layer_id in self.KV_cache_offload_layer_ids and mask == None:   #decode stage
            output_on_cuda = torch.empty(*output.shape, device = "cuda")
            for i in range(bsz): 
                output_on_cuda[i].copy_(output[i])  #back to GPU
            output = output_on_cuda

        #record computation time
        Wo_start_event.record()
        
        output = self.wo(output)
        
        #record computation time
        Wo_end_event.record()

        #record computation time
        computation_end_event.record()
        
        if self.layer_id in self.KV_cache_offload_layer_ids and mask != None:
            self.store_KV_cache_stream.synchronize()  #資料同步

        torch.cuda.current_stream().synchronize()
        
        time_dict["Attention_time"] = computation_start_event.elapsed_time(computation_end_event)
        time_dict["WqWkWvWo_time"] = WqWkWv_start_event.elapsed_time(WqWkWv_end_event)
        time_dict["load_KV_cache_time"] = load_KV_cache_start_event.elapsed_time(load_KV_cache_end_event)
        time_dict["attention_score_time"] = attention_score_start_event.elapsed_time(attention_score_end_event)
        time_dict["WqWkWvWo_time"] = time_dict["WqWkWvWo_time"] + Wo_start_event.elapsed_time(Wo_end_event)

        return output


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_size: int,
    ):
        super().__init__()

        self.w1 = torch.nn.Linear(
            dim, intermediate_size, bias=False
        )
        self.w2 = torch.nn.Linear(
            intermediate_size, dim, bias=False
        )
        self.w3 = torch.nn.Linear(
            dim, intermediate_size, bias=False
        )

    def forward(self, x, time_dict):
        #record computation time
        computation_start_event = torch.cuda.Event(enable_timing=True)
        computation_end_event = torch.cuda.Event(enable_timing=True)
        computation_start_event.record()

        out = self.w2(F.silu(self.w1(x)) * self.w3(x))
    
        #record computation time
        computation_end_event.record()
        torch.cuda.current_stream().synchronize()
        elapsed_ms = computation_start_event.elapsed_time(computation_end_event)
        time_dict["FeedForward_time"] = elapsed_ms

        return out


class TransformerBlock(nn.Module):
    def __init__(
        self, layer_id: int, args: ModelArgs, 
        KV_cache_offload_layer_ids,
        async_offload: bool,
        store_KV_cache_stream: torch.cuda.Stream = None,
    ):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention( 
            args, 
            layer_id=layer_id,
            KV_cache_offload_layer_ids=KV_cache_offload_layer_ids, 
            async_offload=async_offload, 
            store_KV_cache_stream=store_KV_cache_stream
        )
        self.feed_forward = FeedForward(
            dim=args.dim,
            intermediate_size=args.intermediate_size
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        cache_k,
        cache_v,
        time_dict,
        temp_key_value
    ):
        #record computation time
        computation_start_event = torch.cuda.Event(enable_timing=True)
        computation_end_event = torch.cuda.Event(enable_timing=True)
        computation_start_event.record()

        h = x + self.attention(
            self.attention_norm(x), start_pos, freqs_cis, mask, cache_k=cache_k, cache_v=cache_v, time_dict=time_dict, temp_key_value=temp_key_value
        )
        out = h + self.feed_forward(self.ffn_norm(h),time_dict)
        
        #record computation time
        computation_end_event.record()
        torch.cuda.current_stream().synchronize()
        elapsed_ms = computation_start_event.elapsed_time(computation_end_event)
        time_dict["TransformerBlock_time"] = elapsed_ms

        return out
        

class Transformer(nn.Module):
    # 將模型的權重從 CPU 移到 GPU
    def load_module_to_gpu(self, mod: nn.Module, async_offload: bool = False):
        if not hasattr(mod, "_cpu_data_dict"):
            mod._cpu_data_dict = {}
        for name, param in mod.named_parameters(recurse=True):
            # 記錄原本的 CPU Tensor
            mod._cpu_data_dict[name] = param.data
            # 轉到 GPU
            param.data = param.data.to("cuda", non_blocking = async_offload)
    # 將模型的權重從 GPU 清掉
    def return_module_to_cpu(self, mod: nn.Module):
        if not hasattr(mod, "_cpu_data_dict"):
            return
        for name, param in mod.named_parameters(recurse=True):
            cpu_tensor = mod._cpu_data_dict[name]
            # param.data 指回之前保存的 CPU 張量
            param.data = cpu_tensor
        # 釋放記錄
        del mod._cpu_data_dict

    def __init__(
        self, params: ModelArgs, 
        weight_offload_layer_ids, 
        KV_cache_offload_layer_ids, 
        async_offload: bool
    ):
        super().__init__()
        
        #ms
        self.total_TransformerBlock_time = 0.0
        self.total_Attention_time = 0.0
        self.total_WqWkWvWo_time = 0.0
        self.total_attention_score_time = 0.0
        self.total_load_KV_cache_time = 0.0
        self.total_FeedForward_time = 0.0
        self.total_weight_transfer_time = 0.0
        self.total_layer_time = 0.0
        
        self.total_forward_time = 0.0

        self.prefill_latency = 0.0

        self.avg_per_layer_weight_transfer_time = []
        self.avg_per_layer_TransformerBlock_time = []
        self.avg_per_layer_attention_score_time = []
        self.avg_per_layer_WqWkWvWo_time = []
        self.avg_per_layer_load_KV_cache_time = []
        self.avg_per_layer_Attention_time = []
        self.avg_per_layer_FeedForward_time = []
        self.avg_layer_time = []
        
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = torch.nn.Embedding(
            params.vocab_size, params.dim
        )

        self.store_KV_cache_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.weight_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        self.n_heads = params.n_heads
        self.n_kv_heads = params.n_heads if params.n_kv_heads is None else params.n_kv_heads
        self.head_dim = params.dim // params.n_heads

        self.weight_offload_layer_ids = weight_offload_layer_ids
        self.KV_cache_offload_layer_ids = KV_cache_offload_layer_ids
        self.async_offload = async_offload

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(
                TransformerBlock(
                    layer_id, params, 
                    KV_cache_offload_layer_ids=KV_cache_offload_layer_ids,
                    async_offload=async_offload, 
                    store_KV_cache_stream=self.store_KV_cache_stream
                )
            )

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = torch.nn.Linear(
            params.dim, params.vocab_size, bias=False
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

        #KV cache define in from_pretrained()
        self.past_key_values_buffer = None

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):           #有效kv cache 存放在past_key_values_buffer[0:number of layer , 0:2 , 0:batch_size, 0:start_pos, :n_kv_heads, :head_dim]
        forward_start = torch.cuda.Event(enable_timing=True)  
        forward_end = torch.cuda.Event(enable_timing=True)
        forward_start.record()

        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            )

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=tokens.device),
                mask
            ]).type_as(h)

        if 0 in self.weight_offload_layer_ids:  #如果第一層的權重有offload就要先移回GPU
            self.load_module_to_gpu(self.layers[0])  

        weight_transfer_time = 0
        TransformerBlock_time = 0
        attention_score_time = 0
        WqWkWvWo_time = 0
        Attention_time = 0
        FeedForward_time = 0
        layer_time = 0
        load_KV_cache_time = 0

        #used for contiguous when decode stage
        temp_key_value = None
        if len(self.KV_cache_offload_layer_ids) > 0 and mask == None:
            temp_key_value = torch.empty(2, bsz, self.n_heads, start_pos + seqlen, self.head_dim, device="cpu")

        for layer_id, layer in enumerate(self.layers):
            layer_start_event = torch.cuda.Event(enable_timing=True)  #Record layer time
            layer_end_event = torch.cuda.Event(enable_timing=True)  
            layer_start_event.record()
            
            cache_k = self.past_key_values_buffer[layer_id][0]
            cache_v = self.past_key_values_buffer[layer_id][1]
            
            #若下一層有offload，prefetch下一層的權重
            if layer_id+1 in self.weight_offload_layer_ids:
                self.weight_stream.wait_stream(torch.cuda.current_stream())
                #record load time
                weight_start = torch.cuda.Event(enable_timing=True)
                weight_end = torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(self.weight_stream):
                    weight_start.record()
                    self.load_module_to_gpu(self.layers[layer_id+1], async_offload=self.async_offload)
                    weight_end.record()
            

            time_dict = {"TransformerBlock_time": 0.0, "Attention_time": 0.0, "WqWkWvWo_time" : 0.0, "load_KV_cache_time" : 0.0, "attention_score_time" : 0.0, "FeedForward_time": 0.0}
            h = layer(h, start_pos, freqs_cis, mask, cache_k=cache_k, cache_v=cache_v, time_dict=time_dict, temp_key_value=temp_key_value)
            TransformerBlock_time = TransformerBlock_time + time_dict["TransformerBlock_time"]
            attention_score_time = attention_score_time + time_dict['attention_score_time']
            WqWkWvWo_time = WqWkWvWo_time + time_dict["WqWkWvWo_time"]
            Attention_time = Attention_time + time_dict["Attention_time"]
            FeedForward_time = FeedForward_time + time_dict["FeedForward_time"]
            load_KV_cache_time = load_KV_cache_time + time_dict["load_KV_cache_time"]


            if layer_id in self.weight_offload_layer_ids:  #若這一層是有offload的，把移到GPU的權重指回CPU的權重
                self.return_module_to_cpu(layer)
                
            if layer_id+1 in self.weight_offload_layer_ids:  
                self.weight_stream.synchronize()
                elapsed_ms = weight_start.elapsed_time(weight_end)
                weight_transfer_time = weight_transfer_time + elapsed_ms
            
            layer_end_event.record()
            torch.cuda.current_stream().synchronize() 
            elapsed_ms = layer_start_event.elapsed_time(layer_end_event)
            layer_time = layer_time + elapsed_ms

            """
            print(f"[Layer {layer_id}] TransformerBlock_time: {time_dict['TransformerBlock_time']:.3f} ms")
            print(f"[Layer {layer_id}]  ├─ Attention_time: {time_dict['Attention_time']:.3f} ms")
            print(f"[Layer {layer_id}]  │   ├─ WqWkWvWo_time: {time_dict['WqWkWvWo_time']:.3f} ms")
            print(f"[Layer {layer_id}]  │   ├─ load_KV_cache_time: {time_dict['load_KV_cache_time']:.3f} ms")
            print(f"[Layer {layer_id}]  │   └─ attention_score_time: {time_dict['attention_score_time']:.3f} ms")
            print(f"[Layer {layer_id}]  └─ FeedForward_time: {time_dict['FeedForward_time']:.3f} ms")
            print(f"[Layer {layer_id}] layer time: {elapsed_ms:.3f} ms")
            print("-------------------------------------")
            """

        h = self.norm(h)
        output = self.output(h).float()

        forward_end.record()
        torch.cuda.current_stream().synchronize() 
        elapsed_ms = forward_start.elapsed_time(forward_end)
        if mask != None:  #prefill stage
            self.prefill_latency = elapsed_ms

        self.total_forward_time = self.total_forward_time + elapsed_ms
        self.total_TransformerBlock_time = self.total_TransformerBlock_time + TransformerBlock_time
        self.total_Attention_time = self.total_Attention_time + Attention_time
        self.total_WqWkWvWo_time = self.total_WqWkWvWo_time + WqWkWvWo_time
        self.total_attention_score_time = self.total_attention_score_time + attention_score_time
        self.total_load_KV_cache_time = self.total_load_KV_cache_time + load_KV_cache_time
        self.total_FeedForward_time = self.total_FeedForward_time + FeedForward_time
        self.total_weight_transfer_time = self.total_weight_transfer_time + weight_transfer_time
        self.total_layer_time = self.total_layer_time + layer_time

        if len(self.weight_offload_layer_ids) > 0:
            if len(self.weight_offload_layer_ids) == len(self.layers):
                avg_weight_transfer_time = weight_transfer_time/(len(self.layers)-1)
            else:
                avg_weight_transfer_time = weight_transfer_time/len(self.weight_offload_layer_ids)
        else:
            avg_weight_transfer_time = 0
        avg_TransformerBlock_time = TransformerBlock_time/self.n_layers
        avg_WqWkWvWo_time = WqWkWvWo_time/self.n_layers
        avg_attention_score_time = attention_score_time/self.n_layers
        avg_load_KV_cache_time = load_KV_cache_time/self.n_layers
        avg_Attention_time = Attention_time/self.n_layers
        avg_FeedForward_time = FeedForward_time/self.n_layers
        avg_layer_time = layer_time/self.n_layers

        self.avg_layer_time.append(avg_layer_time)
        self.avg_per_layer_weight_transfer_time.append(avg_weight_transfer_time)
        self.avg_per_layer_TransformerBlock_time.append(avg_TransformerBlock_time)
        self.avg_per_layer_WqWkWvWo_time.append(avg_WqWkWvWo_time)
        self.avg_per_layer_attention_score_time.append(avg_attention_score_time)
        self.avg_per_layer_load_KV_cache_time.append(avg_load_KV_cache_time)
        self.avg_per_layer_Attention_time.append(avg_Attention_time)
        self.avg_per_layer_FeedForward_time.append(avg_FeedForward_time)

        print("(average time per layer)")
        print(f"Weight Transfer Time: {avg_weight_transfer_time:.3f} ms")   
        print(f"TransformerBlock_time: {avg_TransformerBlock_time:.3f} ms")
        print(f" ├─ Attention_time: {avg_Attention_time:.3f} ms")
        print(f" │   ├─ WqWkWvWo_time: {avg_WqWkWvWo_time:.3f} ms")
        print(f" │   ├─ load_KV_cache_time: {avg_load_KV_cache_time:.3f} ms")
        print(f" │   └─ attention_score_time: {avg_attention_score_time:.3f} ms")
        print(f" └─ FeedForward_time: {avg_FeedForward_time:.3f} ms")
        print(f" layer time: {avg_layer_time:.3f} ms")
        print(f"========= forward_time: {elapsed_ms:.3f} ms ==========\n")

        return output




def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

class MyLlamaForCausalLM:
    def __init__(self, model: Transformer, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @classmethod  
    def load_config(cls, model_dir: str, max_seq_len: int, max_batch_size: int):
        model_dir = Path(model_dir)
        config_file = model_dir / "config.json"
        assert config_file.exists(), f"Did not find config file:{config_file}"

        with open(config_file, "r") as f:
            config = json.load(f)
        
        model_args = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            dim = config["hidden_size"],
            n_layers = config["num_hidden_layers"],
            n_heads = config["num_attention_heads"],
            n_kv_heads = config["num_key_value_heads"],
            vocab_size = config["vocab_size"],
            intermediate_size = config["intermediate_size"],
            norm_eps = config["rms_norm_eps"],
        )

        return model_args
    
    @classmethod  
    def convert_hf_state_dict(cls, hf_state_dict):
        converted = {}
        for k, v in hf_state_dict.items():
            new_k = k

            new_k = new_k.replace("model.layers.", "layers.")

            new_k = new_k.replace("self_attn.q_proj", "attention.wq")
            new_k = new_k.replace("self_attn.k_proj", "attention.wk")
            new_k = new_k.replace("self_attn.v_proj", "attention.wv")
            new_k = new_k.replace("self_attn.o_proj", "attention.wo")

            new_k = new_k.replace("mlp.gate_proj", "feed_forward.w1")
            new_k = new_k.replace("mlp.down_proj", "feed_forward.w2")
            new_k = new_k.replace("mlp.up_proj", "feed_forward.w3")

            new_k = new_k.replace("input_layernorm", "attention_norm")
            new_k = new_k.replace("post_attention_layernorm", "ffn_norm")

            new_k = new_k.replace("model.embed_tokens", "tok_embeddings")
            new_k = new_k.replace("model.norm", "norm")
            new_k = new_k.replace("lm_head", "output")

            converted[new_k] = v
        return converted

    @classmethod  
    def load_hf_model_state_dict(cls, model_dir: str, map_location: str):
        model_dir = Path(model_dir)
        index_file = model_dir / "pytorch_model.bin.index.json"
        assert index_file.exists(), f"找不到 index 檔案：{index_file}"

        with open(index_file, "r") as f:
            index_data = json.load(f)

        weight_map = index_data.get("weight_map", {})
        full_state_dict = {}

        for tensor_name, file_name in weight_map.items():
            file_path = model_dir / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"找不到權重檔：{file_path}")

            if tensor_name in full_state_dict:
                continue  # 已經載入過

            checkpoint = torch.load(file_path, map_location=map_location, weights_only=True)

            for name, tensor in checkpoint.items():
                full_state_dict[name] = tensor

        return full_state_dict
    
    @classmethod 
    def return_to_original_weight(cls, hf_state_dict, model_args):
        for key, value in hf_state_dict.items():
            if 'attention.wq.weight' in key:
                hf_state_dict[key] = hf_state_dict[key].view(model_args.n_heads, 2, model_args.dim // model_args.n_heads // 2, model_args.dim)
                hf_state_dict[key] = hf_state_dict[key].transpose(1, 2).contiguous()
                hf_state_dict[key] = hf_state_dict[key].view(model_args.dim, model_args.dim)
            if 'attention.wk.weight' in key:
                dims_per_head = model_args.dim // model_args.n_heads
                key_value_dim = dims_per_head * model_args.n_kv_heads  #only different with dim on 70b model
                hf_state_dict[key] = hf_state_dict[key].view(model_args.n_kv_heads, 2, key_value_dim // model_args.n_kv_heads // 2, model_args.dim)
                hf_state_dict[key] = hf_state_dict[key].transpose(1, 2).contiguous()
                hf_state_dict[key] = hf_state_dict[key].view(key_value_dim, model_args.dim)

    @classmethod 
    def to_bf16(cls, checkpoint: dict):
        for key, value in checkpoint.items():
            if value.dtype.is_floating_point:
                checkpoint[key] = value.to(torch.bfloat16)
    
    @classmethod   
    def from_pretrained(
        cls,
        model_path: str, 
        max_seq_len: int, 
        max_batch_size: int, 
        tokenizer, 
        weight_offload_layer_num: int = 0,
        KV_cache_offload_layer_num: int = 0,
        pin_memory: bool = False, 
        async_offload: bool = False
    ):
        
        model_args = cls.load_config(model_path, max_seq_len, max_batch_size)
        
        if weight_offload_layer_num > 0:
            checkpoint = cls.load_hf_model_state_dict(model_path, "cpu")  # 直接載入到 CPU 
            if weight_offload_layer_num > model_args.n_layers:
                weight_offload_layer_num = model_args.n_layers
            weight_offload_layer_ids = set(range(model_args.n_layers - weight_offload_layer_num, model_args.n_layers))
        else:
            checkpoint = cls.load_hf_model_state_dict(model_path, "cuda")  # 直接載入到 GPU
            weight_offload_layer_ids = set()

        if KV_cache_offload_layer_num > 0:
            if KV_cache_offload_layer_num > model_args.n_layers:
                KV_cache_offload_layer_num = model_args.n_layers
            KV_cache_offload_layer_ids = set(range(model_args.n_layers - KV_cache_offload_layer_num, model_args.n_layers))
        else:
            KV_cache_offload_layer_ids = set()

        checkpoint = cls.convert_hf_state_dict(checkpoint)
        cls.return_to_original_weight(checkpoint, model_args)
        cls.to_bf16(checkpoint)
        torch.set_default_device('cuda')
        torch.set_default_dtype(torch.bfloat16)

        with init_empty_weights():
            model = Transformer(
                model_args, 
                weight_offload_layer_ids=weight_offload_layer_ids,
                KV_cache_offload_layer_ids=KV_cache_offload_layer_ids,
                async_offload=async_offload
            )
        model.load_state_dict(checkpoint, strict=False, assign=True)
        del checkpoint
            
        if weight_offload_layer_num > 0:
            for layer_id, layer in enumerate(model.layers): 
                if layer_id in weight_offload_layer_ids:
                    print(f"offloading weight layer {layer_id} to cpu")
                    layer.to("cpu")
                    if pin_memory:
                        for name, param in layer.named_parameters(recurse=True):
                            param.data = param.data.pin_memory()        #need pin memory to prefetch next layer
                else:
                    layer.to("cuda")

            model.norm.to("cuda")
            model.output.to("cuda")
            model.tok_embeddings.to("cuda")
        else:
            model.to("cuda")   

        #KV cache
        if KV_cache_offload_layer_num > 0:
            model.past_key_values_buffer = []
            layer_shape = [2, model.params.max_batch_size, model.params.max_seq_len, model.n_kv_heads, model.head_dim] # for [k_states, v_states] per layer in order
            for i in range(model_args.n_layers):
                if i in KV_cache_offload_layer_ids:
                    model.past_key_values_buffer.append(torch.empty(layer_shape, device="cpu", pin_memory=pin_memory))
                else:
                    model.past_key_values_buffer.append(torch.empty(layer_shape, device="cuda"))
        else:
            buffer_shape = [len(model.layers), 2, model.params.max_batch_size, model.params.max_seq_len, model.n_kv_heads, model.head_dim] # for [k_states, v_states] per layer in order
            model.past_key_values_buffer = torch.empty(buffer_shape, device="cuda")    
        
        return cls(model, tokenizer)
    
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
        force_max_len: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        if force_max_len == False:
            eos_reached = torch.tensor([False] * bsz, device="cuda")

        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        for cur_pos in range(min_prompt_len, total_len):
            #tokens[:, prev_pos:cur_pos] -> tokens fed into the model
            #tokens[:, cur_pos] -> the next generated tokens
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if current position is beyond the prompt
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            prev_pos = cur_pos
            if force_max_len == False:
                eos_reached |= (~input_text_mask[:, cur_pos]) & (
                    next_token == self.tokenizer.eos_id
                )
                if all(eos_reached):
                    break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            if force_max_len == False:
                if self.tokenizer.eos_id in toks:
                    eos_idx = toks.index(self.tokenizer.eos_id)
                    toks = toks[:eos_idx]
                    probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)
    

def warmup(model: MyLlamaForCausalLM, tokenizer, 
           prompt_len: int = 32, 
           decode_steps: int = 10, warm_iters: int = 2):
    torch.cuda.empty_cache()            # 先把殘留顯存清乾淨

    batch_size = model.model.params.max_batch_size
    print("=================================start warmup=================================")
    with torch.no_grad(): 
        for _ in range(warm_iters):
            # === 1. PREFILL ===
            prompt = torch.randint(
                low=0, high=tokenizer.n_words,
                size=(batch_size, prompt_len),
                device="cuda", dtype=torch.long
            )
            model.model.forward(prompt, start_pos=0)

            # === 2. DECODE ===
            ctx = prompt[:, -1:]                      # 只拿最後一個 token
            start_pos = prompt_len                    # cache 已經有 prompt_len
            for _ in range(decode_steps):
                model.model.forward(ctx, start_pos)
                start_pos += 1 
        torch.cuda.synchronize() 
        print("=================================end warmup=================================\n\n\n\n")
        
        model.model.total_TransformerBlock_time = 0.0
        model.model.total_Attention_time = 0.0
        model.model.total_WqWkWvWo_time = 0.0
        model.model.total_attention_score_time = 0.0
        model.model.total_load_KV_cache_time = 0.0
        model.model.total_FeedForward_time = 0.0
        model.model.total_weight_transfer_time = 0.0
        model.model.total_layer_time = 0.0
        model.model.total_forward_time = 0.0

        model.model.prefill_latency = 0.0

        model.model.avg_per_layer_weight_transfer_time = []
        model.model.avg_per_layer_TransformerBlock_time = []
        model.model.avg_per_layer_WqWkWvWo_time = []
        model.model.avg_per_layer_attention_score_time = []
        model.model.avg_per_layer_load_KV_cache_time = []
        model.model.avg_per_layer_Attention_time = []
        model.model.avg_per_layer_FeedForward_time = []
        model.model.avg_layer_time = []