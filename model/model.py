import math
from turtle import forward
from typing import Any, Optional, Tuple
from numpy import repeat
from transformers import GenerationMixin, PreTrainedModel, PretrainedConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutput, CausalLMOutputWithPast

class SanzeninMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        # [MoE官方适配] MiniMind官方代码使用 num_experts 命名，这里保留你的 n_routed_experts 并增加兼容别名。
        num_experts: int = None,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        # [MoE官方适配] 官方专家FFN可单独指定 moe_intermediate_size，不指定时沿用普通FFN中间层维度。
        moe_intermediate_size: int = None,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        # [MoE官方适配] MiniMind官方代码中辅助损失参数名为 router_aux_loss_coef。
        router_aux_loss_coef: float = None,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        # [MoE官方适配] num_experts 是官方命名；n_routed_experts 是你当前代码里的命名，两者保持同步。
        self.num_experts = num_experts if num_experts is not None else n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = self.num_experts
        self.n_shared_experts = n_shared_experts
        # [MoE官方适配] 给之后手敲官方MOEFeedForward时使用。
        self.moe_intermediate_size = moe_intermediate_size if moe_intermediate_size is not None else intermediate_size
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        # [MoE官方适配] 官方名 router_aux_loss_coef 与你已有 aux_loss_alpha 兼容。
        self.router_aux_loss_coef = router_aux_loss_coef if router_aux_loss_coef is not None else aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x: torch.Tensor):
        # 对最后一维 hidden_size 计算 RMS 缩放系数。
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        return (self.weight * self.norm(x.float())).type_as(x)


def precompute_freqs_cis(dim:int,rope_base,end:int=(32*1024),rope_scaling:Optional[dict]=None): #end是要推断的长度
    #分配旋转频率
    freqs,attention_factor=(
        1.0/(rope_base**(torch.arange(0,dim,2)[:dim//2].float()/dim)),
        1.0)
    if rope_scaling is not None:
        orig_max,factor,beta_fast,beta_slow=(
            rope_scaling["original_max_position_embeddings"],
            rope_scaling["factor"],
            rope_scaling["beta_fast"],
            rope_scaling["beta_slow"],
        )
        attention_factor = rope_scaling.get("attention_factor", 1.0)
        #波长b到i的映射
        if end>orig_max:
            inv_dim=lambda b:(dim*math.log(orig_max/(b*2*math.pi)))/(2*math.log(rope_base))

            low,high=(max(math.floor(inv_dim(beta_fast)),0),min(math.ceil(inv_dim(beta_slow)),dim//2-1))

            #计算缩放因子：low之前的高频不缩放，中间平滑缩放，high之后的低频按factor缩放
            ramp=torch.clamp(
                (torch.arange(dim//2,device=freqs.device).float()-low)/max(high-low,0.001),
                0,
                1
            )
            freqs=freqs*(1-ramp+ramp/factor)

    #生成位置索引t
    t=torch.arange(end,device=freqs.device).float()
    #计算外积,t与频率部分相乘,并得到旋转角度
    freqs=torch.outer(t,freqs).float()
    freqs_cos=(
        torch.cat([torch.cos(freqs),torch.cos(freqs)],dim=-1)*attention_factor
    )
    freqs_sin=(
        torch.cat([torch.sin(freqs),torch.sin(freqs)],dim=-1)*attention_factor
    )
    return freqs_cos,freqs_sin
    
def apply_rotary_pos_emb(q:torch.Tensor,k:torch.Tensor,cos:torch.Tensor,sin:torch.Tensor,position_ids=None,unsqueeze_dim=1):
    #为了方便并行计算 实际上是向量前半和后半按顺序配对
    def rotate_half(x):
        return torch.cat([
            -x[...,x.shape[-1]//2:],x[...,:x.shape[-1]//2]
        ],dim=-1)
    q_embed=(q*cos.unsqueeze(unsqueeze_dim)+rotate_half(q)*sin.unsqueeze(unsqueeze_dim))
    k_embed=(k*cos.unsqueeze(unsqueeze_dim)+rotate_half(k)*sin.unsqueeze(unsqueeze_dim))
    return q_embed,k_embed

def repeat_kv(x:torch.Tensor,repeat_nums:int)->torch.Tensor:
    bs,seq_len,num_key_value_heads,head_dim=x.shape
    if repeat_nums==1:
        return x
    
    return (
        x[:,:,:,None,:]
        .expand(bs,seq_len,num_key_value_heads,repeat_nums,head_dim)
        .reshape(bs,seq_len,num_key_value_heads*repeat_nums,head_dim)
    )
KVCache = Tuple[torch.Tensor, torch.Tensor]
PastKeyValues = list[Optional[KVCache]]


class Attention(nn.Module):
    def __init__(self, args:SanzeninMindConfig) -> None:
        super().__init__()
        #全局kv头数
        self.num_key_value_heads=args.num_key_value_heads if args.num_key_value_heads is not None else args.num_attention_heads
        assert args.num_attention_heads%self.num_key_value_heads==0,""
        #当前分配到的头(如果使用了多GPU并行运算)
        self.local_heads=args.num_attention_heads
        self.local_kv_heads=self.num_key_value_heads
        self.repeat_nums=self.local_heads//self.local_kv_heads
        #每个注意力头的维度
        self.head_dim=args.hidden_size//args.num_attention_heads

        #投影计算
        self.q_proj=nn.Linear(args.hidden_size,self.head_dim*self.local_heads,bias=False)
        self.k_proj=nn.Linear(args.hidden_size,self.head_dim*self.local_kv_heads,bias=False)
        self.v_proj=nn.Linear(args.hidden_size,self.head_dim*self.local_kv_heads,bias=False)
        self.o_proj=nn.Linear(self.head_dim*self.local_heads,args.hidden_size,bias=False)
        
        self.attn_dropout=nn.Dropout(args.dropout)
        self.resid_dropout=nn.Dropout(args.dropout)
        self.dropout=args.dropout

        #hasattr检查是否有目标方法
        self.flash=hasattr(nn.functional,'scaled_dot_product_attention') and args.flash_attention
    
    def forward(self,x:torch.Tensor,position_embeddings:Tuple[torch.Tensor,torch.Tensor],past_key_value:Optional[Tuple[torch.Tensor,torch.Tensor]],use_cache=False,attention_mask:Optional[torch.Tensor]=None)->Tuple[torch.Tensor,Optional[KVCache]]:
        batch_size,seq_len,_=x.shape
        xq,xk,xv=self.q_proj(x),self.k_proj(x),self.v_proj(x)
        
        #拆分成多个头
        xq=xq.view(batch_size,seq_len,self.local_heads,self.head_dim)
        xk=xk.view(batch_size,seq_len,self.local_kv_heads,self.head_dim)
        xv=xv.view(batch_size,seq_len,self.local_kv_heads,self.head_dim)

        cos,sin=position_embeddings
        #预计算出的cos和sin通常包含整个序列的所有位置,当前只能处理seq_len个token
        xq,xk=apply_rotary_pos_emb(xq,xk,cos,sin)

        if past_key_value is not None:
            xk=torch.cat([past_key_value[0],xk],dim=1)
            xv=torch.cat([past_key_value[1],xv],dim=1)
        past_kv=(xk,xv) if use_cache else None
        #对于Q  [batch_size, seq_len,local_heads, head_dim]->[batch_size, local_heads,seq_len, head_dim]
        xq,xk,xv,=(
            xq.transpose(1,2),
            repeat_kv(xk,self.repeat_nums).transpose(1,2),
            repeat_kv(xv,self.repeat_nums).transpose(1,2),
        )
        if (
            self.flash
            and (seq_len > 1)
            and (past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            ) #is_causal表示启用因果注意力
        else:
            #相乘后在每一维,行方向对应为Q
            scores=(xq@xk.transpose(-2,-1))/math.sqrt(self.head_dim)
            #应用causal_mask防止看到未来token
            scores[:,:,:,-seq_len:]+=torch.triu(torch.full((seq_len,seq_len),float('-inf'),device=scores.device),diagonal=1)#diagonal=1上三角再向上移一行,矩阵行方向对应Q
            
            #这里的mask是处理用于对齐序列长度对齐的pad部分
            if attention_mask is not None:
                extended_attention_mask=attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask=(1-extended_attention_mask)*(-1e9)
                scores=scores+extended_attention_mask

            scores=F.softmax(scores.float(),dim=-1).type_as(xq)
            scores=self.attn_dropout(scores)
            output=scores@xv #点积
        output=output.transpose(1,2).reshape(batch_size,seq_len,-1)
        output=self.resid_dropout(self.o_proj(output))
        return output,past_kv

class FeedForward(nn.Module):
    # [MoE官方适配] 增加 intermediate_size 参数，方便每个专家传入 config.moe_intermediate_size。
    def __init__(self,config:SanzeninMindConfig,intermediate_size:int=None) -> None:
        super().__init__()
        # [MoE官方适配] 优先使用调用方传入的 intermediate_size；否则使用 config.intermediate_size；都没有时再自动计算。
        intermediate_size=intermediate_size if intermediate_size is not None else config.intermediate_size
        if intermediate_size is None:
            intermediate_size=config.hidden_size*8//3
            intermediate_size=64*((intermediate_size+63)//64)#向上取整为64的倍数,有利于gpu的性能
        self.gate_proj=nn.Linear(config.hidden_size,intermediate_size,bias=False)
        self.down_proj=nn.Linear(intermediate_size,config.hidden_size,bias=False)
        self.up_proj=nn.Linear(config.hidden_size,intermediate_size,bias=False)
        self.dropout=nn.Dropout(config.dropout)
        #通常是SwiGLU激活函数
        self.act_fn=ACT2FN[config.hidden_act]


    def forward(self,x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x))*self.up_proj(x)))

class MoEGate(nn.Module):
    def __init__(self,config:SanzeninMindConfig) -> None:
        super().__init__()
        self.config=config
        self.gate=nn.Linear(config.hidden_size,config.num_experts,bias=False)
        self.experts=nn.ModuleList([
            FeedForward(config,intermediate_size=config.moe_intermediate_size) for i in range(config.num_experts)
        ])
        
        self.shared_experts=nn.ModuleList([
                        FeedForward(config,intermediate_size=config.moe_intermediate_size) for i in range(config.n_shared_experts)
        ])
        self.act_fn=ACT2FN[config.hidden_act]
        self.aux_loss=torch.tensor(0.0)

    def forward(self,x:torch.Tensor):
        batch_size,seq_len,hidden_dim=x.shape

        identity=x

        #展平所有的token
        x_flat=x.view(-1,hidden_dim)
        scores=F.softmax(self.gate(x_flat),dim=-1) #在最后一个维度上进行soft_max

        # topk_weight: [token_count, num_experts_per_tok]为选中专家的概率权重
        # topk_idx:    [token_count, num_experts_per_tok]为选中专家的编号
        topk_weight,topk_idx=torch.topk(
            scores,
            k=self.config.num_experts_per_tok,
            dim=-1,
            sorted=False
        )

        if self.config.norm_topk_prob:
            topk_weight=topk_weight/(topk_weight.sum(dim=-1,keepdim=True)+1e-20)

        #创建输出缓存
        y=torch.zeros_like(x_flat)

        for i,expert in enumerate(self.experts):
            # mask:     [token_count, num_experts_per_tok]
            mask=(topk_idx==i)

            #如果当前专家被至少一个token选中
            if mask.any():
                #选择当前专家的token下标
                token_idx=mask.any(dim=-1).nonzero().flatten()
                #[num_selected_tokens, 1] 方便广播
                weight=topk_weight[mask].view(-1,1) #下标mask会取出所有mask为真的元素

                expert_out=expert(x_flat[token_idx])*weight

                y.index_add_(0,token_idx,expert_out.to(y.dtype))#加到第0维上token_idx索引处
            elif self.training:
                y[0,0]+=0*sum(p.sum() for p in expert.parameters()) #设置梯度

        y = y.view(batch_size, seq_len, hidden_dim)

        for expert in self.shared_experts:
            shared_out = expert(identity)
            y = y + shared_out

        if self.training and self.config.router_aux_loss_coef>0:
            #[token_count, num_experts_per_tok] -> [token_count, num_experts_per_tok, num_experts] -> [num_experts_per_tok, num_experts]
            #对哪一维做mean 哪一维就会消失
            load=F.one_hot(
                topk_idx,
                self.config.num_experts
            ).float().mean(0) #表示每个专家作为第k个topk专家被选中的概率

            self.aux_loss=(
                (load*scores.mean(0)).sum()
                *self.config.num_experts
                *self.config.router_aux_loss_coef
            )
        else:
            self.aux_loss=scores.new_zeros(1).squeeze()

        return y
class SanzeinMindBlock(nn.Module):
    def __init__(self,Layer_id:int,config:SanzeninMindConfig) -> None:
        super().__init__()
        self.num_attention_heads=config.num_attention_heads
        self.hidden_size=config.hidden_size
        self.head_dim=config.head_dim
        self.self_attentionn=Attention(config)
        
        self.Layer_id=Layer_id
        self.input_layer_norm=RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.post_attention_layernorm=RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.mlp=FeedForward(config) if not config.use_moe else MoEGate(config)
        
    def forward(self,hidden_states,position_embeddings,past_key_value=None,use_cache=False,attention_mask=None):
        residual=hidden_states
        hidden_states,present_key_values=self.self_attentionn(
            self.input_layer_norm(hidden_states),position_embeddings,past_key_value,use_cache,attention_mask
        )
        hidden_states=residual+hidden_states
        hidden_states=hidden_states+self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states,present_key_values

class SanzeninMindModel(nn.Module):
    def __init__(self,config:SanzeninMindConfig) -> None:
        super().__init__()
        self.vocab_size=config.vocab_size
        self.num_hidden_layers=config.num_hidden_layers
        self.embed_tokens=nn.Embedding(config.vocab_size,config.hidden_size)
        self.dropout=nn.Dropout(config.dropout)

        self.layers=nn.ModuleList([SanzeinMindBlock(i,config) for i in range(self.num_hidden_layers)])

        self.norm=RMSNorm(config.hidden_size,config.rms_norm_eps)

        #Rope预计算
        self.freqs_cos,self.freqs_sin=precompute_freqs_cis(
            dim=config.hidden_size//config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )
        self.register_buffer("freqs_cos",self.freqs_cos)
        self.register_buffer("freqs_sin",self.freqs_sin)
    
    def forward(
            self,
            input_ids:torch.Tensor,#[batch_size,seq_len] 输入的token id
            attention_mask:Optional[torch.Tensor]=None,
            past_key_values:Optional[PastKeyValues]=None,
            use_cache:bool=False,
            **kwargs,
        ):
            batch_size,seq_len=input_ids.shape
            if hasattr(past_key_values,'layers'):
                past_key_values=None

            past_key_values=past_key_values or [None]*len(self.layers)
            start_pos=(
                past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0 #KV Cache的形状 [batch_size, cached_seq_len, num_key_value_heads, head_dim]
            )
            hidden_states=self.dropout(self.embded_tokens(input_ids))
            position_embeddings=(
                self.freqs_cos[start_pos:start_pos+seq_len],
                self.freqs_sin[start_pos:start_pos+seq_len]
            )
            presents=[]

            for layer_idx,(layer,past_key_value) in enumerate(zip(self.layers,past_key_values)):
                hidden_states,present=layer(
                    hidden_states,
                    position_embeddings,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                    attention_mask=attention_mask,
                )
                presents.append(present)
            hidden_states=self.norm(hidden_states)
            return hidden_states,presents

#继承Hugging face集成类
class SanzeninMindForCausalLM(PreTrainedModel,GenerationMixin):
    config_class=SanzeninMindConfig

    def __init__(self,config:SanzeninMindConfig):
        super().__init__(config)
        # Block 里会读取 head_dim，这里按注意力头数补齐派生配置
        self.config.head_dim=self.config.hidden_size//self.config.num_attention_heads
        # 只在 CausalLM 包装层内组装 backbone，复用前面已经定义好的模块
        self.model=nn.Module()
        self.model.vocab_size=self.config.vocab_size
        self.model.num_hidden_layers=self.config.num_hidden_layers
        self.model.embed_tokens=nn.Embedding(self.config.vocab_size,self.config.hidden_size)
        self.model.dropout=nn.Dropout(self.config.dropout)
        self.model.layers=nn.ModuleList([SanzeinMindBlock(i,self.config) for i in range(self.config.num_hidden_layers)])
        self.model.norm=RMSNorm(self.config.hidden_size,self.config.rms_norm_eps)
        freqs_cos,freqs_sin=precompute_freqs_cis(
            dim=self.config.hidden_size//self.config.num_attention_heads,
            end=self.config.max_position_embeddings,
            rope_base=self.config.rope_theta,
            rope_scaling=self.config.rope_scaling
        )
        # RoPE 频率表作为 buffer 保存，随模型 device/dtype 迁移但不参与梯度更新
        self.model.register_buffer("freqs_cos",freqs_cos)
        self.model.register_buffer("freqs_sin",freqs_sin)
        self.lm_head=nn.Linear(self.config.hidden_size,self.config.vocab_size,bias=False)
        #输入输出层共享权重,输入时按token_id查行转换为词向量,输出时线性映射为整个词表的概率分布
        self.model.embed_tokens.weight=self.lm_head.weight

    def forward(self,input_ids:torch.Tensor|None=None,
                attention_mask:torch.Tensor|None=None,
                labels:torch.Tensor|None=None,
                past_key_values:PastKeyValues|None=None,
                use_cache:bool=False,
                logits_to_keep:int|torch.Tensor=0,
                **kwargs:Any
                ):
        if input_ids is None:
            raise ValueError("input_ids must be provided")

        if hasattr(past_key_values,"layers"):
            past_key_values=None
        past_key_values=past_key_values or [None]*len(self.model.layers)

        batch_size,seq_len=input_ids.shape
        start_pos=past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # token id -> hidden states；这里直接调用 embed_tokens，避开底层 forward 中的拼写问题
        hidden_states=self.model.dropout(self.model.embed_tokens(input_ids))
        position_embeddings=(
            self.model.freqs_cos[start_pos:start_pos+seq_len],
            self.model.freqs_sin[start_pos:start_pos+seq_len],
        )

        presents=[]
        for layer,past_key_value in zip(self.model.layers,past_key_values):
            # 每层只接收自己对应的 KV cache，并在 use_cache=True 时返回新的 cache。
            hidden_states,present=layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)
        hidden_states=self.model.norm(hidden_states)
        aux_loss=sum(
            [
                layer.mlp.aux_loss
                for layer in self.model.layers
                if isinstance(layer.mlp,MoEGate)
            ],
            hidden_states.new_zeros(1).squeeze(),
        )

        # logits_to_keep 用于生成阶段只计算末尾 token 的 logits，训练时默认保留全序列。
        slice_indices=slice(-logits_to_keep,None) if isinstance(logits_to_keep,int) else logits_to_keep
        logits=self.lm_head(hidden_states[:,slice_indices,:])

        loss=None
        if labels is not None:
            shift_logits=logits[...,:-1,:].contiguous()
            shift_labels=labels[...,1:].contiguous()
            loss=F.cross_entropy(
                shift_logits.view(-1,shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        output=CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=presents if use_cache else None,
            hidden_states=hidden_states,
        )
        output.aux_loss=aux_loss
        return output
