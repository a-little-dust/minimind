import math
import struct
import inspect
import time

from .LMConfig import LMConfig
from typing import Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

#归一化
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))#创建一个维度为dim的全1向量。Parameter表示这是一个可训练的参数，在反向传播时会被更新

    def forward(self, x):
        return self.weight * (x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)

# 计算rotate位置编码
def precompute_pos_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # 生成一个从 0 到 end-1 的张量
    freqs = torch.outer(t, freqs).float()  # 计算 t 和 freqs 的外积
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # 先得到一个形状为freqs的全是 1的张量，表示极坐标的振幅
    # 然后使用freqs作为极坐标的角度
    return pos_cis


def apply_rotary_emb(xq, xk, pos_cis):
    def unite_shape(pos_cis, x):#首先调整形状
        """
        调整位置编码的形状以匹配输入张量的维度
        
        Args:
            pos_cis: 位置编码张量
            x: 输入张量
            
        Returns:
            调整后的位置编码张量
            
        注意:
            - 确保输入张量至少有2个维度
            - pos_cis的形状需要与x的第2维和最后一维匹配
        """
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])#分别表示序列长度和维度
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)

    # 将query和key转换为复数形式
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # 调整位置编码形状
    pos_cis = unite_shape(pos_cis, xq_)
    # 应用旋转位置编码并转回实数形式。flatten(3)表示将最后一维展平
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    # 返回与输入相同数据类型的结果
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):#注意力层
    def __init__(self, args: LMConfig):
        # 调用父类初始化
        super().__init__()
        # 设置key-value头的数量,如果未指定则等于注意力头数量
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # 确保注意力头数量能被kv头数量整除
        assert args.n_heads % self.n_kv_heads == 0
        # 设置本地注意力头数量
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        # 计算每个kv头对应的注意力头数量
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 计算每个头的维度
        self.head_dim = args.dim // args.n_heads
        # 定义Q、K、V的线性变换层
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)#q的大小是头的数量*头的维度
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)#k的大小是kv头的数量*头的维度
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)#v的大小是kv头的数量*头的维度
        # 定义输出的线性变换层
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        # 定义注意力和残差的dropout层
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        # 判断是否使用flash attention加速
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # 创建注意力掩码矩阵,用于实现因果注意力
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))#值为-无穷
        mask = torch.triu(mask, diagonal=1)#只保留原来的上三角部分，其余变成0
        self.register_buffer("mask", mask, persistent=False)#把mask注册到buffer中，这样在反向传播时不会被更新

    def forward(self,
                x: torch.Tensor,
                pos_cis: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False):
        bsz, seq_len, _ = x.shape#得到batch_size,序列长度,维度
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)#输入x，得到q、k、v
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)#转换成batch_size,序列长度,头的数量,头的维度
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, pos_cis)#给q、k添加位置编码
        # kv_cache实现
        if past_key_value is not None:#在k、v中添加past_key_value
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),# q的维度转换成batch_size,头的数量,序列长度,头的维度
            # transpose表示对调维度
            repeat_kv(xk, self.n_rep).transpose(1, 2),#每个kv头会分到好几个注意力头，所以要重复
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )
        if self.flash and seq_len != 1:
            dropout_p = self.dropout if self.training else 0.0#如果是训练阶段，就使用dropout，否则使用0.0
            output = F.scaled_dot_product_attention(#F来自torch.nn.functional
                xq, xk, xv,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True#是否是因果注意力
            )
        else:
            # 手动计算注意力
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)# 把q和k的转置矩阵相乘
            scores += self.mask[:, :, :seq_len, :seq_len]# 把掩码添加到scores中
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)#对最后一个维度进行softmax
            scores = self.attn_dropout(scores)#对scores进行dropout
            output = scores @ xv#最后是计算q和v的乘积

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)#把输出转换成batch_size,序列长度,维度
        output = self.resid_dropout(self.wo(output))#进行output，然后进行dropout
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: LMConfig):
        """
        前馈神经网络模块初始化
        
        Args:
            config: LMConfig - 模型配置参数对象
            
        初始化内容:
            1. 如果未指定hidden_dim，则根据dim计算合适的隐藏层维度
            2. 创建三个线性变换层(w1, w2, w3)用于SwiGLU激活函数实现
            3. 设置dropout层用于正则化
        """
        super().__init__()
        if config.hidden_dim is None:
            hidden_dim = 4 * config.dim  # 初始隐藏层大小为输入维度的4倍
            hidden_dim = int(2 * hidden_dim / 3)  # 缩小到原来的2/3
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)  # 向上取整到最近的倍数
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)  # 输入投影层
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)  # 输出投影层
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)  # 门控投影层
        self.dropout = nn.Dropout(config.dropout)  # dropout层用于防止过拟合

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MoEGate(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        # 初始化MoE模型配置
        self.config = config
        # 创建路由专家列表，每个专家是一个前馈神经网络
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])# 有n_routed_experts个，初始时刻每个都一样
        # 创建MoE门控网络，用于选择专家
        self.gate = MoEGate(config)
        # 如果配置了共享专家，则创建一个共享的前馈神经网络
        if config.n_shared_experts is not None:
            self.shared_experts = FeedForward(config)

    def forward(self, x):
        """
        混合专家模型(MoE)的前向传播函数
        
        Args:
            x: torch.Tensor - 输入张量，形状为 [batch_size, seq_len, hidden_dim]
            
        Returns:
            torch.Tensor - 经过专家处理后的输出张量，与输入形状相同
            
        实现逻辑:
        1. 保存输入用于残差连接和形状恢复
        2. 使用门控网络选择每个token应该由哪些专家处理
        3. 根据训练/推理模式采用不同的专家调用策略
        4. 如果配置了共享专家，则添加共享专家的输出
        """
        identity = x  # 保存输入用于残差连接
        orig_shape = x.shape  # 记录原始形状用于后续恢复
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家。把x输入gate，输出为专家索引，权重，辅助损失
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])  # 将输入展平为二维张量，也就是batch_size * seq_len , hidden_dim
        flat_topk_idx = topk_idx.view(-1)  # 将专家索引展平，原本的形状是batch_size * seq_len * num_experts_per_tok
        if self.training:
            # 训练模式下，重复输入数据以便并行处理多个专家
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)#有多少个专家，就重复多少次
            y = torch.empty_like(x, dtype=torch.float16)  # 创建输出张量，和x形状相同
            for i, expert in enumerate(self.experts):#遍历每个专家，然后让输出中这个专家对应的值等于expert(x)
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # flat_topk_idx == i确保只选择专家
            # 让y的形状与 topk_weight 的形状相匹配，然后将结果与topk_weight相乘，再求和
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)  # 加权求和
            y = y.view(*orig_shape)  # 恢复原始形状
        else:
            # 推理模式下，只选择最优专家，使用专门的推理函数提高效率
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)  # 如果有共享专家，输入x得到输出
        self.aux_loss = aux_loss  # 保存辅助损失用于优化专家负载均衡
        return y

    @torch.no_grad()#不跟踪梯度
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        混合专家模型(MoE)的推理函数，用于高效处理不同专家的token分配
        
        Args:
            x: torch.Tensor - 输入张量，形状为 [batch_size*seq_len, hidden_dim]
            flat_expert_indices: torch.Tensor - 展平的专家索引，指示每个token应由哪个专家处理
            flat_expert_weights: torch.Tensor - 展平的专家权重，表示每个token对应专家的权重
            
        Returns:
            torch.Tensor - 经过专家处理后的输出张量，与输入形状相同
        """
        expert_cache = torch.zeros_like(x)  # 创建与输入相同形状的零张量作为输出缓存
        idxs = flat_expert_indices.argsort()  # 对专家索引进行排序，便于按专家分组处理
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)  # 计算每个专家处理的token数量的累积和（例如[2,4,7）
        token_idxs = idxs // self.config.num_experts_per_tok  # 计算每个专家索引对应的token索引
        # 例如当tokens_per_expert=[6, 15, 20, 26, 33, 38, 46, 52]
        # 当token_idxs=[3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...]
        # 意味着当token_idxs[:6] -> [3,  7, 19, 21, 24, 25,  4]位置的token都由专家0处理，token_idxs[6:15]位置的token都由专家1处理......
        for i, end_idx in enumerate(tokens_per_expert):#对于每个专家
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]  # 确定当前专家处理的token范围起始位置,是 0 或者上一个专家处理的token数量
            if start_idx == end_idx:  # 如果当前专家没有需要处理的token，则跳过
                continue
            expert = self.experts[i]  # 获取当前专家模型
            exp_token_idx = token_idxs[start_idx:end_idx]  # 获取当前专家需要处理的token索引
            expert_tokens = x[exp_token_idx]  # 提取需要当前专家处理的token
            expert_out = expert(expert_tokens).to(expert_cache.dtype)  # 使用专家处理token并转换为缓存的数据类型
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])  # 用mul函数将专家输出乘以对应的权重
            # 使用 scatter_add_ 进行 sum 操作，将处理后的token放回对应位置
            # 这里重复了batch_size * seq_len次，从而让每个token只被一个专家处理
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache  # 返回所有专家处理后的结果


class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: LMConfig):
        """
        MiniMindBlock的初始化函数
        
        Args:
            layer_id: 当前层的ID
            config: 模型配置参数
            
        初始化内容包括:
        1. 多头注意力相关参数(n_heads, dim, head_dim)
        2. 注意力层(Attention)
        3. 两个归一化层(attention_norm, ffn_norm) 
        4. 前馈网络层(feed_forward或MOEFeedForward)
        """
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.attention = Attention(config)#自定义的Attention

        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        # 两种前馈网络都是自定义的
        self.feed_forward = FeedForward(config) if not config.use_moe else MOEFeedForward(config)


    def forward(self, x, pos_cis, past_key_value=None, use_cache=False):
        h_attn, past_kv = self.attention(
            self.attention_norm(x),#先归一化再做注意力
            pos_cis,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        h = x + h_attn#加入注意力层，再残差连接
        out = h + self.feed_forward(self.ffn_norm(h))#先归一化再加入前馈网络层，最后残差连接
        return out, past_kv


class MiniMindLM(PreTrainedModel):#预训练大模型
    config_class = LMConfig

    def __init__(self, params: LMConfig = None):
        self.params = params or LMConfig()#从LMConfig中读取params
        super().__init__(self.params)
        self.vocab_size, self.n_layers = params.vocab_size, params.n_layers
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)#输入层，把词汇表映射为embedding
        self.dropout = nn.Dropout(params.dropout)#以droupout的概率丢弃输入
        self.layers = nn.ModuleList([MiniMindBlock(l, params) for l in range(self.n_layers)])#每一层都是一个MiniMindBlock
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)#对dim个维度进行归一化，eps表示归一化时的epsilon
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)#输出层，是一个全连接层，linear表示线性变换
        self.tok_embeddings.weight = self.output.weight#输入embedding层和输出层的权重共享，tok表示token
        self.register_buffer("pos_cis",
                             precompute_pos_cis(dim=params.dim // params.n_heads, theta=params.rope_theta),
                             persistent=False)#设置缓冲区，它不会被视为模型的可学习参数（即不会在反向传播中更新）
        #缓冲区的名称是pos_cis pos表示位置，cis表示cosine，theta表示角度
        # persistent=False表示缓冲区不会被保存到checkpoint中，即不会被保存到硬盘中
        # precompute_pos_cis是自定义的函数，将嵌入维度除以头数，计算每个注意力头的维度
        self.OUT = CausalLMOutputWithPast()#定义模型输出的数据结构,包含logits、past key values等信息
        #causallm表示是自回归语言模型，output表示输出层，withpast表示包含past key values，过去的键值对

# 前向传播
    def forward(self,
                input_ids: Optional[torch.Tensor] = None,# 输入的token id序列
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,# KV缓存，是元组类型
                use_cache: bool = False,
                **args):
        past_key_values = past_key_values or [None] * len(self.layers)#每一层都有KV缓存
        start_pos = args.get('start_pos', 0)#如果没有设置start_pos，则默认为0
        h = self.dropout(self.tok_embeddings(input_ids))#先进行embedding，再进行dropout
        pos_cis = self.pos_cis[start_pos:start_pos + input_ids.size(1)]#截取一段，从start_pos开始，长度等于当前输入序列的长度(序列的第二维大小)
        past_kvs = []
        for l, layer in enumerate(self.layers):
            h, past_kv = layer(
                h, pos_cis,
                past_key_value=past_key_values[l],
                use_cache=use_cache
            )#h（上一层的输出）和位置编码一起传入layer，然后输出h和past_kv
            past_kvs.append(past_kv)
        logits = self.output(self.norm(h))#最后一层做归一化，然后做线性变换
        # 用于moe模式，辅助损失用于优化专家的负载均衡，防止某些专家被过度使用而其他专家闲置
        # 遍历每一层，检查每一层的feed_forward，如果feed_forward是MOEFeedForward，则计算辅助损失
        aux_loss = sum(l.feed_forward.aux_loss for l in self.layers if isinstance(l.feed_forward, MOEFeedForward))
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT

    @torch.inference_mode()#表示在推理模式下（而非训练模式）运行，即不跟踪梯度
    def generate(self, input_ids, eos_token_id=2, max_new_tokens=1024, temperature=0.75, top_p=0.90,
                 stream=False, rp=1., use_cache=True, pad_token_id=0, **args):
        # 流式生成
        if stream:
            return self._stream(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)

        # 直接生成
        generated = []
        for i in range(input_ids.size(0)):#遍历每一个batch
            # 提取非padding的有效token序列，然后在第0维增加一个维度
            non_pad = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0)
            # 调用_stream方法生成新的token序列
            out = self._stream(non_pad, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)
            # 只保留每个生成步骤的最后一个token
            tokens_list = [tokens[:, -1:] for tokens in out]
            # 将生成的token在最后一个维度拼接起来,形成完整token，如果没有生成则使用原序列
            gen = torch.cat(tokens_list, dim=-1) if tokens_list else non_pad
            # 将non_pad和生成的序列拼接，然后放入generated列表
            full_sequence = torch.cat([non_pad, gen], dim=-1)
            generated.append(full_sequence)
        # 找到最长序列的长度
        max_length = max(seq.size(1) for seq in generated)
        # 对所有序列进行padding,使其长度一致
        generated = [
            torch.cat(#用pad_token_id填充
                [seq, torch.full((1, max_length - seq.size(1)), pad_token_id, dtype=seq.dtype, device=seq.device)],
                dim=-1)
            for seq in generated
        ]
        # 将generated数组的所有序列在batch维度上拼接
        return torch.cat(generated, dim=0)

    def _stream(self, input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args):
        """
        流式生成文本的核心方法
        
        Args:
            input_ids: 输入的token id序列
            eos_token_id: 结束符token的id
            max_new_tokens: 最大生成的新token数量
            temperature: 采样温度,控制生成的随机性
            top_p: 核采样的概率阈值
            rp: 重复惩罚因子
            use_cache: 是否使用KV缓存
            **args: 其他参数
            
        Yields:
            torch.Tensor: 每一步生成的token序列
            
        实现逻辑:
        1. 初始化起始位置和KV缓存
        2. 循环生成,直到达到最大长度或生成结束符:
           - 第一次或不使用缓存时,处理整个序列
           - 后续只处理最新生成的token
           - 对logits进行温度缩放和重复惩罚
           - 使用top_p采样生成下一个token
        """
        start, first_seq, past_kvs = input_ids.shape[1], True, None
        while input_ids.shape[1] < max_new_tokens - 1:#这里的shape[1]表示序列的长度
            # 第一次或不使用缓存时,处理整个序列
            if first_seq or not use_cache:
                out, first_seq = self(input_ids, past_key_values=past_kvs, use_cache=use_cache, **args), False
            else:
                # 只处理最新生成的token
                out = self(input_ids[:, -1:], past_key_values=past_kvs, use_cache=use_cache,
                           start_pos=input_ids.shape[1] - 1, **args)
            logits, past_kvs = out.logits[:, -1, :], out.past_key_values
            # 对已生成的token进行重复惩罚，rp是超参数，[0]表示第一个样本
            logits[:, list(set(input_ids.tolist()[0]))] /= rp
            # 温度缩放
            logits /= (temperature + 1e-9)
            # 使用top_p采样
            if top_p is not None and top_p < 1.0:
                # 对logits进行降序排序，得到排序后的logits和对应的索引
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                # 对排序后的logits进行softmax得到概率分布
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                # 计算累积概率，每个元素表示从第一个 token 到当前 token 的概率总和
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                # 找出累积概率大于top_p的位置
                sorted_indices_to_remove = cumulative_probs > top_p
                # 将需要移除的索引向前移动一位，保留第一个token
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False#把第一个设置为False
                # 将刚才的布尔值映射回原始顺序
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                # 将需要移除的位置的logits设为负无穷
                logits[indices_to_remove] = -float('Inf')
            # 采样生成下一个token
            input_ids_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)#默认不放回采样
            input_ids = torch.cat((input_ids, input_ids_next), dim=1)#连接到input_ids
            yield input_ids[:, start:]#这里start是序列长度，所以这里返回了最新的token
            # 如果生成了结束符则停止
            if input_ids_next.item() == eos_token_id:
                break
