import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadSelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SingleHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        
        # 定义 q, k, v 的线性变换
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        
        # 定义输出的线性变换
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        N, seq_length, _ = x.shape  # N: batch size, seq_length: sequence length

        # 计算 Q, K, V
        values = self.values(x)  # (N, seq_length, embed_size)
        keys = self.keys(x)      # (N, seq_length, embed_size)
        queries = self.queries(x)  # (N, seq_length, embed_size)

        # 计算注意力分数
        energy = torch.einsum("nqd,nkd->nqk", [queries, keys])  # (N, seq_length, seq_length)

        # 计算注意力权重
        attention = F.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)  # (N, seq_length, seq_length)

        # 计算加权值
        out = torch.einsum("nqk,nvd->nqv", [attention, values])  # (N, seq_length, embed_size)

        # 注意力机制的最后 要通过全连接层输出
        out = self.fc_out(out)  # (N, seq_length, embed_size)

        return out

class MultiHeadSelfAttention(nn.Module):#多头自注意力
    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size must be divisible by heads"
# 定义qkv和fc_out的形状
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):#对x进行切分，切分成多个head
        N, seq_length, _ = x.shape  # N: batch size, seq_length: sequence length

        # Split the embedding into multiple heads
        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)
# 把qkv切分成多个head
        values = values.view(N, seq_length, self.heads, self.head_dim)
        keys = keys.view(N, seq_length, self.heads, self.head_dim)
        queries = queries.view(N, seq_length, self.heads, self.head_dim)
# 将值张量的维度顺序调整为 (N, heads, seq_length, head_dim)
        values = values.permute(0, 2, 1, 3)  # (N, heads, seq_length, head_dim)
        keys = keys.permute(0, 2, 1, 3)      # (N, heads, seq_length, head_dim)
        queries = queries.permute(0, 2, 1, 3)  # (N, heads, seq_length, head_dim)

        # energy表示注意力得分
        energy = torch.einsum("nqhd,nkhd->nqk", [queries, keys])  # 最后得到(N, heads, seq_length, seq_length)
        attention = F.softmax(energy / (self.head_dim ** (1 / 2)), dim=2)  # 先对energy进行缩放，再对第二个维度进行softmax

        # Get the weighted values
        out = torch.einsum("nqk,nvhd->nqhd", [attention, values])  # (N, heads, seq_length, head_dim)
        out = out.permute(0, 2, 1, 3).contiguous()  # (N, seq_length, heads, head_dim)
        out = out.view(N, seq_length, self.embed_size)  # (N, seq_length, embed_size)

        return self.fc_out(out)  # (N, seq_length, embed_size)

class FeedForward(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embed_size)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))#对第一层的线性变换进行relu激活，再对第二层的线性变换进行计算

class TransformerBlock(nn.Module):
    """
    Transformer编码器块
    
    包含多头自注意力层和前馈神经网络层，每层后接Layer Normalization和残差连接
    
    参数:
        embed_size (int): 输入的嵌入维度
        heads (int): 注意力头的数量
        hidden_size (int): 前馈神经网络的隐藏层维度
        dropout (float): Dropout的概率，默认为0
    """
    def __init__(self, embed_size, heads, hidden_size, dropout=0):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, heads)  # 多头自注意力层
        self.norm1 = nn.LayerNorm(embed_size)  # 第一个Layer Normalization层
        self.norm2 = nn.LayerNorm(embed_size)  # 第二个Layer Normalization层
        self.feed_forward = FeedForward(embed_size, hidden_size)  # 前馈神经网络层
        self.dropout1 = nn.Dropout(dropout)  # 第一个Dropout层，用于注意力输出
        self.dropout2 = nn.Dropout(dropout)  # 第二个Dropout层，用于前馈网络输出

    def forward(self, x):
        # 每一层都要先dropout再进行Layer Normalization
        attention = self.attention(x)
        x = self.dropout1(attention) + x  # 对输出应用dropout（正则化）之后，连接输出（残差连接）
        x = self.norm1(x)#归一化，用于维持数值稳定性

        forward = self.feed_forward(x)
        x = self.dropout2(forward) + x  # Residual connection
        x = self.norm2(x)

        return x

# 测试 TransformerBlock
if __name__ == "__main__":
    embed_size = 256  # 嵌入维度
    heads = 8  # 注意力头数
    hidden_size = 512  # 前馈网络隐藏层大小
    dropout = 0.1  # Dropout 概率
    seq_length = 10  # 序列长度
    batch_size = 2  # 批量大小

    # 创建一个 TransformerBlock 实例
    transformer_block = TransformerBlock(embed_size, heads, hidden_size, dropout)

    # 创建一个随机输入张量
    x = torch.rand(batch_size, seq_length, embed_size)

    # 前向传播
    out = transformer_block(x)
    print(out.shape)  # 应该输出 (batch_size, seq_length, embed_size)