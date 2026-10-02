"""用城市与粒球的 5 维特征计算 POMO 每条路线的下一城市概率。"""

from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor, nn

from .granular_ball import GranularBall
from .tsp import City


def build_city_features(
    cities: list[City], balls: list[GranularBall], mapping: list[int]
) -> Tensor:
    """把普通 Python 城市/粒球对象变成神经网络可读取的特征 Tensor。

    mapping[城市编号] 给出所属粒球下标。输出 [1, 城市数, 5]：
    第 1 维是一个 TSP 算例，第 2 维下标就是城市编号；
    五列依次为城市 x/y、所属球的中心 x/y、所属球半径。
    城市和球心共用同一个原点与尺度，半径只除以尺度，
    因此相对位置和 x/y 比例不会因归一化而改变。
    这个函数只整理数值，没有可训练参数，也不计算下一城市概率。
    """
    city_count = len(cities)
    if city_count == 0:
        raise ValueError("城市列表不能为空")
    if [city.number for city in cities] != list(range(city_count)):
        raise ValueError("城市编号必须从 0 开始，且与列表位置一致")
    if len(mapping) != city_count or any(
        ball_id < 0 or ball_id >= len(balls) for ball_id in mapping
    ):
        raise ValueError("mapping 必须为每个城市指定有效的粒球下标")

    """
        例如取得城市 i 的球，就是 balls[mapping[i]]。
        三组数归一化后，用 torch.cat(..., dim=-1) 横向拼起来：
        城市 i：[城市x, 城市y, 球心x, 球心y, 半径]
    """

    city_xy = torch.tensor([[city.x, city.y] for city in cities], dtype=torch.float32)
    ball_xy = torch.tensor(
        [[balls[mapping[city.number]].center.x, balls[mapping[city.number]].center.y]
         for city in cities],
        dtype=torch.float32,
    )
    radius = torch.tensor(
        [[balls[mapping[city.number]].radius] for city in cities],
        dtype=torch.float32,
    )
    # 用整张地图的城市坐标确定原点和统一尺度；同一球的信息也用这个尺度。
    origin = city_xy.amin(dim=0)
    scale = (city_xy.amax(dim=0) - origin).amax().clamp_min(1.0)
    features = torch.cat(
        ((city_xy - origin) / scale, (ball_xy - origin) / scale, radius / scale),
        dim=-1,
    )

    return features.unsqueeze(0)  # 当前一次输入一个算例，仍保留 batch 维。


class EncoderLayer(nn.Module):
    """Encoder 的一层：城市间注意力 → 残差/归一化 → 前馈层 → 残差/归一化。

    输入和输出都是 [B, N, D]。这一层只更新城市表示，不选择路线。
    """

    def __init__(self, embedding_dim: int = 128, head_num: int = 8) -> None:
        """创建本层需要学习参数的多头注意力、前馈层和归一化层。"""
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embedding_dim, head_num, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )
        self.feed_forward_norm = nn.LayerNorm(embedding_dim)

    def forward(self, nodes: Tensor) -> Tensor:
        """让每个城市与全部城市交换信息，返回形状不变的新表示。"""
        # Q/K/V 都来自全部城市；城市之间的关系在这里被编码。
        attended, _ = self.self_attention(
            nodes, nodes, nodes, need_weights=False
        )
        nodes = self.attention_norm(nodes + attended)
        return self.feed_forward_norm(nodes + self.feed_forward(nodes))


class CityEncoder(nn.Module):
    """5 维输入先映射到 128 维，再经过三层全城市自注意力。

    Encoder：
    输入的是5就是城市的特征，但是这 5 个数字表达能力比较有限，神经网络会先通过一个可学习的 Linear：
    把 5 维特征映射到 128 维 embedding；这个Embedding 理解成：神经网络自己学习出来的“城市内部表示”。
    这个128维的意思是：一个城市 → 用128个数字来描述，我们用 128，主要因为这个规模对 POMO/Attention 类模型很常见
    然后现在每个城市都是128维，Self-Attention 就开始让城市之间互相看。
    每个城市的 128维向量会分别生成：Q，K，V
        城市3 embedding
              ↓
         ┌────┼────┐
         ↓    ↓    ↓
        Q3   K3   V3
        然后 Q3 会去和：
        K1 K2 K3 K4 K5比较
        Attention 算出来的其实是在问：城市3现在应该关注其他哪些城市？
        可能结果是：
        城市3关注：
        城市1  10%
        城市2  40%
        城市5  30%
        其他   20%
        然后把这些城市的 V 按权重汇总。
    Self-Attention
    城市3 contextual embedding
    = 城市3自己
    + 城市3与整个城市集合的关系

    如果只有一个 Attention，它可能只能比较单一的一套关系。
    所以通常同时搞多个 Head。
           128维
            ↓
        分成8个 Attention Head
            ↓
        每个 Head 处理16维
    训练过程中它们自己学习不同关系。
    """

    def __init__(
        self,
        feature_dim: int = 5,
        embedding_dim: int = 128,
        head_num: int = 8,
        encoder_layer_num: int = 3,
    ) -> None:
        """建立 5→D 的特征映射，再按 encoder_layer_num 堆叠 EncoderLayer。"""
        super().__init__()
        if feature_dim != 5:
            raise ValueError("当前城市输入固定为 5 维")
        if embedding_dim < 1 or head_num < 1 or embedding_dim % head_num:
            raise ValueError("embedding_dim 必须为正数且可被 head_num 整除")
        if encoder_layer_num < 1:
            raise ValueError("encoder_layer_num 必须为正数")
        self.embedding = nn.Linear(feature_dim, embedding_dim)
        self.layers = nn.ModuleList(
            EncoderLayer(embedding_dim, head_num)
            for _ in range(encoder_layer_num)
        )

    def forward(self, city_features: Tensor) -> tuple[Tensor, Tensor]:
        """输入 [B,N,5]，同时返回初始 embedding 和多层处理后的 encoded_nodes。

        两者都是 [B,N,D]。rollout 会缓存 encoded_nodes，后续每一步直接给 Decoder。
        """
        if city_features.ndim != 3 or city_features.shape[-1] != 5:
            raise ValueError("city_features 形状必须是 [B, N, 5]")
        """
           self.embedding 是 nn.Linear(5,128)：
           
           每个城市原来有 5 个数，经过可训练的计算后变成 128 个数。因此 [1,9,5] → [1,9,128]。
        """
        embedded_nodes = self.embedding(city_features)

        encoded_nodes = embedded_nodes
        for layer in self.layers:
            encoded_nodes = layer(encoded_nodes)
        return embedded_nodes, encoded_nodes


class POMODecoder(nn.Module):
    """按每条路线的起点、当前位置和候选 Mask 计算下一城市概率。

    Decoder：
    例如 POMO 某条路线：0 → 3 → 5

    起点 = 0
    当前城市 = 5
    Encoder 已经提前得到：
    城市0 → encoded_0
    城市1 → encoded_1
    ...
    城市5 → encoded_5
    每一个都是：128维
    Decoder 就拿：
    起点 embedding：encoded_0
    当前城市 embedding：encoded_5
    构造当前的 Query。“我从 0 出发，现在在 5，根据整张地图的结构，我接下来应该去哪？”
    然后用 Query 去和 Encoder 的每个城市的 K 做比较，得到每个城市的分数。
    最后结合候选城市，得出指定候选城市的score

    当前实现比上面的概念说明多一步：先做多头注意力得到 context，
    再由 context 与所有城市做最终单头打分；候选 Mask 放在 Softmax 前。
    """

    def __init__(
        self, embedding_dim: int = 128, head_num: int = 8, logit_clip: float = 10.0
    ) -> None:
        """建立路线 Query、多头注意力和最终城市打分所需的网络层。"""
        super().__init__()
        if embedding_dim < 1 or head_num < 1 or embedding_dim % head_num:
            raise ValueError("embedding_dim 必须为正数且可被 head_num 整除")
        if logit_clip <= 0:
            raise ValueError("logit_clip 必须为正数")
        self.embedding_dim = embedding_dim
        self.logit_clip = logit_clip
        # 拼接保留了起点和当前城市各自的信息，再投影成一个 Query。
        self.query_projection = nn.Linear(2 * embedding_dim, embedding_dim)
        self.cross_attention = nn.MultiheadAttention(
            embedding_dim, head_num, batch_first=True
        )
        self.final_query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.final_key = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(
        self,
        encoded_nodes: Tensor,
        first_cities: Tensor,
        current_cities: Tensor,
        candidate_mask: Tensor,
    ) -> Tensor:
        """为每条 POMO 路线计算一次下一城市的概率。

        encoded_nodes [B,N,D]：Encoder 算好的全部城市表示；
        first_cities/current_cities [B,P]：P 条路线各自的起点和当前位置；
        candidate_mask [B,P,N]：True 的城市才允许选。
        返回 [B,P,N]：最后一维下标是城市编号，非候选概率为 0。
        """
        if encoded_nodes.ndim != 3 or encoded_nodes.shape[-1] != self.embedding_dim:
            raise ValueError("encoded_nodes 形状必须是 [B, N, embedding_dim]")
        batch_size, city_count, embedding_dim = encoded_nodes.shape
        if first_cities.ndim != 2 or current_cities.shape != first_cities.shape:
            raise ValueError("first_cities 和 current_cities 形状必须相同，均为 [B, P]")
        pomo_size = first_cities.shape[1]
        if first_cities.shape[0] != batch_size or candidate_mask.shape != (
            batch_size, pomo_size, city_count
        ) or candidate_mask.dtype != torch.bool:
            raise ValueError("candidate_mask 形状必须是 [B, P, N] 且类型为 bool")
        if not candidate_mask.any(dim=-1).all().item():
            raise ValueError("每条路线至少需要一个候选城市")
        for city_ids in (first_cities, current_cities):
            if ((city_ids < 0) | (city_ids >= city_count)).any().item():
                raise ValueError("起点或当前城市编号超出范围")

        # gather 按每条路线的城市编号，从同一份 encoded_nodes 中取出对应向量。
        first = encoded_nodes.gather(
            1, first_cities.unsqueeze(-1).expand(-1, -1, embedding_dim)
        )
        current = encoded_nodes.gather(
            1, current_cities.unsqueeze(-1).expand(-1, -1, embedding_dim)
        )
        query = self.query_projection(torch.cat((first, current), dim=-1))
        # 每条路线的 Query 与全体城市的 Key/Value 交互，形成当前路线 context。
        context, _ = self.cross_attention(
            query, encoded_nodes, encoded_nodes, need_weights=False
        )
        # 最终单头分数的最后一维始终对应城市编号。
        scores = torch.matmul(
            self.final_query(context), self.final_key(encoded_nodes).transpose(1, 2)
        ) / sqrt(embedding_dim)
        # 限制分数的绝对值，再屏蔽非候选城市，最后只在候选城市间分配概率。
        scores = self.logit_clip * torch.tanh(scores)
        scores = scores.masked_fill(~candidate_mask, float("-inf"))
        return torch.softmax(scores, dim=-1)


class POMOPolicy(nn.Module):
    """把 Encoder 和 Decoder 组成一套共享的神经网络参数。

    POMO 虽有 N 条路线，也只有这一套 Policy；每条路线用自己的状态调用 Decoder。
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        head_num: int = 8,
        encoder_layer_num: int = 3,
    ) -> None:
        """创建城市 Encoder 和路线 Decoder，默认 D=128、8 个头、3 层 Encoder。"""
        super().__init__()
        self.encoder = CityEncoder(5, embedding_dim, head_num, encoder_layer_num)
        self.decoder = POMODecoder(embedding_dim, head_num)

    def encode(self, city_features: Tensor) -> tuple[Tensor, Tensor]:
        """路线开始前调用一次，得到初始 embedding 与可复用的 encoded_nodes。"""
        return self.encoder(city_features)

    def forward(
        self,
        encoded_nodes: Tensor,
        first_cities: Tensor,
        current_cities: Tensor,
        candidate_mask: Tensor,
    ) -> Tensor:
        """在某一步调用 Decoder，用各路线状态和候选 Mask 得到下一城市概率。"""
        return self.decoder(
            encoded_nodes, first_cities, current_cities, candidate_mask
        )
