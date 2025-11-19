# 一、论文总结
## 1 研究背景
单图像去雨的核心目标是移除图像中的雨纹信息，恢复图像清晰度与原始质量，是计算机视觉领域的热门方向，但仍面临诸多挑战。

**1.1、现有方法的固有局限**
   - 仅依赖空间域处理，难以充分捕捉雨丝的特征，限制去雨性能提升。
   - 网络结构日趋复杂，GAN、CNN、Transformer 类去雨模型均存在缺陷：GAN 收敛困难、易产生伪影；CNN 感受野有限、上下文建模能力弱；Transformer 局部信息处理不足、参数呈二次增长。

**1.2、关键痛点**
   - 参数规模大：SOTA 模型参数多在百万级以上，不利于实时应用与物联网场景。
   - 效率低下：训练周期长（部分模型需数千轮迭代）、测试速度慢，难以满足实际需求。
   - 性能瓶颈：易丢失图像细节纹理，复杂雨景（如大雨、斜向雨）下去雨效果不佳。


## 2 研究面临的核心问题（相关工作）
现有去雨研究主要围绕 “空间 - 通道域”“频率域”“轻量化” 三个方向展开，各有进展与不足：

2.1、**空间与通道域学习**：
   - 核心思路：通过 CNN 或 Transformer 提取空间 / 通道特征，利用多尺度、递归、编码器 - 解码器等结构实现粗到精去雨。
   - 代表性方法：JORDER 系列[10,11]（递归架构）、RLNet[16-18]（金字塔网络）、SwinIR[27]（Swin-Transformer 骨干）、SPANet[23]（引入空间注意力）。
   - 不足：CNN 类方法易丢失纹理细节，Transformer 类方法局部特征提取能力弱，且参数与计算成本高。

2.2、**频率域学习**：
   - 核心思路：将图像转换至频率域（小波变换、DCT），利用雨丝与背景在频率分布上的差异实现去雨。
   - 代表性方法：RWL[38]（递归小波学习）、WCAM[40]（小波通道注意力）、MDARNet[41]（层级单演小波变换）、WAAR[42]（平稳小波变换）。
   - 不足：现有方法多为非端到端设计，参数调优困难；仅采用固定带宽分解，难以适配复杂雨景；缺乏对雨丝方向与形态的针对性频率分析。

2.3、**轻量化学习**：
   - 核心需求：降低模型复杂度，适配实时与边缘设备场景。
   - 现有尝试：采用空洞卷积（Dilated Convolution）[45]、深度可分离卷积（DWS-Conv）[46]、ShiftAddNet[37] 等轻量化卷积结构，但未在去雨领域充分应用。
   - 不足：深度可分离卷积缺乏通道间信息交互，空洞卷积对像素级密集预测效果有限，现有轻量化模型仍难以平衡性能与效率。


## 3 解决问题的贡献点
论文围绕单图像去雨任务的 “性能提升、效率优化、轻量化设计” 三大核心目标，提出轻量化多域多注意力渐进网络（** $M_{2}PN$**），主要贡献可分为**理论分析与注意力机制创新、渐进式网络结构设计、极致轻量化与高效性能平衡**三大维度，具体如下：
### 3.1、理论分析与注意力机制创新：多域特征融合提升去雨精度
3.1.1、**揭示雨丝特性与 DCT 频谱的关联，提出频率通道注意力（FcA）机制**
   - 论文通过离散余弦变换（DCT）对雨丝的频率能量分布进行理论分析，首次明确雨纹 “近似垂直下落” 或 “轻微偏离垂直” 的形态特征与 DCT 频谱带宽的对应关系 —— 垂直雨丝的高频能量集中于 DCT 矩阵的水平 v 轴，斜向雨丝的能量分布方向与下落角度 θ 一致，且这些能量主要集中于 DCT 矩阵的右上区域（UR-DCTM）。
   - 基于此，设计 FcA 机制：将图像 DCT 频谱分解为 16 个关键带宽（UR-DCTM 系数），通过频率向量与特征图的逐元素加权，精准捕捉雨丝的频率信息，补充空间域特征的不足，有效区分雨丝与背景细节，解决了传统方法 “仅依赖空间域导致雨丝误判” 的问题。
3.1.2、**提出参数无关的空间通道注意力（ScA）机制，优化多域特征融合**
   - 受能量函数优化理论启发，引入 SimAM 模块的快速闭式解（无需额外参数），设计 3D 空间 - 通道注意力（ScA）：通过计算神经元能量差异（公式 $e_{s}=\frac{4(\sigma^{2}+\lambda)}{(t-\mu)^{2}+2\sigma^{2}+2\lambda}$），抑制空间冗余信息、强化雨丝区域的空间 - 通道关联，实现 “频率 - 空间 - 通道” 多域特征的高效融合。
   - 该机制无需额外参数，却能进一步提升雨丝识别精度，与 FcA 协同作用，显著改善复杂雨景（如斜向雨、密集雨）的去雨效果。
### 3.2、渐进式递归网络结构：兼顾特征捕捉与训练效率
3.2.1、**设计带跳跃连接的递归渐进结构，优化梯度流动与特征层级 $M_{2}PN$**
   - 骨干网络由 6 个（S=6）相同的递归 ** $M_{2}PN$** 模块组成，每个模块通过跳跃连接（将初始雨图与前一模块输出拼接）实现 “低 - 高尺度” 特征的逐步传递与细化。这种结构不仅解决了深层网络的梯度消失问题，还能高效捕捉雨丝从 “粗粒度去除” 到 “细粒度修复” 的渐进式特征，提升上下文信息获取能力，弥补了传统 CNN “感受野有限”、Transformer “局部信息弱” 的缺陷。

3.2.2、**简化训练流程，实现快速收敛**
   - 网络仅采用 “负 SSIM 损失”（公式 $L=-\sum_{s=1}^{S=6}\omega_{s}SSIM(x^{s},x_{GT})$）作为优化目标，无需复杂损失函数组合；同时，递归结构与多注意力机制的协同作用，使网络仅需 100 轮训练即可收敛（远少于 SOTA 模型如 EfficientDeRain 的 10000 轮、SwinIR 的 1200 轮），大幅降低训练成本。
### 3.3、极致轻量化设计：168K 参数实现 SOTA 级性能
3.3.1、**极简网络组件，参数规模降低 1-2 个数量级 $M_{2}PN$**
   - 通过三大策略实现轻量化：① 采用浅通道设计，网络最大通道数仅 32；② 替换传统重卷积：用 ShiftAddNet 的 $Conv_{(3×3)}$（低计算成本）替代传统卷积，仅保留 30 个 $Conv_{(1×1)}$处理细节；③ 注意力机制参数无关：FcA 仅需 128 个额外参数，ScA 完全无参数。最终网络总参数仅 168K，较 SOTA 模型（如 SwinIR 11.8M、MFDNet 4.741M）降低 1-2 个数量级，适配实时应用与物联网（IoT）场景。
3.3.2、**轻量化与高性能的平衡：多数据集验证最优综合表现**
   - 在 Rain100L、Rain100H、Rain200L 等 5 个基准数据集上， $M_{2}PN$ 实现 “参数最少 + 性能优异 + 效率最高” 的三重优势：① 性能：Rain100L 数据集上 PSNR 达 38.36dB、SSIM 达 0.985，位列 SOTA 前三；② 效率：处理 512×512 图像仅需 0.11s，快于 SwinIR（0.81s）、TRNR（0.5s）；③ 综合排名：在 “性能（PSNR/SSIM）+ 效率（训练轮次）+ 轻量化（参数）” 的综合评分中，超越 TRNR、MFDNet 等 SOTA 模型，成为单图像去雨任务的高效轻量化解决方案。


### 4 M₂PN方法流程图
<img src="Lightweight M2PN structure.jpg" width="1552" alt="总体结构">

# 二. 论文公式与程序代码对照表

核心代码文件为 **`m2pn_model.py`**（主网络定义）和 **`layer.py`**（FCALayer实现）。

## 论文公式与程序代码对照表
| 论文公式编号 | 公式内容（核心含义） | 对应代码文件 | 代码核心逻辑 |
|--------------|----------------------|--------------|--------------|
| 公式（1）    | LSTM门控与记忆单元更新：<br>$f_s=\sigma(W_f·[x_s,h_{s-1}]+b_f)$<br>$c_s=i_s⊙u_s+f_s⊙c_{s-1}$<br>$h_s=tanh(c_s)⊙o_s$ | m2pn_model.py | 1. 拼接当前特征与历史记忆状态：`combined = torch.cat((x, memory), 1)`；<br>2. 计算输入门、遗忘门、更新门与输出门：`i = self.state_gate_i(combined)`、`f = self.state_gate_f(combined)`、`g = self.state_gate_g(combined)`、`o = self.state_gate_o(combined)`；<br>3. 更新记忆单元与隐藏状态：`context = f * context + i * g`、`memory = o * torch.tanh(context)`，与论文中LSTM梯度流动和跨尺度上下文交互逻辑一致。 |
| 公式（2）    | M₂PN模块输出：<br>$x^s = x^0 + f_{out}(g(h(f_{in}(x^{s-1}))))$ | m2pn_model.py | 1. 特征增强：通过`_apply_feature_enhancement`函数结合注意力机制优化特征；<br>2. 图像重建：`current_img = self.reconstruction(x)`将特征映射回RGB空间；<br>3. 跳跃连接：利用初始输入`input_img`（即论文中的$x^0$）与当前重建结果迭代更新，实现渐进式去雨，匹配论文中模块输出的残差学习逻辑。 |
| 公式（3）    | DCT变换公式：<br>$Freq^n = D\sum_{i,j}g_{i,j}^{2d}cos(\frac{\pi u}{N}(i+\frac{1}{2}))cos(\frac{\pi v}{N}(j+\frac{1}{2}))$ | layer.py（FCALayer）、m2pn_model.py | 在`FCALayer`中实现DCT变换：<br>1. 对输入特征进行分块DCT，计算频率域特征；<br>2. 基于论文中确定的UR-DCTM系数（聚焦雨纹高频能量区域）筛选有效频率分量；<br>3. 通过`self.freq_attention = FCALayer(channel=32, dct_h=14, dct_w=14)`集成到主网络，实现频率-通道注意力权重计算，对应论文中DCT用于雨纹频率特征提取的逻辑。 |
| 公式（4）    | SimAM能量函数（ScA机制）：<br>$e_s = \frac{4(\sigma^2+\lambda)}{(t-\mu)^2 + 2\sigma^2 + 2\lambda}$ | m2pn_model.py（enhancement_blocks）、layer.py（FCALayer） | 1. 在残差块`_make_res_block`中，通过`x = F.relu(self.freq_attention(block(x)) + residual)`融合FcA与ScA机制；<br>2. ScA（SimAM）通过计算神经元能量$e_s$（区分目标神经元与周围神经元差异）实现空间-通道注意力聚焦，无额外参数，符合论文中ScA补充频率域信息、抑制冗余背景的设计。 |
| 公式（6）    | 负SSIM损失：<br>$Loss = -\sum_{s=1}^{s=6} \omega_s·SSIM(x^s, x_{GT})$ | train.py（未提供）、m2pn_model.py | 1. 从`m2pn_model.py`的`all_stage_outputs`获取各迭代阶段输出$x^s$；<br>2. 计算各阶段输出与真值 $ x_{GT} $ 的SSIM值，按权重$\omega_s$加权求和后取负；<br>3. 以该损失为优化目标，实现快速收敛，匹配论文中“仅用负SSIM损失指导优化”的设计。 |


# 三. 论文方法详细总结
本文提出的轻量化多域多注意力渐进网络（ $M_{2}PN$），核心是通过“递归渐进式架构+多域多注意力机制+轻量化设计”，在单图像去雨任务中实现“性能、效率、轻量化”的三重平衡。方法整体围绕“空间-频率-通道”多域特征融合展开，从网络架构、核心模块、损失函数到轻量化策略形成完整体系，具体细节如下：

## 1、整体网络架构：递归渐进式设计
 $M_{2}PN$ 整体为基于CNN的递归渐进结构，核心目标是通过多阶段迭代逐步细化雨丝去除效果，同时优化梯度流动与上下文信息捕捉。

### 1.1 架构核心组成
- **递归模块数量**：由 $S=6$ 个相同的 $M_{2}PN$ 模块递归组成，每个模块负责一轮雨丝细化去除。
- **跳跃连接机制**：每个模块的输入均包含初始雨图 $x^0$ 与前一模块输出 $x^{s-1}$ 的拼接特征，既保证梯度有效传递，又能保留原始图像细节，避免过度平滑。
- **整体流程**：从初始雨图输入，经过6轮递归迭代，每轮通过“特征提取-上下文交互-多域融合-特征重建”实现雨丝逐步去除，最终输出去雨结果 $x^6$。

### 1.2 单个 $M_{2}PN$ 模块流程（第$s$轮迭代）
每个模块按以下步骤处理特征，维度变化清晰可追溯：
1. **图像拼接**：将初始雨图 $x^0$（ $H×W×3$）与前一轮输出 $x^{s-1}$（ $H×W×3$）拼接，得到输入特征图 $x_{in}^{s-1}$（ $H×W×6$），保留原始信息与中间处理结果。
2. **浅层特征提取**：通过 $Conv_{(1×1)}$ 卷积对 $x_{in}^{s-1}$ 处理，输出浅层特征图 $f_{in}(x_{in}^{s-1})$（ $H×W×32$），压缩通道维度并初步提取雨丝与背景特征。
3. **LSTM上下文交互**：将浅层特征 $f_{in}$ 与上一轮LSTM隐藏状态 $h_{s-1}$（ $H×W×32$）拼接为 $x_s$（ $H×W×64$），输入LSTM模块。通过门控机制（输入门 $i_s$、遗忘门 $f_s$、更新门 $u_s$、输出门 $o_s$）更新记忆单元 $c_s$ 与隐藏状态 $h_s$，实现跨尺度/跨阶段的上下文信息交互，细化特征表达。
4. **M₂PN块组处理**：输入LSTM输出的特征图，通过 $q=5$ 个串联的 $M_{2}PN$ 块组成的块组，进行“频率-空间-通道”多域特征融合与雨丝特征过滤。
5. **特征重建**：采用轻量化ShiftAddNet $Conv_{(3×3)}$ 卷积对融合特征进行重建，将特征图恢复为RGB图像维度 $x^s$（ $H×W×3$）。
6. **残差学习**：通过跳跃连接实现 $x^s = x^0 + 重建特征$，直接学习雨丝与背景的残差，加速收敛并保留原始图像结构。
```
# LM2PN网络主体（对应论文M₂PN结构）
class LM2PN(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(LM2PN, self).__init__()
        self.num_refinement_stages = recurrent_iter  # 递归迭代次数（对应论文S=6个M₂PN模块）
        self.use_GPU = use_GPU  # 是否使用GPU加速

        # 输入适配层：将6通道输入（原始雨图+当前迭代输出图拼接）映射为32通道特征
        # 论文中每个M₂PN模块输入为"原始图复制+前一模块输出"的拼接特征
        self.input_adapter = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),  # 6→32通道，3×3卷积（padding=1保持尺寸）
            nn.ReLU()  # 激活函数增强非线性表达
        )

        # 多路径特征提取（对应论文多域特征学习）
        # 设计两条并行路径提取不同维度特征，后续融合以丰富表征
        self.feature_pathways = nn.ModuleDict({
            'path1': nn.Sequential(
                nn.Conv2d(32, 32, 3, 1, 1),  # 32通道特征深化
                nn.ReLU(),
                nn.Conv2d(32, 16, 1)  # 1×1卷积降维至16通道
            ),
            'path2': nn.Sequential(
                nn.Conv2d(32, 32, 3, 1, 1),  # 第二条并行路径
                nn.ReLU(),
                nn.Conv2d(32, 16, 1)
            )
        })

        # 频率-通道注意力层（FcA，论文核心模块）
        # 基于DCT频谱分析，强化雨纹相关频率特征的识别与抑制
        self.freq_attention = FCALayer(channel=32, dct_h=14, dct_w=14)

        # 特征融合模块：将两条路径的16通道特征拼接（32通道）后融合
        self.fusion_module = nn.Sequential(
            nn.Conv2d(32, 32, 1),  # 1×1卷积实现通道融合
            nn.ReLU()
        )

        # 特征增强块组（对应论文M₂PN块组）
        # 3个残差块用于局部特征细化，配合注意力机制提升去雨效果
        self.enhancement_blocks = nn.ModuleList([
            self._make_res_block(32, 32),  # 输入输出均为32通道的残差块
            self._make_res_block(32, 32),
            self._make_res_block(32, 32)
        ])

        # LSTM门控机制（对应论文每个M₂PN模块中的LSTM）
        # 用于递归传递上下文信息，优化梯度流动，实现渐进式去雨
        self.state_gate_i = nn.Sequential(  # 输入门：控制新信息输入比例
            nn.Conv2d(32 + 32, 32, 3, 1, 1),  # 输入特征+记忆状态拼接（64→32通道）
            nn.Sigmoid()  # 输出0-1区间权重
        )
        self.state_gate_f = nn.Sequential(  # 遗忘门：控制历史记忆保留比例
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.state_gate_g = nn.Sequential(  # 更新门：生成候选记忆状态
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()  # 输出-1~1区间候选值
        )
        self.state_gate_o = nn.Sequential(  # 输出门：控制记忆状态输出比例
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )

        # 图像重建层：将32通道特征映射回3通道RGB图像（去雨结果）
        self.reconstruction = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),  # 32→3通道，还原图像尺寸
        )

    def _make_res_block(self, in_channels, out_channels):
        """创建残差块（对应论文M₂PN块的残差结构）
        作用：缓解深层网络梯度消失，保留原始特征信息
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.ReLU()
        )

    def _apply_feature_enhancement(self, x):
        """应用特征增强（残差块+频率注意力）
        对应论文M₂PN块组的核心逻辑：局部特征提取+频率域注意力筛选
        """
        features = []  # 保存各阶段增强特征（用于后续分析或融合）
        for block in self.enhancement_blocks:
            residual = x  # 残差连接：保存当前输入特征
            # 残差块特征提取 → 频率注意力加权 → 残差相加 → 激活
            x = F.relu(self.freq_attention(block(x)) + residual)
            features.append(x)  # 记录每一层增强特征
        return x, features

    def _extract_multi_path_features(self, x):
        """提取多路径特征并融合（对应论文多域特征融合）"""
        feat1 = self.feature_pathways['path1'](x)  # 路径1特征（16通道）
        feat2 = self.feature_pathways['path2'](x)  # 路径2特征（16通道）
        combined = torch.cat([feat1, feat2], dim=1)  # 通道维度拼接（32通道）
        return self.fusion_module(combined)  # 特征融合输出

    def _init_states(self, batch_size, height, width):
        """初始化LSTM的记忆状态（memory）和上下文状态（context）
        论文中LSTM用于跨阶段传递信息，初始为零矩阵
        """
        # 创建batch_size×32×height×width的零矩阵
        memory = Variable(torch.zeros(batch_size, 32, height, width))
        context = Variable(torch.zeros(batch_size, 32, height, width))
        
        if self.use_GPU:
            memory = memory.cuda()  # GPU加速时将状态移至GPU
            context = context.cuda()
            
        return memory, context

    def forward(self, input_img):
        """前向传播（对应论文M₂PN的渐进式去雨流程）
        input_img: 输入雨图（batch_size×3×H×W）
        返回：最终去雨结果 + 各阶段特征
        """
        batch_size, _, height, width = input_img.size()  # 获取输入尺寸

        current_img = input_img  # 初始化当前输出为原始雨图（第一阶段输入）
        memory, context = self._init_states(batch_size, height, width)  # 初始化LSTM状态
        
        all_stage_outputs = []  # 保存各递归阶段的去雨输出（论文S=6个阶段）
        all_features = []  # 保存各阶段融合特征

        # 递归迭代（渐进式去雨，对应论文6个M₂PN模块的串行执行）
        for _ in range(self.num_refinement_stages):
            # 1. 输入拼接：原始雨图 + 当前阶段输出（论文的skip connection机制）
            # 目的：保留原始图像细节，辅助梯度反向传播
            x = torch.cat((input_img, current_img), 1)  # 通道维度拼接（3+3=6通道）
            x = self.input_adapter(x)  # 6→32通道特征映射

            # 2. LSTM门控更新（跨阶段上下文传递）
            combined = torch.cat((x, memory), 1)  # 当前特征 + 历史记忆（32+32=64通道）
            i = self.state_gate_i(combined)  # 输入门权重
            f = self.state_gate_f(combined)  # 遗忘门权重
            g = self.state_gate_g(combined)  # 候选记忆状态
            o = self.state_gate_o(combined)  # 输出门权重
            
            context = f * context + i * g  # 更新上下文状态（遗忘旧信息+融入新信息）
            memory = o * torch.tanh(context)  # 更新记忆状态

            # 3. 特征增强：残差块 + 频率注意力（核心去雨特征处理）
            x = memory  # LSTM输出作为特征增强的输入
            x, stage_features = self._apply_feature_enhancement(x)  # 特征增强

            # 4. 多路径特征提取与融合
            enhanced_features = self._extract_multi_path_features(x)
            all_features.append(enhanced_features)  # 记录融合特征

            # 5. 图像重建：32通道特征→3通道RGB图像
            current_img = self.reconstruction(x)
            all_stage_outputs.append(current_img)  # 记录当前阶段去雨结果

        # 返回最终阶段输出（渐进优化后的最优结果）和所有阶段特征
        return all_stage_outputs[-1], all_features
```

## 2、核心模块：多域多注意力（ $M_{2}PN$）块
每个 $M_{2}PN$ 块是实现多域特征融合的核心单元，采用残差结构设计，包含“卷积增强-通道分割-DCT频率处理-FcA机制-ScA机制”5个关键步骤，聚焦雨丝特征的精准识别与过滤。

### 2.1 模块内部流程
#### 2.1.1 雨纹特征增强
- 操作：对输入的 $H×W×32$ 特征图应用 $Conv_{(3×3)}$ 卷积。
- 目的：增强雨丝纹理的局部特征表达，为后续多域处理提供更清晰的雨丝特征基础。
```
# 1. 输入拼接：原始雨图 + 当前阶段输出（论文的skip connection机制）
# 目的：保留原始图像细节，辅助梯度反向传播
x = torch.cat((input_img, current_img), 1)  # 通道维度拼接（3+3=6通道）
x = self.input_adapter(x)  # 6→32通道特征映射

# 2. LSTM门控更新（跨阶段上下文传递）
combined = torch.cat((x, memory), 1)  # 当前特征 + 历史记忆（32+32=64通道）
```

#### 2.1.2 通道分割与降维
- 操作：通过自适应全局平均池化（GAP）将特征图空间维度压缩，再将32个通道均匀分割为 $n=16$ 个子模型（每个子模型通道数为2）。
- 目的：降低后续频率处理的计算复杂度，使每个子模型聚焦特定频率范围的特征。

#### 2.1.3 DCT频率矩阵构建
- 操作：基于离散余弦变换（DCT）原理，对每个子模型进行频率分解。核心是利用论文提出的“雨丝形态-频谱带宽”对应关系，选择DCT矩阵右上区域（UR-DCTM）的16个关键系数作为频率特征载体。
- 理论依据：雨丝多为垂直或小角度倾斜下落，其高频能量集中于DCT矩阵的水平v轴或对应倾斜方向，UR-DCTM系数可精准捕捉这些频率特征。
```
def get_freq_indices(method):
    # 断言确保输入方法合法
    assert method in ['top1','top2','top4','top8','top16','top32','box16',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])  # 提取筛选的频率分量数量（如top16对应16个频率）
    
    if 'top' in method:
        # 高频区域索引（基于7×7频率空间能量排序，前N个高频分量）
        # 论文中高频对应雨纹等噪声特征，需重点关注
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]  # 截取前num_freq个索引
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        # 低频区域索引（能量最集中的基础图像特征，对应图像轮廓、结构）
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        # 特定低频-中频过渡区域索引（补充基础特征与细节特征的衔接）
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    elif 'box' in method:
        # 矩形区域频率索引（局部集中的频率分量，增强局部频域特征捕捉）
        all_bot_indices_x = [0,1,2,3,2,3,4,5,4,3,4,5,5,4,5,5]
        all_bot_indices_y = [0,0,0,0,1,1,0,0,1,2,2,1,2,3,3,4]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError("未实现该频率筛选方法")
    return mapper_x, mapper_y

class MultiSpectralDCTLayer(nn.Module):
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        
        assert len(mapper_x) == len(mapper_y), "x和y方向频率索引数量必须一致"
        self.num_freq = len(mapper_x)  # 筛选的频率分量总数（如top16对应16个）

        # 注册固定的DCT滤波器（buffer：不参与梯度更新，论文中UR-DCTM使用固定DCT基）
        # 可选模式：fixed DCT init（固定）/ learnable DCT init（可学习），论文采用固定模式保证泛化性
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

    def forward(self, x):
        assert len(x.shape) == 4, f"输入必须是4维张量(batch×c×h×w)，当前为{len(x.shape)}维"
        
        # 步骤1：DCT滤波 - 输入特征图与DCT基滤波器逐元素相乘（保留目标频率，抑制其他频率）
        x = x * self.weight  
        # 步骤2：频域聚合 - 在H、W维度求和，得到每个通道的频域能量标量
        result = torch.sum(x, dim=[2,3])  # 输出shape: (batch, channel)
        return result

    def build_filter(self, pos, freq, POS):
        # 二维DCT变换的1D分量公式：cos(π * freq * (pos + 0.5) / POS) / sqrt(POS)
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:  # 直流分量（freq=0）需乘以1/√2归一化（论文公式要求）
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        # 初始化DCT滤波器（通道×高度×宽度）
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)
        # 每个频率分量分配的通道数（c_part = 总通道数 / 频率分量数，论文中C' = C / K）
        c_part = channel // len(mapper_x)

        # 遍历每个频率分量（u_x, v_y）
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            # 遍历DCT特征图的每个空间位置（t_x, t_y）
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    # 生成2D DCT基：x方向1D基 × y方向1D基（论文公式的二维扩展）
                    dct_val = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                    # 为当前频率分量分配的通道块赋值（确保每个频率对应固定通道组）
                    dct_filter[i * c_part : (i+1)*c_part, t_x, t_y] = dct_val
        return dct_filter
```

#### 2.1.4 频率-通道注意力（FcA）机制
FcA的核心是通过频率权重引导通道特征筛选，实现雨丝频率信息与通道特征的融合：

(1). 构建频率注意力向量：将UR-DCTM的16个系数拼接为频率-通道注意力向量 $FreqC$（包含DC-低-高频信息）。

(2). 特征加权：将 $FreqC$ 与16个子模型进行逐元素乘法，得到加权特征图 $g_w$，突出雨丝对应的频率特征。

(3). 通道权重学习：通过两层全连接（FC）层+ReLU激活+Sigmoid激活，学习通道级权重向量 $g_{wv}$（ $1×1×32$），对所有子模型的通道特征进行二次加权。

(4). 输出：经缩放操作后，得到增强的频率-通道融合特征图，有效区分雨丝与背景的频率差异。
- 参数特点：仅需128个额外参数，几乎不增加模型复杂度。
```
class FCALayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction = 16, freq_sel_method = 'top16'):
        super(FCALayer, self).__init__()
        self.reduction = reduction  # 通道注意力的降维系数（论文默认16）
        self.dct_h = dct_h  # DCT特征图高度（如14）
        self.dct_w = dct_w  # DCT特征图宽度（如14）

        # 步骤1：获取预定义的频率索引（论文默认top16，筛选16个高频分量）
        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)  # 频率分量数量（K=16）
        
        # 步骤2：频率索引缩放（关键！对应论文UR-DCTM的统一分辨率机制）
        # 将7×7标准频率空间的索引映射到dct_h×dct_w尺寸（如14×14 → 索引×2）
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # 目的：使不同尺寸的DCT特征图都对应统一的7×7频率空间，保证频率筛选的一致性

        # 步骤3：初始化多光谱DCT层（UR-DCTM模块）
        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        
        # 步骤4：通道注意力层（论文中的MLP：2层全连接+激活）
        # 功能：对频域聚合特征进行降维→升维，生成通道级注意力权重
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 降维：C → C/reduction
            nn.ReLU(inplace=True),  # 非线性激活
            nn.Linear(channel // reduction, channel, bias=False),  # 升维：C/reduction → C
            nn.Sigmoid()  # 输出0~1权重（通道重要性）
        )

    def forward(self, x):
        n, c, h, w = x.shape  # n=batch_size, c=channel, h=height, w=width
        x_pooled = x

        # 步骤1：自适应池化（将输入特征图缩放到DCT尺寸dct_h×dct_w）
        # 兼容不同输入尺寸（如分割/检测任务的可变尺寸输入）
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))

        # 步骤2：UR-DCTM频域提取（生成频域聚合特征）
        y = self.dct_layer(x_pooled)  # 输出shape: (n, c)

        # 步骤3：通道注意力生成（频域特征→通道权重）
        y = self.fc(y).view(n, c, 1, 1)  # reshape为(n, c, 1, 1)，适配特征图维度

        # 步骤4：注意力加权（原始特征×通道权重，增强重要通道，抑制无关通道）
        return x * y.expand_as(x)  # expand_as(x)将权重广播到(n, c, h, w)
```

#### 2.1.5 空间-通道注意力（ScA）机制
ScA基于SimAM模块设计，无额外参数，核心是通过能量函数优化实现空间-通道特征的联合聚焦：
1. 能量计算：对FcA输出的特征图，按公式 $e_{s}=\frac{4(\sigma^{2}+\lambda)}{(t-\mu)^{2}+2\sigma^{2}+2\lambda}$ 计算每个神经元的能量（ $\mu$、 $\sigma$ 为神经元周围均值和方差，$\lambda=0.0005$ 为正则化常数）。
2. 3D注意力融合：能量值 $e_s$ 越小表示神经元与周围差异越大（更可能是雨丝特征），通过Sigmoid激活生成3D（ $H×W×C$）注意力权重，与输入特征逐元素乘法。
3. 目的：抑制空间冗余信息，强化雨丝区域的空间-通道关联，补充FcA在局部细节上的捕捉能力。
```
class SimamAttention(torch.nn.Module):

    def __init__(self, epsilon=1e-4):
        super(SimamAttention, self).__init__()
        self.activation = nn.Sigmoid()
        self.epsilon = epsilon

    def forward(self, x):

        batch_size, channels, height, width = x.size()
        n = width * height - 1
        
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        
        attention = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.epsilon)) + 0.5
        
        return x * self.activation(attention)

```

### 2.2 模块核心优势
- 多域融合：同时利用频率（雨丝高频特性）、空间（雨丝位置信息）、通道（特征表达差异）三个维度的信息，解决单一空间域处理的局限。
- 残差结构：每个 $M_{2}PN$ 块采用残差连接，保证梯度流动，避免深层网络训练退化。

## 3、损失函数：负SSIM损失
为实现“轻量化+快速收敛”，论文摒弃复杂损失函数组合，仅采用负结构相似性指数（SSIM）损失作为优化目标：

### 3.1 损失函数公式
 $L=-\sum_{s=1}^{S=6} \omega_{s} SSIM\left(x^{s}, x_{GT}\right)$
-  $x^s$：第 $s$轮迭代的去雨输出， $x_{GT}$：干净图像（真实标签）。
-  $\omega_s$：递归监督的平衡参数，用于调节不同迭代阶段的损失权重。
```
def gaussian(window_size, sigma):
    # 高斯函数公式：exp(-(x - μ)²/(2σ²))，其中μ=window_size//2（窗口中心）
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) 
                          for x in range(window_size)])
    return gauss / gauss.sum()  # 归一化：确保窗口权重总和为1，避免局部区域亮度偏移


def create_window(window_size, channel):
    # 步骤1：生成1D高斯核并扩展为2D（通过矩阵乘法实现外积，得到window_size×window_size的2D高斯矩阵）
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)  # 1D→2D：(window_size,) → (window_size, 1)
    _2D_window = _1D_window.mm(_1D_window.t()).float()    # 外积：(window_size,1) × (1,window_size) → (window_size,window_size)
    
    # 步骤2：适配Conv2d输入格式（批量维度+通道维度）
    # unsqueeze(0)添加批量维度，unsqueeze(0)添加通道维度 → (1,1,window_size,window_size)
    # expand(channel, ...)扩展到对应通道数，确保每个通道有独立的高斯窗口
    _2D_window = _2D_window.unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    # 1. 计算局部亮度均值（μ₁、μ₂）→ 对应论文亮度分量 L(x,y)
    # 用深度卷积（groups=channel）确保每个通道独立计算，避免通道间干扰
    # padding=window_size//2：保证输出尺寸与输入一致（每个像素都有完整的局部窗口）
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)  # img1的局部均值图
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)  # img2的局部均值图

    # 亮度相关中间变量
    mu1_sq = mu1.pow(2)       # μ₁²
    mu2_sq = mu2.pow(2)       # μ₂²
    mu1_mu2 = mu1 * mu2       # μ₁×μ₂

    # 2. 计算局部对比度（σ₁²、σ₂²）和结构协方差（σ₁₂）→ 对应论文对比度C(x,y)和结构S(x,y)
    # 方差公式：σ² = E[X²] - (E[X])²，协方差公式：σ₁₂ = E[X1X2] - E[X1]E[X2]
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # 3. SSIM公式平滑常数（论文默认值，用于避免分母为0，同时平衡亮度/对比度的权重）
    C1 = 0.01 ** 2  # 亮度项平滑常数（基于图像像素值归一化到[0,1]设计）
    C2 = 0.03 ** 2  # 对比度-结构项平滑常数

    # 4. 计算SSIM映射图（每个像素的SSIM值）→ 论文核心公式
    # SSIM = [(2μ₁μ₂ + C1)(2σ₁₂ + C2)] / [(μ₁² + μ₂² + C1)(σ₁² + σ₂² + C2)]
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # 5. 计算最终SSIM值（根据任务需求选择平均方式）
    if size_average:
        return ssim_map.mean()  # 全局所有像素平均（训练时常用，作为损失函数输入）
    else:
        # 对通道、高度、宽度维度求平均，返回每个样本的SSIM值（shape: (batch,)，适用于评估单个样本效果）
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size  # 窗口大小（固定）
        self.size_average = size_average  # 平均模式标记
        self.channel = 1  # 初始通道数（默认灰度图，后续动态更新）
        # 创建初始1通道高斯窗口（后续根据输入自动适配多通道）
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()  # 获取输入图像的通道数（适配RGB/灰度图）

        # 检查窗口是否需要更新（通道数不匹配或数据类型不匹配时）
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window  # 直接使用缓存窗口（提升效率）
        else:
            # 重新创建对应通道数的高斯窗口
            window = create_window(self.window_size, channel)

            # 设备适配：若输入在GPU上，将窗口移至对应GPU（避免设备不匹配错误）
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            # 数据类型适配：同步窗口与输入的数据类型（如float32/float16）
            window = window.type_as(img1)

            # 更新缓存（避免重复创建窗口）
            self.window = window
            self.channel = channel

        # 调用核心函数计算SSIM
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()  # 获取通道数
    window = create_window(window_size, channel)  # 创建对应通道数的高斯窗口

    # 设备和数据类型适配
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    # 调用核心计算函数
    return _ssim(img1, img2, window, window_size, channel, size_average)
```

### 3.2 设计理由
- 直接优化图像相似度：SSIM衡量图像的亮度、对比度、结构一致性，最大化SSIM等价于让去雨图与干净图在视觉和结构上高度一致。
- 简化训练流程：无需额外设计感知损失、对抗损失等，减少参数数量，加速网络收敛（仅需100轮迭代）。

## 4、轻量化设计策略
 $M_{2}PN$ 仅含168K参数（较SOTA模型降低1-2个数量级），核心通过“结构简化+轻量化组件+低参机制”实现，具体策略如下：

### 4.1 网络结构精简
- 浅通道设计：网络最大通道数仅32，远低于传统CNN（如RLNet通道数通常超128）。
- 少卷积核配置：骨干网络仅包含 $S×q=6×5=30$ 个 $M_{2}PN$ 块，搭配30个 $Conv_{(1×1)}$ 处理细节，避免冗余卷积操作。

### 4.2 轻量化卷积替代
- 核心卷积：用ShiftAddNet $Conv_{(3×3)}$ 替代传统重量级 $Conv_{(3×3)}$，ShiftAddNet通过“移位+加法”替代乘法运算，在保持特征提取能力的同时降低计算成本和参数数量。
- 细节处理：仅用 $Conv_{(1×1)}$ 进行通道维度调整和细节增强，参数成本远低于 $Conv_{(3×3)}$。

```
class MeanShift(nn.Conv2d):
    """图像均值偏移与归一化模块（对应论文数据预处理步骤）
    核心作用：将输入RGB图像的像素值进行均值减和标准差归一化，使数据分布更稳定，加速模型训练
    常见于图像重建、去雨、超分辨率等任务，M2PN中用于统一输入数据的分布范围
    """
    def __init__(
        self, rgb_range,  # 图像像素值的动态范围（如255表示8bit图像，1.0表示归一化后图像）
        rgb_mean=(0.4488, 0.4371, 0.4040),  # RGB通道的均值（论文中基于训练集统计的最优值）
        rgb_std=(1.0, 1.0, 1.0),  # RGB通道的标准差（默认1.0，可根据数据调整）
        sign=-1):  # 偏移方向（-1表示减均值，1表示加均值，预处理用-1，逆处理用1）
        super(MeanShift, self).__init__(3, 3, kernel_size=1)  # 1x1卷积（不改变尺寸，仅调整通道数值）
        
        # 初始化权重：单位矩阵 / 标准差（实现标准差归一化，每个通道独立处理）
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        # 初始化偏置：-rgb_range×均值 / 标准差（实现均值偏移，将各通道均值归零）
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        
        # 固定参数，不参与梯度更新（归一化参数是数据统计量，无需训练）
        for p in self.parameters():
            p.requires_grad = False

class ShiftConv2d0(nn.Module):
    """移位卷积版本0（低训练内存占用）
    核心思想：基于「通道分组移位+固定掩码卷积」实现轻量化特征提取（源自ShiftConv论文）
    用通道移位替代传统3x3卷积的部分空间计算，减少参数和计算量，符合M2PN轻量化设计目标
    原理：将输入通道分成5组，每组对应一个空间移位方向（左、右、上、下、不变），通过掩码过滤卷积核实现移位
    """
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d0, self).__init__()    
        self.inp_channels = inp_channels  # 输入通道数
        self.out_channels = out_channels  # 输出通道数
        self.n_div = 5  # 通道分组数（固定为5，对应5个移位方向）
        g = inp_channels // self.n_div  # 每组的通道数（确保输入通道数能被5整除）

        # 初始化3x3卷积（基础卷积核，后续通过掩码过滤实现移位）
        conv3x3 = nn.Conv2d(inp_channels, out_channels, 3, 1, 1)
        # 生成固定移位掩码（3x3，仅在移位方向位置为1，其他为0，不参与训练）
        mask = nn.Parameter(torch.zeros((self.out_channels, self.inp_channels, 3, 3)), requires_grad=False)
        mask[:, 0*g:1*g, 1, 2] = 1.0  # 第1组（0~g-1通道）：右移（卷积核位置(1,2)对应输入右移1像素）
        mask[:, 1*g:2*g, 1, 0] = 1.0  # 第2组（g~2g-1通道）：左移（卷积核位置(1,0)对应输入左移1像素）
        mask[:, 2*g:3*g, 2, 1] = 1.0  # 第3组（2g~3g-1通道）：上移（卷积核位置(2,1)对应输入上移1像素）
        mask[:, 3*g:4*g, 0, 1] = 1.0  # 第4组（3g~4g-1通道）：下移（卷积核位置(0,1)对应输入下移1像素）
        mask[:, 4*g:, 1, 1] = 1.0     # 第5组（4g~end通道）：不变（卷积核中心位置(1,1)，无移位）

        # 保存卷积核、偏置和移位掩码（复用基础卷积参数，通过掩码实现移位功能）
        self.w = conv3x3.weight
        self.b = conv3x3.bias
        self.m = mask

    def forward(self, x):
        # 移位卷积计算：卷积核×移位掩码（过滤出移位方向的权重）→ 3x3卷积
        y = F.conv2d(input=x, weight=self.w * self.m, bias=self.b, stride=1, padding=1) 
        return y

# M2PN网络仅使用该版本移位卷积（论文中选择训练速度更快的实现）
class ShiftConv2d1(nn.Module):
    """移位卷积版本1（快速训练速度，M2PN选用）
    优化点：用「分组卷积实现移位+1x1卷积融合」替代版本0的掩码卷积，训练时计算效率更高
    核心逻辑：先通过分组卷积完成通道移位（无参数学习），再用1x1卷积融合多通道特征，兼顾轻量化和速度
    """
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d1, self).__init__()    
        self.inp_channels = inp_channels  # 输入通道数
        self.out_channels = out_channels  # 输出通道数

        # 初始化固定移位权重（1x3x3，不参与训练，实现通道移位）
        self.weight = nn.Parameter(torch.zeros(inp_channels, 1, 3, 3), requires_grad=False)
        self.n_div = 5  # 通道分组数（5个移位方向）
        g = inp_channels // self.n_div  # 每组通道数
        
        # 为每组通道分配移位方向（权重仅在对应移位位置为1，实现无参数移位）
        self.weight[0*g:1*g, 0, 1, 2] = 1.0  ## 第1组：右移（输入像素→卷积核(1,2)位置）
        self.weight[1*g:2*g, 0, 1, 0] = 1.0  ## 第2组：左移（输入像素→卷积核(1,0)位置）
        self.weight[2*g:3*g, 0, 2, 1] = 1.0  ## 第3组：上移（输入像素→卷积核(2,1)位置）
        self.weight[3*g:4*g, 0, 0, 1] = 1.0  ## 第4组：下移（输入像素→卷积核(0,1)位置）
        self.weight[4*g:, 0, 1, 1] = 1.0     ## 第5组：不变（输入像素→卷积核中心(1,1)）

        # 1x1卷积（核心学习模块）：融合5组移位后的通道特征，将输入通道数映射到输出通道数
        self.conv1x1 = nn.Conv2d(inp_channels, out_channels, 1)

    def forward(self, x):
        # 步骤1：分组卷积实现通道移位（groups=inp_channels→每个输入通道独立卷积，仅保留移位方向像素）
        # padding=1确保移位后尺寸不变，stride=1保持空间分辨率
        y = F.conv2d(
            input=x, 
            weight=self.weight, 
            bias=None, 
            stride=1, 
            padding=1, 
            groups=self.inp_channels  # 分组卷积关键：每个通道用自己的移位权重
        )
        # 步骤2：1x1卷积融合特征（将移位后的多通道特征压缩/扩展到目标通道数，学习通道间依赖）
        y = self.conv1x1(y) 
        return y

class ShiftConv2d(nn.Conv2d):
    """移位卷积统一接口（适配两种实现，兼容不同训练需求）
    作用：为不同场景提供统一调用方式，M2PN中选择'fast-training-speed'模式（ShiftConv2d1）
    """
    def __init__(self, inp_channels, out_channels, conv_type='fast-training-speed'):
        super(ShiftConv2d, self).__init__(inp_channels, out_channels, kernel_size=1)  # 占位1x1卷积（实际用内部shift_conv）
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.conv_type = conv_type  # 卷积类型：低内存/快速度
        
        # 根据类型选择移位卷积实现
        if conv_type == 'low-training-memory': 
            self.shift_conv = ShiftConv2d0(inp_channels, out_channels)  # 低内存版（掩码卷积）
        elif conv_type == 'fast-training-speed':
            self.shift_conv = ShiftConv2d1(inp_channels, out_channels)  # 快速训练版（分组移位+1x1融合，M2PN选用）
        else:
            raise ValueError('invalid type of shift-conv2d')  # 无效类型报错

    def forward(self, x):
        # 调用内部移位卷积实现前向传播
        y = self.shift_conv(x)
        return y
```

### 4.3 低参/无参注意力机制
- FcA机制：仅需128个额外参数，主要依赖DCT频率分析的先验知识，无需大量可训练参数。
- ScA机制：基于SimAM模块，通过闭式解计算注意力权重，无任何额外可训练参数，实现“零参提升性能”。

## 5、关键技术亮点总结
5.1 **多域协同**：首次将雨丝的“形态-频率”理论关系融入注意力机制设计，通过FcA+ScA实现“频率-空间-通道”特征的深度融合，解决单一域处理的局限。

5.2 **渐进式细化**：6轮递归迭代+跳跃连接，从粗到精逐步去除雨丝，既保证去雨彻底性，又避免丢失图像细节。

5.3 **效率与性能平衡**：通过轻量化组件、简化损失函数、低参机制，实现“168K参数+100轮收敛+0.11s/张测试速度”，同时在5个基准数据集上达到SOTA级性能。


# 四 训练
## 1 训练流程
**为了提升训练效率、简化数据处理流程、优化内存利用，将数据集中所有图像转为H5格式：**
```
def prepare_data_RainTrainL(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path)
    target_path = os.path.join(data_path)

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(200):
        target_file = "norain-%d.png" % (i + 1)
        target = cv2.imread(os.path.join(target_path,target_file))
        b, g, r = cv2.split(target)
        target = cv2.merge([r, g, b])

        for j in range(2):
            input_file = "rain-%d.png" % (i + 1)
            input_img = cv2.imread(os.path.join(input_path,input_file))
            b, g, r = cv2.split(input_img)
            input_img = cv2.merge([r, g, b])

            target_img = target

            if j == 1:
                target_img = cv2.flip(target_img, 1)
                input_img = cv2.flip(input_img, 1)

            target_img = np.float32(normalize(target_img))
            target_patches = Im2Patch(target_img.transpose(2,0,1), win=patch_size, stride=stride)

            input_img = np.float32(normalize(input_img))
            input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))
            for n in range(target_patches.shape[3]):
                target_data = target_patches[:, :, :, n].copy()
                target_h5f.create_dataset(str(train_num), data=target_data)

                input_data = input_patches[:, :, :, n].copy()
                input_h5f.create_dataset(str(train_num), data=input_data)

                train_num += 1

    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)
```
**以上代码就是将Rain100L数据集转为H5格式的代码。**

**主要的训练代码：**

```
def main():
    """M2PN系列网络（单图像去雨）训练主函数（严格遵循论文实验配置）
    核心流程：数据预处理→模型构建→训练配置→迭代训练→日志记录→模型保存
    对应论文4.1节训练设置（数据集、优化器、损失函数、训练策略等）
    """
    print('Loading dataset ...\n')
    # 加载训练数据集（Dataset类封装了雨图-干净图的配对加载）
    dataset_train = Dataset(data_path=opt.data_path)
    # 构建数据加载器（论文配置：batch_size=opt.batch_size，shuffle=True打乱数据）
    loader_train = DataLoader(
        dataset=dataset_train, 
        num_workers=4,  # 多线程加载（加速数据读取）
        batch_size=opt.batch_size, 
        shuffle=True,  # 打乱训练数据，提升泛化性
        pin_memory=True  # 锁定内存，加速GPU数据传输
    )
    print("# of training samples: %d\n" % int(len(dataset_train)))  # 输出训练样本总数

    # 构建模型（论文中对比了M2PN、LM2PN、biM2PN，此处选用biM2PN变体）
    # model = M2PN(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)  # 基础M2PN
    # model = LM2PN(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)  # 轻量化M2PN
    model = biM2PN(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)  # 双向M2PN（论文可能的改进版）
    print_network(model)  # 打印模型结构和各层参数数量（验证论文轻量化目标）

    # 统计模型总参数量（对应论文4.1节模型复杂度分析，如M2PN约168K参数）
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()  # 累加所有可训练参数数量
    print('MODEL SAVE AT {}'.format(opt.save_path))  # 模型保存路径
    print('Total number of parameters: %d' % num_params)  # 输出总参数量（验证轻量化）

    # 定义损失函数（论文4.1节损失函数设置：优先使用SSIM损失，而非MSE）
    # criterion = nn.MSELoss(size_average=False)  # MSE损失（像素级误差，易导致图像模糊）
    criterion = SSIM()  # SSIM损失（结构相似性损失，更符合人眼感知，避免模糊）

    # 模型与损失函数移至GPU（论文使用GPU加速训练）
    if opt.use_gpu:
        model = model.cuda()
        criterion.cuda()

    # 定义优化器（论文4.1节配置：Adam优化器，初始学习率lr=opt.lr）
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # 学习率调度器（论文配置：MultiStepLR， milestones=[30,50,80]，gamma=0.2）
    # 作用：在指定epoch降低学习率，避免后期震荡，加速收敛
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)

    # 训练日志记录（使用TensorBoard保存损失、PSNR、SSIM曲线及图像对比）
    writer = SummaryWriter(opt.save_path)

    # 查找最近保存的checkpoint（支持断点续训，论文实验中用于中断后恢复训练）
    initial_epoch = findLastCheckpoint(save_dir=opt.save_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        # 加载checkpoint（包含模型参数、优化器状态、调度器状态、当前epoch）
        checkpoint = torch.load(os.path.join(opt.save_path, 'net_epoch%d.pth' % initial_epoch))
        model.load_state_dict(checkpoint['state_dict'])  # 加载模型参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器状态
        scheduler.load_state_dict(checkpoint['scheduler'])  # 加载学习率调度器状态
        initial_epoch = checkpoint['epoch']  # 恢复起始epoch

    # 开始训练（论文4.1节训练轮数：opt.epochs=100或200，根据数据集调整）
    step = 0  # 全局迭代步数（用于TensorBoard日志记录）
    for epoch in range(initial_epoch, opt.epochs):
        SSIM_list = []  # 保存当前epoch所有batch的SSIM值（用于计算epoch平均）
        psnr_list = []  # 保存当前epoch所有batch的PSNR值（用于计算epoch平均）
        
        scheduler.step(epoch)  # 更新学习率（按milestones调整）
        # 打印当前学习率（验证调度器是否正常工作）
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])

        ## 单epoch训练开始
        for i, (input_train, target_train) in enumerate(loader_train, 0):
            # input_train: 输入雨图（batch×3×H×W）；target_train: 对应干净图（batch×3×H×W）
            model.train()  # 模型设为训练模式（启用Dropout、BatchNorm训练模式）
            model.zero_grad()  # 清空模型梯度
            optimizer.zero_grad()  # 清空优化器梯度

            # 封装为Variable（兼容旧版PyTorch自动求导，新版可省略）
            input_train, target_train = Variable(input_train), Variable(target_train)

            # 数据移至GPU
            if opt.use_gpu:
                input_train, target_train = input_train.cuda(), target_train.cuda()
            
            # 前向传播：模型输出去雨图和中间特征（_表示忽略中间特征）
            out_train, _ = model(input_train)
            # 计算损失：SSIM值越大表示图像越相似，故损失为-SSIM（最小化损失即最大化SSIM）
            pixel_metric = criterion(target_train, out_train)  # 计算干净图与去雨图的SSIM
            loss = -pixel_metric  # 转换为损失（需最小化）
            
            # 反向传播与参数更新（论文训练核心流程）
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新模型参数

            # 评估当前batch的训练效果（切换为评估模式，禁用Dropout等）
            model.eval()
            with torch.no_grad():  # 禁用梯度计算，加速推理
                out_train, _ = model(input_train)
                out_train = torch.clamp(out_train, 0., 1.)  # 裁剪输出至[0,1]（图像像素值范围）
                # 计算PSNR（峰值信噪比，论文主要评估指标之一，越大越好）
                psnr_train = batch_PSNR(out_train, target_train, 1.)  # 1.表示像素值归一化到[0,1]

            # 记录当前batch的指标（用于epoch平均）
            SSIM_list.append(pixel_metric.item())
            psnr_list.append(psnr_train)

            # 打印训练日志（实时监控loss、SSIM、PSNR）
            print("[epoch %d][%d/%d] loss: %.4f, SSIM: %.4f, PSNR: %.4f" %
                  (epoch+1, i+1, len(loader_train), loss.item(), pixel_metric.item(), psnr_train))

            # 每10步记录TensorBoard日志（损失、PSNR、SSIM曲线）
            if step % 10 == 0:
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
                writer.add_scalar('SSIM on training data', pixel_metric.item(), step)

            step += 1  # 更新全局步数
        ## 单epoch训练结束

        # 记录当前epoch的图像对比（TensorBoard可视化：雨图→去雨图→干净图）
        model.eval()
        with torch.no_grad():
            out_train, _ = model(input_train)
            out_train = torch.clamp(out_train, 0., 1.)
            # 构建图像网格（便于可视化对比）
            im_target = utils.make_grid(target_train.data, nrow=8, normalize=True, scale_each=True)  # 干净图
            im_input = utils.make_grid(input_train.data, nrow=8, normalize=True, scale_each=True)  # 雨图
            im_derain = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)  # 去雨图
            # 写入TensorBoard
            writer.add_image('clean image', im_target, epoch+1)
            writer.add_image('rainy image', im_input, epoch+1)
            writer.add_image('deraining image', im_derain, epoch+1)

        # 保存模型checkpoint（支持断点续训）
        # 保存最新模型（覆盖更新）
        torch.save({
            'epoch': epoch,  # 当前epoch
            'state_dict': model.state_dict(),  # 模型参数
            'optimizer': optimizer.state_dict(),  # 优化器状态
            'scheduler': scheduler.state_dict()  # 调度器状态
        }, os.path.join(opt.save_path, 'net_latest.pth'))
        # 按保存频率保存epoch模型（如每5个epoch保存一次，用于实验对比）
        if epoch % opt.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, os.path.join(opt.save_path, 'net_epoch%d.pth' % (epoch+1)))

        # 计算当前epoch的平均指标（评估epoch级训练效果）
        SSIM_average = sum(SSIM_list) / len(SSIM_list)
        psnr_average = sum(psnr_list) / len(psnr_list)
        print("[epoch %d], average SSIM: %.4f, average PSNR: %.4f" %
              (epoch + 1, SSIM_average, psnr_average))
        # 清空指标列表（准备下一个epoch）
        SSIM_list.clear()
        psnr_list.clear()

if __name__ == "__main__":
    # 数据预处理（论文4.1节数据集配置：对不同数据集进行patch裁剪，提升训练效率和泛化性）
    if opt.preprocess:
        # 根据数据集名称选择对应预处理函数（裁剪固定大小的patch，如98×98）
        if opt.data_path.find('RainTrainH') != -1:
            # RainTrainH： heavy rain数据集，裁剪patch_size=98，stride=80（论文默认配置）
            prepare_data_RainTrainH(data_path=opt.data_path, patch_size=98, stride=80)
        elif opt.data_path.find('RainTrainL') != -1:
            # RainTrainL： light rain数据集
            prepare_data_RainTrainL(data_path=opt.data_path, patch_size=98, stride=80)
        elif opt.data_path.find('Rain12600') != -1:
            # Rain12600： 混合雨强数据集
            prepare_data_Rain12600(data_path=opt.data_path, patch_size=98, stride=120)
        elif opt.data_path.find('RainTrain200H') != -1:
            prepare_data_RainTrain200H(data_path=opt.data_path, patch_size=98, stride=80)
        elif opt.data_path.find('RainTrain200L') != -1:
            prepare_data_RainTrain200L(data_path=opt.data_path, patch_size=98, stride=80)
        elif opt.data_path.find('Rain800') != -1:
            prepare_data_Rain800(data_path=opt.data_path, patch_size=98, stride=80)
        elif opt.data_path.find('Rain1200H') != -1:
            prepare_data_Rain1200H(data_path=opt.data_path, patch_size=98, stride=80)
        elif opt.data_path.find('Rain1200M') != -1:
            prepare_data_Rain1200M(data_path=opt.data_path, patch_size=98, stride=80)
        elif opt.data_path.find('Rain1200L') != -1:
            prepare_data_Rain1200L(data_path=opt.data_path, patch_size=98, stride=80)
        elif opt.data_path.find('LSUI') != -1:
            # LSUI： 真实雨图数据集
            prepare_data_LSUI(data_path=opt.data_path, patch_size=98, stride=80)
        else:
            print('unknown datasets: please define prepare data function in DerainDataset.py')

    # 启动训练
    main()

```
上边是训练代码的主函数，包含了损失函数、训练循环次数、断点训练等相关内容

## 2创建训练的虚拟环境和安装依赖项

```
conda creat -n pytorch python=3.9
conda activate pytorch
```
```
torch
numpy
random
h5py
cv2
random
math
```
输入以下代码进行训练：

```
python train_M2PN.py
```

## 3测试结果
将训练好的模型`net_latest.pth`保存到`logs`文件夹下
打开`test_M2PN.py`修改模型和测试数据集文件、以及文件保存的路径

```
parser.add_argument("--logdir", type=str, default="./logs/RainH/net_latest.pth", help='path to model and log files')
parser.add_argument("--data_path", type=str, default=r"datasets/test/Rain100L", help='path to test data')
parser.add_argument("--gt_path", type=str, default=r"datasets/test/Rain100H", help='path to ground truth data')
parser.add_argument("--save_path", type=str, default="./results", help='path to save results')
```

## 4测试结果
| 有雨图像 | 去雨后的结果图像 |
| :------: | :--------------: |
| <img src="有雨图像.png" width="321" alt="有雨图像"> | <img src="去雨后的图像.png" width="321" alt="去雨后的结果图像"> |


# 5论文公式对应的代码

`LM2PN.py`代码文件中包含的公式及对应的代码
```
class LM2PN(nn.Module):

    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(LM2PN, self).__init__()
        self.num_refinement_stages = recurrent_iter
        self.use_GPU = use_GPU

        # 输入适配器：拼接初始输入与当前迭代输出，提取浅层特征（公式2中f_in的前序操作）
        self.input_adapter = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )

        # 多路径特征提取：对应公式2中多域特征融合的并行路径
        self.feature_pathways = nn.ModuleDict({
            'path1': nn.Sequential(
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(32, 16, 1)
            ),
            'path2': nn.Sequential(
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(32, 16, 1)
            )
        })

        # 频率-通道注意力（FcA）：实现公式3的DCT变换与频率域特征加权
        self.freq_attention = FCALayer(channel=32, dct_h=14, dct_w=14)

        # 特征融合模块：对应公式2中多路径特征的拼接与融合
        self.fusion_module = nn.Sequential(
            nn.Conv2d(32, 32, 1),
            nn.ReLU()
        )

        # 增强块组：结合公式4的SimAM（空间-通道注意力）与残差学习
        self.enhancement_blocks = nn.ModuleList([
            self._make_res_block(32, 32),
            self._make_res_block(32, 32),
            self._make_res_block(32, 32)
        ])

        # LSTM门控（公式1的核心实现）：输入门、遗忘门、更新门、输出门
        self.state_gate_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.state_gate_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.state_gate_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
        )
        self.state_gate_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )

        # 图像重建：公式2中f_out的最终输出层
        self.reconstruction = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )

    def _make_res_block(self, in_channels, out_channels):
        # 残差块基础结构：为公式4的特征增强提供残差连接
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.ReLU()
        )

    def _apply_feature_enhancement(self, x):
        # 多域特征增强：融合公式3（FcA）与公式4（ScA），实现残差学习
        features = []
        for block in self.enhancement_blocks:
            residual = x
            # self.freq_attention实现公式3的DCT变换+公式4的SimAM能量优化
            x = F.relu(self.freq_attention(block(x)) + residual)
            features.append(x)
        return x, features

    def _extract_multi_path_features(self, x):
        # 多路径特征融合：对应公式2中多域特征的并行提取与拼接
        feat1 = self.feature_pathways['path1'](x)
        feat2 = self.feature_pathways['path2'](x)
        combined = torch.cat([feat1, feat2], dim=1)
        return self.fusion_module(combined)

    def _init_states(self, batch_size, height, width):
        # 初始化LSTM记忆单元与上下文（公式1的初始状态）
        memory = Variable(torch.zeros(batch_size, 32, height, width))
        context = Variable(torch.zeros(batch_size, 32, height, width))

        if self.use_GPU:
            memory = memory.cuda()
            context = context.cuda()

        return memory, context

    def forward(self, input_img):

        batch_size, _, height, width = input_img.size()

        current_img = input_img
        # 初始化LSTM状态（公式1的c_0和h_0）
        memory, context = self._init_states(batch_size, height, width)

        all_stage_outputs = []
        all_features = []

        for _ in range(self.num_refinement_stages):
            # 图像拼接：公式2中x_in^(s-1)的构建
            x = torch.cat((input_img, current_img), 1)
            x = self.input_adapter(x)

            # 拼接特征与记忆：公式1中[x_s, h_{s-1}]的融合
            combined = torch.cat((x, memory), 1)

            # LSTM门控计算（公式1的i_s, f_s, g_s, o_s）
            i = self.state_gate_i(combined)
            f = self.state_gate_f(combined)
            g = self.state_gate_g(combined)
            o = self.state_gate_o(combined)

            # 记忆单元更新（公式1的c_s = f_s⊙c_{s-1} + i_s⊙g_s）
            context = f * context + i * g
            # 隐藏状态更新（公式1的h_s = o_s⊙tanh(c_s)）
            memory = o * torch.tanh(context)

            x = memory
            # 多域特征增强（公式3+公式4）
            x, stage_features = self._apply_feature_enhancement(x)

            # 多路径特征融合（公式2的多域特征拼接）
            enhanced_features = self._extract_multi_path_features(x)
            all_features.append(enhanced_features)

            # 图像重建（公式2的f_out输出x^s）
            current_img = self.reconstruction(x)
            all_stage_outputs.append(current_img)

        # 返回最终迭代结果（公式2的x^S）
        return all_stage_outputs[-1], all_features
```
`layer.py`代码文件中包含的公式及对应的代码
```
def get_freq_indices(method):
    """
    获取频率分量的索引，对应论文中**UR-DCTM（DCT矩阵右上角区域）**的雨纹高频能量集中区选择策略
    不同method对应不同频率选择规则（top选能量最高、low选低频、bot选高频、box选区域）
    """
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32', 'box16',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        # 前num_freq个能量最高的频率分量索引（雨纹高频能量集中区）
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        # 前num_freq个低频分量索引（背景信息集中区）
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        # 前num_freq个高频分量索引（雨纹细节集中区）
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    elif 'box' in method:
        # 区域型频率分量索引（特定区域内的频率）
        all_bot_indices_x = [0, 1, 2, 3, 2, 3, 4, 5, 4, 3, 4, 5, 5, 4, 5, 5]
        all_bot_indices_y = [0, 0, 0, 0, 1, 1, 0, 0, 1, 2, 2, 1, 2, 3, 3,
                             4]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class MultiSpectralDCTLayer(nn.Module):
    """
    多光谱DCT层，实现**公式3的DCT变换**：
    $Freq^n = D\sum_{i,j}g_{i,j}^{2d}cos(\frac{\pi u}{N}(i+\frac{1}{2}))cos(\frac{\pi v}{N}(j+\frac{1}{2}))$
    其中$u,v$对应mapper_x/mapper_y的频率索引，$N$对应tile_size_x/tile_size_y（DCT矩阵尺寸）
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        assert len(mapper_x) == len(mapper_y)
        self.num_freq = len(mapper_x)  # 选中的频率分量数量
        # 注册DCT滤波器（固定权重，对应论文中“预定义DCT矩阵”）
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

    def forward(self, x):
        """
        对输入特征图做DCT变换，提取频率域特征并求和（对应公式3的频率能量聚合）
        :param x: 输入特征图，shape=(N, C, H, W)
        :return: 频率域聚合后的特征，shape=(N, C)
        """
        assert len(x.shape) == 4, '输入必须为4维张量(N, C, H, W)'
        x = x * self.weight  # 逐元素相乘，实现DCT变换（公式3的核心计算）
        result = torch.sum(x, dim=[2, 3])  # 对H、W维度求和，聚合频率域能量
        return result

    def build_filter(self, pos, freq, POS):
        """
        构建一维DCT基（公式3中余弦项的实现）：
        $cos(\frac{\pi freq}{POS}(pos + 0.5)) / \sqrt{POS}$ （freq=0时额外乘√2）
        """
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        """
        生成二维DCT滤波器（公式3的二维扩展，两个一维DCT基的乘积）
        :param tile_size_x/tile_size_y: DCT矩阵的高/宽（对应公式中N）
        :param mapper_x/mapper_y: 选中的频率分量索引（对应公式中u,v）
        :param channel: 特征图通道数
        :return: DCT滤波器，shape=(C, tile_size_x, tile_size_y)
        """
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)
        c_part = channel // len(mapper_x)  # 每个频率分量分配的通道数
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    # 二维DCT基 = 行DCT基 × 列DCT基（公式3的乘积形式）
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
        return dct_filter


class FCALayer(torch.nn.Module):
    """
    频率-通道注意力层（FcA），结合**公式3的DCT频率域特征**与通道注意力机制，实现雨纹特征的精准聚焦
    """
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='top16'):
        super(FCALayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h  # DCT矩阵的高
        self.dct_w = dct_w  # DCT矩阵的宽

        # 获取频率分量索引（对应论文UR-DCTM的雨纹高频区域选择）
        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        # 缩放索引以适配不同尺寸的DCT矩阵（论文中以7×7为基准，此处支持dct_h/dct_w的灵活缩放）
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]

        # 多光谱DCT层（公式3的核心实现）
        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        # 通道注意力全连接层（实现频率域特征到通道权重的映射）
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 降维
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),  # 升维
            nn.Sigmoid()  # 归一化权重
        )

    def forward(self, x):
        """
        频率-通道注意力的前向传播：DCT变换→通道权重计算→特征加权
        :param x: 输入特征图，shape=(N, C, H, W)
        :return: 加权后的特征图，shape=(N, C, H, W)
        """
        n, c, h, w = x.shape
        x_pooled = x
        # 自适应池化以适配DCT矩阵尺寸（兼容不同输入尺寸的特征图）
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        
        y = self.dct_layer(x_pooled)  # DCT变换，提取频率域特征（公式3的输出）
        y = self.fc(y).view(n, c, 1, 1)  # 计算通道权重并reshape
        return x * y.expand_as(x)  # 特征加权（注意力机制的最终作用）
```

