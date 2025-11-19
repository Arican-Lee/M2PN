# 1、论文总结
## 1.1 研究背景
单图像去雨的核心目标是移除图像中的雨纹信息，恢复图像清晰度与原始质量，是计算机视觉领域的热门方向，但仍面临诸多挑战。

**(1)、现有方法的固有局限**
   - 仅依赖空间域处理，难以充分捕捉雨丝的特征，限制去雨性能提升。
   - 网络结构日趋复杂，GAN、CNN、Transformer 类去雨模型均存在缺陷：GAN 收敛困难、易产生伪影；CNN 感受野有限、上下文建模能力弱；Transformer 局部信息处理不足、参数呈二次增长。

**(2)、关键痛点**
   - 参数规模大：SOTA 模型参数多在百万级以上，不利于实时应用与物联网场景。
   - 效率低下：训练周期长（部分模型需数千轮迭代）、测试速度慢，难以满足实际需求。
   - 性能瓶颈：易丢失图像细节纹理，复杂雨景（如大雨、斜向雨）下去雨效果不佳。


## 1.2 研究面临的核心问题（相关工作）
现有去雨研究主要围绕 “空间 - 通道域”“频率域”“轻量化” 三个方向展开，各有进展与不足：

(1)、**空间与通道域学习**：
   - 核心思路：通过 CNN 或 Transformer 提取空间 / 通道特征，利用多尺度、递归、编码器 - 解码器等结构实现粗到精去雨。
   - 代表性方法：JORDER 系列[10,11]（递归架构）、RLNet[16-18]（金字塔网络）、SwinIR[27]（Swin-Transformer 骨干）、SPANet[23]（引入空间注意力）。
   - 不足：CNN 类方法易丢失纹理细节，Transformer 类方法局部特征提取能力弱，且参数与计算成本高。

(2)、**频率域学习**：
   - 核心思路：将图像转换至频率域（小波变换、DCT），利用雨丝与背景在频率分布上的差异实现去雨。
   - 代表性方法：RWL[38]（递归小波学习）、WCAM[40]（小波通道注意力）、MDARNet[41]（层级单演小波变换）、WAAR[42]（平稳小波变换）。
   - 不足：现有方法多为非端到端设计，参数调优困难；仅采用固定带宽分解，难以适配复杂雨景；缺乏对雨丝方向与形态的针对性频率分析。

(3)、**轻量化学习**：
   - 核心需求：降低模型复杂度，适配实时与边缘设备场景。
   - 现有尝试：采用空洞卷积（Dilated Convolution）[45]、深度可分离卷积（DWS-Conv）[46]、ShiftAddNet[37] 等轻量化卷积结构，但未在去雨领域充分应用。
   - 不足：深度可分离卷积缺乏通道间信息交互，空洞卷积对像素级密集预测效果有限，现有轻量化模型仍难以平衡性能与效率。


## 1.3 解决问题的贡献点
论文围绕单图像去雨任务的 “性能提升、效率优化、轻量化设计” 三大核心目标，提出轻量化多域多注意力渐进网络（** $M_{2}PN$**），主要贡献可分为**理论分析与注意力机制创新、渐进式网络结构设计、极致轻量化与高效性能平衡**三大维度，具体如下：
### (一)、理论分析与注意力机制创新：多域特征融合提升去雨精度
(1)、**揭示雨丝特性与 DCT 频谱的关联，提出频率通道注意力（FcA）机制**
   - 论文通过离散余弦变换（DCT）对雨丝的频率能量分布进行理论分析，首次明确雨纹 “近似垂直下落” 或 “轻微偏离垂直” 的形态特征与 DCT 频谱带宽的对应关系 —— 垂直雨丝的高频能量集中于 DCT 矩阵的水平 v 轴，斜向雨丝的能量分布方向与下落角度 θ 一致，且这些能量主要集中于 DCT 矩阵的右上区域（UR-DCTM）。
   - 基于此，设计 FcA 机制：将图像 DCT 频谱分解为 16 个关键带宽（UR-DCTM 系数），通过频率向量与特征图的逐元素加权，精准捕捉雨丝的频率信息，补充空间域特征的不足，有效区分雨丝与背景细节，解决了传统方法 “仅依赖空间域导致雨丝误判” 的问题。
(2)、**提出参数无关的空间通道注意力（ScA）机制，优化多域特征融合**
   - 受能量函数优化理论启发，引入 SimAM 模块的快速闭式解（无需额外参数），设计 3D 空间 - 通道注意力（ScA）：通过计算神经元能量差异（公式 $e_{s}=\frac{4(\sigma^{2}+\lambda)}{(t-\mu)^{2}+2\sigma^{2}+2\lambda}$），抑制空间冗余信息、强化雨丝区域的空间 - 通道关联，实现 “频率 - 空间 - 通道” 多域特征的高效融合。
   - 该机制无需额外参数，却能进一步提升雨丝识别精度，与 FcA 协同作用，显著改善复杂雨景（如斜向雨、密集雨）的去雨效果。
### (二)、渐进式递归网络结构：兼顾特征捕捉与训练效率
(1)、**设计带跳跃连接的递归渐进结构，优化梯度流动与特征层级 $M_{2}PN$**
   - 骨干网络由 6 个（S=6）相同的递归 ** $M_{2}PN$** 模块组成，每个模块通过跳跃连接（将初始雨图与前一模块输出拼接）实现 “低 - 高尺度” 特征的逐步传递与细化。这种结构不仅解决了深层网络的梯度消失问题，还能高效捕捉雨丝从 “粗粒度去除” 到 “细粒度修复” 的渐进式特征，提升上下文信息获取能力，弥补了传统 CNN “感受野有限”、Transformer “局部信息弱” 的缺陷。

(2)、**简化训练流程，实现快速收敛**
   - 网络仅采用 “负 SSIM 损失”（公式 $L=-\sum_{s=1}^{S=6}\omega_{s}SSIM(x^{s},x_{GT})$）作为优化目标，无需复杂损失函数组合；同时，递归结构与多注意力机制的协同作用，使网络仅需 100 轮训练即可收敛（远少于 SOTA 模型如 EfficientDeRain 的 10000 轮、SwinIR 的 1200 轮），大幅降低训练成本。
### (三)、极致轻量化设计：168K 参数实现 SOTA 级性能
(1)、**极简网络组件，参数规模降低 1-2 个数量级 $M_{2}PN$**
   - 通过三大策略实现轻量化：① 采用浅通道设计，网络最大通道数仅 32；② 替换传统重卷积：用 ShiftAddNet 的 $Conv_{(3×3)}$（低计算成本）替代传统卷积，仅保留 30 个 $Conv_{(1×1)}$处理细节；③ 注意力机制参数无关：FcA 仅需 128 个额外参数，ScA 完全无参数。最终网络总参数仅 168K，较 SOTA 模型（如 SwinIR 11.8M、MFDNet 4.741M）降低 1-2 个数量级，适配实时应用与物联网（IoT）场景。
(2)、**轻量化与高性能的平衡：多数据集验证最优综合表现**
   - 在 Rain100L、Rain100H、Rain200L 等 5 个基准数据集上， $M_{2}PN$ 实现 “参数最少 + 性能优异 + 效率最高” 的三重优势：① 性能：Rain100L 数据集上 PSNR 达 38.36dB、SSIM 达 0.985，位列 SOTA 前三；② 效率：处理 512×512 图像仅需 0.11s，快于 SwinIR（0.81s）、TRNR（0.5s）；③ 综合排名：在 “性能（PSNR/SSIM）+ 效率（训练轮次）+ 轻量化（参数）” 的综合评分中，超越 TRNR、MFDNet 等 SOTA 模型，成为单图像去雨任务的高效轻量化解决方案。


### 1.4 M₂PN方法流程图


<img src="Lightweight M2PN structure.jpg" width="1552" alt="总体结构">

# 2. 论文公式与程序代码对照表

假设核心代码文件为 **`m2pn_model.py`**（主网络定义）和 **`layer.py`**（FCALayer实现）。

## 论文公式与程序代码对照表
| 论文公式编号 | 公式内容（核心含义） | 对应代码文件 | 代码核心逻辑 |
|--------------|----------------------|--------------|--------------|
| 公式（1）    | LSTM门控与记忆单元更新：<br>$f_s=\sigma(W_f·[x_s,h_{s-1}]+b_f)$<br>$c_s=i_s⊙u_s+f_s⊙c_{s-1}$<br>$h_s=tanh(c_s)⊙o_s$ | m2pn_model.py | 1. 拼接当前特征与历史记忆状态：`combined = torch.cat((x, memory), 1)`；<br>2. 计算输入门、遗忘门、更新门与输出门：`i = self.state_gate_i(combined)`、`f = self.state_gate_f(combined)`、`g = self.state_gate_g(combined)`、`o = self.state_gate_o(combined)`；<br>3. 更新记忆单元与隐藏状态：`context = f * context + i * g`、`memory = o * torch.tanh(context)`，与论文中LSTM梯度流动和跨尺度上下文交互逻辑一致。 |
| 公式（2）    | M₂PN模块输出：<br>$x^s = x^0 + f_{out}(g(h(f_{in}(x^{s-1}))))$ | m2pn_model.py | 1. 特征增强：通过`_apply_feature_enhancement`函数结合注意力机制优化特征；<br>2. 图像重建：`current_img = self.reconstruction(x)`将特征映射回RGB空间；<br>3. 跳跃连接：利用初始输入`input_img`（即论文中的$x^0$）与当前重建结果迭代更新，实现渐进式去雨，匹配论文中模块输出的残差学习逻辑。 |
| 公式（3）    | DCT变换公式：<br>$Freq^n = D\sum_{i,j}g_{i,j}^{2d}cos(\frac{\pi u}{N}(i+\frac{1}{2}))cos(\frac{\pi v}{N}(j+\frac{1}{2}))$ | layer.py（FCALayer）、m2pn_model.py | 在`FCALayer`中实现DCT变换：<br>1. 对输入特征进行分块DCT，计算频率域特征；<br>2. 基于论文中确定的UR-DCTM系数（聚焦雨纹高频能量区域）筛选有效频率分量；<br>3. 通过`self.freq_attention = FCALayer(channel=32, dct_h=14, dct_w=14)`集成到主网络，实现频率-通道注意力权重计算，对应论文中DCT用于雨纹频率特征提取的逻辑。 |
| 公式（4）    | SimAM能量函数（ScA机制）：<br>$e_s = \frac{4(\sigma^2+\lambda)}{(t-\mu)^2 + 2\sigma^2 + 2\lambda}$ | m2pn_model.py（enhancement_blocks）、layer.py（FCALayer） | 1. 在残差块`_make_res_block`中，通过`x = F.relu(self.freq_attention(block(x)) + residual)`融合FcA与ScA机制；<br>2. ScA（SimAM）通过计算神经元能量$e_s$（区分目标神经元与周围神经元差异）实现空间-通道注意力聚焦，无额外参数，符合论文中ScA补充频率域信息、抑制冗余背景的设计。 |
| 公式（6）    | 负SSIM损失：<br>$Loss = -\sum_{s=1}^{s=6} \omega_s·SSIM(x^s, x_{GT})$ | train.py（未提供）、m2pn_model.py | 1. 从`m2pn_model.py`的`all_stage_outputs`获取各迭代阶段输出$x^s$；<br>2. 计算各阶段输出与真值 $ x_{GT} $ 的SSIM值，按权重$\omega_s$加权求和后取负；<br>3. 以该损失为优化目标，实现快速收敛，匹配论文中“仅用负SSIM损失指导优化”的设计。 |

# 3 训练
## 3.1 训练流程
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
    # -------------------------- 1. 加载训练数据集 --------------------------
    print('Loading dataset ...\n')
    # 初始化训练数据集（opt.data_path为配置的数据集路径）
    dataset_train = Dataset(data_path=opt.data_path)
    # 构建数据加载器：num_workers=4（4个线程加载数据）、batch_size=配置值、打乱数据、内存锁定加速
    loader_train = DataLoader(
        dataset=dataset_train, 
        num_workers=4, 
        batch_size=opt.batch_size, 
        shuffle=True, 
        pin_memory=True
    )
    # 打印训练样本总数
    print("# of training samples: %d\n" % int(len(dataset_train)))

    # -------------------------- 2. 构建模型并统计参数 --------------------------
    # 构建模型：选择双向M₂PN（biM2PN），recurrent_iter=配置的递归迭代次数（论文中S=6），use_GPU=是否用GPU
    # （注释掉的M2PN、LM2PN为其他变体，可根据实验需求切换）
    # model = M2PN(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    # model = LM2PN(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    model = biM2PN(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    # 打印网络结构详情（自定义工具函数，如层名称、输出维度等）
    print_network(model)
    # 统计模型总参数量（论文中M₂PN参数量为168K，此处用于验证）
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()  # 累加每个参数的元素数量
    # 打印模型保存路径和总参数量
    print('MODEL SAVE AT {}'.format(opt.save_path))
    print('Total number of parameters: %d' % num_params)

    # -------------------------- 3. 定义损失函数 --------------------------
    # 选择损失函数：论文中用负SSIM损失（因SSIM越大表示图像越相似，最小化负SSIM等价于最大化SSIM）
    # （注释掉的MSELoss为均方误差损失，是备选方案）
    # criterion = nn.MSELoss(size_average=False)
    criterion = SSIM()  # 自定义SSIM计算类（需返回单个数值损失）


    # -------------------------- 4. 配置GPU环境 --------------------------
    if opt.use_gpu and torch.cuda.is_available():  # 若配置使用GPU且设备支持
        model = model.cuda()  # 模型移至GPU
        criterion = criterion.cuda()  # 损失函数移至GPU（若含可训练参数）

    # -------------------------- 5. 配置优化器与学习率调度 --------------------------
    # 优化器：Adam优化器，学习率=配置值（论文中初始lr=4e-4）
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # 学习率调度：MultiStepLR（多步衰减），milestones=衰减节点（论文中为[30,50,80]），gamma=衰减系数（0.2）
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)

    # -------------------------- 6. 初始化训练日志（TensorBoard） --------------------------
    # SummaryWriter：用于记录训练过程中的标量（损失、PSNR等）和图像（雨图、去雨图等）
    writer = SummaryWriter(opt.save_path)

    # -------------------------- 7. 断点续训（加载最近保存的模型） --------------------------
    # findLastCheckpoint：自定义函数，查找保存目录下最近一次训练的epoch
    initial_epoch = findLastCheckpoint(save_dir=opt.save_path)
    if initial_epoch > 0:  # 若存在历史训练记录
        print('resuming by loading epoch %d' % initial_epoch)
        # 加载checkpoint文件（含模型参数、优化器状态、调度器状态）
        checkpoint = torch.load(os.path.join(opt.save_path, 'net_epoch%d.pth' % initial_epoch))
        model.load_state_dict(checkpoint['state_dict'])  # 加载模型参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数（保证学习率连续性）
        scheduler.load_state_dict(checkpoint['scheduler'])  # 加载调度器参数
        initial_epoch = checkpoint['epoch']  # 恢复起始epoch

    # -------------------------- 8. 开始训练循环 --------------------------
    step = 0  # 记录训练总步数（用于TensorBoard日志）
    # 外层循环：遍历epoch（从初始epoch到配置的总epoch数）
    for epoch in range(initial_epoch, opt.epochs):
        # 初始化当前epoch的SSIM和PSNR列表（用于计算 epoch 平均指标）
        SSIM_list = []
        psnr_list = []
        # 更新学习率调度器（需传入当前epoch，确保衰减时机正确）
        scheduler.step(epoch)
        # 打印当前epoch的学习率
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])

        ## -------------------------- 8.1 单epoch训练（遍历所有batch） --------------------------
        for i, (input_train, target_train) in enumerate(loader_train, 0):
            # 1. 模型切换为训练模式（启用Dropout、BatchNorm更新等）
            model.train()
            # 2. 清零模型梯度和优化器梯度（避免梯度累积）
            model.zero_grad()
            optimizer.zero_grad()

            # 3. 数据转换为Variable（兼容PyTorch旧版本，新版本可省略）
            input_train, target_train = Variable(input_train), Variable(target_train)
            # 4. 数据移至GPU（若使用）
            if opt.use_gpu:
                input_train, target_train = input_train.cuda(), target_train.cuda()

            # 5. 模型前向传播：输入雨图，输出去雨图（_为其他返回值，如中间迭代结果）
            out_train, _ = model(input_train)
            # 6. 计算损失：论文中用负SSIM作为损失（criterion返回SSIM值，取负后最小化）
            pixel_metric = criterion(target_train, out_train)  # pixel_metric为SSIM值
            loss = -pixel_metric  # 负SSIM损失

            # 7. 反向传播与参数更新
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新模型参数

            # 8. 计算当前batch的评估指标（PSNR、SSIM）—— 模型切换为评估模式（禁用Dropout等）
            model.eval()
            with torch.no_grad():  # 禁用梯度计算，加速推理
                out_train, _ = model(input_train)
                out_train = torch.clamp(out_train, 0., 1.)  # 限制输出在[0,1]（图像像素值范围）
                # 计算batch级PSNR（自定义函数，输入去雨图、干净图、最大像素值1.0）
                psnr_train = batch_PSNR(out_train, target_train, 1.)

            # 9. 记录当前batch的指标
            SSIM_list.append(pixel_metric.item())  # 保存SSIM值（转为Python数值）
            psnr_list.append(psnr_train)  # 保存PSNR值
            # 打印当前batch的训练信息：epoch、batch序号、损失、SSIM、PSNR
            print("[epoch %d][%d/%d] loss: %.4f, SSIM: %.4f, PSNR: %.4f" %
                  (epoch+1, i+1, len(loader_train), loss.item(), pixel_metric.item(), psnr_train))

            # 10. 每10步记录日志到TensorBoard
            if step % 10 == 0:
                writer.add_scalar('loss', loss.item(), step)  # 记录损失
                writer.add_scalar('PSNR on training data', psnr_train, step)  # 记录训练PSNR
                writer.add_scalar('SSIM on training data', pixel_metric.item(), step)  # 记录训练SSIM
            step += 1  # 总步数+1
        ## -------------------------- 8.2 单epoch训练结束 --------------------------

        # -------------------------- 8.3 记录当前epoch的图像结果（TensorBoard） --------------------------
        model.eval()
        with torch.no_grad():
            # 用当前epoch最后一个batch的数据生成图像日志
            out_train, _ = model(input_train)
            out_train = torch.clamp(out_train, 0., 1.)  # 限制像素值范围
            # 转换为网格图像（nrow=8：每行显示8张图，normalize=True：归一化像素值）
            im_target = utils.make_grid(target_train.data, nrow=8, normalize=True, scale_each=True)  # 干净图（GT）
            im_input = utils.make_grid(input_train.data, nrow=8, normalize=True, scale_each=True)    # 雨图（输入）
            im_derain = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)     # 去雨图（输出）
            # 记录图像到TensorBoard
            writer.add_image('clean image (GT)', im_target, epoch+1)
            writer.add_image('rainy image (input)', im_input, epoch+1)
            writer.add_image('deraining image (output)', im_derain, epoch+1)

        # -------------------------- 8.4 保存模型 --------------------------
        # 保存最新模型（覆盖式，方便中断后快速恢复）
        torch.save(
            {
                'epoch': epoch,  # 当前epoch
                'state_dict': model.state_dict(),  # 模型参数
                'optimizer': optimizer.state_dict(),  # 优化器参数
                'scheduler': scheduler.state_dict()  # 调度器参数
            },
            os.path.join(opt.save_path, 'net_latest.pth')
        )
        # 按配置频率保存epoch模型（如每5个epoch保存一次，避免频繁存储）
        if epoch % opt.save_freq == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                },
                os.path.join(opt.save_path, 'net_epoch%d.pth' % (epoch+1))
            )
        # -------------------------- 8.5 计算并打印当前epoch的平均指标 --------------------------
        SSIM_average = sum(SSIM_list) / len(SSIM_list)  # 平均SSIM
        psnr_average = sum(psnr_list) / len(psnr_list)  # 平均PSNR
        print("[epoch %d], average SSIM: %.4f, average PSNR: %.4f" %
              (epoch + 1, SSIM_average, psnr_average))
        # 清空指标列表，准备下一个epoch
        SSIM_list.clear()
        psnr_list.clear()
# 主函数入口
if __name__ == "__main__":
    main()

```
上边是训练代码的主函数，包含了损失函数、训练循环次数、断点训练等相关内容

## 3.2创建训练的虚拟环境和安装依赖项

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

## 3.3测试结果
将训练好的模型`net_latest.pth`保存到`logs`文件夹下
打开`test_M2PN.py`修改模型和测试数据集文件、以及文件保存的路径

```
parser.add_argument("--logdir", type=str, default="./logs/RainH/net_latest.pth", help='path to model and log files')
parser.add_argument("--data_path", type=str, default=r"datasets/test/Rain100L", help='path to test data')
parser.add_argument("--gt_path", type=str, default=r"datasets/test/Rain100H", help='path to ground truth data')
parser.add_argument("--save_path", type=str, default="./results", help='path to save results')
```

## 3.4测试结果
| 有雨图像 | 去雨后的结果图像 |
| :------: | :--------------: |
| <img src="有雨图像.png" width="321" alt="有雨图像"> | <img src="去雨后的图像.png" width="321" alt="去雨后的结果图像"> |


# 4论文公式对应的代码

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

