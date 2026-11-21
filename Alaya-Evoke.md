# Alaya-EVOKE (Evoke) 论文深度总结

## 1. 核心设计理念

Evoke 旨在解决交互式世界模型中的三大冲突需求：**持久记忆、实时交互和长时程生成**。其核心设计原则是**解耦**：

- **空间状态解耦**：将持久世界状态从去噪器（Denoiser）上下文中剥离，存入外部的世界状态库（World State Bank, $M_k$）。
- **监督能力解耦**：重新设计教师模型（Teacher），使其具备长时程监督和动态条件控制能力，再通过蒸馏转移给少步学生模型（Student）。

---

## 2. 系统架构与组件

### 2.1 学生模型 (Student)

- **基础架构**：基于 Helios (Yuan et al., 2026) 的 Diffusion Transformer。
- **生成策略**：3 步金字塔采样（Coarse-to-Fine Pyramid）。
  - **分辨率**：$12 \times 20 \rightarrow 24 \times 40 \rightarrow 48 \times 80$。
  - 每步执行 1 次去噪评估，共 3 次。
  - **无 CFG**（Classifier-Free Guidance）。
- **上下文管理**：
  - **局部历史**：仅保留最近 19 个潜在帧（约 3.2 秒）。
    - 长程层：16 帧
    - 中程层：2 帧
    - 短程层：1 帧
  - **位置编码**：每个 chunk 复用相同的局部位置布局，不随会话时长扩展。

### 2.2 世界状态库 (World State Bank, $M_k$)

- **存储内容**：世界空间几何（World-space Geometry）。
- **写入机制 (Write)**：
  - 使用单目深度模型估计生成 chunk 中 12 帧的深度。
  - 通过相机内外参反投影为 3D 点云/几何。
  - **独立插入**：不同 chunk 的几何独立存入，避免长序列融合导致的尺度漂移。
  - **保留预算**：保留最近 90 秒（2160 像素帧，每 3 帧存 1 帧，共 720 活跃源帧）。
- **读取机制 (Read)**：
  - **几何检索**：根据当前相机位姿，计算与存储源视图的共可见性（Co-visibility）。
  - **选择**：选取最多 8 个足够不同的源视图。
  - **渲染**：通过批量投影和 Z-Buffering 渲染出视图对齐的扭曲观察图（Warped Observation）。
  - **可见性掩码**：生成逐像素掩码，指示哪些区域有可靠几何支持。
- **注入去噪器**：
  - 仅在**最粗粒度阶段**（$12 \times 20$）注入几何条件。
  - **可见区域**：低噪声（$\sigma \in [0, 0.135]$），强约束。
  - **不可见区域**：高噪声（$\sigma = 1$），无约束，允许自由生成。
  - **Token 选择**：移除无几何支持的历史 Token。

### 2.3 教师模型 (Evoke Teacher)

- **基础架构**：Wan2.2 A14B Diffusion Transformer。
- **专家路由**：保留高噪声/低噪声专家，根据采样时间步选择。
- **注意力机制**：Chunk-wise Sparse Attention。
  - 将序列分为 9 潜在帧的 Chunk。
  - 每个 Query Chunk 访问：
    - 首帧全局 Sink。
    - 局部上下文（1 帧重叠）。
    - 空间压缩的邻近帧。
    - 少量选定的远距离帧。
    - 线性注意力全局状态。
  - **复杂度**：从二次方降至线性增长。
- **条件控制**：Per-chunk Conditioning。
  - 每个 Chunk 拥有独立的文本条件。
  - 支持在单个长序列中表示提示词变化（Prompt Switching）。
- **Critic**：与 Teacher 共享骨干，通过 LoRA 适配器切换。启用 LoRA 为 Critic，禁用为 Teacher。

---

## 3. 训练流程详解

Evoke 的训练分为三个阶段，逐步构建最终的学生模型。

### Stage 1: Camera Control (相机控制训练)

- **目标**：学习相机轨迹控制下的视频生成。
- **输入**：
  - 文本条件 $c$。
  - 相机轨迹 $P$。
  - 历史 Token（短历史）。
  - 噪声潜在帧。
- **输出**：去噪后的潜在帧。
- **注意力机制**：标准 DiT 注意力（具体细节文献未详述，推测为局部或全局）。
- **损失函数**：Flow Matching Loss（Rectified Flow）。
  - 与 Ground Truth 进行匹配。
- **World State Bank**：不使用（或仅使用极短历史，不涉及长时程几何记忆）。

### Stage 2: Few-step Distillation (少步蒸馏)

- **目标**：将多步教师蒸馏为 3 步学生，实现低延迟。
- **输入**：
  - 学生模型：3 步金字塔采样。
  - 教师模型：Wan2.2 A14B（多步，双向注意力）。
  - Critic：Wan2.2 A14B + LoRA。
- **输出**：学生生成的 3 步潜在帧。
- **注意力机制**：
  - 学生：Helios 架构。
  - 教师：双向注意力（Bidirectional）。
- **损失函数**：DMD (Distribution Matching Distillation) Loss。
  - $\Delta s = s_{\text{fake}} - s_{\text{real}}$。
  - $s_{\text{real}}$：教师分数。
  - $s_{\text{fake}}$：Critic 分数。
  - 梯度仅应用于生成的 Chunk（排除 GT 前缀和第一个生成 Chunk，以避免边界闪烁）。
- **World State Bank**：不使用（或仅使用短历史，重点在于步数压缩）。

### Stage 3: Long Distillation (长时程蒸馏)

- **目标**：将长时程一致性和动态条件控制能力转移给学生。
- **输入**：
  - 学生模型：3 步金字塔采样，Self-forced Rollout。
  - 教师模型：Evoke Teacher（Chunk-wise Sparse Attention, Per-chunk Conditioning）。
  - Critic：Evoke Teacher + LoRA。
  - World State Bank：使用。学生在 Rollout 过程中读取几何记忆。
- **输出**：学生生成的长序列潜在帧（约 31.4 秒，20 个 Chunk）。
- **注意力机制**：
  - 学生：Helios 架构 + 局部历史 + Warp Conditioning。
  - 教师：Chunk-wise Sparse Attention（线性复杂度）。
- **损失函数**：Full-window DMD Loss。
  - 监督窗口 $W$ 覆盖整个 31.4 秒 Rollout。
  - 梯度截断：历史在 Chunk 之间断开（Detached），每个 Chunk 独立计算梯度。
  - 监督范围：教师/Critic 对整个 189 潜在帧（753 像素帧）联合打分。
- **条件控制**：
  - 文本条件按 12 秒间隔分段，映射到对应 Chunk。
  - 支持 Prompt 切换（Evocation）。

---

## 4. 推理流程 (Inference)

### 4.1 递归生成循环

对于每个递归步骤 $k$：

1. **读取 (Read)**：
   - 根据当前相机位姿 $P_k$，从 $M_k$ 中检索几何。
   - 渲染出 Warp Observation 和 Visibility Mask。
2. **生成 (Generate)**：
   - 输入：Warp Observation + 局部历史 $h_k$ + 文本条件 $c_k$。
   - 执行 3 步金字塔去噪。
   - 输出：Chunk $x_k$（9 潜在帧，1.5 秒）。
3. **写入 (Write)**：
   - 估计 $x_k$ 的深度。
   - 反投影为几何。
   - 更新 $M_{k+1} = \text{Write}(M_k, x_k, P_k)$。
   - 更新局部历史 $h_{k+1}$。

### 4.2 性能指标

- **延迟**：单张 H200，$384 \times 640$ 分辨率，每个 1.5s Chunk 生成耗时 **2.11s**。
- **长时程稳定性**：
  - 支持 **2 小时** 连续生成。
  - 光度统计（亮度、饱和度）在一小时内保持平坦。
  - 场景身份余弦相似度稳定在 **0.523**（与真实视频 60 秒间隔的自相似度相当）。
- **基准测试**：
  - **WBench**：Video Quality, Setting, Physical 组领先。
  - **VBench-2.0**：66.77 分（排名第 1，略高于 Veo 3）。
  - **VBench-Long**：85.11 分（排名第 7，接近 Veo 3）。

---

## 5. 关键实验发现

### 5.1 长时程教师监督的效果

- **光度稳定性**：长时程教师蒸馏的学生亮度保持 **101%**，短时程教师仅 **74%**。
- **内容漂移**：长时程监督显著改善颜色漂移，但对锐度无显著影响。
- **结论**：长时程监督主要转移**光度稳定性**，而非所有内容一致性。

### 5.2 几何召回与定时交互

- **几何召回**：
  - 保留 90 秒几何的覆盖率（0.99）远高于保留 1.5 秒（0.81）。
  - 世界状态库能在相机返回时恢复之前观察到的区域。
- **定时提示切换 (Evocation)**：
  - 未锚定内容（新物体）：引入成功率 **67%**。
  - 已锚定内容（覆盖背景）：成功率仅 **4%**。
  - **结论**：文本控制可引入新元素，但持久几何记忆会抵抗对已锚定内容的覆盖。

### 5.3 计算成本

- **几何路径成本**：占去噪器成本的 **38%**，占跳过几何路径的递归步骤成本的 **93%**。
- **有界性**：存储和每步几何条件成本相对于会话时长保持有界。

---

## 6. 总结

Evoke 通过**外部几何记忆（World State Bank）**和**长时程教师监督（Chunk-wise Sparse Attention Teacher）**实现了小时级稳定生成。其核心权衡是：

- 几何记忆提供空间一致性，但抵抗内容覆盖。
- 长时程监督提供光度稳定性，但未完全解决内容漂移。
- 3 步无 CFG 采样在保持低延迟的同时，在基准测试中达到 SOTA 或竞争性水平。