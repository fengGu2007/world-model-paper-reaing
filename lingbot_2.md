# LingBot-World 2.0 / LingBot-World-Infinity 回顾

## 1. 整体定位

LingBot-World 2.0，也称 LingBot-World-Infinity，是 LingBot-World 的进阶版本。核心目标是构建一个**可实时交互、可长时间稳定生成、支持丰富语义交互**的世界模型。

相比 1.0，2.0 的主要升级包括：

- 支持更长的交互 horizon，甚至接近无限时长生成；
- 通过蒸馏得到实时模型，支持 720p、60 fps 视频流；
- 支持更丰富的交互动作，如攻击、射箭、施法、射击等；
- 支持文本驱动的环境事件，如下雪、下雨、昼夜变化、实体生成等；
- 引入 Director-Pilot agentic harness，让 VLM 负责语义推理和事件规划，视频生成模型负责渲染世界状态；
- 提供多玩家共享交互界面。

---

## 2. Data Engine 部分

### 2.1 与 1.0 的关系

Data engine 整体框架与 1.0 基本一致，仍然是：

1. Data Acquisition；
2. Data Profiling；
3. Multi-granularity / Multi-dimensional Annotation。

数据来源包括：

- 自采第一人称视频；
- 游戏 / Unreal Engine 合成数据；
- 大规模网络视频。

数据 profiling 包括：

- 解码校验；
- 镜头切分；
- 基础约束过滤；
- 技术质量打分；
- VLM profiling；
- 质量属性与语义属性标注。

### 2.2 2.0 的关键新增：chunk-wise 文本 caption

2.0 相比 1.0 最重要的数据侧变化是：不仅使用 video-level global caption，还构建了 chunk-wise local caption。

也就是说，每个视频片段除了有一个全局描述，还会被切成多个时间 chunk，每个 chunk 有自己的局部语义描述。

目的很明确：

> 让模型在训练阶段就见过“时间局部化的文本/事件条件”，从而在推理时能够根据实时文本 prompt 或事件改变后续世界内容。

如果只用 global caption，会出现 train-inference mismatch：

- **训练时**：整段视频只有一个总描述；
- **推理时**：用户可能在生成过程中不断输入新事件，例如“下雪”“开门”“角色射箭”“场景切换”。

因此 2.0 的 annotation 更强调：

- video-level global context；
- chunk-wise local description；
- multi-track event-level annotation；
- action context 对齐；
- chunk-wise optimization。

### 2.3 Multi-track event-level annotation

2.0 不是简单给每个 chunk 写一句 caption，而是分多个事件轨道标注：

- subject visibility；
- motion state；
- interaction state；
- environmental dynamics；
- static scene state。

这样做是为了减少时间歧义，避免把不同时间边界的事件混在一起。

例如一个 chunk 内可能同时发生：

- 角色移动；
- 门打开；
- 天气变化；
- 背景保持静止。

分开 track 标注后，可以分别表示它们的发生状态，再融合成该 chunk 的局部 caption。

### 2.4 Action context 的作用

如果数据中有控制信号，例如相机轨迹、游戏动作、合成动作标签，会作为 action context 辅助标注。

它主要用于：

- 提供时间边界线索；
- 帮助恢复短交互；
- 弥补稀疏视觉采样可能漏掉的动作。

但最终 caption 仍以可见视觉证据为准，避免推测性动作或状态变化。

---

## 3. 训练部分

### 3.1 阶段变化：从 3 个阶段变成 2 个阶段

根据我的理解，1.0 可能是多阶段训练，而 2.0 明确组织为两个阶段：

1. **Pre-training**：训练因果世界模型；
2. **Post-training**：蒸馏成实时模型，并抑制长时漂移。

### 3.2 Pre-training：训练因果世界模型

Pre-training 的目标是训练一个 causal, action-conditioned world model。

模型输入包括：

- 初始帧；
- 历史视频 latent；
- 相机姿态；
- chunk-wise prompt / 文本事件。

模型输出：

- 下一段视频 latent / 世界状态。

整体形式是因果生成：

$$
p_\theta(x_{1:T} \mid a_{1:T})
=
\prod_t p_\theta(x_t \mid x_{<t}, a_{\le t})
$$

也就是说，每个未来状态只依赖过去状态和当前/历史输入，不依赖未来。

### 3.3 基座模型

论文没有非常详细展开基座模型细节，但从方法看，2.0 应该仍然是在一个视频生成模型 / DiT backbone 基础上继续训练成因果世界模型。

Pre-training 阶段学习的是：

- 视频动态；
- 相机控制；
- 文本/事件条件；
- 长时一致性；
- 动作导致的状态变化。

### 3.4 MoBA Attention Mask

2.0 提出了 MoBA mask，即 Mixture of Bidirectional and Autoregressive Attention Mask。

它是在 teacher forcing mask 基础上加入 bidirectional component。

可以理解为：

- **Autoregressive / teacher forcing 部分**：当前 noisy frame 只能看自己和之前的 clean context；
- **Bidirectional 部分**：允许一定程度的全局/双向注意力，帮助模型适应变长视频生成，并缓解纯 teacher forcing 的过拟合和画质退化。

作用是：

> 既保持因果生成能力，又保留基础视频生成模型的视觉保真度。

### 3.5 动作与文本注入

动作和文本注入方式与之前类似：

- 相机姿态使用 Plücker embedding；
- 通过 AdaLN 注入 diffusion / flow-matching 过程；
- 文本 prompt 通过 cross-attention 注入；
- 2.0 更强调 chunk-wise prompt，即每个 chunk 可以有自己的文本条件。

相机控制：

$$
p_{\le i}
$$

文本/语义控制：

$$
a_{\le i}
$$

其中 $i$ 表示当前 chunk。

### 3.6 训练目标：Rectified Flow / Flow Matching

2.0 的 pre-training 使用 conditional flow-matching objective，而不是传统 DDPM 噪声预测。

对于第 $i$ 帧，构造 noisy latent：

$$
x_i^t = (1-t)x_i + t\epsilon
$$

其中：

$$
t \sim U(0,1), \quad \epsilon \sim \mathcal{N}(0,I)
$$

训练网络 $v_\theta$ 预测 flow velocity：

$$
L_{\text{fm}}
=
\mathbb{E}_{x,i,t,\epsilon}
\left[
\left\|
v_\theta(x_i^t, t \mid x_{<i}, p_{\le i}, a_{\le i})
-
(\epsilon - x_i)
\right\|^2
\right]
$$

目标速度是：

$$
\epsilon - x_i
$$

这对应 rectified-flow 的线性插值路径。

相比 DDPM，这里不是直接预测噪声，而是学习从噪声到数据的路径速度场。

---

## 4. Post-training：CD + DMD

Post-training 的目标是：

> 把多步、高质量的 pre-trained causal world model 压缩成少步、实时、可交互的模型，同时抑制长时间 self-rollout 带来的漂移。

主要包括两个技术：

1. Consistency Distillation, CD；
2. Distribution Matching Distillation, DMD。

### 4.1 Consistency Distillation

CD 的目标是减少去噪步数。

Teacher 是多步 causal flow-matching / diffusion world model。Student 学习从 teacher 轨迹上的不同噪声水平出发，预测到一致的结果。

CD loss：

$$
L_{\text{CD}}
=
\mathbb{E}
\left[
d\left(
G_\theta(x_i^t, t \mid c),
G_{\theta^-}(\tilde{x}_i^{t-\Delta t}, t-\Delta t \mid c)
\right)
\right]
$$

其中：

$$
c = (x_{<i}, p_{\le i}, a_{\le i})
$$

直观理解：

> 同一条 teacher 去噪轨迹上的相邻/不同噪声状态，应该被 student 映射到同一个干净结果。

所以 CD 的核心不是简单让学生逐帧模仿老师，而是让学生继承 teacher 的轨迹结构，并实现少步/一步生成。

### 4.2 Distribution Matching Distillation

CD 之后，student 可以少步生成，但可能出现：

- 画质下降；
- 细节退化；
- 几何不稳；
- 长时自回归漂移。

DMD 用于进一步对齐真实数据分布。

DMD 优化：

$$
D_{\text{KL}}(p_{\theta,t} \parallel p_{\text{data},t})
$$

即让 student 生成样本加噪后的分布接近真实数据加噪后的分布。

论文中梯度形式为：

$$
\nabla_\theta
\mathbb{E}
\left[
D_{\text{KL}}(p_{\theta,t} \parallel p_{\text{data},t})
\right]
=
-
\mathbb{E}
\left[
\left(
s_{\text{real}}(\hat{x}_i^t, t \mid c)
-
s_{\text{fake}}(\hat{x}_i^t, t \mid c)
\right)
\frac{\partial \hat{x}_i}{\partial \theta}
\right]
$$

其中：

- $\hat{x}_i$：student 生成样本；
- $\hat{x}_i^t$：student 样本加噪后的版本；
- $s_{\text{real}}$：真实数据分布 score；
- $s_{\text{fake}}$：student 当前生成分布 score。

### 4.3 DMD 的关键：在 self-rollout 轨迹上训练

论文特别强调：

> DMD 不只用于 teacher-forced states，而是用于 long self-rollout trajectories。

也就是说，训练时让 student 自己连续 rollout 一段轨迹，然后在这段“自己生成的历史”上优化。

这很重要，因为实际部署时：

$$
x_1 \rightarrow x_2 \rightarrow x_3 \rightarrow \cdots
$$

后面的历史帧都是模型自己生成的。如果只在真实历史上训练，模型可能无法处理自己生成误差累积的问题。

因此 DMD over self-rollout 的作用是：

> 让模型在自己生成的长轨迹上仍然保持真实分布、视觉质量和世界一致性，从而抑制 drift。

### 4.4 CD 与 DMD 的关系

可以概括为：

| 方法 | 作用 |
|---|---|
| CD | 少步化，让学生从不同噪声水平一致预测结果 |
| DMD | 分布对齐，让学生少步生成更像真实数据，并抑制长时漂移 |

更口语化：

> CD 管“快且一致”，DMD 管“像且稳”。

---

## 5. Deployment 部分

Deployment 部分重点是把模型变成可实时交互系统。

主要包括：

1. Systematic Optimization；
2. Agentic Interaction Harness；
3. Visual Quality Enhancement；
4. User Interface Design。

### 5.1 Systematic Optimization

目标是低延迟、高吞吐。

包括两个层面：

#### Model inference efficiency

提高视频生成 backbone 的推理速度：

- compiler-based optimization；
- efficient attention kernels；
- hybrid parallel inference；
- 多 GPU 并行；
- latent generation 与 frame reconstruction 异步 pipeline；
- VAE decoding 并行执行。

#### Streaming responsiveness

降低用户输入到可见响应之间的延迟：

- 不等整个 chunk 解码完，而是流式输出；
- decoding、transmission、generation 重叠；
- 低延迟视频流式 pipeline。

### 5.2 Agentic Interaction Harness：Director-Pilot

这是 2.0 的重点之一。

核心思想：

> 视频生成模型擅长渲染视觉动态，但缺乏开放世界所需的因果推理和语义规划。
> 因此用 VLM 作为 Director，用视频生成模型作为 Pilot。

#### Director：VLM

负责高层语义推理：

- 分析当前画面；
- 理解用户动作或文本指令；
- 预测交互后果；
- 提出 event proposals；
- 决定世界下一步应发生什么。

例如：

- 用户推门 → VLM 判断门应打开；
- 用户输入“下雪” → VLM 判断天气如何变化；
- 用户召唤鸟群 → VLM 决定鸟群如何进入场景。

#### Pilot：Video Generator / DiT

负责低层视觉/物理动态：

- 根据 Director 的语义事件；
- 结合当前世界状态；
- 生成下一段视频帧；
- 渲染视觉一致、时间连贯的结果。

整体流程：

```text
User action / prompt
→ VLM Director 推理事件
→ Video Generator Pilot 渲染世界状态