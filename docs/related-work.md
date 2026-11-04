# Adaptation, RSI and transfer

The keyboard, cameras and robot used for these demonstrations did not move.
The observations support useful program repair, memory and composition within
that setup. They leave an empirical question: which learned behavior transfers
when the text, target key, geometry or hardware changes?

These are separate tests rather than one binary label:

| Change | What it probes |
|---|---|
| New strings on known keys | Composition and execution reliability |
| Keys without a supplied demonstration | Local transfer from references and feedback |
| Moving the keyboard or camera | Whether learned relationships survive a geometric change |
| Another keyboard or robot | Portability of skills and learned device understanding |
| Adapting after each change | Attempts, elapsed time and tokens needed to recover |

Experience can continue accumulating during ordinary use. For comparisons, we
can record a knowledge/program snapshot and measure immediate transfer plus the
subsequent adaptation curve. A measurement snapshot does not require disabling
online learning in the application.

## Primary references

- **[EmbodiedEvo](https://github.com/Einsia/EmbodiedEvo)** acquires skills and
  lessons in a Playground and evaluates a fixed learned library in fresh Test
  scenes. This provides a concrete within-task cross-scene comparison.
  [Paper, v2](https://arxiv.org/abs/2610.18867v2).
- **[PhysicalRSI](https://github.com/PhysicalRSI/PhysicalRSI)** describes a
  [demo workbench](https://github.com/PhysicalRSI/PhysicalRSI/blob/main/docs/demo-workbench.md)
  that uses separate training and paired validation layouts for its mouse-click
  example. Its same-task layout claim is more specific than cross-task or
  cross-device transfer.
- **[RoboDoctor](https://github.com/THUSI-Lab/RoboDoctor)** studies fixed-weight
  policy improvement with accumulated experience. Its
  [paper](https://arxiv.org/abs/2610.35604v1) leaves controlled isolation of
  cross-episode memory benefits as further work. Improvement traces and causal
  attribution are therefore distinct kinds of evidence.
- **[SenseNova-RoboRSI](https://github.com/OpenSenseNova/SenseNova-RoboRSI)** and
  [Has GPT-6-Astra Achieved Embodied Intelligence?, v2](https://arxiv.org/abs/2610.22471v2)
  informed this project's use of demonstrations and interface information as
  context. The report separates demonstration-based task knowledge (§2.2) from
  a real-robot profile assembled using documentation and non-motion probes
  (§4.3). Cross-robot context sources are proposed directions, not a result we
  claim to have reproduced.

Sources reviewed on 2026-11-04. The paper versions above are explicit; mutable
repository pages are not presented as pinned benchmark results. These protocols
help formulate tests for our project; they do not establish that all RSI is
overfitting or that our fixed-keyboard demonstrations establish general RSI.

## 中文

我们关注的是一组可以逐项测的问题：新文本的组合能力、未示范键位的迁移、
键盘／相机移动后的保持能力，以及另一套设备上的重新适应成本。
固定场景中的改进可以很有用，但不能单凭它判断跨场景能力。

EmbodiedEvo 的新测试场景、PhysicalRSI workbench 的训练／验证布局划分、
RoboDoctor 对经验贡献的开放问题，都提醒我们把“应用中逐渐变好”与
“哪些机制导致改善、能迁移多远”分别研究。我们保持日常使用中的持续累积，
再用有版本记录的对照和适应曲线回答这些问题。

SenseNova-RoboRSI 对本项目的重要启发是：将示范以及机器人接口资料
组织为模型能理解和复用的上下文。其报告中的示范 ICL 与实机 robot profile
来源不同；我们借鉴思路，不声称直接复用了其代码或完成了其全部 ICL 实验。
