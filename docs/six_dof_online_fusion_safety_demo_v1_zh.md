# 六自由度在线诊断与 FTC 安全恢复演示（V1）

## 演示目的

本演示用一个可解释的单故障场景，展示“真实故障（仅用于验证显示）—在线 AI 输出—ESC 规则确认—FTC 动作—安全状态”的因果链。仿真在约 200 m 局部深度运行，属于项目当前重点的 0–300 m 工况范围。

## 场景与结果

- 仿真时长：18.0 s，步长：0.05 s，随机种子：20260909。
- 唯一注入故障：V1 垂向推进器从 10.00 s 起完全无输出，推力效率为 0。
- ESC 直接证据在 10.20 s 达到 possible；在线 BiLSTM-attention 在 11.55 s 首次输出 thrust loss，原始故障概率约 100%，并在 12.05 s 输出 no output，原始故障概率约 100%。
- ESC 规则在 12.90 s 确认并锁存 V1，FTC 同时执行 V1 定向重分配；垂向推进器安全保持在下一控制步 12.95 s 生效，任务目标暂停并切换到局部保持目标。
- 故障后最大倾斜角为 6.06°；保持阶段最大深度误差为 0.53 m，最大水平误差为 0.58 m；最大归一化分配残差为 0.089。未触发 safe abort。
- 诊断目标与真实故障一致：Ground truth 为 V1 no output，规则/FTC 最终目标为 V1。

环境扰动下出现过短暂的深度或 IMU `log_only` 记录，但没有被确认为传感器故障，也没有触发传感器隔离。这类记录用于事后审计，不等同于误维护指令。

## 在线 AI 的实际作用边界

AI 在仿真循环内在线执行，并不是仿真结束后回填结果。本次共有 19 次模型推理更新，平均推理时间 2.97 ms，P95 为 4.13 ms，最大 7.60 ms，均低于 50 ms 控制步预算。

但本演示必须如实说明：在“ESC 遥测直接确认 V1 完全无输出”这一强证据场景中，FTC 仍由独立 ESC 确认规则主导，AI 作为并行诊断和门控证据，不拥有独立控制权限。因此当前实现已经完成“在线 AI + 规则/门控 + FTC 同屏因果展示”，但尚不能描述为“AI 决策直接驱动 FTC”。这一限制是当前安全策略，不是程序离线回放造成的。

## 演示画面阅读顺序

1. `GROUND TRUTH`：仿真注入的真实故障，仅用于验证，不进入诊断输入。
2. `AI MODEL`：当前或最近一次有效的在线模型原始输出；动作切换后因上下文重置会短暂重新预热。
3. `RULE / GATE`：ESC 直接规则、确认锁存状态与融合门控状态。
4. `OUTCOME`：诊断是否与真实注入一致，以及确认延迟。
5. `Fault-tolerant control`：FTC 动作、目标推进器、任务/保持模式和安全边界余量。

## 复现命令

在项目根目录运行：

```powershell
& 'D:\Anaconda_envs\envs\auv_gpu\python.exe' `
  'math-model\examples\demo_six_dof_unified_diagnostics.py' `
  --duration 18 --dt 0.05 --seed 20260909 `
  --injection-mode safety_recovery `
  --diagnosis-mode fusion --model-device cuda `
  --fps 12 --max-video-frames 216 `
  --output-dir 'math-model\results\six_dof_online_fusion_safety_demo_v1_20260909'
```

## 验证

完整测试结果：`260 passed, 5 subtests passed`。

仓库只归档演示 MP4、总览图、技术细节图、CSV、精简指标和本文档。完整逐帧 JSON 与在线模型调试轨迹可由上述命令重新生成，不纳入版本控制。
