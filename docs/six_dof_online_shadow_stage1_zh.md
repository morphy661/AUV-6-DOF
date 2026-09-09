# 六自由度AI在线影子检测：第一阶段（2026-09-07）

## 当前完成内容

统一演示入口已将V3.5 BiLSTM-Attention接入 `SixDOFSimulator.step()`。
每一步在当前传感器、ESC可信状态和规则FTC字段生成后，调用一次模型观察器，
保存当时的诊断结果，然后才返回并进入下一仿真步。演示入口不再运行仿真后的
`enrich_logs()`；该方法仅保留用于参考回放验证。

这是在线影子检测阶段。当前步AI输出不传给FTC；规则FTC仍独立决策。
观察器收到当前步日志的深拷贝，修改副本不会改变控制日志。
下一阶段需要显式实现AI证据与规则融合，不能将本阶段称为AI驱动的FTC闭环。

模型权重和训练集归一化参数保持不变。本模型的类别是正常、推进器无输出、推进器推力损失，
位置为H1–H4、V1–V2。深度、IMU和DVL故障当前仍来自规则诊断。

## 时间与输出约定

- 本轮采样周期0.05秒，窗口50点，首次数据窗口约2.5秒形成。
- 稳定上下文中每10步推理一次，即每0.5秒刷新一次；其余步保留上一条结果。
- 原有上下文门控会在转向上下文、传感器异常或ESC通信变化后重建窗口，
  因而不能承诺整个任务始终每0.5秒产生有效诊断。
- `updated`标识本步是否真正推理；`available`标识有无可用窗口结果；
  `inference_time_s`和`result_age_s`标识原推理时刻和结果年龄。
- `raw_mode_probabilities`顺序为normal/no_output/thrust_loss；
  `raw_location_probabilities`顺序为H1/H2/H3/H4/V1/V2。
- 原始softmax概率在门控抑制期间仍保留，不能把抑制后的正常显示理解为原始模型认定正常。
- `mode_entropy_normalized`和`location_entropy_normalized`是归一化熵，
  衡量输出分散程度；它们不是经过校准的错误概率。位置概率只适合在存在故障证据时解释。
- 无窗口时原始概率与熵写为null；不应把初始化占位的normal当作一次模型推理。
- `execution_mode=online_shadow`、`affects_ftc=false`明确本阶段权限。

GPU计算内核在仿真开始前用零张量准备3次，不加入真实数据窗口、不更新诊断状态。
首次冷启动试验中观察到约401毫秒的模型更新开销，准备步骤将其移到任务开始前；
启动准备耗时另行保存在报告中。

## 验证结果

环境：用户指定的auv_gpu Python环境、CUDA、冻结的V3.5 seed43模型。
四组任务各30秒、600步，包含200米正常机动、固定V1无输出、随机V2推力损失、随机V1推力损失。
后三组同时包含传感器和ESC通信异常。

| 场景 | 实际模型推理次数 | 推理更新平均耗时(ms) | 推理更新最大耗时(ms) | 完整仿真步超50ms次数 |
|---|---:|---:|---:|---:|
| 正常，seed20260907 | 39 | 5.56 | 11.53 | 0/600 |
| 固定故障，seed20260907 | 26 | 11.76 | 19.54 | 0/600 |
| 随机故障，seed20260911 | 27 | 10.66 | 15.58 | 1/600 |
| 随机故障，seed20260922 | 31 | 15.56 | 32.18 | 1/600 |

这里“推理更新耗时”包括当前日志副本、特征处理、GPU推理、结果回传和诊断后处理，
不是仅计神经网络内核耗时。两次完整步超时分别约68.26和63.66毫秒，
其中一次发生在没有执行新模型推理的观察器更新步；具体耗时来源尚未剖析。
123次实际推理均未超50毫秒，完整步2/2400超时。因此功能验收通过，严格实时期限验收未通过。
本测试不包含视频渲染耗时，仿真按同步计算推进，没有按墙钟定时调度。

四组任务均通过以下逐项检查：

1. 开启/关闭AI影子观察器时，所有原有物理、传感器和FTC日志完全一致（只排除计时和新增模型字段）。
2. 在线与事后因果回放的全部诊断字段完全一致，包括概率、时序判定和上下文门控。
3. 只回放前半段且删除声明的仿真特权真值字段，诊断仍与在线前半段完全一致。
4. 每次推理窗口的结束时刻不晚于当前仿真时刻。
5. 观察器回调在step返回前执行，复位清空历史；独立单元测试验证副本修改不会影响FTC/运动。

项目回归测试：232 passed，5 subtests passed。
这些测试验证接入正确性，不构成新的误报率或诊断准确率评估，不能将四组演示结果推广为全域性能。

## 文件与运行

完整验收结果位于 `math-model/results/six_dof_online_shadow_stage1_20260907/validation_summary.json`，
同目录包含四条在线概率轨迹。`cold_start_validation_summary.json`保留加入启动准备前的计时证据。

在项目根目录运行下面命令可复现验收：

```powershell
& 'D:/Anaconda_envs/envs/auv_gpu/python.exe' -B math-model/examples/validate_six_dof_online_shadow.py --device cuda --output-dir math-model/results/six_dof_online_shadow_recheck
```

生成在线影子演示视频：

```powershell
& 'D:/Anaconda_envs/envs/auv_gpu/python.exe' -B math-model/examples/demo_six_dof_unified_diagnostics.py --duration 30 --dt 0.05 --seed 20260911 --injection-mode random --model-device cuda --output-dir math-model/results/six_dof_online_shadow_demo_20260907
```

主JSON和视频沿用原有简洁展示；`online_model_trace.jsonl`额外保存每个仿真步真实产生的
原始概率、平滑/门控结果、熵、窗口时刻、结果年龄、计时及FTC动作。
视频在仿真结束后编码，其中AI数据来源已经是在线保存结果。
`--disable-model`可运行规则基线。

## 下一阶段入口

先基于这些在线结果设计AI持续证据与规则融合，仍保留明确硬故障的规则安全通道。
现有置信度不能直接解释为确认概率，AI熵不能单独当FTC阈值。
若需要50毫秒严格墙钟周期，应先剖析完整步长尾，并验证工作线程/任务调度及过期结果处理。
本阶段没有修改模型权重、诊断阈值或FTC动作规则。
