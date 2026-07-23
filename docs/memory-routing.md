# 记忆路由规格（type 统一）

## Types

| type | 含义 | run_id | 跨局 |
|------|------|--------|------|
| `world` | 世界观 | 否 | 是 |
| `persona` | 角色设定 | 否 | 是 |
| `dialogue` | 对话；tier=`daily`\|`important` | 建议有 | 是 |
| `run_event` | **本局事件摘要** | **必填** | **默认否**（查当前 run） |
| `reflection` | **反思/教训** | source_run_id | **是** |

## run_id

- 每局游戏启动生成 UUID（前端 `state.runId`）
- chat / think / run_event 写入均携带
- **判断「本局」= metadata.run_id 等于当前 run_id**，不是靠时间猜

## 存什么

- **dialogue**：玩家↔NPC 对话后分类入库
- **run_event**：倒地、姿态切换、进层、通关等**摘要句**（非每帧）
- **reflection**：局末 1 句偏好/教训
- **不入库**：scene 帧、RL 动作、Working/Commitment、反射模板刷屏

## 怎么存

统一 metadata：`memory_type, player_id, npc_id, run_id?, tier?, source, created_at, floor?`

API：

- `POST /api/memory/run_event`
- `POST /api/memory/reflection`
- 对话仍由 chat 结束后 `add_dialogue_memory`

## 怎么查 / 何时查

| channel | 场景 | 主查 |
|---------|------|------|
| `chat` | 玩家说话 | persona + dialogue + run_event(当前) + reflection |
| `think_combat` | 战斗自主 | run_event + important + reflection |
| `think_safe` | 安全自主 | persona + daily + important |
| `run_start` | 新局 | reflection + important（**不**灌旧 run_event） |

## Prompt 分块（空则省略）

```text
[角色设定]
[世界设定]
[长期反思]
[对话记忆-重要]
[对话记忆-日常]
[本局事件]
[工作记忆]   ← 不入库：commitment / 最近指令 / 场面摘要
```
