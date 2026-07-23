# 2D Roguelite 智能 NPC 战友设计方案

> 整理来源：Grok 搜索 + 项目代码分析（AI-roguelite-web-suanfa）
> 日期：2026-07-08

---

## 核心结论：别让 LLM 打帧，让它「像战友」

业界和论文的共识（CODA 双脑架构、LLM NPC 跨平台记忆研究）都指向同一点：

| 层级 | 负责什么 | 频率 | 技术 |
|------|----------|------|------|
| **大脑（Cerebrum）** | 人格、对话、战术意图、关系记忆 | 秒级 / 事件触发 | LLM + RAG 记忆 |
| **小脑（Cerebellum）** | 走位、躲弹、风筝、开火时机 | 60fps | RL 策略网络 |
| **脊髓（Reflex）** | 危急反应、模板喊话 | <1ms | 规则 FSM |

**2D Roguelite 的特殊约束**：战斗节奏快、弹幕密、单局短。若让 LLM 直接输出移动/攻击，延迟和随机性会毁掉手感。因此**最优解是 LLM 管「说什么 + 做什么战术」，RL 管「怎么打」**。

---

## 参考来源

### 学术论文与业界方案

1. **CODA: Coordinating the Cerebrum and Cerebellum** — 双脑解耦：Planner（大脑）负责高层规划，Executor（小脑）负责精确执行；解耦 RL 训练更高效。
   - https://arxiv.org/html/2508.20096v1

2. **LLM-Driven NPCs: Cross-Platform Dialogue System** — LLM NPC 跨平台记忆、好感度机制、RAG 长期记忆。
   - https://arxiv.org/html/2504.13928v1

3. **NVIDIA ACE** — 游戏 AI 伴侣：高频动作本地化，低频人格云端化。
   - https://developer.nvidia.com/ace-for-games

4. **Generative Agents (Stanford)** — 基于 LLM 的生成式智能体，记忆-反思-规划循环。
   - https://arxiv.org/abs/2304.03442

### 本项目已有实现（AI-roguelite-web-suanfa）

| 模块 | 路径 | 职责 |
|------|------|------|
| LLM 对话引擎 | `server/npc_backend/graph.py` | 流式对话、自主发言、记忆写入 |
| 反射层 | `server/npc_backend/reflex.py` | 危急局面零 LLM 响应（<1ms） |
| 触发器 | `server/npc_backend/triggers.py` | P0-P3 优先级战术/社交模板 |
| 长期记忆 | `server/npc_backend/memory.py` | ChromaDB（world/persona/dialogue） |
| RL 环境 | `rl/limbs/assault_skirmish/env.py` | AssaultEnv，109 维观测，Gymnasium |
| RL 训练 | `rl/limbs/assault_skirmish/train.py` | RecurrentPPO（LSTM） |
| ONNX 导出 | `rl/limbs/assault_skirmish/export_onnx.py` | → `game/resources/limbs/...` |
| 游戏前端 | `game/game.js` | scene_info 上报、assault FSM、ONNX 推理 |
| 寻路 | `game/pathfinding.js` | A* 绕障（assault 接近 / guard 跟随） |

---

## 推荐架构：双脑分层 + Roguelite 元成长

```
玩家层
  ├── 聊天输入
  └── Roguelite 祝福/构筑
        ↓
大脑 LLM（~1-3s）
  ├── ChromaDB 记忆（world / persona / dialogue）
  ├── /api/npc/think 自主发言
  ├── /api/chat/stream 对话
  └── 输出: dialogue | command（stance / focus_target / retreat）
        ↓
脊髓 Reflex（<1ms）
  ├── reflex_decide（圈/低血/危急）
  └── P0-P3 触发器模板
        ↓
小脑 RL（60fps）
  ├── guard: A* 贴玩家 + 减伤
  └── assault: ONNX LSTM（109维 obs → 9向微操）
        ↓
2D 战斗仿真（弹幕/障碍/Boss）
        ↓
scene_info 快照 → 回传大脑/脊髓
```

---

## 玩法设计

### 1. 两种姿态 = 玩家可理解的战术语言

| 姿态 | 玩家感受 | 底层行为 |
|------|----------|----------|
| **guard（护卫）** | 「贴身保我」 | A* 跟随、挡火力、减伤祝福叠加 |
| **assault（突击）** | 「你去清场」 | A* 接近 → RL 风筝/躲弹/输出 |

玩家指令示例：
- 「贴着我」→ `guard`
- 「去清怪」→ `assault`
- 「先打 Boss」→ `focus_target: boss`

LLM 的价值：把自然语言映射成语义清晰的战术状态，而非直接操控坐标。

### 2. 三层对话节奏（Roguelite 友好）

| 优先级 | 触发 | 示例 | 是否走 LLM |
|--------|------|------|------------|
| **P0 反射** | 圈即将炸、玩家濒死 | 「先出圈！」 | 否（reflex.py） |
| **P1 战术** | 怪潮/Boss 二阶段/精英怒 | 模板 + 可选 LLM 润色 | 规则为主 |
| **P2 社交** | 玩家主动聊天、战后休息 | 完整 LLM + 记忆检索 | 是 |
| **P3 闲谈** | 安全区、走廊 | 轻量 banter | 可选 LLM |

战斗中 P0/P1 占 80%，保证不卡顿；安全区才深度聊天。

### 3. Roguelite 元循环

**Run 内（单局祝福）**
- 祝福改变 RL 观测偏置或奖励塑形，而非重训模型
- `ally_guard+` → guard 减伤 ↑，assault 时 RL 对「离玩家过远」惩罚加重
- `ally_aggressive` → assault 攻速 ↑，RL 对「进射程」奖励 ↑
- LLM 在获得祝福时可解说（构筑叙事）

**Run 间（Meta 记忆）**
- dialogue 记忆按 `daily | important` 分层
- 下一局开局引用历史（「上次 Boss 房切 guard，这次还要吗？」）
- 好感/信任度影响默认姿态倾向

---

## LLM 与 RL 的接口设计

### LLM 只输出离散「战术 API」

```json
{
  "action_type": "command",
  "dialogue": "这波怪多，我先顶上去，你跟紧。",
  "emotion": "focused",
  "stance": "assault",
  "focus": "lowest_hp",
  "duration_sec": 8
}
```

**禁止** LLM 输出 `(vx, vy)` 或每帧动作。

### RL 只在 assault 射程内接管

- Episode 在交战距离内开局
- 脱离包络（dist > attack_range × 1.3）交回 A*
- 奖励只看战斗：伤害、受伤、射程区间、躲弹、动作平滑
- guard 永远不用 RL；RL 专精「突击微操」

### scene_info 共享黑板

```javascript
{
  mode: "battle",
  floor: 3,
  enemy_count: 5,
  boss_alive: true,
  player_hp: 45,
  ally_hp: 120,
  ally_stance: "assault",
  hazard_near_player: false,
  blessings: ["ally_guard", "focus_fire"],
  combat_mood: "surge"
}
```

---

## 强化学习训练建议

### 观测空间（109 维，战斗专家版）

- 自身：攻速 CD、HP、敌人数
- 主目标 8 维 + 次要敌人（最多 4 个）+ 威胁子弹/AOE（最多 8 个）
- 16 向射线 + 四壁距离
- 上帧动作 one-hot（LSTM 时序）
- 最危险威胁 TTA

### 奖励塑形（战友职责）

| 奖励项 | 含义 |
|--------|------|
| `+dmg_dealt` | 输出贡献 |
| `-dmg_taken` | 生存 |
| `+in_kite_range × LOS` | 风筝质量 |
| `+dodge_threat` | 躲弹成功 |
| `+player_proximity_penalty`（assault 时） | 别跑太远 |
| `+focus_target_bonus` | 集火指定目标 |

### 课程学习

1. Phase 1：1 个慢速 mob，开阔地
2. Phase 2：多 mob + 障碍布局（6 种 layout）
3. Phase 3：精英 + 弹幕密度 ↑
4. Phase 4：Boss 技能模式

使用 RecurrentPPO（LSTM）保留弹幕时序记忆，导出 ONNX 供浏览器推理。

---

## 延迟与成本策略

| 场景 | 策略 |
|------|------|
| 战斗帧循环 | 仅 RL + 规则，零 LLM |
| 战术切换 | LLM think 或规则，2~5s 冷却 |
| 玩家聊天 | 流式 LLM，异步 |
| 记忆写入 | 流结束后 daemon 线程 |
| LLM 失败 | 降级到 triggers.py 模板 |

---

## 落地优先级

1. **巩固双脑边界** — assault=RL+A*；guard=FSM+寻路；LLM 输出限制为 dialogue|stance|focus|noop
2. **Roguelite 构筑联动** — 每个 ally_* 祝福映射到 RL obs 偏置或 guard 参数
3. **跨局记忆玩法** — important 记忆影响开局默认姿态、对话语气
4. **可选：战术 LLM 微调** — 收集「玩家指令→stance」轨迹，微调 intent 分类小模型
5. **评测指标** — RL：存活/DPS/躲弹率/同框率；LLM：延迟/指令采纳率/主观评分

---

## 一句话总结

> **2D Roguelite 智能战友的最优玩法 = LLM 赋予「人格、记忆、战术意图」，RL 赋予「突击微操」，规则层兜底「危急反射」；Roguelite 通过祝福改参数、通过记忆改关系，而不是每局重训模型。**

当前项目架构已踩中业界最佳实践。下一步重点是把 Roguelite 构筑和跨局记忆做成玩法主轴。