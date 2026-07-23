# AI Roguelite Web Demo

《无间行录》网页演示：2D 俯视角 Roguelite 战斗 + 可对话、可自主思考的鬼差同伴「乌枭」。

游戏前端（`game/`）与 NPC 后端（`server/`）完全分离：浏览器负责战斗与渲染，Python 服务负责 LLM 对话、战术指令解析和自主决策。智能同伴采用 **LLM 大脑 + RL/规则四肢** 分层：LLM 负责说什么、切什么姿态；突击微操由浏览器内 ONNX 策略网络执行。

![模拟游戏界面-1](images/img1.png)

---

## 快速启动

### 环境要求

- Python 3.10+
- 可访问的 LLM API（默认配置为 DeepSeek 兼容接口）
- 现代浏览器（Chrome / Edge / Firefox 等）

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

复制配置模板并填入你的 API Key：

```bash
cp config.example.yaml config.yaml
```

编辑 `config.yaml`，至少修改 `llm.api_key`。也可用环境变量覆盖（优先级更高）：

```bash
export AI_NPC_LLM_API_KEY="sk-xxxx"
export AI_NPC_LLM_BASE_URL="https://api.deepseek.com"
export AI_NPC_LLM_MODEL="deepseek-chat"
```

配置解析顺序（后者覆盖前者）：

1. `server/npc_backend/config.py` 默认值
2. 项目根目录 `config.yaml`
3. 环境变量 `AI_NPC_LLM_API_KEY` / `AI_NPC_LLM_BASE_URL` / `AI_NPC_LLM_MODEL`

**嵌入模型说明：** 默认使用 `BAAI/bge-small-zh-v1.5`，缓存目录为 `models/`。若 `embeddings.local_files_only: true`（默认），需事先把模型放到 `models/`，否则启动会失败。首次使用可改为 `local_files_only: false`，让 HuggingFace 自动下载。

### 3. 导入世界观与角色设定（首次运行）

```bash
python scripts/import_world_setting.py
python scripts/import_persona_setting.py --npc-id wuxiao_01
```

- 世界观 / 角色设定为覆盖导入，可重复执行
- 运行中产生的对话记忆、本局事件、反思由游戏/服务自动追加，不会被导入脚本清空
- NPC ID 请使用 **`wuxiao_01`**（与前端硬编码一致）

### 4. 启动服务（需要两个终端）

**终端 1 — NPC API：**

```bash
python run.py
```

监听 `http://127.0.0.1:5100`（`0.0.0.0:5100`），提供对话、战术指令、自主思考与记忆写入接口。

**终端 2 — 游戏页面：**

```bash
python run_game.py
```

默认监听 `http://127.0.0.1:8082`（端口可通过参数修改，如 `python run_game.py 8080`）。

### 5. 打开游戏

浏览器访问：

```
http://127.0.0.1:8082
```

若对话或乌枭自主发言无响应，先确认 NPC API 已启动，并访问 `http://127.0.0.1:5100/health` 应返回 `{"status":"ok"}`。

**内网分享：** 前端可只部署 `game/` 目录（须包含 `game/resources/`）。请把 `game/game.js` 里的 `NPC_API` 改成你本机内网 IP（例如 `http://192.168.x.x:5100`），并保持后端在运行。

---

## 玩法介绍

### 背景

你是坠入阴司的魂体，在多层「狱」中向前探索。身边的鬼差**乌枭**嘴臭但靠谱——战斗里他会跟着你打，聊天框里可以用自然语言指挥他，他也会根据战况自己开口。

### 操作

| 操作 | 按键 |
|------|------|
| 移动 | WASD / 方向键 |
| 射击 | 空格 |
| 闪避 | Shift |
| 技能 | E |
| 语音（按住） | L |
| 与乌枭对话 | 右侧聊天框输入后发送 |

### 关卡流程

1. **探索狱房**：每层的房间由走廊、战斗房、精英房、Boss 房等组成，清完当前房敌人后门会打开。
2. **靠门前进**：清房后移动到右侧门边进入下一间。
3. **镇压 Boss**：每层尽头击败 Boss。
4. **选择狱印**：Boss 后从三张「狱印」中选一张强化本局，然后进入下一层。

狱印分三类，可重复叠层（有上限）：

- **魂体印**：强化玩家（移速、伤害、闪避、生命等）
- **鬼差印**：强化乌枭（守护减伤、回血、集火等）
- **契约印**：强化配合效果

### 战斗要点

- 场景中有柱子、断墙等掩体，可利用障碍物阻挡视线和弹幕。
- 敌人类型包括小怪、精英、Boss；部分 Boss 会释放范围技能（如地面圈、荆棘墙等）。
- 玩家与乌枭都有独立血量。
- **乌枭倒地（hp≤0）**：不攻击，自动寻路贴到玩家身边「挂着」；过层回血或其它方式使 hp>0 后恢复正常行动。

### 乌枭战斗姿态（四肢 / Limb）

通过聊天下达战术指令，无需额外按钮。当前实现两种姿态（对应不同控制器）：

| 姿态 | 控制器 | 行为概要 |
|------|--------|----------|
| **守护（guard）** | 规则 + A* | 默认。贴身跟随、射程内攻击；贴近时为你减伤。玩家危急时可救援（回血 + 护盾）。倒地时走贴挂逻辑。 |
| **突击（assault）** | A* 接近 + **RL ONNX** | 前压交战；进入战斗距离后由 `assault_skirmish` 策略网络控制走位与输出。加载失败则降级规则 AI。 |

策略模型路径（唯一）：

```text
game/resources/limbs/assault_skirmish/policy.onnx
```

注册表见 `game/resources/limbs/manifest.json`。

指令示例：

- 「回来守护我」「贴着我」→ 守护
- 「上去打」「突击」→ 突击
- 其他内容 → 正常对话

HUD 会显示当前姿态；突击时还会标注当前是策略模型还是规则 AI 在控制。

---

## 智能 NPC 设计

乌枭不是纯聊天机器人。后端把**玩家主动对话**和**战斗中的自主行为**分成两条链路，共用记忆与角色设定，但决策节奏不同。

### 整体架构

```text
浏览器（game/）
  ├─ 战斗循环：scene_info + run_id
  ├─ POST /api/chat/stream     ← 玩家发消息
  ├─ POST /api/npc/think       ← 自主思考
  ├─ POST /api/memory/run_event  ← 本局事件
  └─ POST /api/memory/reflection ← 局末反思
  └─ 突击微操：本地加载 resources/limbs/*/policy.onnx

NPC API（server/）
  ├─ 意图分类：对话 vs 战术指令（guard / assault）
  ├─ 记忆检索：world / persona / dialogue / run_event / reflection
  ├─ 工作记忆：commitment、最近指令（不入库）
  ├─ 流式生成：对话 + 情绪标签
  └─ 自主决策：noop / 切姿态 / 主动说话
```

### 玩家对话（`/api/chat/stream`）

- 识别为**战术指令** → 返回 `command`，前端切换姿态，写入本局事件与工作记忆承诺
- 识别为**对话** → 流式回复；结束后分级写入 `dialogue`（daily / important）

请求需带 `run_id`（前端每局自动生成）。

### 自主思考（`/api/npc/think`）

| 优先级 | 典型场景 | 处理方式 |
|--------|----------|----------|
| P0 危急 | 乌枭/玩家濒死、玩家发呆挨打 | 规则优先：求援、切守护等 |
| P1 场景 | 换层、Boss、敌群突变 | 规则提示 + LLM |
| P2 战术 | 弹幕、地面圈、视线被挡 | 战术提示 + LLM |
| P3 日常 | 长时间无对话 | LLM 决定是否碎嘴；可 noop |

结果：`noop` / `command` / `dialogue`。带发言冷却、去重、姿态冷却等约束。

### 记忆系统

统一用 `memory_type` 区分；**本局**靠 `run_id` 界定。

| type | 存储 | 作用范围 | 用途 |
|------|------|----------|------|
| `world` | ChromaDB | 跨局 | 世界观（导入） |
| `persona` | ChromaDB | 跨局 | 角色设定（导入） |
| `dialogue` | ChromaDB | 跨局 | 对话；tier=`daily` / `important` |
| `run_event` | ChromaDB | **默认仅当前 run_id** | 本局事件摘要（倒地、切姿态、进层等） |
| `reflection` | ChromaDB | 跨局 | 局末反思/偏好 |
| 工作记忆 | 进程内 | 会话 | 承诺 limb、最近指令（**不入向量库**） |

检索按 channel 控制预算（如战斗 think 偏 `run_event` + important；新局 `run_start` 不灌上一局事件）。Prompt 分块：`[角色设定][世界设定][长期反思][对话记忆-*][本局事件][工作记忆]`。

详见 `docs/memory-routing.md`。

### 情绪与表达

对话末尾可由模型输出 `<emotion>...</emotion>`，前端映射为颜文字气泡。自主发言同样走气泡。

### 角色：乌枭（`wuxiao_01`）

设定见 `lore/persona_setting.md`：嘴臭、话痨、重规矩的鬼差。语气随楼层、敌情、血量、姿态与近期记忆变化。

---

## 强化学习（四肢）

每个策略网络**独占一个肢目录**，训练产物与代码同目录；导出只写入 `game/resources/`。

```text
rl/
  limbs/
    assault_skirmish/          # 现有「突击」网络
      env.py train.py export_onnx.py
      checkpoints/ logs/       # 训练中间产物（gitignore）
      config.yaml README.md

game/resources/limbs/
  manifest.json
  assault_skirmish/
    policy.onnx                # 浏览器加载（可提交）
    meta.json
```

### 训练与导出

在项目根目录：

```bash
# 训练
python -m rl.limbs.assault_skirmish.train
python -m rl.limbs.assault_skirmish.train --timesteps 50000 --run-name debug
python -m rl.limbs.assault_skirmish.train \
  --resume rl/limbs/assault_skirmish/checkpoints/assault_best/best_model.zip

# 导出到 game/resources（无需再手动 cp）
python -m rl.limbs.assault_skirmish.export_onnx --verify
```

- **不要**在 `rl/` 或 `game/` 根目录再放 onnx / checkpoints。
- 新肢：复制 `rl/limbs/assault_skirmish/` 为新目录，改 env 与 `LIMB_ID`，并更新 `manifest.json`。

详见 `docs/rl-limb-layout.md`、`rl/limbs/assault_skirmish/README.md`。

---

## 项目结构

```text
├─ run.py / run_game.py           # 启动入口
├─ config.example.yaml            # 配置模板
├─ game/                          # 游戏客户端
│  ├─ index.html / game.js / ...
│  └─ resources/limbs/            # 部署用 RL 模型（唯一）
├─ server/                        # NPC API（Flask）
│  └─ npc_backend/                # 对话、记忆、自主、工作记忆
├─ rl/limbs/<limb_id>/            # 一肢一目录：env/train/export/checkpoints
├─ lore/                          # 世界观与角色文本
├─ scripts/                       # ChromaDB 导入脚本
├─ docs/                          # 设计与规格
│  ├─ memory-routing.md
│  ├─ rl-limb-layout.md
│  └─ 2d-roguelite-intelligent-npc-companion-design.md
├─ data/ / models/                # 运行时数据（gitignore）
└─ images/                        # README 配图
```

### 主要 API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/api/chat/stream` | 玩家对话 / 指令（NDJSON 流） |
| POST | `/api/npc/think` | 自主思考（NDJSON 流） |
| POST | `/api/memory/run_event` | 写入本局事件 |
| POST | `/api/memory/reflection` | 写入跨局反思 |

---

## 相关文档

| 文档 | 内容 |
|------|------|
| `docs/memory-routing.md` | 记忆 type、run_id、存查与 Prompt 规格 |
| `docs/rl-limb-layout.md` | 多肢目录与导出规范 |
| `docs/2d-roguelite-intelligent-npc-companion-design.md` | 智能 NPC 双脑设计方案 |
| `rl/limbs/assault_skirmish/README.md` | 突击肢训练说明 |
| `game/RETRAIN_DEFERRED.md` | 环境同步与微调注意点 |
