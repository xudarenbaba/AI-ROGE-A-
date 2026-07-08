const NPC_API = "http://127.0.0.1:5100";
const NPC_AUTONOMY_BUILD = "autonomy9";
const GAME_BUILD = "upgrade1";

// ── RL 推理模块（assault 姿态）────────────────────────────────────────────────
// onnxruntime-web 从 CDN 加载；如需离线部署可改为本地路径。
// 模型文件 assault_policy.onnx 放在 game/ 目录下。
// 在模型加载完成前，assault 姿态降级为规则 AI。

const _RL_MODEL_PATH = "assault_policy.onnx";

// 与 rl/env.py ACTION_VECTORS 严格对齐（索引 0-8）
const _RL_ACTION_VECTORS = [
  [ 0.0,  0.0],          // 0 静止
  [ 0.0, -1.0],          // 1 上
  [ 0.7071, -0.7071],    // 2 右上
  [ 1.0,  0.0],          // 3 右
  [ 0.7071,  0.7071],    // 4 右下
  [ 0.0,  1.0],          // 5 下
  [-0.7071,  0.7071],    // 6 左下
  [-1.0,  0.0],          // 7 左
  [-0.7071, -0.7071],    // 8 左上
];

// 与 rl/env.py 常量对齐（战斗专家版：射线检测 + LSTM，OBS_DIM=109）
const _RL_CANVAS_W        = 900.0;
const _RL_CANVAS_H        = 540.0;
const _RL_DIAG            = Math.hypot(_RL_CANVAS_W, _RL_CANVAS_H);  // ~1051
const _RL_MAX_ENEMIES     = 5;
const _RL_MAX_BULLETS     = 8;
const _RL_MAX_BULLET_DIST = 200.0;
const _RL_ASSAULT_INTERVAL = 0.45;
const _RL_BULLET_TTL       = 2.2;
const _RL_N_ACTIONS       = 9;

// assault FSM / 动作后处理（与优化方案对齐）
const _COMBAT_ENTER_LOS_FRAMES = 10;
const _COMBAT_EXIT_DIST_MUL    = 1.3;
const _COMBAT_HOLD_MAX_SEC     = 2.5;    // 1c：已废除 hold 阶段，常量保留避免牵连
const _COMBAT_LOS_LOST_EXIT_FRAMES = 24; // 1c：combat 中连续丢 LOS ≥24帧(0.4s)才退回 approach
const _ENGAGE_POINT_REACH      = 20;
const _SAFE_HOLD_TTA_SEC       = 0.40;  // 方案A：不再用于强制静止（保留常量备用）
const _EMERGENCY_DODGE_TTA_SEC = 0.25;
const _MIN_ACTION_HOLD_FRAMES  = 3;     // 方案A：5 → 3（约 50ms，防抖更轻）
const _ACTION_SWITCH_MARGIN    = 0.10;  // 方案A：0.30 → 0.10（降低移动间切换门槛）

// 射线检测（与 env.py N_RAYS / RAY_MAX_DIST / RAY_STEPS 严格对齐）
const _RL_N_RAYS       = 16;
const _RL_RAY_MAX_DIST = 260.0;
const _RL_RAY_STEPS    = 26;
// 16 方向单位向量（每 22.5°）
const _RL_RAY_DIRS = [];
for (let i = 0; i < _RL_N_RAYS; i++) {
  const a = 2 * Math.PI * i / _RL_N_RAYS;
  _RL_RAY_DIRS.push([Math.cos(a), Math.sin(a)]);
}

// 段长与 env.py 严格对齐（战斗专家版：删绝对坐标/距离变化量/历史最小距离）
const _RL_SEG1 = 3;                            // 自身：攻击冷却, hp, 敌人数量
const _RL_SEG2 = 8;
const _RL_SEG3 = (_RL_MAX_ENEMIES - 1) * 5;   // 20
const _RL_SEG4 = _RL_MAX_BULLETS * 6;          // 48
const _RL_SEG5 = _RL_N_RAYS;                   // 16（射线检测）
const _RL_SEG6 = 4;
const _RL_SEG7 = _RL_N_ACTIONS;               // 9
const _RL_SEG8 = 1;                            // 最危险子弹 TTA
const _RL_OBS_DIM = _RL_SEG1 + _RL_SEG2 + _RL_SEG3 + _RL_SEG4 + _RL_SEG5 + _RL_SEG6 + _RL_SEG7 + _RL_SEG8; // 109

// LSTM hidden state 维度（与训练模型对齐，导出后从 meta json 也可读取）
const _RL_LSTM_HIDDEN = 128;
const _RL_LSTM_LAYERS = 1;

// 帧间状态
let _rlPrevAction = 0;
let _rlRawAction = 0;
let _rlAppliedAction = 0;
let _rlActionHoldRemain = 0;
let _rlLastLogits = null;
// 方案A 诊断：统计 RL 原始动作(argmax)分布；若几乎全集中在 index 0(静止)，说明模型已退化为站桩，需走方案B 重训
let _rlRawHist = new Array(_RL_N_ACTIONS).fill(0);
let _rlRawHistCount = 0;
// LSTM hidden state（Float32Array，形状 [layers, 1, hidden]）
let _rlHidden = null;   // h state
let _rlCell   = null;   // c state

function _rlResetLstmState() {
  const size = _RL_LSTM_LAYERS * 1 * _RL_LSTM_HIDDEN;
  _rlHidden = new Float32Array(size);   // 全零
  _rlCell   = new Float32Array(size);
  _rlPrevAction = 0;
  _rlRawAction = 0;
  _rlAppliedAction = 0;
  _rlActionHoldRemain = 0;
  _rlLastLogits = null;
}

// 状态：null = 未加载，"loading" = 加载中，InferenceSession = 就绪，"error" = 失败
let _rlSession = null;
let _rlLoadState = "idle";  // "idle" | "loading" | "ready" | "error"

async function _rlLoadModel() {
  if (_rlLoadState !== "idle") return;
  _rlLoadState = "loading";
  try {
    if (typeof ort === "undefined") {
      throw new Error("onnxruntime-web not loaded — check network or CDN");
    }
    ort.env.wasm.numThreads = 1;  // 避免 SharedArrayBuffer 跨域限制
    _rlSession = await ort.InferenceSession.create(_RL_MODEL_PATH, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    _rlResetLstmState();   // 初始化 LSTM hidden state 为全零
    _rlLoadState = "ready";
    console.log("[RL] assault_policy.onnx (LSTM) loaded, RL mode active.");
  } catch (e) {
    _rlLoadState = "error";
    console.error("[RL] Failed to load model:", e);
  }
}

// LOS 射线步进检测（与 env.py _has_line_of_sight 对齐，steps=16）
function _rlHasLOS(ax, ay, bx, by) {
  const steps = 16;
  for (let i = 1; i < steps; i++) {
    const t  = i / steps;
    const px = ax + (bx - ax) * t;
    const py = ay + (by - ay) * t;
    for (const o of state.obstacles) {
      if (px >= o.x && px <= o.x + o.w && py >= o.y && py <= o.y + o.h) return false;
    }
  }
  return true;
}

// 圆盘碰撞 TTA（秒）；未命中或背向返回 BULLET_TTL（与 env.py _bullet_time_to_ally 对齐）
function _rlBulletTTASec(b, ally) {
  const rx = ally.x - b.x;
  const ry = ally.y - b.y;
  const vx = b.vx;
  const vy = b.vy;
  const v2 = vx * vx + vy * vy;
  if (v2 < 1e-6) return _RL_BULLET_TTL;
  const hitR = ally.radius + (b.radius || 4);
  const a = v2;
  const bCoef = 2 * (rx * vx + ry * vy);
  const c = rx * rx + ry * ry - hitR * hitR;
  const disc = bCoef * bCoef - 4 * a * c;
  if (disc < 0) return _RL_BULLET_TTL;
  const sqrtDisc = Math.sqrt(disc);
  let tHit = null;
  for (const t of [(-bCoef - sqrtDisc) / (2 * a), (-bCoef + sqrtDisc) / (2 * a)]) {
    if (t >= 0 && (tHit === null || t < tHit)) tHit = t;
  }
  if (tHit === null || tHit > _RL_BULLET_TTL) return _RL_BULLET_TTL;
  return tHit;
}

function _rlBulletWillHit(b, ally) {
  return _rlBulletTTASec(b, ally) < _RL_BULLET_TTL - 1e-4;
}

// 归一化 TTA（obs / 排序用）
function _rlBulletTTA(b, ally) {
  return Math.min(1.0, Math.max(0.0, _rlBulletTTASec(b, ally) / _RL_BULLET_TTL));
}

function _rlMinBulletTTASec() {
  const ally = state.ally;
  let minSec = _RL_BULLET_TTL;
  for (const b of state.enemyBullets) {
    if (Math.hypot(b.x - ally.x, b.y - ally.y) > _RL_MAX_BULLET_DIST) continue;
    if (!_rlBulletWillHit(b, ally)) continue;
    minSec = Math.min(minSec, _rlBulletTTASec(b, ally));
  }
  return minSec;
}

function _rlEmergencyDodgeAction() {
  const ally = state.ally;
  let worst = null;
  let worstTta = Infinity;
  for (const b of state.enemyBullets) {
    if (Math.hypot(b.x - ally.x, b.y - ally.y) > _RL_MAX_BULLET_DIST) continue;
    const ttaSec = _rlBulletTTASec(b, ally);
    if (ttaSec >= _EMERGENCY_DODGE_TTA_SEC || !_rlBulletWillHit(b, ally)) continue;
    if (ttaSec < worstTta) { worstTta = ttaSec; worst = b; }
  }
  if (!worst) return null;
  const spd = Math.hypot(worst.vx, worst.vy) || 1;
  const ux = worst.vx / spd;
  const uy = worst.vy / spd;
  const dx = ally.x - worst.x;
  const dy = ally.y - worst.y;
  const lateral = -uy * dx + ux * dy;
  const sign = lateral >= 0 ? 1 : -1;
  const ex = -uy * sign;
  const ey = ux * sign;
  let bestA = 1;
  let bestDot = -Infinity;
  for (let i = 1; i < _RL_ACTION_VECTORS.length; i += 1) {
    const [vx, vy] = _RL_ACTION_VECTORS[i];
    const dot = vx * ex + vy * ey;
    if (dot > bestDot) { bestDot = dot; bestA = i; }
  }
  return bestA;
}

function _rlResolveAction(los, minTtaSec) {
  // 方案A：移除 Safe-Hold 强制静止，把“安全时动不动”的决定权交还 RL 模型；
  // 仅保留“紧急闪避”硬逻辑 + 温和防抖（只抑制移动方向间的高频互切）。
  // 参数 los 暂保留以兼容调用处签名（当前逻辑不再使用）。
  void los;

  // 1) 紧急闪避：子弹即将命中 → 强制横向规避（最高优先级）
  const dodge = minTtaSec < _EMERGENCY_DODGE_TTA_SEC ? _rlEmergencyDodgeAction() : null;
  if (dodge !== null) {
    _rlAppliedAction = dodge;
    _rlActionHoldRemain = _MIN_ACTION_HOLD_FRAMES;
    _rlPrevAction = dodge;
    return dodge;
  }

  // 2) 期望动作 = RL 原始输出（不再被 Safe-Hold 摁成静止）
  let desired = _rlRawAction;

  // 3) 防抖：只抑制“移动→移动”的高频互切，绝不阻止“静止→移动”的启动
  if (_rlLastLogits
      && desired !== _rlAppliedAction
      && desired !== 0 && _rlAppliedAction !== 0
      && _rlLastLogits[desired] - _rlLastLogits[_rlAppliedAction] < _ACTION_SWITCH_MARGIN) {
    desired = _rlAppliedAction;
  }

  // 4) 最短保持帧：切换后保持若干帧，平滑残余抖动
  if (desired !== _rlAppliedAction && _rlActionHoldRemain > 0) {
    _rlActionHoldRemain -= 1;
    _rlPrevAction = _rlAppliedAction;
    return _rlAppliedAction;
  }
  _rlActionHoldRemain = desired !== _rlAppliedAction
    ? _MIN_ACTION_HOLD_FRAMES
    : Math.max(0, _rlActionHoldRemain - 1);
  _rlAppliedAction = desired;
  _rlPrevAction = desired;
  return desired;
}

// 射线检测：从 (ox,oy) 沿 (dx,dy) 步进，返回到障碍物/边界的归一化距离
// 与 env.py _raycast_obstacle 严格对齐
function _rlRaycast(ox, oy, dx, dy) {
  const stepLen = _RL_RAY_MAX_DIST / _RL_RAY_STEPS;
  for (let i = 1; i <= _RL_RAY_STEPS; i++) {
    const d  = i * stepLen;
    const px = ox + dx * d;
    const py = oy + dy * d;
    if (px < 0 || px > _RL_CANVAS_W || py < 0 || py > _RL_CANVAS_H) {
      return Math.min(1.0, d / _RL_RAY_MAX_DIST);
    }
    for (const o of state.obstacles) {
      if (px >= o.x && px <= o.x + o.w && py >= o.y && py <= o.y + o.h) {
        return d / _RL_RAY_MAX_DIST;
      }
    }
  }
  return 1.0;
}

// === RL FROZEN: _rlBuildObs / _rlResolveAction / ONNX — 训完前勿改 ===
// 构建 109 维观测向量，与 rl/env.py _get_obs() 严格对齐（战斗专家版）
function _rlBuildObs(attackCd) {
  const obs  = new Float32Array(_RL_OBS_DIM);
  const ally = state.ally;
  let idx    = 0;

  // ── 段1：自身状态 (3维) ─────────────────────────────────────────────────
  // 战斗专家：去掉绝对坐标(战斗平移不变)，改放 hp（按血量调节冒险程度）
  obs[idx++] = Math.min(1.0, attackCd / _RL_ASSAULT_INTERVAL);
  obs[idx++] = ally.hp / ally.maxHp;
  obs[idx++] = state.enemies.length / 11.0;

  // ── 段2：主目标敌人（最近，8维）──────────────────────────────────────────
  const target = _rlNearestEnemy();
  if (target !== null) {
    const dx   = target.x - ally.x;
    const dy   = target.y - ally.y;
    const dist = Math.hypot(dx, dy);
    const shootCdMax = target.kind === "boss" ? 1.2 : 1.6;
    const los  = _rlHasLOS(ally.x, ally.y, target.x, target.y);
    const inRange = (dist > 65 && dist < 110) ? 1.0 : 0.0;  // 与 env ASSAULT_KITE_RANGE(65) 对齐
    obs[idx++] = dx / _RL_CANVAS_W;
    obs[idx++] = dy / _RL_CANVAS_H;
    obs[idx++] = dist / _RL_DIAG;
    obs[idx++] = target.hp / target.maxHp;
    obs[idx++] = target.kind === "boss" ? 1.0 : 0.0;
    obs[idx++] = target.shootCd / shootCdMax;
    obs[idx++] = los ? 1.0 : 0.0;   // LOS 标志
    obs[idx++] = inRange;            // 有效射程标志
  } else {
    idx += 8;
  }

  // ── 段3：其余最多 4 个敌人 (5维/敌) ────────────────────────────────────
  const others = state.enemies
    .filter(e => e !== target)
    .map(e => ({ e, d: Math.hypot(e.x - ally.x, e.y - ally.y) }))
    .sort((a, b) => a.d - b.d)
    .slice(0, _RL_MAX_ENEMIES - 1);
  for (const { e } of others) {
    const dx = e.x - ally.x;
    const dy = e.y - ally.y;
    obs[idx++] = dx / _RL_CANVAS_W;
    obs[idx++] = dy / _RL_CANVAS_H;
    obs[idx++] = Math.hypot(dx, dy) / _RL_DIAG;
    obs[idx++] = e.hp / e.maxHp;
    obs[idx++] = e.kind === "boss" ? 1.0 : 0.0;
  }
  idx += (_RL_MAX_ENEMIES - 1 - others.length) * 5;

  // ── 段4：最多 8 颗威胁子弹（6维/颗，按 TTA 排序）────────────────────────
  const threatBullets = state.enemyBullets
    .filter(b => Math.hypot(b.x - ally.x, b.y - ally.y) < _RL_MAX_BULLET_DIST)
    .map(b => ({ b, tta: _rlBulletTTA(b, ally) }))
    .sort((a, b) => a.tta - b.tta)   // TTA 越小越危险，排在前面
    .slice(0, _RL_MAX_BULLETS);
  for (const { b, tta } of threatBullets) {
    const dx   = b.x - ally.x;
    const dy   = b.y - ally.y;
    const dist = Math.hypot(dx, dy);
    const bspd = Math.hypot(b.vx, b.vy) || 1.0;
    obs[idx++] = dx / _RL_CANVAS_W;
    obs[idx++] = dy / _RL_CANVAS_H;
    obs[idx++] = b.vx / bspd;
    obs[idx++] = b.vy / bspd;
    obs[idx++] = dist / _RL_MAX_BULLET_DIST;
    obs[idx++] = tta;                 // 预测碰撞时间（0=即将命中）
  }
  idx += (_RL_MAX_BULLETS - threatBullets.length) * 6;

  // ── 段5：射线检测（16 方向，替换矩形障碍物 obs）─────────────────────────
  for (const [rdx, rdy] of _RL_RAY_DIRS) {
    obs[idx++] = _rlRaycast(ally.x, ally.y, rdx, rdy);
  }

  // ── 段6：到四壁距离 (4维) ────────────────────────────────────────────────
  obs[idx++] = ally.y / _RL_CANVAS_H;
  obs[idx++] = (_RL_CANVAS_H - ally.y) / _RL_CANVAS_H;
  obs[idx++] = ally.x / _RL_CANVAS_W;
  obs[idx++] = (_RL_CANVAS_W - ally.x) / _RL_CANVAS_W;

  // ── 段7：上帧动作 one-hot (9维) ─────────────────────────────────────────
  obs[idx + _rlPrevAction] = 1.0;
  idx += _RL_N_ACTIONS;

  // ── 段8：最危险子弹 TTA (1维) ────────────────────────────────────────────
  // 所有威胁子弹中 TTA 最小值，0=即将命中，1=无威胁
  let minTTA = 1.0;
  for (const b of state.enemyBullets) {
    if (Math.hypot(b.x - ally.x, b.y - ally.y) < _RL_MAX_BULLET_DIST) {
      const tta = _rlBulletTTA(b, ally);
      if (tta < minTTA) minTTA = tta;
    }
  }
  obs[idx++] = minTTA;

  return obs;
}

function _rlNearestEnemy() {
  // LOS 加权评分，与 rl/env.py _nearest_enemy 严格对齐
  // 评分 = 距离 × (LOS通畅 ? 1.0 : 1.3)，轻度偏好有视线且近的敌人
  if (state.enemies.length === 0) return null;
  const ally = state.ally;
  let best = null;
  let bestScore = Infinity;
  for (const e of state.enemies) {
    const d     = Math.hypot(e.x - ally.x, e.y - ally.y);
    const los   = _rlHasLOS(ally.x, ally.y, e.x, e.y);
    const score = d * (los ? 1.0 : 1.3);
    if (score < bestScore) { bestScore = score; best = e; }
  }
  return best;
}

// 异步推理：提交本帧 obs，下帧用 _rlRawAction + 后处理层输出实际移动
let _rlInferring = false;

function _rlInferAsync(attackCd) {
  if (_rlLoadState !== "ready" || _rlInferring) return;
  if (!_rlHidden || !_rlCell) _rlResetLstmState();
  _rlInferring = true;

  const obs = _rlBuildObs(attackCd);
  const hShape = [_RL_LSTM_LAYERS, 1, _RL_LSTM_HIDDEN];
  const feeds = {
    obs:  new ort.Tensor("float32", obs, [1, _RL_OBS_DIM]),
    h_in: new ort.Tensor("float32", _rlHidden, hShape),
    c_in: new ort.Tensor("float32", _rlCell,   hShape),
  };
  _rlSession.run(feeds).then(output => {
    const logits = output.logits.data;
    _rlLastLogits = logits;
    let best = 0;
    for (let i = 1; i < logits.length; i++) {
      if (logits[i] > logits[best]) best = i;
    }
    _rlRawAction = best;
    // 方案A 诊断：每 120 次推理打印一次原始动作分布（idx0=静止, 1..8=八方向）
    _rlRawHist[best] += 1;
    if (++_rlRawHistCount >= 120) {
      console.log("[RL] rawAction 分布/120帧 [静,上,右上,右,右下,下,左下,左,左上]:", _rlRawHist.join(","));
      _rlRawHist = new Array(_RL_N_ACTIONS).fill(0);
      _rlRawHistCount = 0;
    }
    _rlHidden = output.h_out.data;
    _rlCell   = output.c_out.data;
    _rlInferring = false;
  }).catch((e) => {
    console.error("[RL] inference error:", e);
    _rlRawAction = 0;
    _rlAppliedAction = 0;
    _rlPrevAction = 0;
    state.ally.combatPhase = "approach";
    state.ally.navPath = null;
    _rlInferring = false;
  });
}

const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");
const hudStats = document.getElementById("hudStats");
const chatLog = document.getElementById("chatLog");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");

const FLOOR_META = [
  { name: "拔舌狱", tone: "#542127", accent: "#c65f52", haze: "#2a0f12", hint: "妄言会在这里变成锁链，少站直线位。" },
  { name: "剪刀狱", tone: "#4a202b", accent: "#ce7a6e", haze: "#1f1016", hint: "敌群会连线增伤，优先切断牵引目标。" },
  { name: "铁树狱", tone: "#40272c", accent: "#ad7c5a", haze: "#171115", hint: "冲刺别贪，刺林会惩罚错误位移。" },
  { name: "孽镜狱", tone: "#2f2638", accent: "#8f7fe0", haze: "#120f1d", hint: "镜像会复刻行动，先辨认真身再集火。" },
  { name: "蒸笼狱", tone: "#5b2a1d", accent: "#de8a58", haze: "#24120f", hint: "热压会叠层，别在闷区久留。" },
  { name: "铜柱狱", tone: "#5a2e19", accent: "#d88946", haze: "#28150a", hint: "火线扩散很快，卡掩体边缘走。" },
  { name: "刀山狱", tone: "#402826", accent: "#ba8f88", haze: "#1b1314", hint: "地形比怪更危险，先抢安全落脚点。" },
  { name: "冰山狱", tone: "#1d2a3f", accent: "#7ab8d1", haze: "#101722", hint: "冻结后伤害翻倍，别硬吃连续弹道。" },
  { name: "油锅狱", tone: "#52301a", accent: "#d48e4f", haze: "#23160b", hint: "油溅有抛物轨迹，保持横向机动。" },
  { name: "牛坑狱", tone: "#3f281f", accent: "#be7c57", haze: "#190f0a", hint: "冲锋波次会连段，留一个位移保命。" },
  { name: "石压狱", tone: "#3f3132", accent: "#a6978d", haze: "#161314", hint: "塌陷有读秒，宁可少打也别贪刀。" },
  { name: "舂臼狱", tone: "#49352b", accent: "#b89271", haze: "#19120f", hint: "节拍是活路，跟着环境节律移动。" },
  { name: "血池狱", tone: "#5a1f29", accent: "#d66b73", haze: "#2a0e14", hint: "侵蚀会滚雪球，别在血潭硬站桩。" },
  { name: "枉死狱", tone: "#30212d", accent: "#a88db9", haze: "#18101d", hint: "怨魂会扰乱目标，先解控再输出。" },
  { name: "磔刑狱", tone: "#4f2623", accent: "#d37669", haze: "#220f0f", hint: "分部位破坏更高效，别均摊伤害。" },
  { name: "火山狱", tone: "#66261c", accent: "#ee7b42", haze: "#2e120a", hint: "喷发前有前兆，提前换位。" },
  { name: "石磨狱", tone: "#41332a", accent: "#b8a27d", haze: "#1c1410", hint: "碾压周期固定，记住三拍后撤。" },
  { name: "无间边狱", tone: "#2a182b", accent: "#b070da", haze: "#140b16", hint: "终层按因果结算，稳住比抢速更重要。" },
];

const state = {
  player: {
    x: 180, y: 260, radius: 14, hp: 160, maxHp: 160, speed: 220,
    attackCd: 0, shieldCd: 0, dashCd: 0, dashInvuln: 0,
  },
  ally: {
    x: 220,
    y: 290,
    radius: 13,
    hp: 160,
    maxHp: 160,
    speed: 200,
    attackCd: 0,
    rescueCd: 0,
    stance: "guard",
    bubble: "",
    bubbleUntil: 0,
    dead: false,   // 死亡 flag，防止 checkDefeat 每帧重置气泡
    // ── assault 分层控制 FSM ──────────────────────────────────────────────
    combatPhase: "approach",  // 1c：两态 "approach"(A* 接近) | "combat"(RL 战斗)
    navPath: null,
    navReplanCd: 0,
    navGoal: null,
    engagePoint: null,
    losStableFrames: 0,
    losLostFrames: 0,         // 1c：combat 中连续丢 LOS 帧数，≥阈值退回 approach
    focusMode: null,
    focusUntil: 0,
  },
  combo: 0,
  comboTimer: 0,
  maxCombo: 0,
  playerDamageMul: 1,
  guardDamageReduction: 0.15,
  dashCdMul: 1,
  blessingShieldMul: 1,
  blessingsTaken: [],
  blessingChoices: [],
  enemies: [],
  playerBullets: [],
  allyBullets: [],
  enemyBullets: [],
  bossAlive: true,
  obstacles: [],
  keys: {},
  result: "",
  playerId: "player_web_demo",
  floor: 1,
  floorState: "playing", // "playing" | "blessing_pick" | "clear"
  transitionTimer: 0,
  cinders: [],
};

// ── 关卡配置 ──────────────────────────────────────────────────────────────────

function floorScale() {
  const f = state.floor - 1;
  return {
    hpMul:    1 + f * 0.30,
    speedMul: 1 + f * 0.06,
    mobCount: Math.min(3 + Math.floor(f * 1.2), 10),
  };
}

const MOB_BASE_POSITIONS = [
  { x: 520, y: 100 }, { x: 650, y: 150 }, { x: 780, y: 100 },
  { x: 560, y: 380 }, { x: 700, y: 430 }, { x: 820, y: 360 },
  { x: 700, y: 270 }, { x: 820, y: 200 }, { x: 760, y: 430 },
  { x: 850, y: 130 },
];

// 在障碍物附近随机采样安全出生点，与 rl/env.py _safe_spawn 逻辑对齐
function safeSpawnPos(baseX, baseY, radius, fallbackX, fallbackY) {
  for (let i = 0; i < 20; i += 1) {
    const x = baseX + (Math.random() - 0.5) * 40;
    const y = baseY + (Math.random() - 0.5) * 40;
    if (!collidesWithObstacle(x, y, radius)) return { x, y };
  }
  return { x: fallbackX, y: fallbackY };
}

function spawnEnemies(emitNpcEvents = true) {
  const { hpMul, speedMul, mobCount } = floorScale();
  const mobs = [];
  for (let i = 0; i < mobCount; i += 1) {
    const base = MOB_BASE_POSITIONS[i % MOB_BASE_POSITIONS.length];
    const pos  = safeSpawnPos(base.x, base.y, 12, 750, 270);
    mobs.push({
      kind: "mob",
      x: pos.x, y: pos.y,
      hp: Math.round(30 * hpMul),
      maxHp: Math.round(30 * hpMul),
      radius: 12,
      speed: 42 * speedMul,
      shootCd: Math.random() * 1.0 + 0.8,
    });
  }
  const bossPos = safeSpawnPos(820, 270, 20, 830, 400);
  mobs.push({
    kind: "boss",
    x: bossPos.x, y: bossPos.y,
    hp: Math.round(200 * hpMul),
    maxHp: Math.round(200 * hpMul),
    radius: 20,
    speed: 32 * speedMul,
    shootCd: 0.8,
  });
  state.enemies = mobs;
  state.bossAlive = true;
  if (emitNpcEvents) npcPushEvent("boss_spawn", { floor: state.floor });
}

// ── 障碍物（布局见 obstacles/layouts.js，碰撞盒不变）────────────────────────────

// assault APPROACH 阶段的 A* 导航栅格（障碍物变更时重建）
let _navGrid = null;

function generateObstacles() {
  state.obstacles = window.GameObstacles.getForFloor(state.floor);
  _navGrid = buildNavGrid(state.obstacles, state.ally.radius, canvas.width, canvas.height);
}

// 圆形实体与矩形障碍物碰撞检测
function collidesWithObstacle(cx, cy, radius) {
  return state.obstacles.some((o) => {
    const nearX = Math.max(o.x, Math.min(cx, o.x + o.w));
    const nearY = Math.max(o.y, Math.min(cy, o.y + o.h));
    return Math.hypot(cx - nearX, cy - nearY) < radius;
  });
}

// 子弹（点）与障碍物碰撞
function bulletHitsObstacle(bx, by) {
  return state.obstacles.some(
    (o) => bx >= o.x && bx <= o.x + o.w && by >= o.y && by <= o.y + o.h
  );
}

// 带障碍物分量滑动的移动辅助
function moveWithCollision(entity, dx, dy) {
  const r = entity.radius;
  const newX = clampUnit(entity.x + dx, r, canvas.width - r);
  const newY = clampUnit(entity.y + dy, r, canvas.height - r);
  if (!collidesWithObstacle(newX, newY, r)) {
    entity.x = newX;
    entity.y = newY;
  } else if (!collidesWithObstacle(newX, entity.y, r)) {
    entity.x = newX;
  } else if (!collidesWithObstacle(entity.x, newY, r)) {
    entity.y = newY;
  }
}

// ── 关卡过渡 ──────────────────────────────────────────────────────────────────

function nextFloor() {
  state.playerBullets = [];
  state.allyBullets = [];
  state.enemyBullets = [];
  state.player.x = 180; state.player.y = 260;
  state.ally.x = 220;   state.ally.y = 290;
  state.player.hp = Math.min(state.player.maxHp, state.player.hp + 40);
  state.ally.hp   = Math.min(state.ally.maxHp,   state.ally.hp   + 40);
  state.ally.dead = false;   // 进入下一层时复活
  // 关卡切换 = 新 episode，重置 LSTM hidden state 与寻路 FSM（障碍物已变）
  if (_rlLoadState === "ready") _rlResetLstmState();
  state.ally.combatPhase = "approach";
  state.ally.navPath = null;
  state.ally.engagePoint = null;
  state.ally.losStableFrames = 0;
  state.ally.losLostFrames = 0;
  generateObstacles();
  spawnEnemies();
  const meta = FLOOR_META[(state.floor - 1) % FLOOR_META.length];
  setAllyBubble(`第 ${state.floor} 层 ${meta.name}。${meta.hint}`);
}

function updateFloorTransition(dt) {
  if (state.floorState !== "clear") return;
  state.transitionTimer -= dt;
  if (state.transitionTimer <= 0) {
    state.floor += 1;
    npcPushEvent("floor_enter", { floor: state.floor });
    nextFloor();
    state.floorState = "playing";
  }
}

// ── 初始化第一关 ──────────────────────────────────────────────────────────────

generateObstacles();
spawnEnemies(false); // 脚本前部初始化，NPC 模块尚未就绪

// ── 工具函数 ──────────────────────────────────────────────────────────────────

function nowSeconds() {
  return performance.now() / 1000;
}

function setAllyBubble(text) {
  state.ally.bubble = text;
  const duration = Math.min(12, Math.max(3, (text || "").length * 0.12));
  state.ally.bubbleUntil = nowSeconds() + duration;
}

function appendMessage(role, text) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = role === "player" ? `你：${text}` : `${NPC_DISPLAY_NAME}：${text}`;
  chatLog.appendChild(el);
  chatLog.scrollTop = chatLog.scrollHeight;
}

const EMOTION_KAOMOJI = {
  neutral:   "( ・_・)",
  focused:   "(•̀ᴗ•́)و",
  annoyed:   "(╯°□°）╯",
  worried:   "(；ω；)",
  happy:     "(＾▽＾)",
  tense:     "(°ロ°!)",
  sarcastic: "(¬‿¬)",
};

function emotionKaomoji(emotion) {
  return EMOTION_KAOMOJI[emotion] || EMOTION_KAOMOJI.neutral;
}

function appendStreamingNpcMessage() {
  const el = document.createElement("div");
  el.className = "msg npc";
  el.textContent = `${NPC_DISPLAY_NAME}：`;
  chatLog.appendChild(el);
  chatLog.scrollTop = chatLog.scrollHeight;
  return {
    append(delta) {
      el.textContent += delta;
      chatLog.scrollTop = chatLog.scrollHeight;
    },
    finish(finalText, emotion) {
      const kaomoji = emotionKaomoji(emotion);
      el.textContent = `${NPC_DISPLAY_NAME} ${kaomoji}：${finalText}`;
      chatLog.scrollTop = chatLog.scrollHeight;
    },
  };
}

function normalize(dx, dy) {
  const len = Math.hypot(dx, dy);
  if (len < 0.0001) return [0, 0];
  return [dx / len, dy / len];
}

function clampUnit(val, min, max) {
  return Math.max(min, Math.min(max, val));
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function findNearestEnemy(from) {
  let nearest = null;
  let minDist = Number.POSITIVE_INFINITY;
  state.enemies.forEach((enemy) => {
    const d = distance(from, enemy);
    if (d < minDist) { minDist = d; nearest = enemy; }
  });
  return [nearest, minDist];
}

function gameInputBlocked() {
  return !!state.result || state.floorState === "blessing_pick";
}

function findTargetForAlly() {
  const now = performance.now() / 1000;
  if (state.ally.focusMode === "lowest_hp" && state.ally.focusUntil > now) {
    let low = null;
    state.enemies.forEach((e) => {
      if (!low || e.hp < low.hp) low = e;
    });
    if (low) return low;
  }
  const [nearest] = findNearestEnemy(state.ally);
  return nearest;
}

function showBlessingPick() {
  state.floorState = "blessing_pick";
  state.blessingChoices = window.GameBlessings.pickThree();
  const overlay = document.getElementById("blessingOverlay");
  const container = document.getElementById("blessingChoices");
  if (!overlay || !container) {
    state.floorState = "clear";
    state.transitionTimer = 2.5;
    return;
  }
  container.innerHTML = "";
  state.blessingChoices.forEach((b, i) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "blessing-card";
    btn.innerHTML = `<strong>${b.name}</strong><span>${b.desc}</span>`;
    btn.addEventListener("click", () => pickBlessing(i));
    container.appendChild(btn);
  });
  overlay.classList.remove("hidden");
}

function pickBlessing(idx) {
  const b = state.blessingChoices[idx];
  if (b) window.GameBlessings.apply(state, b);
  const overlay = document.getElementById("blessingOverlay");
  if (overlay) overlay.classList.add("hidden");
  state.floorState = "clear";
  state.transitionTimer = 2.5;
  npcPushEvent("floor_clear", { floor: state.floor });
  setAllyBubble(b ? `狱印「${b.name}」已铭刻，下一层见。` : "下一层见。");
}

// ── 游戏逻辑更新 ──────────────────────────────────────────────────────────────

function removeDeadEnemies() {
  const dead = state.enemies.filter((e) => e.hp <= 0);
  if (dead.length > 0) {
    dead.forEach((e) => {
      window.GameFx.burst(e.x, e.y, e.kind === "boss" ? "#ffaa88" : "#ffb366", e.kind === "boss" ? 14 : 7);
      if (e.kind === "boss") window.GameFx.shake(5, 0.18);
    });
    state.combo += dead.length;
    state.comboTimer = 2.2;
    state.maxCombo = Math.max(state.maxCombo, state.combo);
    window.GameFx.floatText(state.player.x, state.player.y - 28, `+${dead.length} 连击 ${state.combo}`, "#ffe08a");
    npcPushEvent("kill", { count: dead.length });
  }
  state.enemies = state.enemies.filter((e) => e.hp > 0);
  state.bossAlive = state.enemies.some((e) => e.kind === "boss");
  if (state.enemies.length === 0 && state.floorState === "playing" && !state.result) {
    showBlessingPick();
    setAllyBubble(`第 ${state.floor} 层清除！挑一枚狱印再走。`);
  }
}

function createBullet(owner, from, to, speed, damage) {
  const [nx, ny] = normalize(to.x - from.x, to.y - from.y);
  return { owner, x: from.x, y: from.y, vx: nx * speed, vy: ny * speed, radius: 4, damage, ttl: 2.2 };
}

function playerAttack() {
  if (state.player.attackCd > 0 || gameInputBlocked() || state.floorState === "clear") return;
  const [target] = findNearestEnemy(state.player);
  if (!target) return;
  const dmg = Math.max(1, Math.round(18 * (state.playerDamageMul || 1)));
  state.player.attackCd = 0.3;
  state.playerBullets.push(createBullet("player", state.player, target, 430, dmg));
  npcMarkPlayerAction();
}

function tryPlayerDash() {
  if (state.player.dashCd > 0 || gameInputBlocked() || state.floorState === "clear") return;
  let dx = 0;
  let dy = 0;
  if (state.keys.ArrowUp || state.keys.KeyW) dy -= 1;
  if (state.keys.ArrowDown || state.keys.KeyS) dy += 1;
  if (state.keys.ArrowLeft || state.keys.KeyA) dx -= 1;
  if (state.keys.ArrowRight || state.keys.KeyD) dx += 1;
  const [nx, ny] = normalize(dx || 1, dy);
  const burst = state.player.speed * 0.55;
  moveWithCollision(state.player, nx * burst, ny * burst);
  state.player.dashCd = 1.1 * (state.dashCdMul || 1);
  state.player.dashInvuln = 0.16;
  npcMarkPlayerAction();
}

function updatePlayer(dt) {
  if (state.result) return;
  if (gameInputBlocked()) return;
  let dx = 0;
  let dy = 0;
  if (state.keys.ArrowUp    || state.keys.KeyW) dy -= 1;
  if (state.keys.ArrowDown  || state.keys.KeyS) dy += 1;
  if (state.keys.ArrowLeft  || state.keys.KeyA) dx -= 1;
  if (state.keys.ArrowRight || state.keys.KeyD) dx += 1;

  const [nx, ny] = normalize(dx, dy);
  if (nx !== 0 || ny !== 0) npcMarkPlayerAction();
  moveWithCollision(state.player, nx * state.player.speed * dt, ny * state.player.speed * dt);
  state.player.attackCd = Math.max(0, state.player.attackCd - dt);
  state.player.shieldCd = Math.max(0, state.player.shieldCd - dt);
  state.player.dashCd = Math.max(0, state.player.dashCd - dt);
  state.player.dashInvuln = Math.max(0, state.player.dashInvuln - dt);
  if (state.comboTimer > 0) {
    state.comboTimer -= dt;
    if (state.comboTimer <= 0) state.combo = 0;
  }
}

function allyConfig() {
  if (state.ally.stance === "assault") {
    return { attackRange: 110, kiteRange: 65, interval: 0.45, speedMul: 1.2, damage: 13 };
  }
  // guard（默认）
  return { attackRange: 0, kiteRange: 0, interval: 0.75, speedMul: 1.0, damage: 10 };
}

// 在敌人环带上选 A* 可达、有 LOS 的交战锚点（寻路终点，非敌人中心）
function findEngagePoint(target, cfg) {
  const ally = state.ally;
  const lo = cfg.kiteRange + 5;
  const hi = cfg.attackRange * 0.95;
  let best = null;
  let bestScore = Infinity;
  for (let i = 0; i < 16; i += 1) {
    const angle = (2 * Math.PI * i) / 16;
    for (const r of [lo, (lo + hi) * 0.5, hi]) {
      const px = target.x + Math.cos(angle) * r;
      const py = target.y + Math.sin(angle) * r;
      if (px < ally.radius || px > canvas.width - ally.radius) continue;
      if (py < ally.radius || py > canvas.height - ally.radius) continue;
      if (collidesWithObstacle(px, py, ally.radius)) continue;
      if (!_rlHasLOS(px, py, target.x, target.y)) continue;
      const score = Math.hypot(px - ally.x, py - ally.y);
      if (score < bestScore) { bestScore = score; best = { x: px, y: py }; }
    }
  }
  return best || { x: target.x, y: target.y };
}

// APPROACH 阶段：按节流策略重规划到交战锚点的 A* 路径
function maybeReplanNav(target, dt) {
  const a = state.ally;
  a.navReplanCd -= dt;
  const moved = a.navGoal ? Math.hypot(target.x - a.navGoal.x, target.y - a.navGoal.y) : Infinity;
  // navPath 有效时：正常节流（计时器 + 目标位移）
  // navPath 为 null（上次寻路失败）时：也走节流，避免每帧重复搜索无解情况
  if (a.navReplanCd > 0 && moved < 36) return;
  a.navReplanCd = 0.35;
  a.navGoal = { x: target.x, y: target.y };
  if (!_navGrid) _navGrid = buildNavGrid(state.obstacles, a.radius, canvas.width, canvas.height);
  const raw = findPath(_navGrid, a, target);
  a.navPath = raw ? smoothPath(raw, state.obstacles, a.radius) : null;
}

function updateAlly(dt) {
  const cfg = allyConfig();
  state.ally.attackCd = Math.max(0, state.ally.attackCd - dt);
  state.ally.rescueCd = Math.max(0, state.ally.rescueCd - dt);
  const speed = state.ally.speed * cfg.speedMul;

  if (state.ally.stance === "guard" || state.player.hp <= 40) {
    const d = distance(state.ally, state.player);
    if (d > 50) {
      const [nx, ny] = normalize(state.player.x - state.ally.x, state.player.y - state.ally.y);
      moveWithCollision(state.ally, nx * speed * dt, ny * speed * dt);
    }
    const target = findTargetForAlly();
    if (target && state.ally.attackCd <= 0) {
      state.allyBullets.push(createBullet("ally", state.ally, target, 380, cfg.damage));
      state.ally.attackCd = cfg.interval;
    }

  } else if (state.ally.stance === "assault") {
    if (_rlLoadState === "idle") _rlLoadModel();

    const target = _rlNearestEnemy();
    if (target) {
      const d   = distance(state.ally, target);
      const los = _rlHasLOS(state.ally.x, state.ally.y, target.x, target.y);
      const inEnvelope = d <= cfg.attackRange * _COMBAT_EXIT_DIST_MUL;

      state.ally.losStableFrames = los ? state.ally.losStableFrames + 1 : 0;
      state.ally.losLostFrames   = los ? 0 : state.ally.losLostFrames + 1;

      const engagePt = findEngagePoint(target, cfg);
      state.ally.engagePoint = engagePt;
      const atEngage = distance(state.ally, engagePt) < _ENGAGE_POINT_REACH;
      const pathNearlyDone = !state.ally.navPath || state.ally.navPath.length <= 2;

      if (state.ally.combatPhase === "approach") {
        // ── APPROACH：A* 绕障接近交战锚点，到位且 LOS 稳定后把控制权交给 RL ──
        maybeReplanNav(engagePt, dt);
        const [mx, my] = state.ally.navPath
          ? steerAlong(state.ally.navPath, state.ally, state.obstacles, state.ally.radius)
          : [0, 0];
        moveWithCollision(state.ally, mx * speed * dt, my * speed * dt);

        const canEnterCombat = d <= cfg.attackRange
          && state.ally.losStableFrames >= _COMBAT_ENTER_LOS_FRAMES
          && _rlLoadState === "ready"
          && (atEngage || pathNearlyDone);

        if (canEnterCombat) {
          _rlResetLstmState();
          state.ally.combatPhase = "combat";
          state.ally.navPath = null;
          state.ally.losLostFrames = 0;
        }

      } else {
        // ── COMBAT：RL 全权走位/风筝/躲弹（1c：已废除 hold 站桩）─────────────
        _rlInferAsync(state.ally.attackCd);
        const minTtaSec = _rlMinBulletTTASec();
        const action = _rlResolveAction(los, minTtaSec);
        const [mx, my] = _RL_ACTION_VECTORS[action];
        moveWithCollision(state.ally, mx * speed * dt, my * speed * dt);

        // 退出 COMBAT 仅两种情况，均回 approach 用 A* 移动重定位（绝不站桩）：
        //   ① 脱离交战包络（敌人跑远，!inEnvelope）
        //   ② 被障碍长期(≥0.4s)隔断视线；瞬时遮挡不退出，交给 RL 自己走出墙缝
        if (!inEnvelope || state.ally.losLostFrames >= _COMBAT_LOS_LOST_EXIT_FRAMES) {
          state.ally.combatPhase = "approach";
          state.ally.navPath = null;
          state.ally.losLostFrames = 0;
        }
      }

      if (state.ally.attackCd <= 0) {
        state.allyBullets.push(createBullet("ally", state.ally, target, 400, cfg.damage));
        state.ally.attackCd = cfg.interval;
      }
    }

  }

  // 边界夹紧
  state.ally.x = clampUnit(state.ally.x, 10, canvas.width - 10);
  state.ally.y = clampUnit(state.ally.y, 10, canvas.height - 10);

  if (state.player.hp <= 45 && state.ally.rescueCd <= 0 && state.ally.stance !== "assault") {
    state.player.hp = Math.min(state.player.maxHp, state.player.hp + 28);
    state.player.shieldCd = 2.2 * (state.blessingShieldMul || 1);
    state.ally.rescueCd = 9.0;
    setAllyBubble("先后撤，我给你护盾。");
  }
}

function updateEnemies(dt) {
  if (state.floorState !== "playing") return;
  state.enemies.forEach((enemy) => {
    enemy.shootCd = Math.max(0, enemy.shootCd - dt);
    if (state.result) return;
    if (window.GameBullets.tickWarn(enemy, dt)) return;

    const distToPlayer = distance(enemy, state.player);
    const distToAlly = state.ally.hp > 0 ? distance(enemy, state.ally) : Number.POSITIVE_INFINITY;
    const primary = distToPlayer <= distToAlly ? state.player : state.ally;
    const [nx, ny] = normalize(primary.x - enemy.x, primary.y - enemy.y);
    moveWithCollision(enemy, nx * enemy.speed * dt, ny * enemy.speed * dt);

    if (enemy.shootCd <= 0) {
      const bulletSpeed = enemy.kind === "boss" ? 200 : 170;
      const bulletDamage = enemy.kind === "boss" ? 9 : 5;
      if (enemy.kind === "boss" && Math.random() < 0.38) {
        window.GameBullets.startWarn(enemy, 0.45);
        enemy._pendingFan = { primary, bulletSpeed, bulletDamage };
        enemy.shootCd = 0.5;
      } else {
        window.GameBullets.aimedShot(state.enemyBullets, enemy, primary, bulletSpeed, bulletDamage);
        enemy.shootCd = enemy.kind === "boss" ? 1.2 : 1.6;
      }
    } else if (enemy._pendingFan && !enemy.warnT) {
      const p = enemy._pendingFan;
      window.GameBullets.fanShot(state.enemyBullets, enemy, p.primary, p.bulletSpeed, p.bulletDamage, 42, 5);
      enemy._pendingFan = null;
      enemy.shootCd = 1.35;
    }
  });
}

function updateBullets(dt) {
  const moveBullets = (arr) => {
    for (let i = arr.length - 1; i >= 0; i -= 1) {
      const b = arr[i];
      b.x += b.vx * dt;
      b.y += b.vy * dt;
      b.ttl -= dt;
      if (
        b.ttl <= 0 ||
        b.x < -10 || b.x > canvas.width + 10 ||
        b.y < -10 || b.y > canvas.height + 10 ||
        bulletHitsObstacle(b.x, b.y)
      ) {
        arr.splice(i, 1);
      }
    }
  };
  moveBullets(state.playerBullets);
  moveBullets(state.allyBullets);
  moveBullets(state.enemyBullets);

  for (let i = state.playerBullets.length - 1; i >= 0; i -= 1) {
    const b = state.playerBullets[i];
    let hit = false;
    for (let j = 0; j < state.enemies.length; j += 1) {
      const e = state.enemies[j];
      if (distance(b, e) <= b.radius + e.radius) {
        e.hp -= b.damage;
        window.GameFx.burst(b.x, b.y, "#6bc8ff", 4);
        window.GameFx.floatText(e.x, e.y - 18, String(b.damage), "#9fe4ff");
        hit = true;
        break;
      }
    }
    if (hit) state.playerBullets.splice(i, 1);
  }

  for (let i = state.allyBullets.length - 1; i >= 0; i -= 1) {
    const b = state.allyBullets[i];
    let hit = false;
    for (let j = 0; j < state.enemies.length; j += 1) {
      const e = state.enemies[j];
      if (distance(b, e) <= b.radius + e.radius) {
        e.hp -= b.damage;
        window.GameFx.burst(b.x, b.y, "#9af19b", 4);
        hit = true;
        break;
      }
    }
    if (hit) state.allyBullets.splice(i, 1);
  }

  for (let i = state.enemyBullets.length - 1; i >= 0; i -= 1) {
    const b = state.enemyBullets[i];
    let consumed = false;
    if (distance(b, state.player) <= b.radius + state.player.radius) {
      if (state.player.dashInvuln <= 0) {
        const raw = b.damage;
        const guardNear = state.ally.stance === "guard" && !state.ally.dead
          && distance(state.ally, state.player) <= 80;
        let final = raw;
        if (state.player.shieldCd > 0) final = Math.max(2, Math.floor(raw * 0.3));
        else if (guardNear) final = Math.max(2, Math.floor(raw * (1 - (state.guardDamageReduction || 0.15))));
        state.player.hp -= final;
        window.GameFx.shake(2.5, 0.08);
      }
      consumed = true;
    } else if (state.ally.hp > 0 && distance(b, state.ally) <= b.radius + state.ally.radius) {
      state.ally.hp -= b.damage;
      consumed = true;
    }
    if (consumed) state.enemyBullets.splice(i, 1);
  }
}

function checkDefeat() {
  if (state.result || state.floorState === "clear" || state.floorState === "blessing_pick") return;
  if (state.player.hp <= 0) {
    state.player.hp = 0;
    state.result = `战败。坚持到了第 ${state.floor} 层。`;
    setAllyBubble("这次没守住，下轮我们换打法。");
  }
  if (state.ally.hp <= 0 && !state.ally.dead) {
    state.ally.hp    = 0;
    state.ally.dead  = true;
    state.ally.stance = "guard";   // 自动切回守护
    setAllyBubble("灵核失稳...你先继续前进。");
    // 气泡只设一次（3秒），后续不再重置
  }
}

// ── 渲染 ──────────────────────────────────────────────────────────────────────

function ensureCinders() {
  if (state.cinders.length > 0) return;
  for (let i = 0; i < 90; i += 1) {
    state.cinders.push({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vy: -(18 + Math.random() * 20),
      size: Math.random() < 0.2 ? 2 : 1,
      flicker: Math.random() * Math.PI * 2,
    });
  }
}

function drawBackground(dt) {
  ensureCinders();
  const meta = FLOOR_META[(state.floor - 1) % FLOOR_META.length];
  const grad = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
  grad.addColorStop(0, meta.tone);
  grad.addColorStop(1, meta.haze);
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "rgba(255, 191, 160, 0.07)";
  for (let i = 0; i < canvas.width; i += 40) {
    ctx.beginPath();
    ctx.moveTo(i, 0);
    ctx.lineTo(i, canvas.height);
    ctx.stroke();
  }
  for (let j = 0; j < canvas.height; j += 40) {
    ctx.beginPath();
    ctx.moveTo(0, j);
    ctx.lineTo(canvas.width, j);
    ctx.stroke();
  }

  const sigilX = canvas.width * 0.78;
  const sigilY = canvas.height * 0.5;
  ctx.strokeStyle = `${meta.accent}55`;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(sigilX, sigilY, 74, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(sigilX, sigilY, 46, 0, Math.PI * 2);
  ctx.stroke();
  ctx.lineWidth = 1;

  state.cinders.forEach((c) => {
    c.y += c.vy * dt;
    c.x += Math.sin((performance.now() / 900) + c.flicker) * 0.2;
    if (c.y < -4) {
      c.y = canvas.height + Math.random() * 20;
      c.x = Math.random() * canvas.width;
    }
    ctx.fillStyle = Math.sin((performance.now() / 500) + c.flicker) > 0 ? `${meta.accent}bb` : `${meta.accent}66`;
    ctx.fillRect(c.x, c.y, c.size, c.size);
  });
}

function drawObstacles() {
  const meta = FLOOR_META[(state.floor - 1) % FLOOR_META.length];
  window.GameObstacles.draw(ctx, state.obstacles, meta);
}

function drawPixelSprite(entity, palette, dir = 1) {
  const px = 4;
  const originX = Math.floor(entity.x - 6 * px);
  const originY = Math.floor(entity.y - 8 * px);
  const head = [
    "0011111100",
    "0112222210",
    "0123333321",
    "0123333321",
    "0012332100",
  ];
  const body = [
    "0001441000",
    "0014444100",
    "0144544410",
    "0144444410",
    "0014545100",
    "0015005100",
  ];
  const rows = [...head, ...body];
  const map = { "1": palette.outline, "2": palette.skin, "3": palette.eye, "4": palette.cloth, "5": palette.trim };

  // 先画一个高可见底座，避免深色关卡里角色不可辨认
  ctx.beginPath();
  ctx.fillStyle = palette.base || "rgba(245, 220, 200, 0.28)";
  ctx.arc(entity.x, entity.y, entity.radius + 2, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.strokeStyle = palette.baseStroke || "rgba(255, 230, 200, 0.45)";
  ctx.lineWidth = 1.5;
  ctx.arc(entity.x, entity.y, entity.radius + 2, 0, Math.PI * 2);
  ctx.stroke();
  ctx.lineWidth = 1;

  rows.forEach((row, rowIdx) => {
    [...row].forEach((cell, colIdx) => {
      if (cell === "0") return;
      const drawCol = dir > 0 ? colIdx : row.length - colIdx - 1;
      ctx.fillStyle = map[cell];
      ctx.fillRect(originX + drawCol * px, originY + rowIdx * px, px, px);
    });
  });

  const w = entity.radius * 2;
  const hpRatio = clampUnit(entity.hp / (entity.maxHp || 100), 0, 1);
  ctx.fillStyle = "#101317";
  ctx.fillRect(entity.x - entity.radius, entity.y - entity.radius - 10, w, 4);
  ctx.fillStyle = palette.hp || "#9fe4ff";
  ctx.fillRect(entity.x - entity.radius, entity.y - entity.radius - 10, w * hpRatio, 4);
}

function drawBullets() {
  const drawSet = (arr, color) => {
    arr.forEach((b) => {
      const c = b.color || color;
      ctx.fillStyle = `${c}55`;
      ctx.fillRect(b.x - b.vx * 0.02 - 2, b.y - b.vy * 0.02 - 2, 4, 4);
      ctx.fillStyle = c;
      ctx.beginPath();
      ctx.arc(b.x, b.y, b.radius || 3, 0, Math.PI * 2);
      ctx.fill();
    });
  };
  drawSet(state.playerBullets, "#6bc8ff");
  drawSet(state.allyBullets, "#9af19b");
  drawSet(state.enemyBullets, "#ff6e58");
}

function drawCombatFx() {
  state.enemies.forEach((e) => {
    if (e.warnT > 0) {
      ctx.strokeStyle = `rgba(255, 120, 90, ${0.35 + Math.sin(performance.now() / 80) * 0.25})`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(e.x, e.y, e.radius + 16, 0, Math.PI * 2);
      ctx.stroke();
      ctx.lineWidth = 1;
    }
  });
  if (state.player.shieldCd > 0) {
    ctx.strokeStyle = "rgba(120, 200, 255, 0.75)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(state.player.x, state.player.y, state.player.radius + 6, 0, Math.PI * 2);
    ctx.stroke();
  }
  if (state.player.dashInvuln > 0) {
    ctx.strokeStyle = "rgba(255, 240, 160, 0.8)";
    ctx.beginPath();
    ctx.arc(state.player.x, state.player.y, state.player.radius + 4, 0, Math.PI * 2);
    ctx.stroke();
  }
  if (state.ally.stance === "guard" && state.ally.hp > 0) {
    ctx.strokeStyle = "rgba(180, 255, 170, 0.35)";
    ctx.beginPath();
    ctx.arc(state.ally.x, state.ally.y, state.ally.radius + 8, 0, Math.PI * 2);
    ctx.stroke();
  } else if (state.ally.stance === "assault" && state.ally.hp > 0) {
    ctx.strokeStyle = "rgba(255, 170, 110, 0.45)";
    ctx.beginPath();
    ctx.arc(state.ally.x, state.ally.y, state.ally.radius + 10, 0, Math.PI * 2);
    ctx.stroke();
  }
}

function drawAllyBubble() {
  if (!state.ally.bubble || nowSeconds() > state.ally.bubbleUntil) return;
  const text = state.ally.bubble.slice(0, 90);
  ctx.font = "13px Segoe UI";
  const pad = 8;
  const width  = ctx.measureText(text).width + pad * 2;
  const height = 26;
  const x = clampUnit(state.ally.x - width / 2, 6, canvas.width - width - 6);
  const y = state.ally.y - state.ally.radius - 40;
  ctx.fillStyle = "rgba(20, 24, 38, 0.92)";
  ctx.fillRect(x, y, width, height);
  ctx.strokeStyle = "#8ea3ff";
  ctx.strokeRect(x, y, width, height);
  ctx.fillStyle = "#e7ecfb";
  ctx.fillText(text, x + pad, y + 17);
}

function drawOverlay() {
  const meta = FLOOR_META[(state.floor - 1) % FLOOR_META.length];
  ctx.fillStyle = "#f8d1b4";
  ctx.font = "bold 16px Segoe UI";
  ctx.fillText(`第 ${state.floor} 层：${meta.name}`, 16, 26);
  ctx.fillStyle = "#f2b48f";
  ctx.font = "13px Segoe UI";
  ctx.fillText(`狱律提示：${meta.hint}`, 16, 46);

  if (state.floorState === "clear") {
    ctx.fillStyle = "rgba(6, 8, 12, 0.55)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#ecf1ff";
    ctx.font = "bold 22px Segoe UI";
    ctx.textAlign = "center";
    ctx.fillText(
      `${Math.ceil(state.transitionTimer)} 秒后进入下一层…`,
      canvas.width / 2,
      canvas.height / 2,
    );
    ctx.textAlign = "left";
  }
  if (state.combo > 1) {
    ctx.fillStyle = "#ffe6a0";
    ctx.font = "bold 14px Segoe UI";
    ctx.fillText(`连击 ×${state.combo}`, canvas.width - 110, 24);
  }
  if (state.result) {
    ctx.fillStyle = "rgba(6, 8, 12, 0.65)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#ecf1ff";
    ctx.font = "bold 24px Segoe UI";
    ctx.textAlign = "center";
    ctx.fillText(state.result, canvas.width / 2, canvas.height / 2);
    ctx.textAlign = "left";
  }
}

function render(dt) {
  ctx.save();
  window.GameFx.applyShake(ctx);
  drawBackground(dt);
  drawObstacles();
  drawPixelSprite(state.player, { outline: "#1f2f39", skin: "#e1bfa1", eye: "#6ec4ff", cloth: "#3f86c2", trim: "#9fe4ff", hp: "#8dd0ff" }, 1);
  drawPixelSprite(state.ally, { outline: "#2f1d14", skin: "#d7b399", eye: "#ffb77f", cloth: "#6e3d2a", trim: "#d08c55", base: "rgba(255, 188, 132, 0.30)", baseStroke: "rgba(255, 202, 160, 0.60)", hp: "#b8f7b6" }, 1);
  state.enemies.forEach((enemy) => {
    drawPixelSprite(
      enemy,
      enemy.kind === "boss"
        ? { outline: "#320d0d", skin: "#d7a299", eye: "#ffd1d1", cloth: "#a72f2f", trim: "#ff7d62", base: "rgba(255, 110, 94, 0.32)", baseStroke: "rgba(255, 138, 116, 0.60)", hp: "#ffd7d2" }
        : { outline: "#321a13", skin: "#c79f86", eye: "#ffcc9c", cloth: "#8f4c35", trim: "#d98b5a", base: "rgba(240, 148, 112, 0.26)", baseStroke: "rgba(255, 178, 140, 0.5)", hp: "#ffd7a3" },
      enemy.x < state.player.x ? -1 : 1,
    );
  });
  drawBullets();
  drawCombatFx();
  window.GameFx.draw(ctx);
  drawAllyBubble();
  drawOverlay();
  ctx.restore();
}

function updateHud() {
  const stanceLabel = STANCE_LABELS[state.ally.stance] || state.ally.stance;
  const floorName = FLOOR_META[(state.floor - 1) % FLOOR_META.length].name;

  // assault 姿态时显示 FSM 阶段 + RL 模型加载状态
  let rlTag = "";
  if (state.ally.stance === "assault") {
    if (state.ally.combatPhase === "combat") {
      rlTag = " [战斗·RL]";
    } else if (_rlLoadState === "ready") {
      rlTag = " [寻路]";
    } else if (_rlLoadState === "loading") {
      rlTag = " [寻路·RL加载中…]";
    } else if (_rlLoadState === "error") {
      rlTag = " [寻路·RL失败]";
    } else {
      rlTag = " [寻路]";
    }
  }

  const comboTag = state.combo > 1 ? ` | 连击 ${state.combo}` : "";
  hudStats.textContent =
    `${floorName}（第 ${state.floor} 层） | 玩家 HP ${Math.floor(state.player.hp)}/${state.player.maxHp}`
    + ` | 乌枭 HP ${Math.floor(state.ally.hp)}/${state.ally.maxHp}`
    + ` | 姿态 ${stanceLabel}${rlTag} | 敌人 ${state.enemies.length}${comboTag}`;
}

// ── NPC 对话与自主思考 ────────────────────────────────────────────────────────

const NPC_ID   = "wuxiao_01";
const NPC_NAME = "乌枭";
const NPC_DISPLAY_NAME = "乌枭";
const STANCE_LABELS = { assault: "突击", guard: "守护" };

const NPC_AUTONOMY = {
  enabled: true,
  idleThresholdMs: 8000,
  thinkIntervalMs: 12000,
  speechCooldownMs: 15000,
  socialBeatMs: 45000,
  periodicFallbackMs: 75000,
  playerIdleForHelpMs: 8000,
  allyCriticalPct: 0.25,
  playerDangerPct: 0.30,
};

const npcIO = {
  busy: false,
  mode: null,
  abortCtrl: null,
  currentPriority: 99,
  combatMood: "observing",
  lastPlayerMsgAt: 0,
  lastPlayerActionAt: 0,
  lastThinkCompletedAt: 0,
  lastNpcSpeechAt: 0,
  lastSceneHash: "",
  prevEnemyCount: -1,
  prevPlayerHp: -1,
  prevAllyHp: -1,
  autonomyStartAt: 0,
  lastPlayerHpDropAt: 0,
  lastStanceChangeAt: 0,
  openingThinkDone: false,
  consecutiveNoop: 0,
  skipLogAt: 0,
  events: [],
};

function npcPushEvent(kind, data = {}) {
  npcIO.events.push({ kind, at: performance.now(), ...data });
  if (npcIO.events.length > 12) npcIO.events.shift();
}

function npcRecentEventKinds(maxAgeMs = 12000) {
  const now = performance.now();
  return npcIO.events
    .filter((e) => now - e.at < maxAgeMs)
    .map((e) => e.kind);
}

function npcMarkPlayerAction() {
  npcIO.lastPlayerActionAt = performance.now();
}

function npcSceneHash() {
  const s = buildSceneInfo("autonomous");
  return `${s.floor}|${s.player_hp}|${s.ally_hp}|${s.enemy_count}|${s.ally_stance}|${s.floor_state}|${s.boss_alive}`;
}

function npcSinceLastSpeechS(now = performance.now()) {
  if (npcIO.lastNpcSpeechAt > 0) return (now - npcIO.lastNpcSpeechAt) / 1000;
  if (npcIO.autonomyStartAt > 0) return (now - npcIO.autonomyStartAt) / 1000;
  return 999;
}

function npcBulletThreatCount(entity, maxDist = 150, maxTta = 1.5) {
  let n = 0;
  for (const b of state.enemyBullets) {
    if (Math.hypot(b.x - entity.x, b.y - entity.y) > maxDist) continue;
    if (_rlBulletTTASec(b, entity) <= maxTta) n += 1;
  }
  return n;
}

function npcBuildTacticalContext() {
  const [playerNearest, playerNearestDist] = findNearestEnemy(state.player);
  const [allyNearest, allyNearestDist] = state.ally.hp > 0
    ? findNearestEnemy(state.ally)
    : [null, -1];
  const now = performance.now();
  const playerLos = !playerNearest
    || _rlHasLOS(state.player.x, state.player.y, playerNearest.x, playerNearest.y);
  const allyLos = !allyNearest
    || _rlHasLOS(state.ally.x, state.ally.y, allyNearest.x, allyNearest.y);
  const incomingPlayer = npcBulletThreatCount(state.player);
  const incomingAlly = state.ally.hp > 0 ? npcBulletThreatCount(state.ally) : 0;
  const enemiesInPlayerRange = state.enemies.filter((e) => distance(state.player, e) <= 200).length;
  const enemiesInAllyRange = state.ally.hp > 0
    ? state.enemies.filter((e) => distance(state.ally, e) <= 150).length
    : 0;
  const allyNavStuck = state.ally.stance === "assault"
    && state.ally.combatPhase === "approach"
    && !state.ally.navPath
    && allyNearestDist > 100;
  return {
    nearest_enemy_distance: playerNearest ? Math.round(playerNearestDist) : -1,
    ally_nearest_enemy_distance: allyNearest ? Math.round(allyNearestDist) : -1,
    player_has_los: playerLos,
    ally_has_los: allyLos,
    los_blocked: !!(playerNearest && (!playerLos || !allyLos)),
    incoming_bullets_player: incomingPlayer,
    incoming_bullets_ally: incomingAlly,
    enemies_in_player_range: enemiesInPlayerRange,
    enemies_in_ally_range: enemiesInAllyRange,
    ally_nav_stuck: allyNavStuck,
    player_under_fire: npcIO.lastPlayerHpDropAt > 0 && (now - npcIO.lastPlayerHpDropAt) < 6000,
  };
}

function npcInitAutonomy() {
  const t = performance.now();
  npcIO.lastPlayerMsgAt = t;
  npcIO.lastPlayerActionAt = t;
  npcIO.autonomyStartAt = t;
  npcIO.lastThinkCompletedAt = 0;
  npcIO.lastNpcSpeechAt = 0;
  npcIO.lastPlayerHpDropAt = 0;
  npcIO.lastStanceChangeAt = t;
  npcIO.lastSceneHash = npcSceneHash();
  npcIO.prevEnemyCount = state.enemies.length;
  npcIO.prevPlayerHp = Math.floor(state.player.hp);
  npcIO.prevAllyHp = Math.floor(state.ally.hp);
  npcIO.consecutiveNoop = 0;
  npcIO.combatMood = "observing";
  npcIO.openingThinkDone = false;
  npcIO.events = [];
}

function npcTrackHpEvents() {
  const playerHp = Math.floor(state.player.hp);
  const allyHp = Math.floor(state.ally.hp);
  if (npcIO.prevPlayerHp >= 0 && playerHp < npcIO.prevPlayerHp - 15) {
    npcIO.lastPlayerHpDropAt = performance.now();
    npcPushEvent("hp_drop", { who: "player", from: npcIO.prevPlayerHp, to: playerHp });
  }
  if (npcIO.prevAllyHp >= 0 && allyHp < npcIO.prevAllyHp - 15) {
    npcPushEvent("hp_drop", { who: "ally", from: npcIO.prevAllyHp, to: allyHp });
  }
  npcIO.prevPlayerHp = playerHp;
  npcIO.prevAllyHp = allyHp;
}

function npcSkipLog(reason) {
  const now = performance.now();
  if (now - npcIO.skipLogAt < 3000) return;
  npcIO.skipLogAt = now;
  console.info("[NPC] think skipped:", reason);
}

function buildSceneInfo(trigger = "reactive") {
  const playerHp = Math.floor(state.player.hp);
  const allyHp = Math.floor(state.ally.hp);
  const now = performance.now();
  const playerActionS = (now - npcIO.lastPlayerActionAt) / 1000;
  const playerHpPct = playerHp / state.player.maxHp;
  const allyHpPct = allyHp / state.ally.maxHp;
  const allyPlayerDist = Math.round(distance(state.ally, state.player));
  const allyNearPlayer = allyPlayerDist <= 80;
  const allyGuardingPlayer = state.ally.stance === "guard" && allyNearPlayer;
  const tactical = npcBuildTacticalContext();
  return {
    mode: "battle",
    floor: state.floor,
    floor_state: state.floorState,
    ally_stance: state.ally.stance,
    ally_combat_phase: state.ally.combatPhase,
    ally_player_distance: allyPlayerDist,
    ally_near_player: allyNearPlayer,
    ally_already_guarding: allyGuardingPlayer,
    player_hp: playerHp,
    player_hp_pct: playerHpPct,
    player_max_hp: state.player.maxHp,
    ally_hp: allyHp,
    ally_hp_pct: allyHpPct,
    ally_max_hp: state.ally.maxHp,
    enemy_count: state.enemies.length,
    prev_enemy_count: npcIO.prevEnemyCount,
    boss_alive: state.bossAlive,
    player_last_action_s: playerActionS,
    player_is_active: playerActionS < 6,
    player_is_idle: playerActionS >= NPC_AUTONOMY.playerIdleForHelpMs / 1000,
    ally_guard_duration_s: state.ally.stance === "guard"
      ? (now - npcIO.lastStanceChangeAt) / 1000
      : 0,
    ally_in_danger: allyHpPct <= NPC_AUTONOMY.allyCriticalPct,
    player_in_danger: playerHpPct <= NPC_AUTONOMY.playerDangerPct,
    recent_events: npcRecentEventKinds(),
    trigger,
    idle_seconds: trigger === "autonomous" ? (now - npcIO.lastPlayerMsgAt) / 1000 : 0,
    since_last_npc_speech: npcSinceLastSpeechS(now),
    ...tactical,
  };
}

function npcStartRequest(mode, priority = 99) {
  if (mode === "chat" && npcIO.mode === "think") npcIO.abortCtrl?.abort();
  if (mode === "think" && npcIO.mode === "think" && priority < npcIO.currentPriority) {
    npcIO.abortCtrl?.abort();
  }
  npcIO.busy = true;
  npcIO.mode = mode;
  npcIO.currentPriority = priority;
  npcIO.abortCtrl = new AbortController();
  return npcIO.abortCtrl;
}

function npcEndRequest(mode) {
  if (npcIO.mode === mode) {
    npcIO.busy = false;
    npcIO.mode = null;
    npcIO.abortCtrl = null;
  }
}

function npcMarkSpeech() {
  npcIO.lastNpcSpeechAt = performance.now();
}

function applyStance(stance, reply, opts = {}) {
  if (!stance) return;
  if (stance === "assault" && state.ally.dead) return;
  const stanceChanged = opts.stanceChanged !== undefined
    ? opts.stanceChanged
    : state.ally.stance !== stance;
  if (stance === "assault" && stanceChanged) {
    state.ally.combatPhase = "approach";
    state.ally.navPath = null;
    state.ally.engagePoint = null;
    state.ally.losStableFrames = 0;
    state.ally.losLostFrames = 0;
    if (_rlLoadState === "ready") _rlResetLstmState();
  }
  state.ally.stance = stance;
  const label = STANCE_LABELS[stance] || stance;
  const bubble = reply || `姿态切换：${label}。`;
  setAllyBubble(bubble);
  if (stanceChanged || !opts.autonomous) {
    appendMessage("npc", bubble);
    npcMarkSpeech();
    if (stanceChanged) npcIO.lastStanceChangeAt = performance.now();
  }
}

async function consumeNpcStream(body, opts = {}) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let msgNode = null;
  let accumulated = "";
  let buffer = "";
  let bubbleThrottle = 0;
  let outcome = "unknown";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      let evt;
      try { evt = JSON.parse(trimmed); } catch { continue; }

      if (evt.type === "meta") {
        if (evt.combat_mood) npcIO.combatMood = evt.combat_mood;
        continue;
      }
      if (evt.type === "noop") {
        outcome = "noop";
        continue;
      }
      if (evt.type === "command") {
        outcome = `command:${evt.stance}:${evt.reply || ""}`;
        applyStance(evt.stance, evt.reply, {
          autonomous: opts.source === "autonomous",
          stanceChanged: evt.stance_changed !== false,
        });
      } else if (evt.type === "delta") {
        outcome = "dialogue(streaming)";
        if (!msgNode) msgNode = appendStreamingNpcMessage();
        accumulated += evt.text;
        msgNode.append(evt.text);
        const now = Date.now();
        if (now - bubbleThrottle > 100) { setAllyBubble(accumulated); bubbleThrottle = now; }
      } else if (evt.type === "done") {
        const finalText = evt.action?.dialogue || accumulated || "收到，我会继续和你协同。";
        const emotion = evt.action?.emotion || "neutral";
        outcome = `dialogue:${finalText}`;
        if (msgNode) msgNode.finish(finalText, emotion);
        else appendMessage("npc", finalText);
        setAllyBubble(`${emotionKaomoji(emotion)} ${finalText}`);
        npcMarkSpeech();
      } else if (evt.type === "polish") {
        const polished = evt.dialogue || "";
        if (polished) {
          outcome = `polish:${polished}`;
          setAllyBubble(`${emotionKaomoji(evt.emotion || "focused")} ${polished}`);
        }
      } else if (evt.type === "error") {
        const fallbackText = evt.fallback?.dialogue || "连接中断。我会继续执行上一条指令。";
        outcome = `error:${fallbackText}`;
        if (msgNode) msgNode.finish(fallbackText, "neutral");
        else appendMessage("npc", fallbackText);
        setAllyBubble(fallbackText);
        npcMarkSpeech();
      }
    }
  }

  if (opts.source === "autonomous") {
    if (outcome === "noop") npcIO.consecutiveNoop += 1;
    else npcIO.consecutiveNoop = 0;
    console.info("[NPC] think response:", outcome);
  }
  return outcome;
}

function npcNoopBackoffMs() {
  const steps = [15000, 30000, 60000];
  const idx = Math.min(Math.max(0, npcIO.consecutiveNoop - 2), steps.length - 1);
  return npcIO.consecutiveNoop >= 2 ? steps[idx] : 0;
}

function npcEvaluateTriggers() {
  if (!NPC_AUTONOMY.enabled) return null;
  if (state.result || state.ally.dead) return null;

  const now = performance.now();
  const scene = buildSceneInfo("autonomous");
  const hash = npcSceneHash();
  const events = npcRecentEventKinds();

  if (events.includes("floor_clear") && state.floorState === "clear") {
    return {
      priority: 1,
      trigger: "scene_change",
      reason: "floor_clear",
      scene_change_kind: "floor_clear",
    };
  }

  if (state.floorState !== "playing") return null;

  if (
    !npcIO.openingThinkDone
    && now - npcIO.autonomyStartAt < 25000
    && scene.enemy_count > 0
    && scene.ally_stance === "guard"
    && now - npcIO.lastPlayerMsgAt >= NPC_AUTONOMY.idleThresholdMs
  ) {
    npcIO.openingThinkDone = true;
    return {
      priority: 1,
      trigger: "scene_change",
      reason: "opening",
      scene_change_kind: "opening",
    };
  }

  if (scene.ally_hp_pct <= 0.10) {
    return { priority: 0, trigger: "critical", reason: "ally_dire" };
  }
  if (scene.ally_in_danger) {
    return { priority: 0, trigger: "critical", reason: "ally_critical" };
  }
  if (scene.player_in_danger && scene.ally_stance === "assault") {
    return { priority: 0, trigger: "critical", reason: "player_danger_assault" };
  }
  if (scene.player_in_danger && scene.player_is_idle) {
    return { priority: 0, trigger: "critical", reason: "player_idle_danger" };
  }

  if (events.includes("floor_enter") || hash !== npcIO.lastSceneHash) {
    const ecDelta = npcIO.prevEnemyCount >= 0
      ? scene.enemy_count - npcIO.prevEnemyCount
      : 0;
    if (events.includes("floor_enter") || Math.abs(ecDelta) >= 2) {
      return {
        priority: 1,
        trigger: "scene_change",
        reason: events.includes("floor_enter") ? "floor_enter" : "enemy_delta",
        scene_change_kind: events.includes("floor_enter") ? "floor_enter" : "enemy_change",
      };
    }
  }

  if (npcSinceLastSpeechS(now) * 1000 >= NPC_AUTONOMY.socialBeatMs) {
    if (now - npcIO.lastPlayerMsgAt >= NPC_AUTONOMY.idleThresholdMs) {
      return { priority: 2, trigger: "social", reason: "social_beat" };
    }
  }

  if (now - npcIO.lastPlayerMsgAt < NPC_AUTONOMY.idleThresholdMs) return null;

  if (npcIO.lastThinkCompletedAt && now - npcIO.lastThinkCompletedAt < NPC_AUTONOMY.thinkIntervalMs) {
    return null;
  }
  const backoff = npcNoopBackoffMs();
  if (backoff && npcIO.lastThinkCompletedAt && now - npcIO.lastThinkCompletedAt < backoff) {
    return null;
  }
  if (!npcIO.lastThinkCompletedAt
      || now - npcIO.lastThinkCompletedAt >= NPC_AUTONOMY.periodicFallbackMs) {
    return { priority: 3, trigger: "periodic", reason: "periodic_fallback" };
  }
  return null;
}

function npcEnqueueThink(evaluation) {
  if (!evaluation) return;
  if (npcIO.busy && evaluation.priority >= npcIO.currentPriority) {
    npcSkipLog(`busy_block_p${evaluation.priority}`);
    return;
  }

  const ctrl = npcStartRequest("think", evaluation.priority);
  const scene = buildSceneInfo("autonomous");
  scene.scene_hash = npcSceneHash();
  scene.scene_change_kind = evaluation.scene_change_kind || "";
  scene.trigger_reason = evaluation.reason;
  console.info("[NPC] think", evaluation.trigger, evaluation);

  fetch(`${NPC_API}/api/npc/think`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal: ctrl.signal,
    body: JSON.stringify({
      player_id: state.playerId,
      npc_id: NPC_ID,
      npc_name: NPC_NAME,
      trigger: evaluation.trigger,
      priority: evaluation.priority,
      trigger_reason: evaluation.reason,
      scene_info: scene,
    }),
  })
    .then((resp) => {
      if (!resp.ok || !resp.body) throw new Error(`think failed: HTTP ${resp.status}`);
      return consumeNpcStream(resp.body, { source: "autonomous" });
    })
    .catch((err) => {
      if (err.name === "AbortError") return;
      console.warn("[NPC] think error:", err);
    })
    .finally(() => {
      npcIO.lastThinkCompletedAt = performance.now();
      npcIO.prevEnemyCount = state.enemies.length;
      npcIO.lastSceneHash = npcSceneHash();
      npcIO.currentPriority = 99;
      npcEndRequest("think");
    });
}

function npcTickAutonomy() {
  if (state.floorState === "playing" && !state.result && !state.ally.dead) {
    npcTrackHpEvents();
  }
  const evaluation = npcEvaluateTriggers();
  if (evaluation) npcEnqueueThink(evaluation);
}

async function sendPlayerChat(message) {
  npcIO.lastPlayerMsgAt = performance.now();
  const ctrl = npcStartRequest("chat");

  const payload = {
    player_id: state.playerId,
    npc_id: NPC_ID,
    npc_name: NPC_NAME,
    message,
    scene_info: buildSceneInfo("reactive"),
  };

  try {
    const resp = await fetch(`${NPC_API}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: ctrl.signal,
      body: JSON.stringify(payload),
    });
    if (!resp.ok || !resp.body) throw new Error(`chat failed: ${resp.status}`);
    await consumeNpcStream(resp.body, { source: "reactive" });
  } catch (err) {
    if (err.name === "AbortError") return;
    const text = "连接中断。我会继续执行上一条指令。";
    appendMessage("npc", text);
    setAllyBubble(text);
    npcMarkSpeech();
  } finally {
    npcEndRequest("chat");
  }
}

// ── 主循环 ────────────────────────────────────────────────────────────────────

let lastTs = performance.now();
function loop(ts) {
  const dt = Math.min(0.033, (ts - lastTs) / 1000);
  lastTs = ts;

  updatePlayer(dt);
  updateAlly(dt);
  updateEnemies(dt);
  updateBullets(dt);
  removeDeadEnemies();
  checkDefeat();
  updateFloorTransition(dt);
  window.GameFx.update(dt);
  render(dt);
  updateHud();
  requestAnimationFrame(loop);
}

// ── 输入 ──────────────────────────────────────────────────────────────────────

document.addEventListener("keydown", (e) => {
  if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "KeyW", "KeyA", "KeyS", "KeyD"].includes(e.code)) {
    e.preventDefault();
  }
  state.keys[e.code] = true;
  if (e.code === "Space") { e.preventDefault(); playerAttack(); }
  if (e.code === "ShiftLeft" || e.code === "ShiftRight") {
    e.preventDefault();
    tryPlayerDash();
  }
});
document.addEventListener("keyup", (e) => { state.keys[e.code] = false; });

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;
  chatInput.value = "";
  appendMessage("player", message);
  await sendPlayerChat(message);
});

appendMessage("npc", "黑签鬼差乌枭到位。你别乱送，我就能把你带到无间边狱。");
npcInitAutonomy();
console.info(`[NPC] autonomy9 + game ${GAME_BUILD} (safe upgrades, RL frozen)`);
setInterval(npcTickAutonomy, 1000);
requestAnimationFrame(loop);
