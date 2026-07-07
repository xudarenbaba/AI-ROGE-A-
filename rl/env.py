"""
AssaultEnv — 与 game.js 物理完全对齐的 gymnasium 环境。

战斗专家版：导航（绕障接近）由 game.js 的 A* 寻路负责，本环境只训练
"射程内战斗微操"。因此 episode 直接在敌人射程内开局，agent 输出 9 个离散
移动方向之一（含静止），负责走位/风筝/躲子弹；攻击逻辑不变（始终朝 LOS
加权最近敌人开火）。脱离交战包络（dist > attack_range×1.3）即截断，镜像
运行时 FSM 把控制权交回 A*。

物理常量全部来自 game.js，禁止随意修改。

设计要点：
  - reset() 射程内开局：ally 生成在随机一个敌人周围 [kite+5, attack×1.3] 且 LOS 通畅处
  - OBS 109 维：删绝对坐标/距离变化量/历史最小距离等导航信号，加自身 hp
  - 奖励只评战斗：伤害、受伤、射程区间×LOS、躲弹、遮挡开火、动作平滑
  - 无课程学习（导航课程已无意义）；楼层难度仍随机
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── 从 game.js 精确对齐的常量 ─────────────────────────────────────────────────

CANVAS_W: float = 900.0
CANVAS_H: float = 540.0

ALLY_RADIUS: float = 13.0
ALLY_MAX_HP: float = 160.0
ALLY_BASE_SPEED: float = 200.0

ASSAULT_ATTACK_RANGE: float = 110.0
ASSAULT_KITE_RANGE: float   = 65.0   # 从 55 调整到 65，保证 ally 与敌人的视觉间距
ASSAULT_INTERVAL: float     = 0.45
ASSAULT_SPEED_MUL: float    = 1.2
ASSAULT_DAMAGE: float       = 13.0
ASSAULT_BULLET_SPEED: float = 400.0

MOB_RADIUS: float        = 12.0
MOB_BASE_HP: float       = 30.0
MOB_BASE_SPEED: float    = 42.0
MOB_SHOOT_CD_MIN: float  = 0.8
MOB_SHOOT_CD_MAX: float  = 1.8
MOB_BULLET_SPEED: float  = 170.0
MOB_BULLET_DAMAGE: float = 5.0
MOB_SHOOT_CD_RESET: float = 1.6

BOSS_RADIUS: float        = 20.0
BOSS_BASE_HP: float       = 200.0
BOSS_BASE_SPEED: float    = 32.0
BOSS_BULLET_SPEED: float  = 200.0
BOSS_BULLET_DAMAGE: float = 9.0
BOSS_SHOOT_CD_RESET: float = 1.2

BULLET_RADIUS: float = 4.0
BULLET_TTL: float    = 2.2
ALLY_BULLET_SPEED: float = 400.0

DT: float       = 1.0 / 60.0
MAX_STEPS: int  = 60 * 20     # 战斗回合短：20s 足够（导航不再占用步数）
FLOOR_RANGE     = (1, 6)

OBSTACLE_LAYOUTS: list[list[dict]] = [
    # 布局 0：中央横墙 + 两侧竖柱 + 斜角掩体
    [
        {"x": 360, "y": 255, "w": 180, "h": 22},
        {"x": 240, "y": 170, "w": 22,  "h": 130},
        {"x": 660, "y": 220, "w": 22,  "h": 130},
        {"x": 480, "y": 140, "w": 140, "h": 20},
        {"x": 420, "y": 360, "w": 140, "h": 20},
        {"x": 300, "y": 360, "w": 80,  "h": 20},
    ],
    # 布局 1：走廊型（上下各一道长墙，中间留缺口）
    [
        {"x": 280, "y": 145, "w": 200, "h": 20},
        {"x": 560, "y": 145, "w": 160, "h": 20},
        {"x": 280, "y": 375, "w": 160, "h": 20},
        {"x": 520, "y": 375, "w": 200, "h": 20},
        {"x": 235, "y": 220, "w": 20,  "h": 110},
        {"x": 660, "y": 210, "w": 20,  "h": 110},
        {"x": 410, "y": 245, "w": 100, "h": 20},
    ],
    # 布局 2：分散长条（斜向交错）
    # 中竖柱 x=450 → x=460，与左上横条(右端 x=430)间隙 30px ≥ 26px 最小通道
    [
        {"x": 270, "y": 160, "w": 160, "h": 20},
        {"x": 580, "y": 200, "w": 20,  "h": 150},
        {"x": 340, "y": 340, "w": 160, "h": 20},
        {"x": 630, "y": 330, "w": 140, "h": 20},
        {"x": 240, "y": 280, "w": 20,  "h": 100},
        {"x": 460, "y": 150, "w": 20,  "h": 120},
    ],
    # 布局 3：十字形 + 外围长条
    [
        {"x": 390, "y": 230, "w": 120, "h": 20},
        {"x": 445, "y": 165, "w": 20,  "h": 140},
        {"x": 240, "y": 155, "w": 130, "h": 20},
        {"x": 620, "y": 155, "w": 130, "h": 20},
        {"x": 240, "y": 365, "w": 130, "h": 20},
        {"x": 620, "y": 365, "w": 130, "h": 20},
    ],
]

MOB_BASE_POSITIONS = [
    (520, 100), (650, 150), (780, 100),
    (560, 380), (700, 430), (820, 360),
    (700, 270), (820, 200), (760, 430),
    (850, 130),
]

_SQ2 = math.sqrt(2) / 2
ACTION_VECTORS: list[tuple[float, float]] = [
    (0.0,   0.0),    # 0 静止
    (0.0,  -1.0),    # 1 上
    ( _SQ2, -_SQ2),  # 2 右上
    (1.0,   0.0),    # 3 右
    ( _SQ2,  _SQ2),  # 4 右下
    (0.0,   1.0),    # 5 下
    (-_SQ2,  _SQ2),  # 6 左下
    (-1.0,  0.0),    # 7 左
    (-_SQ2, -_SQ2),  # 8 左上
]
N_ACTIONS = len(ACTION_VECTORS)  # 9

# ── OBS 维度（战斗专家版：射线检测 + LSTM，删除导航信号）──────────────────────
#
# 段1  自身状态              3   (攻击冷却, hp比例, 敌人数量)
# 段2  主目标敌人            8   (dx, dy, 距离, hp, is_boss, 射击CD, LOS, 射程标志)
# 段3  其余最多4个敌人       4×5 = 20
# 段4  最多8颗威胁子弹       8×6 = 48
# 段5  射线检测              N_RAYS = 16  (战斗靠它贴掩体/躲进墙缝)
# 段6  到四壁距离            4
# 段7  上帧动作 one-hot      9
# 段8  最危险子弹 TTA        1
# ─────────────────────────────────────────────────────
# 合计                      109
#
# 相对 v9 的删改（导航交给 A*，战斗用不到）：
#   - 删 ally 绝对坐标(段1)：战斗平移不变，只需相对几何 + 四壁距离 + 射线
#   - 删 到目标距离变化量：LSTM 已含时序信息
#   - 删 历史最小距离：纯反绕路导航信号
#   + 加 自身 hp 比例：战斗专家按血量调节冒险程度

MAX_ENEMIES   = 5
MAX_BULLETS   = 8

# 射线检测：从 ally 向 16 个均匀方向发射，返回到障碍物的归一化距离
N_RAYS       = 16
RAY_MAX_DIST = 260.0   # 射线最大检测距离，超出视为无障碍（归一化为 1.0）
RAY_STEPS    = 26      # 每条射线步进采样次数（步长 = RAY_MAX_DIST / RAY_STEPS ≈ 10px）

_SEG1 = 3                        # 自身：攻击冷却, hp比例, 敌人数量
_SEG2 = 8
_SEG3 = (MAX_ENEMIES - 1) * 5   # 20
_SEG4 = MAX_BULLETS * 6          # 48
_SEG5 = N_RAYS                   # 16（射线检测）
_SEG6 = 4
_SEG7 = N_ACTIONS                # 9
_SEG8 = 1                        # 最危险子弹 TTA
OBS_DIM = _SEG1 + _SEG2 + _SEG3 + _SEG4 + _SEG5 + _SEG6 + _SEG7 + _SEG8  # 109

# 16 方向射线的单位向量（每 22.5°）
_RAY_DIRS = [
    (math.cos(2 * math.pi * i / N_RAYS), math.sin(2 * math.pi * i / N_RAYS))
    for i in range(N_RAYS)
]

MAX_BULLET_DIST = 200.0  # 超出此范围的子弹视为无威胁

# v4 躲弹强化：稠密弹道威胁阈值（归一化 tta，×BULLET_TTL≈0.77s）。
# 会命中 ally 且归一化 tta < 此值的敌弹，按逼近程度每帧持续扣分（见 _compute_reward 段4）。
_THREAT_TTA_NORM = 0.35


# ── 数据类 ────────────────────────────────────────────────────────────────────

@dataclass
class Entity:
    x: float
    y: float
    radius: float
    hp: float
    max_hp: float
    speed: float

    def dist(self, other: "Entity | Bullet") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class Enemy(Entity):
    kind: str      = "mob"
    shoot_cd: float = 0.0


@dataclass
class Bullet:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    damage: float
    ttl: float


# ── 物理工具函数（精确复刻 game.js）─────────────────────────────────────────

def _normalize(dx: float, dy: float) -> tuple[float, float]:
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return 0.0, 0.0
    return dx / length, dy / length


def _collides_with_obstacle(cx: float, cy: float, radius: float,
                             obstacles: list[dict]) -> bool:
    for o in obstacles:
        near_x = max(o["x"], min(cx, o["x"] + o["w"]))
        near_y = max(o["y"], min(cy, o["y"] + o["h"]))
        if math.hypot(cx - near_x, cy - near_y) < radius:
            return True
    return False


def _bullet_hits_obstacle(bx: float, by: float, obstacles: list[dict]) -> bool:
    for o in obstacles:
        if o["x"] <= bx <= o["x"] + o["w"] and o["y"] <= by <= o["y"] + o["h"]:
            return True
    return False


def _move_with_collision(entity: Entity, dx: float, dy: float,
                          obstacles: list[dict]) -> None:
    r = entity.radius
    new_x = max(r, min(entity.x + dx, CANVAS_W - r))
    new_y = max(r, min(entity.y + dy, CANVAS_H - r))
    if not _collides_with_obstacle(new_x, new_y, r, obstacles):
        entity.x = new_x
        entity.y = new_y
    elif not _collides_with_obstacle(new_x, entity.y, r, obstacles):
        entity.x = new_x
    elif not _collides_with_obstacle(entity.x, new_y, r, obstacles):
        entity.y = new_y


def _create_bullet(owner_x: float, owner_y: float,
                   target_x: float, target_y: float,
                   speed: float, damage: float) -> Bullet:
    nx, ny = _normalize(target_x - owner_x, target_y - owner_y)
    return Bullet(x=owner_x, y=owner_y,
                  vx=nx * speed, vy=ny * speed,
                  radius=BULLET_RADIUS, damage=damage, ttl=BULLET_TTL)


def _has_line_of_sight(ax: float, ay: float,
                        bx: float, by: float,
                        obstacles: list[dict],
                        steps: int = 16) -> bool:
    """
    射线步进检测两点间是否有障碍物遮挡。
    steps=16 在精度和性能间取得平衡（每帧调用一次，约 16 次 AABB 检测）。
    返回 True 表示视线通畅，False 表示被障碍物遮挡。
    """
    for i in range(1, steps):
        t = i / steps
        px = ax + (bx - ax) * t
        py = ay + (by - ay) * t
        for o in obstacles:
            if o["x"] <= px <= o["x"] + o["w"] and o["y"] <= py <= o["y"] + o["h"]:
                return False
    return True


def _raycast_obstacle(ox: float, oy: float,
                      dx: float, dy: float,
                      obstacles: list[dict],
                      max_dist: float = RAY_MAX_DIST,
                      steps: int = RAY_STEPS) -> float:
    """
    从 (ox, oy) 沿单位方向 (dx, dy) 步进，检测第一个障碍物碰撞点的距离。
    返回归一化距离 [0, 1]：0 = 紧贴障碍物，1 = max_dist 内无障碍。
    也把画布边界当作障碍（射线打到墙壁也返回距离）。
    """
    step_len = max_dist / steps
    for i in range(1, steps + 1):
        d = i * step_len
        px = ox + dx * d
        py = oy + dy * d
        # 画布边界
        if px < 0 or px > CANVAS_W or py < 0 or py > CANVAS_H:
            return min(1.0, d / max_dist)
        # 障碍物 AABB
        for o in obstacles:
            if o["x"] <= px <= o["x"] + o["w"] and o["y"] <= py <= o["y"] + o["h"]:
                return d / max_dist
    return 1.0


def _bullet_time_to_ally(b: Bullet, ally: Entity) -> float:
    """
    圆盘碰撞 TTA（秒），归一化到 [0,1]（÷ BULLET_TTL）。
    未命中、背向或超出 TTL 返回 1.0（与 game.js _rlBulletTTA 对齐）。
    """
    rx = ally.x - b.x
    ry = ally.y - b.y
    vx, vy = b.vx, b.vy
    v2 = vx * vx + vy * vy
    if v2 < 1e-6:
        return 1.0
    hit_r = ally.radius + b.radius
    a = v2
    b_coef = 2.0 * (rx * vx + ry * vy)
    c = rx * rx + ry * ry - hit_r * hit_r
    disc = b_coef * b_coef - 4.0 * a * c
    if disc < 0:
        return 1.0
    sqrt_disc = math.sqrt(disc)
    t_hit: float | None = None
    for t in ((-b_coef - sqrt_disc) / (2.0 * a), (-b_coef + sqrt_disc) / (2.0 * a)):
        if t >= 0 and (t_hit is None or t < t_hit):
            t_hit = t
    if t_hit is None or t_hit > b.ttl:
        return 1.0
    return min(1.0, max(0.0, t_hit / BULLET_TTL))


# ── 环境主体 ──────────────────────────────────────────────────────────────────

class AssaultEnv(gym.Env):
    """
    战斗专家环境（射程内开局，导航交给运行时 A*）。

    动作空间：Discrete(9)，0=静止，1-8=8方向移动
    观测空间：109 维 float32（段定义见上方 OBS 维度注释），配合 RecurrentPPO 的 LSTM
    """

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._ally: Entity = None          # type: ignore[assignment]
        self._enemies: list[Enemy] = []
        self._ally_bullets: list[Bullet] = []
        self._enemy_bullets: list[Bullet] = []
        self._obstacles: list[dict] = []
        self._attack_cd: float = 0.0
        self._step_count: int = 0
        self._floor: int = 1
        self._hp_mul: float = 1.0
        self._speed_mul: float = 1.0
        self._prev_action: int = 0
        self._still_frames: int = 0
        self._oor_frames: int = 0          # 连续"脱离交战包络"帧数

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None,
              options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        floor = random.randint(*FLOOR_RANGE)
        self._floor = floor
        f = floor - 1
        self._hp_mul    = 1.0 + f * 0.30
        self._speed_mul = 1.0 + f * 0.06
        mob_count = min(3 + int(f * 1.2), 10)

        # 障碍物随机选一套布局（无课程：导航不再是 RL 的任务）
        self._obstacles = random.choice(OBSTACLE_LAYOUTS)

        # 先放满敌人（全图基准位置 + boss）
        self._enemies = []
        for i in range(mob_count):
            bx, by = MOB_BASE_POSITIONS[i % len(MOB_BASE_POSITIONS)]
            ex, ey = self._safe_spawn(bx, by, MOB_RADIUS, fallback_x=750.0, fallback_y=270.0)
            self._enemies.append(Enemy(
                x=ex, y=ey,
                radius=MOB_RADIUS,
                hp=round(MOB_BASE_HP * self._hp_mul),
                max_hp=round(MOB_BASE_HP * self._hp_mul),
                speed=MOB_BASE_SPEED * self._speed_mul,
                kind="mob",
                shoot_cd=random.uniform(MOB_SHOOT_CD_MIN, MOB_SHOOT_CD_MAX),
            ))
        bx, by = self._safe_spawn(820.0, 270.0, BOSS_RADIUS, fallback_x=830.0, fallback_y=400.0)
        self._enemies.append(Enemy(
            x=bx, y=by,
            radius=BOSS_RADIUS,
            hp=round(BOSS_BASE_HP * self._hp_mul),
            max_hp=round(BOSS_BASE_HP * self._hp_mul),
            speed=BOSS_BASE_SPEED * self._speed_mul,
            kind="boss",
            shoot_cd=0.8,
        ))

        # 射程内开局：ally 生成在随机一个敌人周围 [kite+5, attack×1.3] 且 LOS 通畅处
        self._ally = self._spawn_ally_in_range()

        self._ally_bullets  = []
        self._enemy_bullets = []
        self._attack_cd     = 0.0
        self._step_count    = 0
        self._prev_action   = 0
        self._still_frames  = 0
        self._oor_frames    = 0

        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._ally is not None, "call reset() first"
        self._step_count += 1

        ally_hp_before = self._ally.hp

        # 1. ally 移动
        speed = self._ally.speed * ASSAULT_SPEED_MUL
        mx, my = ACTION_VECTORS[action]
        _move_with_collision(self._ally, mx * speed * DT, my * speed * DT,
                             self._obstacles)

        # 2. ally 攻击（朝 LOS 加权最近敌人）
        self._attack_cd = max(0.0, self._attack_cd - DT)
        target_before = self._nearest_enemy()
        fired_this_step = False
        if target_before is not None and self._attack_cd <= 0.0:
            self._ally_bullets.append(
                _create_bullet(self._ally.x, self._ally.y,
                               target_before.x, target_before.y,
                               ALLY_BULLET_SPEED, ASSAULT_DAMAGE)
            )
            self._attack_cd = ASSAULT_INTERVAL
            fired_this_step = True

        # 3. 敌人移动 + 开火
        for enemy in self._enemies:
            enemy.shoot_cd = max(0.0, enemy.shoot_cd - DT)
            nx, ny = _normalize(self._ally.x - enemy.x, self._ally.y - enemy.y)
            _move_with_collision(enemy, nx * enemy.speed * DT, ny * enemy.speed * DT,
                                 self._obstacles)
            if enemy.shoot_cd <= 0.0:
                spd = BOSS_BULLET_SPEED  if enemy.kind == "boss" else MOB_BULLET_SPEED
                dmg = BOSS_BULLET_DAMAGE if enemy.kind == "boss" else MOB_BULLET_DAMAGE
                cd  = BOSS_SHOOT_CD_RESET if enemy.kind == "boss" else MOB_SHOOT_CD_RESET
                self._enemy_bullets.append(
                    _create_bullet(enemy.x, enemy.y,
                                   self._ally.x, self._ally.y, spd, dmg)
                )
                enemy.shoot_cd = cd

        # 4. 子弹物理
        damage_dealt = self._update_bullets()

        # 5. 清除死亡敌人
        self._enemies = [e for e in self._enemies if e.hp > 0]

        # 6. 判断本步开火是否有 LOS（用开火前的目标位置）
        fired_without_los = False
        if fired_this_step and target_before is not None:
            fired_without_los = not _has_line_of_sight(
                self._ally.x, self._ally.y,
                target_before.x, target_before.y,
                self._obstacles,
            )

        # 7. 奖励（只评战斗微操）
        hp_lost = max(0.0, ally_hp_before - self._ally.hp)
        target_now = self._nearest_enemy()
        dist_now   = self._ally.dist(target_now) if target_now is not None else 0.0
        min_bullet_tta = (
            min((_bullet_time_to_ally(b, self._ally) for b in self._enemy_bullets), default=1.0)
            if self._enemy_bullets else 1.0
        )
        reward = self._compute_reward(
            damage_dealt, hp_lost, target_now, action,
            dist_now, fired_without_los, self._still_frames, min_bullet_tta,
        )

        # 8. 更新帧间状态
        self._still_frames = self._still_frames + 1 if action == 0 else 0
        self._prev_action  = action

        # 9. 终止 / 截断
        terminated = self._ally.hp <= 0 or len(self._enemies) == 0
        if target_now is not None and dist_now > ASSAULT_ATTACK_RANGE * 1.3:
            self._oor_frames += 1
        else:
            self._oor_frames = 0
        truncated = self._step_count >= MAX_STEPS or self._oor_frames > 30

        # 脱离交战包络：运行时此刻控制权交回 A* 寻路，给一次性小惩罚
        if truncated and self._oor_frames > 30:
            reward -= 5.0

        return self._get_obs(), reward, terminated, truncated, {
            "damage_dealt": damage_dealt,
            "hp_lost": hp_lost,
            "enemies_left": len(self._enemies),
        }

    # ── 观测构建 ──────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """
        构建 109 维归一化观测向量（战斗专家版）。

        归一化约定：
          相对位置 dx/dy  / CANVAS_W|H，有符号   → [-1,1]
          距离            / DIAG(≈1051)           → [0,1]
          hp              / max_hp                → [0,1]
          冷却            / 最大冷却时间           → [0,1]
          速度方向        单位向量                → [-1,1]
          one-hot         0/1
        """
        DIAG = math.hypot(CANVAS_W, CANVAS_H)
        obs  = np.zeros(OBS_DIM, dtype=np.float32)
        idx  = 0
        ally = self._ally
        target = self._nearest_enemy()

        # ── 段1：自身状态 (3维)：攻击冷却, hp比例, 敌人数量 ───────────────────
        # 战斗专家：去绝对坐标(平移不变)，加 hp（按血量调节冒险程度）
        obs[idx]   = min(1.0, self._attack_cd / ASSAULT_INTERVAL)
        obs[idx+1] = ally.hp / ally.max_hp
        obs[idx+2] = len(self._enemies) / 11.0
        idx += _SEG1

        # ── 段2：主目标敌人（最近，8维）──────────────────────────────────────
        # dx, dy, 距离, hp比例, is_boss, 射击冷却比例, LOS标志, 到目标的有效射程标志
        if target is not None:
            dx   = target.x - ally.x
            dy   = target.y - ally.y
            dist = math.hypot(dx, dy)
            shoot_cd_max = BOSS_SHOOT_CD_RESET if target.kind == "boss" else MOB_SHOOT_CD_RESET
            los  = _has_line_of_sight(ally.x, ally.y, target.x, target.y, self._obstacles)
            # 有效射程标志：在 [kite_range, attack_range] 内为 1.0，线性衰减
            in_range = 1.0 if ASSAULT_KITE_RANGE < dist < ASSAULT_ATTACK_RANGE else 0.0
            obs[idx]   = dx / CANVAS_W
            obs[idx+1] = dy / CANVAS_H
            obs[idx+2] = dist / DIAG
            obs[idx+3] = target.hp / target.max_hp
            obs[idx+4] = 1.0 if target.kind == "boss" else 0.0
            obs[idx+5] = target.shoot_cd / shoot_cd_max
            obs[idx+6] = 1.0 if los else 0.0   # LOS 标志：1=视线通畅
            obs[idx+7] = in_range
        idx += _SEG2

        # ── 段3：其余最多 4 个敌人（5维/敌）────────────────────────────────────
        others = sorted(
            [e for e in self._enemies if e is not target],
            key=lambda e: math.hypot(e.x - ally.x, e.y - ally.y)
        )[: MAX_ENEMIES - 1]
        for e in others:
            dx   = e.x - ally.x
            dy   = e.y - ally.y
            dist = math.hypot(dx, dy)
            obs[idx]   = dx / CANVAS_W
            obs[idx+1] = dy / CANVAS_H
            obs[idx+2] = dist / DIAG
            obs[idx+3] = e.hp / e.max_hp
            obs[idx+4] = 1.0 if e.kind == "boss" else 0.0
            idx += 5
        idx += (MAX_ENEMIES - 1 - len(others)) * 5

        # ── 段4：最多 8 颗威胁子弹（6维/颗）─────────────────────────────────
        # dx, dy, 速度方向vx, 速度方向vy, 距离归一化, 预测碰撞时间归一化
        # 排序：优先按预测碰撞时间（越短越危险）
        threat_bullets = [
            b for b in self._enemy_bullets
            if math.hypot(b.x - ally.x, b.y - ally.y) < MAX_BULLET_DIST
        ]
        threat_bullets.sort(
            key=lambda b: _bullet_time_to_ally(b, ally)
        )
        threat_bullets = threat_bullets[: MAX_BULLETS]
        for b in threat_bullets:
            dx   = b.x - ally.x
            dy   = b.y - ally.y
            dist = math.hypot(dx, dy)
            bspd = math.hypot(b.vx, b.vy) or 1.0
            tta  = _bullet_time_to_ally(b, ally)
            obs[idx]   = dx / CANVAS_W
            obs[idx+1] = dy / CANVAS_H
            obs[idx+2] = b.vx / bspd
            obs[idx+3] = b.vy / bspd
            obs[idx+4] = dist / MAX_BULLET_DIST
            obs[idx+5] = tta                    # 预测碰撞时间（0=即将命中）
            idx += 6
        idx += (MAX_BULLETS - len(threat_bullets)) * 6

        # ── 段5：射线检测（16 方向，替换矩形障碍物 obs）─────────────────────
        # 从 ally 向 16 个均匀方向发射射线，返回到障碍物/边界的归一化距离
        # 0 = 紧贴障碍，1 = 该方向 RAY_MAX_DIST 内无障碍
        for rdx, rdy in _RAY_DIRS:
            obs[idx] = _raycast_obstacle(ally.x, ally.y, rdx, rdy, self._obstacles)
            idx += 1

        # ── 段6：到四壁距离（4维）────────────────────────────────────────────
        obs[idx]   = ally.y / CANVAS_H
        obs[idx+1] = (CANVAS_H - ally.y) / CANVAS_H
        obs[idx+2] = ally.x / CANVAS_W
        obs[idx+3] = (CANVAS_W - ally.x) / CANVAS_W
        idx += _SEG6

        # ── 段7：上帧动作 one-hot（9维）──────────────────────────────────────
        obs[idx + self._prev_action] = 1.0
        idx += _SEG7

        # ── 段8：最危险子弹 TTA（1维）────────────────────────────────────────
        if threat_bullets:
            obs[idx] = min(_bullet_time_to_ally(b, ally) for b in threat_bullets)
        else:
            obs[idx] = 1.0
        idx += _SEG8

        assert idx == OBS_DIM, f"obs dim mismatch: {idx} != {OBS_DIM}"
        return obs

    # ── 奖励函数 ──────────────────────────────────────────────────────────────

    def _compute_reward(
        self,
        damage_dealt: float,
        hp_lost: float,
        target: Enemy | None,
        action: int,
        dist_now: float,
        fired_without_los: bool,
        still_frames: int,
        min_bullet_tta: float,
    ) -> float:
        """
        战斗专家奖励（v4 躲弹强化版）。相对 v3 的关键改动：
          + 段4 稠密弹道威胁惩罚：站在会命中的敌弹路径上每帧持续扣分，越逼近越疼，
            移出弹道立刻止损 —— 为“往哪躲”提供稠密梯度（v3 只有挨弹瞬间的稀疏负反馈）
          + 段4 危险时主动机动奖励：有威胁且移动给正奖励，打破静止惯性
          ↑ 段2 被命中惩罚 0.25 → 0.6（挨弹要显著比少打亏）
          − 删除 v3 的“安全奖励静止”（不再奖励不动，走位/风筝交给策略）
          ~ 段8 站桩惩罚改温和无条件（豁免 30 帧、最优位减半）
        段1 伤害 / 段3 最优射程位 / 段9 胜负锚点维持，避免训练出只躲不打的怂包。

        段位：1伤害 2受伤 3射程×LOS 4弹道威胁&机动 6遮挡开火 7动作平滑
              8连续静止 9死亡/胜利 10时间。（参数 min_bullet_tta 现由段4 自行遍历，保留以兼容签名）
        """
        reward = 0.0
        ally   = self._ally

        # 1. 伤害奖励（锚住输出动机）
        reward += damage_dealt * 0.08

        # 2. 被命中惩罚（加重：挨弹要显著比少打亏）
        reward -= hp_lost * 0.6

        # 3. 射程区间 × LOS：维持 [kite, attack] 攻击位且视线通畅
        in_optimal_pos = False
        if target is not None:
            los = _has_line_of_sight(
                ally.x, ally.y, target.x, target.y, self._obstacles
            )
            if dist_now < ASSAULT_KITE_RANGE:
                reward -= 0.012   # 过近
            elif dist_now < ASSAULT_ATTACK_RANGE:
                if los:
                    reward += 0.020   # 最优攻击位
                    in_optimal_pos = True
                else:
                    reward -= 0.015   # 射程内遮挡=隔墙苟着
            elif dist_now < 130.0:
                reward += 0.006 if los else -0.008

        # 4. ★躲弹核心：稠密弹道威胁惩罚 + 危险时主动机动奖励
        #    threat = Σ 会命中且逼近的敌弹危险度（每颗 [0,1]，越近越大）。
        #    站在弹道上每帧持续疼 → 移出弹道 threat 下降 → 形成“往哪躲”的稠密梯度。
        threat = 0.0
        for b in self._enemy_bullets:
            tta = _bullet_time_to_ally(b, ally)          # 不会命中的子弹返回 1.0
            if tta < _THREAT_TTA_NORM:
                threat += 1.0 - tta / _THREAT_TTA_NORM   # [0,1]
        if threat > 0.0:
            reward -= 0.035 * min(threat, 3.0)           # 弹幕封顶，避免数值爆炸
            if action != 0:                              # 危险时移动 → 打破静止惯性
                reward += 0.012 * min(threat, 1.0)

        # 6. LOS 遮挡开火惩罚（别隔墙瞎打）
        if fired_without_los:
            reward -= 0.04

        # 7. 动作平滑惩罚：反向/正交快速切换（反向略放宽，给躲弹变向留空间）
        prev_vec = ACTION_VECTORS[self._prev_action]
        curr_vec = ACTION_VECTORS[action]
        dot = prev_vec[0] * curr_vec[0] + prev_vec[1] * curr_vec[1]
        if action != 0 and self._prev_action != 0:
            if dot < -0.7:
                reward -= 0.006
            elif dot < 0.3:
                reward -= 0.003

        # 8. 连续静止惩罚：温和无条件，打破安全站桩吸引子（最优位减半）
        if still_frames > 30:
            overage = min(still_frames - 30, 120)
            base_penalty = 0.004 + overage * (0.010 / 120)
            reward -= base_penalty * (0.5 if in_optimal_pos else 1.0)

        # 9. 死亡 / 胜利
        if ally.hp <= 0:
            reward -= 30.0
        if len(self._enemies) == 0:
            reward += 50.0

        # 10. 时间惩罚
        reward -= 0.003

        return float(reward)

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    def _spawn_ally_in_range(self) -> Entity:
        """
        射程内开局：在随机一个敌人周围 [kite+5, attack×1.3] 且 LOS 通畅、
        不与障碍物碰撞处放置 ally。镜像运行时 A* 把 ally 带到射程内的交接时刻。
        """
        engage = random.choice(self._enemies)
        lo = ASSAULT_KITE_RANGE + 5.0
        hi = ASSAULT_ATTACK_RANGE * 1.3
        for _ in range(60):
            ang = random.uniform(0, 2 * math.pi)
            r   = random.uniform(lo, hi)
            ax  = max(ALLY_RADIUS, min(engage.x + math.cos(ang) * r, CANVAS_W - ALLY_RADIUS))
            ay  = max(ALLY_RADIUS, min(engage.y + math.sin(ang) * r, CANVAS_H - ALLY_RADIUS))
            if (not _collides_with_obstacle(ax, ay, ALLY_RADIUS, self._obstacles)
                    and _has_line_of_sight(ax, ay, engage.x, engage.y, self._obstacles)):
                return Entity(x=ax, y=ay, radius=ALLY_RADIUS,
                              hp=ALLY_MAX_HP, max_hp=ALLY_MAX_HP, speed=ALLY_BASE_SPEED)
        # 兜底：沿 8 个方向找一个无碰撞落点（放在 0.9×attack）
        r = ASSAULT_ATTACK_RANGE * 0.9
        for k in range(8):
            ang = k * math.pi / 4
            ax  = max(ALLY_RADIUS, min(engage.x + math.cos(ang) * r, CANVAS_W - ALLY_RADIUS))
            ay  = max(ALLY_RADIUS, min(engage.y + math.sin(ang) * r, CANVAS_H - ALLY_RADIUS))
            if not _collides_with_obstacle(ax, ay, ALLY_RADIUS, self._obstacles):
                return Entity(x=ax, y=ay, radius=ALLY_RADIUS,
                              hp=ALLY_MAX_HP, max_hp=ALLY_MAX_HP, speed=ALLY_BASE_SPEED)
        return Entity(x=max(ALLY_RADIUS, engage.x - r), y=engage.y, radius=ALLY_RADIUS,
                      hp=ALLY_MAX_HP, max_hp=ALLY_MAX_HP, speed=ALLY_BASE_SPEED)

    def _safe_spawn(self, base_x: float, base_y: float,
                    radius: float,
                    fallback_x: float, fallback_y: float,
                    max_tries: int = 20,
                    jitter: float = 20.0) -> tuple[float, float]:
        """
        在 base 附近随机采样一个不与障碍物碰撞的坐标。
        最多尝试 max_tries 次，失败后返回 fallback 坐标。
        """
        for _ in range(max_tries):
            x = base_x + random.uniform(-jitter, jitter)
            y = base_y + random.uniform(-jitter, jitter)
            x = max(radius, min(x, CANVAS_W - radius))
            y = max(radius, min(y, CANVAS_H - radius))
            if not _collides_with_obstacle(x, y, radius, self._obstacles):
                return x, y
        return fallback_x, fallback_y

    def _nearest_enemy(self) -> Enemy | None:
        """
        LOS 加权评分选取目标。
        评分 = 直线距离 × LOS惩罚系数（LOS通畅=1.0，遮挡=1.3）
        系数从 1.8 降到 1.3：轻度偏好有 LOS 的目标，
        但不过度放大遮挡目标的"有效距离"，避免 agent 因为全部目标都被遮挡时
        不知道该往哪走、倾向于在原地等待。
        """
        if not self._enemies:
            return None
        ally = self._ally
        best = None
        best_score = float("inf")
        for e in self._enemies:
            dist = ally.dist(e)
            los = _has_line_of_sight(ally.x, ally.y, e.x, e.y, self._obstacles)
            score = dist * (1.0 if los else 1.3)
            if score < best_score:
                best_score = score
                best = e
        return best

    def _update_bullets(self) -> float:
        damage_dealt = 0.0

        new_ally_bullets = []
        for b in self._ally_bullets:
            b.x += b.vx * DT
            b.y += b.vy * DT
            b.ttl -= DT
            if (b.ttl <= 0
                    or b.x < -10 or b.x > CANVAS_W + 10
                    or b.y < -10 or b.y > CANVAS_H + 10
                    or _bullet_hits_obstacle(b.x, b.y, self._obstacles)):
                continue
            hit = False
            for e in self._enemies:
                if math.hypot(b.x - e.x, b.y - e.y) <= b.radius + e.radius:
                    e.hp -= b.damage
                    damage_dealt += b.damage
                    hit = True
                    break
            if not hit:
                new_ally_bullets.append(b)
        self._ally_bullets = new_ally_bullets

        new_enemy_bullets = []
        for b in self._enemy_bullets:
            b.x += b.vx * DT
            b.y += b.vy * DT
            b.ttl -= DT
            if (b.ttl <= 0
                    or b.x < -10 or b.x > CANVAS_W + 10
                    or b.y < -10 or b.y > CANVAS_H + 10
                    or _bullet_hits_obstacle(b.x, b.y, self._obstacles)):
                continue
            if math.hypot(b.x - self._ally.x, b.y - self._ally.y) <= b.radius + self._ally.radius:
                self._ally.hp -= b.damage
            else:
                new_enemy_bullets.append(b)
        self._enemy_bullets = new_enemy_bullets

        return damage_dealt
