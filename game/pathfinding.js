// ── pathfinding.js — 栅格 A* 寻路模块（ally 绕障移动）──────────────────────────
//
// 职责：assault 接近阶段绕障到交战锚点；guard 回贴玩家时绕障跟随。
// 战斗微操（射程内走位/风筝/躲弹）由 RL 负责，与本模块零重叠。
//
// 设计：
//   • 栅格化整张可走画布；格心落在"膨胀后障碍"内则标记阻塞（膨胀量 = ally 半径）。
//   • A*（8 邻接 + octile 启发式 + 防穿墙角）求最短格路。
//   • string-pulling 把格路拉直成顺滑直线段。
//   • steerAlong 返回朝"可直线到达的最远航点"的归一化方向。
//
// 全部纯函数 + 不依赖游戏全局 state；坐标单位为画布像素。
// 由 index.html 在 game.js 之前加载，函数挂在全局供 game.js 调用。

const PF_CELL = 18;        // 栅格边长(px)
const PF_SMOOTH_STEP = 6;  // 路径平滑/线段可达性采样步长(px)

// 圆-AABB 碰撞（与 game.js collidesWithObstacle 同款逻辑，独立实现以保模块自洽）
function pfCollides(cx, cy, radius, obstacles) {
  for (const o of obstacles) {
    const nx = Math.max(o.x, Math.min(cx, o.x + o.w));
    const ny = Math.max(o.y, Math.min(cy, o.y + o.h));
    if (Math.hypot(cx - nx, cy - ny) < radius) return true;
  }
  return false;
}

// 构建导航栅格：格心落在(按 agentRadius 膨胀的)障碍内则阻塞。
// 障碍膨胀 = Minkowski，使路径中心线天然离墙 ≥ agentRadius，杜绝贴墙卡死。
function buildNavGrid(obstacles, agentRadius, worldW, worldH, cell = PF_CELL) {
  const cols = Math.ceil(worldW / cell);
  const rows = Math.ceil(worldH / cell);
  const blocked = new Uint8Array(cols * rows);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const cx = (c + 0.5) * cell;
      const cy = (r + 0.5) * cell;
      if (pfCollides(cx, cy, agentRadius, obstacles)) blocked[r * cols + c] = 1;
    }
  }
  return { cols, rows, cell, blocked, worldW, worldH };
}

function pfCellOf(grid, x, y) {
  const c = Math.max(0, Math.min(grid.cols - 1, Math.floor(x / grid.cell)));
  const r = Math.max(0, Math.min(grid.rows - 1, Math.floor(y / grid.cell)));
  return [c, r];
}

function pfIsFree(grid, c, r) {
  if (c < 0 || c >= grid.cols || r < 0 || r >= grid.rows) return false;
  return grid.blocked[r * grid.cols + c] === 0;
}

// 就近找自由格（环形扩散），用于起/终点落在障碍内时吸附
function pfNearestFree(grid, c, r) {
  if (pfIsFree(grid, c, r)) return [c, r];
  const maxR = Math.max(grid.cols, grid.rows);
  for (let radius = 1; radius < maxR; radius += 1) {
    for (let dc = -radius; dc <= radius; dc += 1) {
      for (let dr = -radius; dr <= radius; dr += 1) {
        if (Math.abs(dc) !== radius && Math.abs(dr) !== radius) continue; // 只看最外环
        if (pfIsFree(grid, c + dc, r + dr)) return [c + dc, r + dr];
      }
    }
  }
  return null;
}

// 极简二叉最小堆（A* open 集，懒删除：弹出时跳过已 closed 的项）
class PfHeap {
  constructor() { this.a = []; }
  get size() { return this.a.length; }
  push(p, v) {
    const a = this.a;
    a.push([p, v]);
    let i = a.length - 1;
    while (i > 0) {
      const par = (i - 1) >> 1;
      if (a[par][0] <= a[i][0]) break;
      [a[par], a[i]] = [a[i], a[par]];
      i = par;
    }
  }
  pop() {
    const a = this.a;
    const top = a[0];
    const last = a.pop();
    if (a.length) {
      a[0] = last;
      let i = 0;
      const n = a.length;
      for (;;) {
        const l = 2 * i + 1;
        const r = 2 * i + 2;
        let m = i;
        if (l < n && a[l][0] < a[m][0]) m = l;
        if (r < n && a[r][0] < a[m][0]) m = r;
        if (m === i) break;
        [a[m], a[i]] = [a[i], a[m]];
        i = m;
      }
    }
    return top;
  }
}

// A*：start/goal 为世界坐标 {x,y}。返回世界坐标航点数组；无解返回 null。
function findPath(grid, start, goal) {
  let [sc, sr] = pfCellOf(grid, start.x, start.y);
  const [rawGc, rawGr] = pfCellOf(grid, goal.x, goal.y);
  const goalCellFree = pfIsFree(grid, rawGc, rawGr);
  const s = pfNearestFree(grid, sc, sr);
  const g = pfNearestFree(grid, rawGc, rawGr);
  if (!s || !g) return null;
  [sc, sr] = s;
  const [gc, gr] = g;

  const cols = grid.cols;
  const idOf = (c, r) => r * cols + c;
  const startId = idOf(sc, sr);
  const goalId = idOf(gc, gr);
  if (startId === goalId) return [{ x: goal.x, y: goal.y }];

  const SQRT2 = Math.SQRT2;
  const hEst = (c, r) => {
    const dc = Math.abs(c - gc);
    const dr = Math.abs(r - gr);
    return (dc + dr) + (SQRT2 - 2) * Math.min(dc, dr); // octile
  };

  const gScore = new Map([[startId, 0]]);
  const came = new Map();
  const closed = new Set();
  const open = new PfHeap();
  open.push(hEst(sc, sr), startId);

  const NEI = [
    [1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1],
    [1, 1, SQRT2], [1, -1, SQRT2], [-1, 1, SQRT2], [-1, -1, SQRT2],
  ];

  let found = false;
  while (open.size) {
    const [, curId] = open.pop();
    if (closed.has(curId)) continue;
    if (curId === goalId) { found = true; break; }
    closed.add(curId);
    const cc = curId % cols;
    const cr = (curId - cc) / cols;
    const cg = gScore.get(curId);
    for (const [dc, dr, cost] of NEI) {
      const nc = cc + dc;
      const nr = cr + dr;
      if (!pfIsFree(grid, nc, nr)) continue;
      // 对角防穿墙角：两侧正交格都必须自由
      if (dc !== 0 && dr !== 0 && (!pfIsFree(grid, cc + dc, cr) || !pfIsFree(grid, cc, cr + dr))) continue;
      const nId = idOf(nc, nr);
      if (closed.has(nId)) continue;
      const ng = cg + cost;
      if (ng < (gScore.get(nId) ?? Infinity)) {
        gScore.set(nId, ng);
        came.set(nId, curId);
        open.push(ng + hEst(nc, nr), nId);
      }
    }
  }
  if (!found && !gScore.has(goalId)) return null;

  // 回溯格路 → 世界坐标
  const path = [];
  let cur = goalId;
  while (cur !== undefined) {
    const cc = cur % cols;
    const cr = (cur - cc) / cols;
    path.push({ x: (cc + 0.5) * grid.cell, y: (cr + 0.5) * grid.cell });
    if (cur === startId) break;
    cur = came.get(cur);
  }
  path.reverse();
  // 终点用真实目标坐标收尾——仅当其所在格自由，避免把航点钉进障碍内部
  if (goalCellFree) path.push({ x: goal.x, y: goal.y });
  return path;
}

// 线段 a→b 在 agentRadius 膨胀下是否无碰撞（沿线采样）
function pfSegmentClear(ax, ay, bx, by, radius, obstacles, step = PF_SMOOTH_STEP) {
  const dist = Math.hypot(bx - ax, by - ay);
  const n = Math.max(1, Math.ceil(dist / step));
  for (let i = 0; i <= n; i += 1) {
    const t = i / n;
    if (pfCollides(ax + (bx - ax) * t, ay + (by - ay) * t, radius, obstacles)) return false;
  }
  return true;
}

// string-pulling：把格路拉直，删掉可被直线段跨越的中间点
function smoothPath(path, obstacles, agentRadius) {
  if (!path || path.length <= 2) return path;
  const out = [path[0]];
  let anchor = 0;
  for (let i = 2; i < path.length; i += 1) {
    if (!pfSegmentClear(path[anchor].x, path[anchor].y, path[i].x, path[i].y, agentRadius, obstacles)) {
      out.push(path[i - 1]);
      anchor = i - 1;
    }
  }
  out.push(path[path.length - 1]);
  return out;
}

// 路径跟随：返回朝"从当前位置可直线到达的最远航点"的归一化方向 [dx,dy]
// 修复：去掉 targetIdx+1 的提前跳跃——该点在定义上已超出当前 LOS 范围，
// 跳过去会让 moveWithCollision 把 ally 顶回来，产生转角抖动。
// string-pulling 已经把路径拉直，到达最远可见点后下一帧自然会看到更远的点。
function steerAlong(path, pos, obstacles, agentRadius) {
  if (!path || path.length === 0) return [0, 0];
  let targetIdx = 0;
  for (let i = path.length - 1; i >= 0; i -= 1) {
    if (pfSegmentClear(pos.x, pos.y, path[i].x, path[i].y, agentRadius, obstacles)) {
      targetIdx = i;
      break;
    }
  }
  const wp = path[targetIdx];
  const dx = wp.x - pos.x;
  const dy = wp.y - pos.y;
  const len = Math.hypot(dx, dy);
  if (len < 1e-4) return [0, 0];
  return [dx / len, dy / len];
}

function segmentClear(ax, ay, bx, by, radius, obstacles) {
  return pfSegmentClear(ax, ay, bx, by, radius, obstacles);
}
