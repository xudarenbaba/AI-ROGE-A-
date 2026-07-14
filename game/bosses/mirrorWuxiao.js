/** 孽镜假乌枭：独立 LSTM + 同权重 ONNX，观测主目标=玩家 */
(function () {
  const N_ACTIONS = 9;
  const LSTM_HIDDEN = 128;
  const LSTM_LAYERS = 1;
  const OBS_DIM = 109;
  const MAX_ENEMIES = 5;
  const MAX_BULLETS = 8;
  const MAX_BULLET_DIST = 200;
  const CANVAS_W = 900;
  const CANVAS_H = 540;
  const DIAG = Math.hypot(CANVAS_W, CANVAS_H);
  const N_RAYS = 16;
  const RAY_MAX = 260;
  const RAY_STEPS = 26;
  const ASSAULT_INTERVAL = 0.45;
  const BULLET_TTL = 2.2;

  const ACTION_VECTORS = [
    [0, 0], [0, -1], [0.7071, -0.7071], [1, 0], [0.7071, 0.7071],
    [0, 1], [-0.7071, 0.7071], [-1, 0], [-0.7071, -0.7071],
  ];

  const RAY_DIRS = [];
  for (let i = 0; i < N_RAYS; i++) {
    const a = (2 * Math.PI * i) / N_RAYS;
    RAY_DIRS.push([Math.cos(a), Math.sin(a)]);
  }

  const st = {
    prevAction: 0,
    rawAction: 0,
    appliedAction: 0,
    holdRemain: 0,
    lastLogits: null,
    hidden: null,
    cell: null,
    inferring: false,
    frame: 0,
  };

  function reset() {
    const size = LSTM_LAYERS * 1 * LSTM_HIDDEN;
    st.hidden = new Float32Array(size);
    st.cell = new Float32Array(size);
    st.prevAction = 0;
    st.rawAction = 0;
    st.appliedAction = 0;
    st.holdRemain = 0;
    st.lastLogits = null;
    st.inferring = false;
  }

  function norm(x, y) {
    const l = Math.hypot(x, y) || 1;
    return [x / l, y / l];
  }

  function bulletTTASec(b, entity) {
    const rx = entity.x - b.x;
    const ry = entity.y - b.y;
    const vx = b.vx;
    const vy = b.vy;
    const v2 = vx * vx + vy * vy;
    if (v2 < 1e-6) return BULLET_TTL;
    const hitR = entity.radius + (b.radius || 4);
    const a = v2;
    const bCoef = 2 * (rx * vx + ry * vy);
    const c = rx * rx + ry * ry - hitR * hitR;
    const disc = bCoef * bCoef - 4 * a * c;
    if (disc < 0) return BULLET_TTL;
    const sqrtDisc = Math.sqrt(disc);
    let tHit = null;
    for (const t of [(-bCoef - sqrtDisc) / (2 * a), (-bCoef + sqrtDisc) / (2 * a)]) {
      if (t >= 0 && (tHit === null || t < tHit)) tHit = t;
    }
    if (tHit === null || tHit > BULLET_TTL) return BULLET_TTL;
    return tHit;
  }

  function bulletTTA(b, entity) {
    return Math.min(1, Math.max(0, bulletTTASec(b, entity) / BULLET_TTL));
  }

  function hazardTTA(h, entity) {
    const distC = Math.hypot(entity.x - h.x, entity.y - h.y);
    const edge = distC - h.r - entity.radius;
    if (h.warnT > 0) return Math.min(1, (h.warnT / 1.1) * 0.5 + Math.max(0, edge) / 180);
    if (h.activeT > 0 && edge <= 0) return 0;
    if (edge <= 0) return 0;
    return Math.min(1, edge / 120);
  }

  function collectThreats(boss, state, hasLOS) {
    const entries = [];
    // 假乌枭要躲玩家弹 + 真乌枭弹
    for (const b of state.playerBullets || []) {
      if (Math.hypot(b.x - boss.x, b.y - boss.y) > MAX_BULLET_DIST) continue;
      entries.push({ kind: "b", obj: b, tta: bulletTTA(b, boss) });
    }
    for (const b of state.allyBullets || []) {
      if (Math.hypot(b.x - boss.x, b.y - boss.y) > MAX_BULLET_DIST) continue;
      entries.push({ kind: "b", obj: b, tta: bulletTTA(b, boss) });
    }
    for (const h of state.hazards || []) {
      if (Math.hypot(boss.x - h.x, boss.y - h.y) > h.r + MAX_BULLET_DIST) continue;
      entries.push({ kind: "h", obj: h, tta: hazardTTA(h, boss) });
    }
    entries.sort((a, b) => a.tta - b.tta);
    return entries.slice(0, MAX_BULLETS);
  }

  function raycast(ox, oy, dx, dy, obstacles) {
    const stepLen = RAY_MAX / RAY_STEPS;
    for (let i = 1; i <= RAY_STEPS; i++) {
      const d = i * stepLen;
      const px = ox + dx * d;
      const py = oy + dy * d;
      if (px < 0 || px > CANVAS_W || py < 0 || py > CANVAS_H) {
        return Math.min(1, d / RAY_MAX);
      }
      for (const o of obstacles || []) {
        if (px >= o.x && px <= o.x + o.w && py >= o.y && py <= o.y + o.h) {
          return d / RAY_MAX;
        }
      }
    }
    return 1;
  }

  /**
   * 观测重映射：boss=自身，target(玩家/乌枭)=主目标敌，
   * 另一友军单位=其他敌人，player/ally 弹=威胁
   */
  function buildObs(boss, target, state, obstacles, hasLOS, attackCd) {
    const obs = new Float32Array(OBS_DIM);
    let idx = 0;

    // 段1：自身 3
    obs[idx++] = Math.min(1, (attackCd || 0) / ASSAULT_INTERVAL);
    obs[idx++] = boss.hp / Math.max(1, boss.maxHp);
    // 「敌人数」：场上可打的单位数（玩家+真乌枭）
    let unitN = 1;
    if (state.ally && state.ally.hp > 0) unitN += 1;
    obs[idx++] = unitN / 11;

    // 段2：主目标 8
    if (target) {
      const dx = target.x - boss.x;
      const dy = target.y - boss.y;
      const dist = Math.hypot(dx, dy);
      const los = hasLOS(boss.x, boss.y, target.x, target.y);
      const inRange = dist > 65 && dist < 110 ? 1 : 0;
      obs[idx++] = dx / CANVAS_W;
      obs[idx++] = dy / CANVAS_H;
      obs[idx++] = dist / DIAG;
      obs[idx++] = (target.hp || 1) / Math.max(1, target.maxHp || 1);
      obs[idx++] = 0; // 非 boss 标记（目标是玩家）
      obs[idx++] = 0; // shootCd 未知
      obs[idx++] = los ? 1 : 0;
      obs[idx++] = inRange;
    } else {
      idx += 8;
    }

    // 段3：其他单位（真乌枭等）最多 4×5
    const others = [];
    if (state.ally && state.ally.hp > 0 && state.ally !== target) {
      others.push(state.ally);
    }
    // 也把小怪算作障碍性敌人（若有）
    (state.enemies || []).forEach((e) => {
      if (e === boss || e.hp <= 0) return;
      if (e.bossArchetype === "mirror_wuxiao") return;
      others.push(e);
    });
    others.sort((a, b) => Math.hypot(a.x - boss.x, a.y - boss.y) - Math.hypot(b.x - boss.x, b.y - boss.y));
    const take = others.slice(0, MAX_ENEMIES - 1);
    for (const e of take) {
      const dx = e.x - boss.x;
      const dy = e.y - boss.y;
      obs[idx++] = dx / CANVAS_W;
      obs[idx++] = dy / CANVAS_H;
      obs[idx++] = Math.hypot(dx, dy) / DIAG;
      obs[idx++] = (e.hp || 1) / Math.max(1, e.maxHp || 1);
      obs[idx++] = e.kind === "boss" ? 1 : 0;
    }
    idx += (MAX_ENEMIES - 1 - take.length) * 5;

    // 段4：威胁 8×6
    const threats = collectThreats(boss, state, hasLOS);
    for (const { kind, obj, tta } of threats) {
      if (kind === "b") {
        const b = obj;
        const dx = b.x - boss.x;
        const dy = b.y - boss.y;
        const dist = Math.hypot(dx, dy);
        const bspd = Math.hypot(b.vx, b.vy) || 1;
        obs[idx++] = dx / CANVAS_W;
        obs[idx++] = dy / CANVAS_H;
        obs[idx++] = b.vx / bspd;
        obs[idx++] = b.vy / bspd;
        obs[idx++] = dist / MAX_BULLET_DIST;
        obs[idx++] = tta;
      } else {
        const h = obj;
        const dx = h.x - boss.x;
        const dy = h.y - boss.y;
        const dist = Math.hypot(dx, dy);
        const [ex, ey] = norm(boss.x - h.x, boss.y - h.y);
        obs[idx++] = dx / CANVAS_W;
        obs[idx++] = dy / CANVAS_H;
        obs[idx++] = ex;
        obs[idx++] = ey;
        obs[idx++] = Math.max(0, dist - h.r) / MAX_BULLET_DIST;
        obs[idx++] = tta;
      }
    }
    idx += (MAX_BULLETS - threats.length) * 6;

    // 段5：射线
    for (const [rdx, rdy] of RAY_DIRS) {
      obs[idx++] = raycast(boss.x, boss.y, rdx, rdy, obstacles);
    }

    // 段6：四壁
    obs[idx++] = boss.y / CANVAS_H;
    obs[idx++] = (CANVAS_H - boss.y) / CANVAS_H;
    obs[idx++] = boss.x / CANVAS_W;
    obs[idx++] = (CANVAS_W - boss.x) / CANVAS_W;

    // 段7：上帧动作
    obs[idx + st.prevAction] = 1;
    idx += N_ACTIONS;

    // 段8：最危险 TTA
    obs[idx++] = threats.length ? threats[0].tta : 1;

    return obs;
  }

  function minThreatTtaSec(boss, state) {
    let minSec = BULLET_TTL;
    for (const arr of [state.playerBullets, state.allyBullets]) {
      for (const b of arr || []) {
        if (Math.hypot(b.x - boss.x, b.y - boss.y) > MAX_BULLET_DIST) continue;
        minSec = Math.min(minSec, bulletTTASec(b, boss));
      }
    }
    return minSec;
  }

  function bestDodge(ex, ey) {
    let bestA = 1;
    let bestDot = -Infinity;
    for (let i = 1; i < ACTION_VECTORS.length; i++) {
      const [vx, vy] = ACTION_VECTORS[i];
      const dot = vx * ex + vy * ey;
      if (dot > bestDot) { bestDot = dot; bestA = i; }
    }
    return bestA;
  }

  function emergencyDodge(boss, state) {
    let worst = null;
    let worstT = Infinity;
    for (const arr of [state.playerBullets, state.allyBullets]) {
      for (const b of arr || []) {
        if (Math.hypot(b.x - boss.x, b.y - boss.y) > MAX_BULLET_DIST) continue;
        const t = bulletTTASec(b, boss);
        if (t >= 0.25) continue;
        if (t < worstT) { worstT = t; worst = b; }
      }
    }
    if (!worst) return null;
    const spd = Math.hypot(worst.vx, worst.vy) || 1;
    const ux = worst.vx / spd;
    const uy = worst.vy / spd;
    const dx = boss.x - worst.x;
    const dy = boss.y - worst.y;
    const lateral = -uy * dx + ux * dy;
    const sign = lateral >= 0 ? 1 : -1;
    return bestDodge(-uy * sign, ux * sign);
  }

  function resolveAction(boss, state) {
    const minT = minThreatTtaSec(boss, state);
    const dodge = minT < 0.25 ? emergencyDodge(boss, state) : null;
    if (dodge !== null) {
      st.appliedAction = dodge;
      st.holdRemain = 3;
      st.prevAction = dodge;
      return dodge;
    }
    let desired = st.rawAction;
    if (st.lastLogits && desired !== st.appliedAction && desired !== 0 && st.appliedAction !== 0) {
      if (st.lastLogits[desired] - st.lastLogits[st.appliedAction] < 0.1) {
        desired = st.appliedAction;
      }
    }
    if (desired !== st.appliedAction && st.holdRemain > 0) {
      st.holdRemain -= 1;
      st.prevAction = st.appliedAction;
      return st.appliedAction;
    }
    st.holdRemain = desired !== st.appliedAction ? 3 : Math.max(0, st.holdRemain - 1);
    st.appliedAction = desired;
    st.prevAction = desired;
    return desired;
  }

  function inferAsync(session, boss, target, state, obstacles, hasLOS) {
    if (!session || st.inferring) return;
    if (!st.hidden || !st.cell) reset();
    // 错帧：每 2 帧推一次，减轻与友方抢算力
    st.frame += 1;
    if (st.frame % 2 === 1 && st.rawAction !== undefined) return;

    st.inferring = true;
    const obs = buildObs(boss, target, state, obstacles, hasLOS, boss.attackCd || 0);
    const hShape = [LSTM_LAYERS, 1, LSTM_HIDDEN];
    const feeds = {
      obs: new ort.Tensor("float32", obs, [1, OBS_DIM]),
      h_in: new ort.Tensor("float32", st.hidden, hShape),
      c_in: new ort.Tensor("float32", st.cell, hShape),
    };
    session.run(feeds).then((output) => {
      const logits = output.logits.data;
      st.lastLogits = logits;
      let best = 0;
      for (let i = 1; i < logits.length; i++) {
        if (logits[i] > logits[best]) best = i;
      }
      st.rawAction = best;
      st.hidden = output.h_out.data;
      st.cell = output.c_out.data;
      st.inferring = false;
    }).catch((e) => {
      console.warn("[MirrorRL] infer error", e);
      st.rawAction = 0;
      st.inferring = false;
    });
  }

  function actionVector(actionIdx) {
    return ACTION_VECTORS[actionIdx] || ACTION_VECTORS[0];
  }

  function create(pos, scale, meta) {
    const hpMul = (scale.hpMul || 1) * (meta.hpMul || 1);
    return {
      kind: "boss",
      bossArchetype: "mirror_wuxiao",
      looksLike: "ally",
      noDialogue: true,
      bossName: meta.name || "镜影·乌枭",
      ultName: meta.ult || "镜裂天坠",
      skillId: "mirror_shot",
      skillName: meta.skillName || "镜返",
      x: pos.x,
      y: pos.y,
      hp: Math.round(400 * hpMul),
      maxHp: Math.round(400 * hpMul),
      radius: 14,
      speed: 200 * (scale.speedMul || 1),
      shootCd: 0,
      dmgMul: (scale.dmgMul || 1) * 0.95,
      shootCdMul: scale.shootCdMul || 1,
      fanChance: 0,
      ultInterval: meta.ultInterval || 5,
      ultCd: 2.5,
      aoeColor: meta.aoeColor || "#aa88ff",
      phase: 1,
      combatPhase: "approach",
      navPath: null,
      navGoal: null,
      navReplanCd: 0,
      engagePoint: null,
      losStableFrames: 0,
      losLostFrames: 0,
      attackCd: 0,
    };
  }

  function isMirror(e) {
    return e && e.bossArchetype === "mirror_wuxiao";
  }

  window.GameMirrorBoss = {
    create,
    isMirror,
    reset,
    inferAsync,
    resolveAction,
    actionVector,
    buildObs,
    st,
  };
})();
