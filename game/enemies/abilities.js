/**
 * 敌人技能 + A* 寻路移动（小怪/影怪/精英/Boss 通用）
 * 依赖：pathfinding.js, bullets/patterns.js, bosses/patterns.js, eliteRegistry.js
 */
(function () {
  const BOSS_STOP_DIST = 70;
  const MAX_SUMMONS = 2;
  const MAX_SUMMONS_ENRAGED = 3;
  const SKILL_DMG_BONUS = 1.28;
  const MIN_DAMAGE_SKILL_MS = 1200;
  const ELITE_DAMAGE_SKILL_MS = 800;
  const ELITE_ANCHORS = [{ x: 400, y: 230 }, { x: 540, y: 310 }];

  function norm(dx, dy) {
    const len = Math.hypot(dx, dy);
    if (len < 0.0001) return [0, 0];
    return [dx / len, dy / len];
  }

  function dist(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
  }

  function skillDamage(enemy, base) {
    return Math.round(base * (enemy.dmgMul || 1) * SKILL_DMG_BONUS);
  }

  function eliteSkillCd(enemy, baseCd) {
    let cd = baseCd;
    if (enemy._fierce) cd *= 0.8;
    if (enemy.phase >= 2) cd *= 0.68;
    return cd;
  }

  function bossMechanicCdReset(enemy) {
    let cd = 2.8 + Math.random() * 0.9;
    if (enemy.phase >= 2) cd *= 0.65;
    return cd;
  }

  function clampCoord(x, y) {
    return {
      x: Math.max(60, Math.min(x, 900)),
      y: Math.max(60, Math.min(y, 480)),
    };
  }

  function tickEntityMotion(entity, dt) {
    if (!entity || entity.hp <= 0) return;
    if (!entity._motionPrev) {
      entity._motionPrev = { x: entity.x, y: entity.y };
      entity._vx = 0;
      entity._vy = 0;
      return;
    }
    const inv = 1 / Math.max(0.001, dt);
    entity._vx = (entity.x - entity._motionPrev.x) * inv;
    entity._vy = (entity.y - entity._motionPrev.y) * inv;
    entity._motionPrev.x = entity.x;
    entity._motionPrev.y = entity.y;
  }

  function predictPosition(entity, leadSec, floor = 1) {
    const lead = leadSec * (floor >= 3 ? 1 : 0.5);
    const vx = entity._vx || 0;
    const vy = entity._vy || 0;
    const speed = Math.hypot(vx, vy);
    const cap = speed > 120 ? 120 / speed : 1;
    return clampCoord(
      entity.x + vx * lead * cap,
      entity.y + vy * lead * cap,
    );
  }

  function getObstacles(H) {
    return H?.allObstacles?.() ?? [];
  }

  function hasSkillLOS(enemy, target, obstacles) {
    if (!enemy || !target) return false;
    const tx = target.x;
    const ty = target.y;
    if (tx == null || ty == null) return false;
    return window.GameShooting?.hasBulletLOS(
      enemy.x, enemy.y, tx, ty, obstacles,
    ) ?? true;
  }

  function livingTargets(state) {
    const out = [{ entity: state.player, id: "player" }];
    if (state.ally.hp > 0) out.push({ entity: state.ally, id: "ally" });
    return out;
  }

  function pickNearestVisibleTarget(state, enemy, obstacles) {
    const candidates = livingTargets(state);
    const visible = candidates.filter((c) => hasSkillLOS(enemy, c.entity, obstacles));
    const pool = visible.length > 0 ? visible : candidates;
    let best = pool[0];
    let bestD = dist(enemy, best.entity);
    for (let i = 1; i < pool.length; i += 1) {
      const d = dist(enemy, pool[i].entity);
      if (d < bestD) {
        best = pool[i];
        bestD = d;
      }
    }
    return best;
  }

  function pickSkillTarget(state, enemy, obstacles) {
    return pickNearestVisibleTarget(state, enemy, obstacles);
  }

  function lockSkillTarget(state, enemy, opts = {}) {
    const obstacles = getObstacles(opts.H);
    const pick = pickSkillTarget(state, enemy, obstacles);
    const jitter = opts.jitter ?? 10;
    const floor = state.floor || 1;
    const pred = predictPosition(pick.entity, opts.leadSec ?? 0.15, floor);
    const locked = clampCoord(
      pred.x + (Math.random() - 0.5) * jitter,
      pred.y + (Math.random() - 0.5) * jitter,
    );
    locked.entityId = pick.id;
    return locked;
  }

  function lockFeetTarget(state, enemy, opts = {}) {
    const obstacles = getObstacles(opts.H);
    const pick = pickNearestVisibleTarget(state, enemy, obstacles);
    const floor = state.floor || 1;
    const jitter = opts.jitter ?? 10;
    const pred = predictPosition(pick.entity, opts.leadSec ?? 0.1, floor);
    return {
      ...clampCoord(
        pred.x + (Math.random() - 0.5) * jitter,
        pred.y + (Math.random() - 0.5) * jitter,
      ),
      entityId: pick.id,
    };
  }

  function canUseDamageSkill(enemy) {
    const last = enemy._lastDamageSkillAt || 0;
    const gap = enemy.kind === "elite" ? ELITE_DAMAGE_SKILL_MS : MIN_DAMAGE_SKILL_MS;
    return performance.now() - last >= gap;
  }

  function hookWindupActive(enemy) {
    return (enemy._hookWindup || 0) > 0;
  }

  function tickEliteCooldowns(enemy, dt) {
    if (enemy.kind !== "elite" || enemy._dashActive) return;
    if (enemy.skillCd != null) enemy.skillCd -= dt;
    if (enemy.skill2Cd != null) enemy.skill2Cd -= dt;
  }

  function markDamageSkill(enemy) {
    enemy._lastDamageSkillAt = performance.now();
  }

  function attackCdMul(enemy) {
    let m = enemy.shootCdMul || 1;
    if ((enemy.kind === "elite" || enemy.kind === "boss") && enemy.hp / enemy.maxHp <= 0.25) m *= 0.8;
    return m;
  }

  function chaseSpeedMul(enemy) {
    if ((enemy.kind === "elite" || enemy.kind === "boss") && enemy.hp / enemy.maxHp <= 0.25) return 1.08;
    return 1;
  }

  function maxSummonsFor(elite) {
    return elite.phase >= 2 ? MAX_SUMMONS_ENRAGED : MAX_SUMMONS;
  }

  function checkPhaseTransition(enemy) {
    if (enemy._phaseChecked || enemy.kind === "shade" || enemy.kind === "mob") return;
    const threshold = enemy.kind === "boss" ? 0.55 : 0.6;
    if (enemy.hp / enemy.maxHp > threshold) return;
    enemy.phase = 2;
    enemy._phaseChecked = true;
    if (enemy.kind === "boss") {
      window.GameFx.floatText(enemy.x, enemy.y - 52, "狱怒", enemy.aoeColor || "#ff8866");
      window.__npcPushEvent?.("boss_enrage", { name: enemy.bossName });
    } else if (enemy.kind === "elite") {
      window.GameFx.floatText(enemy.x, enemy.y - 36, "厉怒", enemy.eliteColor || "#aaccff");
      window.__npcPushEvent?.("elite_enrage", { id: enemy.eliteId, name: enemy.eliteName });
    }
  }

  function primaryTarget(state, enemy) {
    if (enemy.isGuard && state.ally.hp > 0 && Math.random() < 0.6) return state.ally;
    const dp = dist(enemy, state.player);
    const da = state.ally.hp > 0 ? dist(enemy, state.ally) : Infinity;
    return dp <= da ? state.player : state.ally;
  }

  function shootRangeFor(enemy) {
    const kind = enemy.kind === "shade" ? "shade" : enemy.kind;
    return window.GameShooting?.rangeFor(kind) ?? 380;
  }

  function canBulletHit(enemy, target, H) {
    return window.GameShooting?.canShoot(
      enemy, target, shootRangeFor(enemy), H.allObstacles(),
    ) ?? true;
  }

  function markShootRetry(enemy) {
    enemy.shootCd = window.GameShooting?.RETRY_CD ?? 0.25;
  }

  function initEnemy(enemy, floor, roomIndex) {
    enemy.navPath = null;
    enemy.navReplanCd = 0;
    enemy.navGoal = null;
    enemy._skillSlowMul = 1;
    enemy._ringBurstDone = false;
    enemy._summonCount = 0;
    enemy.phase = 1;

    if (enemy.kind === "elite") {
      const def = window.GameEliteRegistry.pick(floor, roomIndex);
      enemy.eliteId = def.id;
      enemy.eliteName = def.name;
      enemy.eliteColor = def.color;
      enemy.skillCd = 0.8 + Math.random() * 0.6;
      enemy.skill2Cd = def.skill2.cd || 6;
    }
    if (enemy.kind === "boss") {
      enemy.mechanicCd = 2.0 + Math.random() * 0.8;
      enemy.ultCd = enemy.ultCd ?? 2.5;
      enemy._mechanicStreak = 0;
      enemy._entranceT = 2;
      enemy._invuln = true;
    }
    if (enemy.kind === "shade") {
      enemy.ttl = enemy.ttl ?? 28;
    }
  }

  function moveEnemy(enemy, target, dt, H) {
    const speedMul = (enemy._skillSlowMul ?? 1) * chaseSpeedMul(enemy);
    const spd = enemy.speed * speedMul * dt;

    if (enemy._dashActive) {
      const d = enemy._dashActive;
      d.t -= dt;
      const step = d.speed * dt;
      const [nx, ny] = norm(d.vx, d.vy);
      H.moveWithCollision(enemy, nx * step, ny * step);
      if (d.t <= 0 || step <= 0.1) {
        if (enemy._hookDash) {
          landingRingBurst(enemy._dashState, enemy);
          enemy._hookDash = false;
        }
        enemy._dashActive = null;
      }
      return;
    }

    enemy.navReplanCd -= dt;
    const goalMoved = enemy.navGoal
      ? dist(enemy.navGoal, target)
      : Infinity;
    if (enemy.navReplanCd <= 0 || goalMoved > 42) {
      enemy.navReplanCd = 0.38;
      enemy.navGoal = { x: target.x, y: target.y };
      const grid = H.getNavGrid();
      if (grid) {
        const raw = findPath(grid, enemy, target);
        enemy.navPath = raw ? smoothPath(raw, H.allObstacles(), enemy.radius) : null;
      }
    }

    if (enemy.kind === "boss" && dist(enemy, target) < BOSS_STOP_DIST) return;

    let mx = 0;
    let my = 0;
    if (enemy.navPath) {
      [mx, my] = steerAlong(enemy.navPath, enemy, H.allObstacles(), enemy.radius);
    } else {
      [mx, my] = norm(target.x - enemy.x, target.y - enemy.y);
    }
    H.moveWithCollision(enemy, mx * spd, my * spd);
  }

  function startDash(enemy, target, speed, duration, state) {
    const [vx, vy] = norm(target.x - enemy.x, target.y - enemy.y);
    enemy._dashActive = { vx, vy, speed, t: duration };
    enemy._skillSlowMul = 0.85;
    enemy._dashState = state;
  }

  function ringBurst(state, enemy, bulletSpeed, bulletDamage) {
    const n = 6;
    for (let i = 0; i < n; i += 1) {
      const ang = (2 * Math.PI * i) / n;
      state.enemyBullets.push({
        owner: "enemy",
        x: enemy.x,
        y: enemy.y,
        vx: Math.cos(ang) * bulletSpeed,
        vy: Math.sin(ang) * bulletSpeed,
        radius: 3,
        damage: bulletDamage,
        ttl: 2,
        color: enemy.eliteColor || "#88aaff",
      });
    }
    window.GameFx.burst(enemy.x, enemy.y, enemy.eliteColor || "#88aaff", 8);
  }

  function landingRingBurst(state, enemy) {
    if (!state) return;
    const n = 4;
    const bulletSpeed = 140;
    const bulletDamage = skillDamage(enemy, 4.5);
    for (let i = 0; i < n; i += 1) {
      const ang = (2 * Math.PI * i) / n;
      state.enemyBullets.push({
        owner: "enemy",
        x: enemy.x,
        y: enemy.y,
        vx: Math.cos(ang) * bulletSpeed,
        vy: Math.sin(ang) * bulletSpeed,
        radius: 2.5,
        damage: bulletDamage,
        ttl: 1.4,
        color: enemy.eliteColor || "#88aaff",
      });
    }
    window.GameFx.burst(enemy.x, enemy.y, enemy.eliteColor || "#88aaff", 5);
  }

  function doubleShot(bullets, enemy, target, speed, damage) {
    const [nx, ny] = norm(target.x - enemy.x, target.y - enemy.y);
    const px = -ny * 0.22;
    const py = nx * 0.22;
    bullets.push({
      owner: "enemy", x: enemy.x, y: enemy.y,
      vx: (nx + px) * speed, vy: (ny + py) * speed,
      radius: 4, damage, ttl: 2.2,
    });
    bullets.push({
      owner: "enemy", x: enemy.x, y: enemy.y,
      vx: (nx - px) * speed, vy: (ny - py) * speed,
      radius: 4, damage, ttl: 2.2,
    });
  }

  function spawnHazard(state, opts) {
    window.GameHazards?.push(state, opts);
  }

  function spawnLineHazards(state, from, to, opts) {
    const hz = window.GameHazards;
    const steps = opts.steps ?? 6;
    const warn = opts.warn ?? hz.WARN_DEFAULT;
    const active = opts.active ?? hz.activeForPhase(opts.phase, hz.ACTIVE_DEFAULT);
    const [nx, ny] = norm(to.x - from.x, to.y - from.y);
    const len = dist(from, to);
    for (let i = 1; i <= steps; i += 1) {
      const t = i / steps;
      hz.push(state, {
        x: from.x + nx * len * t,
        y: from.y + ny * len * t,
        r: opts.r ?? 28,
        warn,
        active,
        damage: opts.damage,
        color: opts.color,
        owner: opts.owner,
        _lockMarker: true,
      });
    }
  }

  function spawnThornWall(state, boss, lock) {
    const ttl = boss.phase >= 2 ? 10 : 8;
    const preferX = lock?.x ?? state.player.x;
    const preferY = lock?.y ?? state.player.y;
    const anchor = window.GameSpawn?.findClearSpawnPos(preferX, preferY, 9, {})
      ?? { x: preferX, y: preferY };
    const segments = [
      { dx: -42, dy: -8, w: 18, h: 88 },
      { dx: 24, dy: -8, w: 18, h: 88 },
      { dx: -44, dy: 42, w: 88, h: 18 },
    ];
    if (boss.phase >= 2) {
      segments.push({ dx: -8, dy: -58, w: 18, h: 72 });
    }
    const walls = segments.map((seg) => ({
      x: Math.max(60, Math.min(anchor.x + seg.dx, 900 - seg.w)),
      y: Math.max(60, Math.min(anchor.y + seg.dy, 480 - seg.h)),
      w: seg.w,
      h: seg.h,
      kind: "rubble",
      ttl,
    }));
    state.tempObstacles = state.tempObstacles || [];
    walls.forEach((w) => {
      state.tempObstacles.push({ ...w, _ttl: w.ttl });
    });
    state.tempObstacles._dirty = true;
    window.GameFx.floatText(anchor.x, anchor.y - 28, "荆棘墙", boss.aoeColor || "#ccaa66");
  }

  function mirrorShot(state, enemy, target) {
    const baseDir = state.lastPlayerShotDir || norm(target.x - enemy.x, target.y - enemy.y);
    const spd = 220;
    const dmg = skillDamage(enemy, 12);
    const angles = enemy.phase >= 2 ? [-12, 12] : [0];
    angles.forEach((deg) => {
      const rad = (deg * Math.PI) / 180;
      const cos = Math.cos(rad);
      const sin = Math.sin(rad);
      const vx = baseDir[0] * cos - baseDir[1] * sin;
      const vy = baseDir[0] * sin + baseDir[1] * cos;
      state.enemyBullets.push({
        owner: "enemy", x: enemy.x, y: enemy.y,
        vx: vx * spd, vy: vy * spd,
        radius: 5, damage: dmg, ttl: 2.4, color: "#aa88ff",
      });
    });
  }

  function summonShade(state, elite, scale) {
    if ((elite._summonCount || 0) >= maxSummonsFor(elite)) return;
    const preferX = elite.x + (Math.random() - 0.5) * 80;
    const preferY = elite.y + (Math.random() - 0.5) * 60;
    const pos = window.GameSpawn?.findClearSpawnPos(
      preferX, preferY, 9, { others: state.enemies },
    ) ?? { x: preferX, y: preferY };
    state.enemies.push({
      kind: "shade",
      x: pos.x,
      y: pos.y,
      hp: Math.round(28 * scale.hpMul * (scale.mobHpMul || 1)),
      maxHp: Math.round(28 * scale.hpMul * (scale.mobHpMul || 1)),
      radius: 9,
      speed: 48 * scale.speedMul,
      shootCd: 1.2,
      dmgMul: scale.dmgMul * (scale.mobDmgMul || 1) * 0.7,
      shootCdMul: scale.shootCdMul,
      ttl: 28,
      parentElite: elite,
    });
    elite._summonCount = (elite._summonCount || 0) + 1;
    window.GameFx.burst(pos.x, pos.y, "#aa8866", 6);
  }

  function pauseGuardFire(state) {
    state.enemies.forEach((g) => {
      if (g.isGuard) g.shootCd = Math.max(g.shootCd || 0, 0.8);
    });
  }

  function tryEliteSkill(enemy, dt, state, target, H) {
    if (enemy.kind !== "elite" || enemy.warnT > 0 || enemy._dashActive) return false;
    const def = window.GameEliteRegistry.get(enemy.eliteId);
    const scale = state._combatScale || { hpMul: 1, dmgMul: 1, speedMul: 1, shootCdMul: 1 };
    const bulletSpeed = 185;
    const bulletDamage = skillDamage(enemy, 8);

    if (def.id === "chain") {
      if (!enemy._ringBurstDone && enemy.hp / enemy.maxHp <= def.skill2.hpPct) {
        if (!canUseDamageSkill(enemy)) return false;
        enemy._ringBurstDone = true;
        window.GameFx.floatText(enemy.x, enemy.y - 32, def.skill2.name, def.color);
        window.__npcPushEvent?.("elite_skill", { id: def.id, name: def.skill2.name });
        ringBurst(state, enemy, bulletSpeed, bulletDamage);
        markDamageSkill(enemy);
        enemy.shootCd = 1;
        return true;
      }
      if (enemy.skillCd <= 0 && !hookWindupActive(enemy)) {
        if (!canUseDamageSkill(enemy)) {
          enemy.skillCd = 0.12;
          return false;
        }
        enemy.skillCd = eliteSkillCd(enemy, def.skill1.cd);
        enemy._hookLock = lockSkillTarget(state, enemy, { leadSec: 0.2, jitter: 8, H });
        enemy._hookWindup = 0.65;
        window.GameFx.floatText(enemy.x, enemy.y - 32, def.skill1.name, def.color);
        window.__npcPushEvent?.("elite_skill", {
          id: def.id, name: def.skill1.name, target: enemy._hookLock.entityId,
        });
        return true;
      }
    }

    if (def.id === "brand") {
      if (enemy.skill2Cd <= 0) {
        const before = enemy._summonCount || 0;
        summonShade(state, enemy, scale);
        if ((enemy._summonCount || 0) > before) {
          enemy.skill2Cd = eliteSkillCd(enemy, def.skill2.cd);
          window.GameFx.floatText(enemy.x, enemy.y - 32, def.skill2.name, def.color);
          window.__npcPushEvent?.("elite_skill", { id: def.id, name: def.skill2.name });
        } else {
          enemy.skill2Cd = 1.2;
        }
      }
      if (enemy.skillCd <= 0) {
        if (!canUseDamageSkill(enemy)) {
          enemy.skillCd = 0.12;
          return false;
        }
        enemy.skillCd = eliteSkillCd(enemy, def.skill1.cd);
        const lock = lockSkillTarget(state, enemy, { leadSec: 0.15, jitter: 8, H });
        window.GameFx.floatText(enemy.x, enemy.y - 32, def.skill1.name, def.color);
        window.__npcPushEvent?.("elite_skill", {
          id: def.id, name: def.skill1.name, target: lock.entityId,
        });
        const hz = window.GameHazards;
        const warn = hz.warnForFloor(state.floor);
        const active = hz.activeForPhase(enemy.phase, hz.ACTIVE_DEFAULT);
        spawnHazard(state, {
          x: lock.x, y: lock.y, r: 42,
          warn, active, damage: skillDamage(enemy, 17),
          color: def.color, owner: "elite",
          _lockMarker: true, _targetId: lock.entityId,
        });
        markDamageSkill(enemy);
        if (enemy.phase >= 2 && Math.random() < 0.4 && state.ally.hp > 0) {
          const other = lock.entityId === "player" ? state.ally : state.player;
          const oId = other === state.player ? "player" : "ally";
          if (hasSkillLOS(enemy, other, getObstacles(H))) {
            const lock2 = predictPosition(other, 0.15, state.floor);
            spawnHazard(state, {
              x: lock2.x, y: lock2.y, r: 38,
              warn: warn * 0.85, active, damage: skillDamage(enemy, 14),
              color: def.color, owner: "elite",
              _lockMarker: true, _targetId: oId,
            });
          }
        }
        window.GameBullets.startWarn(enemy, warn);
        enemy.shootCd = 0.6;
        return true;
      }
    }
    return false;
  }

  function tryBossMechanic(enemy, dt, state, target, H) {
    if (enemy.kind !== "boss" || enemy.warnT > 0 || enemy._ultWindup > 0) return false;
    enemy.mechanicCd -= dt;
    if (enemy.mechanicCd > 0) return false;

    const skill = enemy.skillId || "silence_zone";
    const damages = skill === "cut_line" || skill === "mirror_shot";
    if (damages && !canUseDamageSkill(enemy)) return false;

    enemy.mechanicCd = bossMechanicCdReset(enemy);
    const name = enemy.skillName || "狱技";
    const color = enemy.aoeColor || "#ff8866";
    const lock = lockSkillTarget(state, enemy, { leadSec: 0.15, jitter: 8, H });
    window.GameFx.floatText(enemy.x, enemy.y - 44, name, color);
    window.__npcPushEvent?.("boss_skill", { skill, name, target: lock.entityId });
    pauseGuardFire(state);

    enemy._mechanicStreak = (enemy._mechanicStreak || 0) + 1;
    if (enemy._mechanicStreak >= 2) {
      enemy._forceFan = true;
      enemy._mechanicStreak = 0;
    }

    const p2 = enemy.phase >= 2;
    switch (skill) {
      case "silence_zone": {
        const hz = window.GameHazards;
        const warn = hz.warnForFloor(state.floor);
        spawnHazard(state, {
          x: lock.x, y: lock.y, r: p2 ? 80 : 70,
          warn,
          active: hz.activeForPhase(enemy.phase, hz.ACTIVE_SILENCE),
          damage: 0,
          color: "#aa88ff",
          owner: "boss",
          _silence: true,
          _lockMarker: true,
          _targetId: lock.entityId,
        });
        window.GameBullets.startWarn(enemy, warn);
        break;
      }
      case "cut_line": {
        const hz = window.GameHazards;
        const warn = hz.warnForFloor(state.floor);
        window.GameBullets.startWarn(enemy, warn);
        spawnLineHazards(state, enemy, lock, {
          steps: p2 ? 9 : 7,
          r: 26,
          warn,
          active: hz.activeForPhase(enemy.phase, hz.ACTIVE_DEFAULT),
          phase: enemy.phase,
          damage: skillDamage(enemy, 16),
          color,
          owner: "boss",
        });
        enemy._skillSlowMul = 0.4;
        enemy.shootCd = Math.max(enemy.shootCd || 0, warn + 0.2);
        enemy._cutFollowUpT = warn + 0.15;
        enemy._cutFollowUpTarget = { x: lock.x, y: lock.y };
        markDamageSkill(enemy);
        break;
      }
      case "thorn_wall":
        spawnThornWall(state, enemy, lock);
        break;
      case "mirror_shot":
        if (canBulletHit(enemy, target, H)) {
          mirrorShot(state, enemy, target);
          markDamageSkill(enemy);
        }
        break;
      case "steam_fog":
        state.screenFogT = Math.max(state.screenFogT || 0, p2 ? 4.5 : 3);
        break;
      case "magnetic_pull":
        state.player.pullT = p2 ? 2.2 : 1.5;
        state.player.pullFrom = { x: enemy.x, y: enemy.y };
        state.player.pullStrength = p2 ? 114 : 95;
        break;
      default:
        break;
    }
    return true;
  }

  function resolvePending(enemy, state) {
    if (enemy._pendingCut && !enemy.warnT) {
      enemy._pendingCut = null;
      enemy._skillSlowMul = 1;
    }
  }

  function eliteAttack(enemy, target, state) {
    const cdMul = attackCdMul(enemy);
    const bulletSpeed = 185;
    const bulletDamage = Math.round(8 * (enemy.dmgMul || 1));
    const def = window.GameEliteRegistry.get(enemy.eliteId);

    if (def.id === "brand") {
      doubleShot(state.enemyBullets, enemy, target, bulletSpeed, bulletDamage);
      enemy.shootCd = 1.15 * cdMul;
    } else {
      window.GameBullets.aimedShot(state.enemyBullets, enemy, target, bulletSpeed, bulletDamage);
      enemy.shootCd = 1.15 * cdMul;
    }
  }

  function mobAttack(enemy, target, state) {
    const dmgMul = enemy.dmgMul || 1;
    const cdMul = enemy.shootCdMul || 1;
    window.GameBullets.aimedShot(state.enemyBullets, enemy, target, 170, Math.round(6 * dmgMul));
    enemy.shootCd = 1.6 * cdMul;
  }

  function bossAttack(enemy, target, state) {
    const cdMul = attackCdMul(enemy);
    const bulletSpeed = 200;
    const bulletDamage = Math.round(11 * (enemy.dmgMul || 1));
    const fanChance = enemy.fanChance ?? 0.42;

    if (enemy._forceFan || Math.random() < fanChance) {
      enemy._forceFan = false;
      window.GameBullets.startWarn(enemy, 0.45);
      enemy._pendingFan = { primary: target, bulletSpeed, bulletDamage };
      enemy.shootCd = 0.5;
    } else {
      window.GameBullets.aimedShot(state.enemyBullets, enemy, target, bulletSpeed, bulletDamage);
      enemy.shootCd = 1.2 * cdMul;
    }
  }

  function updateEliteRoomAnchors(state, dt, H) {
    const room = state.dungeon?.rooms?.[state.roomIndex];
    if (!room || room.type !== "elite" || state.floorState !== "playing" || state.result) return;
    state._brandAnchorCd = (state._brandAnchorCd ?? 12) - dt;
    if (state._brandAnchorCd > 0) return;
    state._brandAnchorCd = 12;
    const scale = state._combatScale || { dmgMul: 1 };
    const hz = window.GameHazards;
    const warn = hz.warnForFloor(state.floor);
    const active = hz.ACTIVE_DEFAULT;
    const elite = (state.enemies || []).find((e) => e.kind === "elite" && e.hp > 0);
    if (Math.random() < 0.5) {
      const anchor = ELITE_ANCHORS[Math.floor(Math.random() * ELITE_ANCHORS.length)];
      spawnHazard(state, {
        x: anchor.x,
        y: anchor.y,
        r: 38,
        warn,
        active,
        damage: skillDamage({ dmgMul: scale.dmgMul }, 14),
        color: "#ffaa66",
        owner: "elite_anchor",
      });
    } else if (elite) {
      const lock = lockFeetTarget(state, elite, { leadSec: 0.12, jitter: 8, H });
      spawnHazard(state, {
        x: lock.x,
        y: lock.y,
        r: 40,
        warn,
        active,
        damage: skillDamage({ dmgMul: scale.dmgMul }, 14),
        color: "#ffaa66",
        owner: "elite_anchor",
        _lockMarker: true,
        _targetId: lock.entityId,
      });
    }
  }

  function tickCombatMotion(state, dt) {
    tickEntityMotion(state.player, dt);
    if (state.ally.hp > 0) tickEntityMotion(state.ally, dt);
  }

  function updateEnemyCombat(enemy, dt, state, H) {
    if (enemy.kind === "shade") {
      enemy.ttl -= dt;
      if (enemy.ttl <= 0) enemy.hp = 0;
    }

    checkPhaseTransition(enemy);

    if (enemy.kind === "boss" && enemy._entranceT > 0) {
      enemy._entranceT -= dt;
      if (enemy._entranceT <= 0) enemy._invuln = false;
      const target = primaryTarget(state, enemy);
      moveEnemy(enemy, target, dt, H);
      return;
    }

    enemy.shootCd = Math.max(0, (enemy.shootCd || 0) - dt);
    if (enemy._ultWindup > 0) enemy._ultWindup = Math.max(0, enemy._ultWindup - dt);

    if (enemy._cutFollowUpT > 0) {
      enemy._cutFollowUpT -= dt;
      if (enemy._cutFollowUpT <= 0 && enemy._cutFollowUpTarget) {
        const t = enemy._cutFollowUpTarget;
        const aim = { x: t.x, y: t.y };
        if (canBulletHit(enemy, aim, H)) {
          window.GameBullets.aimedShot(
            state.enemyBullets, enemy, aim,
            200, Math.round(10 * (enemy.dmgMul || 1)),
          );
        }
        enemy._cutFollowUpTarget = null;
      }
    }

    if (hookWindupActive(enemy)) {
      enemy._hookWindup -= dt;
      enemy._skillSlowMul = 0.45;
      if (enemy._hookWindup <= 0) {
        const point = enemy._hookLock || primaryTarget(state, enemy);
        enemy._hookDash = true;
        markDamageSkill(enemy);
        startDash(enemy, point, 300, 0.38, state);
        enemy._hookLock = null;
        enemy._hookWindup = 0;
      }
    } else if (!enemy._dashActive) {
      if (enemy.warnT > 0) enemy._skillSlowMul = 0.5;
      else if (!enemy._pendingCut) enemy._skillSlowMul = 1;
    }

    const warning = enemy.warnT > 0;
    if (warning) window.GameBullets.tickWarn(enemy, dt);
    resolvePending(enemy, state);

    const target = primaryTarget(state, enemy);
    enemy.aggroTarget = target;

    tickEliteCooldowns(enemy, dt);

    if (!hookWindupActive(enemy) && !tryEliteSkill(enemy, dt, state, target, H)) {
      tryBossMechanic(enemy, dt, state, target, H);
    }

    if (enemy.kind === "boss") {
      enemy.ultCd = Math.max(0, (enemy.ultCd || 0) - dt);
      if (
        enemy.ultCd <= 0
        && !enemy.warnT
        && !enemy._pendingFan
        && !enemy._ultWindup
        && !enemy._pendingCut
        && canUseDamageSkill(enemy)
      ) {
        const extra = enemy.phase >= 2 ? 2 : 0;
        window.GameBossPatterns.spawnGroundAoE(state, enemy, target, {
          count: 5 + Math.min(2, state.floor - 1) + extra,
          damage: skillDamage(enemy, 15),
          color: enemy.aoeColor || "#ff6644",
        });
        markDamageSkill(enemy);
        enemy.ultCd = enemy.ultInterval || 6;
      }
    }

    moveEnemy(enemy, target, dt, H);

    if (!warning && enemy.shootCd <= 0 && !enemy._dashActive && !enemy._pendingCut && !hookWindupActive(enemy)) {
      if (canBulletHit(enemy, target, H)) {
        if (enemy.kind === "boss") bossAttack(enemy, target, state);
        else if (enemy.kind === "elite") eliteAttack(enemy, target, state);
        else mobAttack(enemy, target, state);
      } else {
        markShootRetry(enemy);
      }
    } else if (enemy._pendingFan && !enemy.warnT) {
      const p = enemy._pendingFan;
      if (canBulletHit(enemy, p.primary, H)) {
        window.GameBullets.fanShot(state.enemyBullets, enemy, p.primary, p.bulletSpeed, p.bulletDamage, 42, 5);
        enemy.shootCd = 1.15 * attackCdMul(enemy);
      } else {
        markShootRetry(enemy);
      }
      enemy._pendingFan = null;
    }
  }

  function updateStatusEffects(state, dt) {
    const p = state.player;
    if (p.silenceT > 0) p.silenceT -= dt;
    if (p.pullT > 0 && p.pullFrom) {
      p.pullT -= dt;
      const [nx, ny] = norm(p.pullFrom.x - p.x, p.pullFrom.y - p.y);
      const pull = (p.pullStrength || 80) * dt;
      p.x += nx * pull;
      p.y += ny * pull;
    }
    if (state.screenFogT > 0) state.screenFogT -= dt;

    const temps = state.tempObstacles || [];
    for (let i = temps.length - 1; i >= 0; i -= 1) {
      temps[i]._ttl -= dt;
      if (temps[i]._ttl <= 0) temps.splice(i, 1);
    }
    if (temps._dirty && state._rebuildNav) {
      temps._dirty = false;
      state._rebuildNav();
    }
  }

  function updateHazardDebuffs(state, dt) {
    (state.hazards || []).forEach((h) => {
      if (h._silence && window.GameHazards?.isActive(h)) {
        const hit = dist(state.player, h) < h.r + state.player.radius;
        if (hit) state.player.silenceT = Math.max(state.player.silenceT || 0, 0.18);
      }
    });
  }

  function drawFx(ctx, state) {
    (state.enemies || []).forEach((e) => {
      if (e.eliteName && e.kind === "elite") {
        ctx.font = "10px PingFang SC, Arial, sans-serif";
        ctx.fillStyle = e.eliteColor || "#aaccff";
        ctx.textAlign = "center";
        ctx.fillText(e.eliteName, e.x, e.y - e.radius - 22);
        ctx.textAlign = "left";
      }
      if (e._hookLock && hookWindupActive(e)) {
        ctx.strokeStyle = "rgba(136, 170, 255, 0.55)";
        ctx.setLineDash([5, 5]);
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(e.x, e.y);
        ctx.lineTo(e._hookLock.x, e._hookLock.y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.strokeStyle = "rgba(255, 220, 180, 0.7)";
        ctx.lineWidth = 1.5;
        ctx.arc(e._hookLock.x, e._hookLock.y, 8, 0, Math.PI * 2);
        ctx.stroke();
        ctx.lineWidth = 1;
      }
      if (e._dashActive) {
        ctx.strokeStyle = "rgba(120, 160, 255, 0.7)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(e.x, e.y);
        ctx.lineTo(e.x - e._dashActive.vx * 20, e.y - e._dashActive.vy * 20);
        ctx.stroke();
        ctx.lineWidth = 1;
      }
      if (e._invuln) {
        ctx.strokeStyle = `rgba(255, 200, 120, ${0.4 + Math.sin(performance.now() / 80) * 0.2})`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(e.x, e.y, e.radius + 8, 0, Math.PI * 2);
        ctx.stroke();
        ctx.lineWidth = 1;
      }
    });

    if (state.player.silenceT > 0) {
      ctx.strokeStyle = `rgba(170, 130, 255, ${0.35 + Math.sin(performance.now() / 100) * 0.2})`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(state.player.x, state.player.y, state.player.radius + 12, 0, Math.PI * 2);
      ctx.stroke();
      ctx.lineWidth = 1;
    }

    if (state.screenFogT > 0) {
      const fogPeak = state.screenFogT > 3.5 ? 4.5 : 3;
      const grd = ctx.createRadialGradient(
        state.player.x, state.player.y, 80,
        state.player.x, state.player.y, 420,
      );
      grd.addColorStop(0, "rgba(0,0,0,0)");
      grd.addColorStop(1, `rgba(8, 12, 20, ${Math.min(0.82, state.screenFogT / fogPeak * 0.75)})`);
      ctx.fillStyle = grd;
      ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    }
  }

  window.GameEnemyAI = {
    initEnemy,
    tickCombatMotion,
    updateEnemyCombat,
    updateEliteRoomAnchors,
    updateStatusEffects,
    updateHazardDebuffs,
    drawFx,
    primaryTarget,
  };
})();