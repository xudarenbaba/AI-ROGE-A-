/** Boss 大范围落点 AOE */
(function () {
  const H = () => window.GameHazards;

  function jitter(scale = 16) {
    return (Math.random() - 0.5) * scale;
  }

  function footAt(entity, leadSec, floor) {
    const lead = floor >= 3 ? leadSec : leadSec * 0.5;
    const vx = entity._vx || 0;
    const vy = entity._vy || 0;
    const speed = Math.hypot(vx, vy);
    const cap = speed > 120 ? 120 / speed : 1;
    return {
      x: Math.max(60, Math.min(entity.x + vx * lead * cap + jitter(10), 900)),
      y: Math.max(60, Math.min(entity.y + vy * lead * cap + jitter(10), 480)),
    };
  }

  function pickAoEPoints(state, count, boss) {
    const player = state.player;
    const ally = state.ally;
    const floor = state.floor || 1;
    const p2 = (boss?.phase || 1) >= 2;
    const lead = 0.25;
    const pts = [];

    if (count >= 1) {
      const p = footAt(player, lead, floor);
      pts.push({ x: p.x, y: p.y, entityId: "player", _lockMarker: true });
    }
    if (count >= 2 && ally.hp > 0) {
      const a = footAt(ally, lead, floor);
      pts.push({ x: a.x, y: a.y, entityId: "ally", _lockMarker: true });
    }

    for (let i = pts.length; i < count; i += 1) {
      const roll = Math.random();
      const midW = p2 ? 0.35 : 0.2;
      if (ally.hp > 0 && roll < 0.5) {
        const p = footAt(player, lead, floor);
        pts.push({ x: p.x, y: p.y, entityId: "player", _lockMarker: true });
      } else if (ally.hp > 0 && roll < 0.8) {
        const a = footAt(ally, lead, floor);
        pts.push({ x: a.x, y: a.y, entityId: "ally", _lockMarker: true });
      } else if (ally.hp > 0 && roll < 0.8 + midW) {
        pts.push({
          x: Math.max(60, Math.min((player.x + ally.x) / 2 + jitter(12), 900)),
          y: Math.max(60, Math.min((player.y + ally.y) / 2 + jitter(12), 480)),
          entityId: "midpoint",
          _lockMarker: true,
        });
      } else {
        const p = footAt(player, lead, floor);
        pts.push({ x: p.x, y: p.y, entityId: "player", _lockMarker: true });
      }
    }
    return pts;
  }

  function spawnGroundAoE(state, boss, target, opts) {
    const hz = H();
    const count = opts?.count ?? 4;
    const radius = opts?.radius ?? 52;
    const warn = opts?.warn ?? hz.WARN_BOSS_ULT;
    const active = hz.activeForPhase(boss?.phase, hz.ACTIVE_DEFAULT);
    const dmg = opts?.damage ?? 14;
    const color = opts?.color ?? "#ff6644";
    const pts = pickAoEPoints(state, count, boss);
    pts.forEach((pt) => {
      hz.push(state, {
        x: pt.x,
        y: pt.y,
        r: radius,
        warn,
        active,
        damage: dmg,
        color,
        owner: boss.kind,
        _lockMarker: pt._lockMarker,
        _targetId: pt.entityId,
      });
    });
    window.GameBullets?.startWarn(boss, warn);
    boss._ultWindup = warn;
    const primary = pts[0]?.entityId || "player";
    window.__npcPushEvent?.("boss_ult", { target: primary, count });
  }

  function spawnEntranceAoE(state, boss) {
    const hz = H();
    const pts = [
      { x: boss.x, y: boss.y },
      { x: state.player.x + 80, y: state.player.y },
      { x: (boss.x + state.player.x) / 2, y: (boss.y + state.player.y) / 2 },
    ];
    pts.forEach((pt) => {
      hz.push(state, {
        x: Math.max(80, Math.min(pt.x, 880)),
        y: Math.max(80, Math.min(pt.y, 460)),
        r: 58,
        warn: hz.WARN_ENTRANCE,
        active: 0.5,
        damage: 0,
        color: boss.aoeColor || "#ff6644",
        owner: "boss_entrance",
      });
    });
    window.GameFx?.floatText(boss.x, boss.y - 56, "狱威", boss.aoeColor || "#ff8866");
  }

  function updateHazards(state, dt) {
    H().update(state, dt);
  }

  function drawHazards(ctx, hazards) {
    H().draw(ctx, hazards);
  }

  window.GameBossPatterns = { spawnGroundAoE, spawnEntranceAoE, updateHazards, drawHazards, pickAoEPoints };
})();