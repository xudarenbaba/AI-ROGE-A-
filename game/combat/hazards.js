/**
 * 地面 hazard：预告(warnT) → 生效(activeT) → 消退(fadeT)
 */
(function () {
  const WARN_DEFAULT = 1.0;
  const WARN_BOSS_ULT = 1.1;
  const WARN_ENTRANCE = 2.0;
  const ACTIVE_DEFAULT = 0.6;
  const ACTIVE_SILENCE = 0.8;
  const ACTIVE_P2_BONUS = 0.15;
  const FADE_DEFAULT = 0.15;

  function warnForFloor(floor) {
    return (floor || 1) <= 2 ? 1.1 : WARN_DEFAULT;
  }

  function activeForPhase(phase, base) {
    return base + ((phase || 1) >= 2 ? ACTIVE_P2_BONUS : 0);
  }

  function create(opts) {
    const warn = opts.warn ?? opts.warnT ?? WARN_DEFAULT;
    const active = opts.active ?? opts.activeT ?? ACTIVE_DEFAULT;
    const fade = opts.fade ?? opts.fadeT ?? FADE_DEFAULT;
    return {
      x: opts.x,
      y: opts.y,
      r: opts.r ?? 40,
      warnT: warn,
      activeT: active,
      fadeT: fade,
      ttl: warn + active + fade,
      damage: opts.damage ?? 12,
      color: opts.color ?? "#ff8866",
      owner: opts.owner ?? "enemy",
      _lockMarker: opts._lockMarker ?? false,
      _targetId: opts._targetId,
      _silence: opts._silence ?? false,
    };
  }

  function push(state, opts) {
    if (!state.hazards) state.hazards = [];
    state.hazards.push(create(opts));
  }

  function isActive(h) {
    return h.warnT <= 0 && h.activeT > 0;
  }

  function applyDamage(state, h) {
    if (h.damage <= 0) return;
    if (!h._hitPlayer && state.player.dashInvuln <= 0) {
      const hitPlayer = Math.hypot(state.player.x - h.x, state.player.y - h.y) < h.r + state.player.radius;
      if (hitPlayer) {
        const raw = h.damage;
        let final = raw;
        if (state.player.shieldCd > 0) final = Math.max(2, Math.floor(raw * 0.3));
        state.player.hp -= final;
        window.GameFx?.floatText(state.player.x, state.player.y - 20, `-${final}`, "#ff8866");
        h._hitPlayer = true;
      }
    }
    if (state.ally.hp > 0 && !h._hitAlly) {
      const hitAlly = Math.hypot(state.ally.x - h.x, state.ally.y - h.y) < h.r + state.ally.radius;
      if (hitAlly) {
        state.ally.hp -= h.damage * 0.85;
        h._hitAlly = true;
      }
    }
  }

  function update(state, dt) {
    const hazards = state.hazards || [];
    let activated = false;
    for (let i = hazards.length - 1; i >= 0; i -= 1) {
      const h = hazards[i];
      h.ttl -= dt;

      if (h.warnT > 0) {
        h.warnT -= dt;
        if (h.warnT <= 0) activated = true;
        continue;
      }

      if (h.activeT > 0) {
        applyDamage(state, h);
        h.activeT -= dt;
        continue;
      }

      if (h.ttl <= 0) hazards.splice(i, 1);
    }
    state.hazards = hazards;
    if (activated) window.GameFx?.shake(2, 0.08);
  }

  function draw(ctx, hazards) {
    const pulse = 0.5 + Math.sin(performance.now() / 85) * 0.35;
    (hazards || []).forEach((h) => {
      const telegraph = h.warnT > 0;
      const active = isActive(h);
      const locked = h._lockMarker && telegraph;

      if (telegraph) {
        const shrink = h.warnT < 0.2 ? 0.92 + (0.2 - h.warnT) * 0.4 : 1;
        const r = h.r * shrink;
        ctx.beginPath();
        ctx.fillStyle = `rgba(255, 45, 55, ${0.14 + pulse * 0.1})`;
        ctx.arc(h.x, h.y, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.setLineDash([7, 5]);
        ctx.strokeStyle = `rgba(255, 55, 70, ${0.55 + pulse * 0.35})`;
        ctx.lineWidth = locked ? 3 : 2.5;
        ctx.stroke();
        ctx.setLineDash([]);
        if (locked) {
          ctx.beginPath();
          ctx.strokeStyle = "rgba(255, 200, 180, 0.75)";
          ctx.lineWidth = 1.5;
          ctx.arc(h.x, h.y, 7, 0, Math.PI * 2);
          ctx.stroke();
        }
      } else if (active) {
        const alpha = 0.62 + Math.sin(performance.now() / 120) * 0.08;
        ctx.beginPath();
        ctx.fillStyle = h._silence
          ? `rgba(170, 130, 255, ${alpha})`
          : `${h.color}${Math.floor(alpha * 255).toString(16).padStart(2, "0")}`;
        ctx.arc(h.x, h.y, h.r, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = h._silence ? "#cc99ff" : h.color;
        ctx.lineWidth = 3.5;
        ctx.stroke();
        if (h._silence) {
          ctx.strokeStyle = "rgba(200, 160, 255, 0.55)";
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(h.x, h.y, h.r + 4, 0, Math.PI * 2);
          ctx.stroke();
        }
      } else {
        const fadeA = Math.max(0, h.ttl / (h.fadeT || FADE_DEFAULT)) * 0.35;
        if (fadeA > 0.02) {
          ctx.beginPath();
          ctx.fillStyle = `${h.color}${Math.floor(fadeA * 255).toString(16).padStart(2, "0")}`;
          ctx.arc(h.x, h.y, h.r * 0.9, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      ctx.lineWidth = 1;
    });
  }

  window.GameHazards = {
    WARN_DEFAULT,
    WARN_BOSS_ULT,
    WARN_ENTRANCE,
    ACTIVE_DEFAULT,
    ACTIVE_SILENCE,
    FADE_DEFAULT,
    warnForFloor,
    activeForPhase,
    create,
    push,
    update,
    draw,
    isActive,
  };
})();