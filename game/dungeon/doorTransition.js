/** 过门过场：光柱吸入 → 廊道闪现 → 落点 */
(function () {
  const PHASES = [
    { id: "pull", dur: 0.28 },
    { id: "corridor", dur: 0.52 },
    { id: "land", dur: 0.34 },
  ];

  let active = null;
  const particles = [];

  function start(opts) {
    active = {
      phaseIdx: 0,
      phaseT: PHASES[0].dur,
      door: opts.door,
      nextLabel: opts.nextLabel || "下一间",
      nextIndex: opts.nextIndex,
      pullFrom: { x: opts.playerX, y: opts.playerY },
      pullTo: {
        x: opts.door.x + opts.door.w / 2,
        y: opts.door.y + opts.door.h / 2,
      },
      landInvuln: 0.32,
    };
    particles.length = 0;
    for (let i = 0; i < 48; i += 1) {
      particles.push({
        x: opts.playerX + (Math.random() - 0.5) * 120,
        y: opts.playerY + (Math.random() - 0.5) * 80,
        vx: 0,
        vy: 0,
        life: 0.4 + Math.random() * 0.3,
        size: 1 + Math.random() * 2,
      });
    }
    window.GameFx?.burst(active.pullTo.x, active.pullTo.y, "#ffe8b0", 12);
  }

  function isActive() {
    return !!active;
  }

  function phaseId() {
    return active ? PHASES[active.phaseIdx].id : null;
  }

  function update(dt, state) {
    if (!active) return { done: false };
    const phase = PHASES[active.phaseIdx];
    active.phaseT -= dt;

    if (phase.id === "pull") {
      const total = PHASES[0].dur;
      const t = 1 - active.phaseT / total;
      const ease = t * t * (3 - 2 * t);
      const tx = active.pullTo.x;
      const ty = active.pullTo.y;
      state.player.x = active.pullFrom.x + (tx - active.pullFrom.x) * ease * 0.85;
      state.player.y = active.pullFrom.y + (ty - active.pullFrom.y) * ease * 0.85;
      particles.forEach((p) => {
        p.life -= dt;
        const dx = tx - p.x;
        const dy = ty - p.y;
        p.vx += dx * 6 * dt;
        p.vy += dy * 6 * dt;
        p.x += p.vx * dt;
        p.y += p.vy * dt;
      });
    }

    if (phase.id === "land" && active.phaseT <= 0) {
      state.player.dashInvuln = Math.max(state.player.dashInvuln, active.landInvuln);
      window.GameFx?.burst(state.player.x, state.player.y, "#c8e0ff", 10);
      const done = { done: true, nextIndex: active.nextIndex };
      active = null;
      particles.length = 0;
      return done;
    }

    if (active.phaseT <= 0) {
      active.phaseIdx += 1;
      if (active.phaseIdx >= PHASES.length) {
        const done = { done: true, nextIndex: active.nextIndex };
        active = null;
        particles.length = 0;
        return done;
      }
      active.phaseT = PHASES[active.phaseIdx].dur;
    }
    return { done: false, phase: phaseId() };
  }

  function draw(ctx, canvasW, canvasH) {
    if (!active) return;
    const phase = PHASES[active.phaseIdx].id;
    const door = active.door;
    const cx = door.x + door.w / 2;
    const cy = door.y + door.h / 2;

    if (phase === "pull") {
      const pulse = 0.55 + Math.sin(performance.now() / 70) * 0.25;
      const grd = ctx.createLinearGradient(cx, door.y, cx, door.y + door.h);
      grd.addColorStop(0, `rgba(255, 220, 140, ${pulse * 0.15})`);
      grd.addColorStop(0.5, `rgba(255, 200, 100, ${pulse * 0.55})`);
      grd.addColorStop(1, `rgba(255, 180, 80, ${pulse * 0.2})`);
      ctx.fillStyle = grd;
      ctx.fillRect(door.x - 8, door.y, door.w + 16, door.h);
      ctx.strokeStyle = `rgba(255, 240, 200, ${pulse})`;
      ctx.lineWidth = 3;
      ctx.strokeRect(door.x - 4, door.y + 4, door.w + 8, door.h - 8);
      ctx.lineWidth = 1;
      particles.forEach((p) => {
        if (p.life <= 0) return;
        ctx.globalAlpha = Math.min(1, p.life * 2);
        ctx.fillStyle = "#ffe8a8";
        ctx.fillRect(p.x, p.y, p.size, p.size);
      });
      ctx.globalAlpha = 1;
    }

    if (phase === "corridor" || phase === "land") {
      const fade = Math.min(1, active.phaseT / 0.2);
      ctx.fillStyle = `rgba(4, 6, 14, ${phase === "corridor" ? 0.82 : 0.55 * fade})`;
      ctx.fillRect(0, 0, canvasW, canvasH);
      const scroll = (performance.now() / 40) % 40;
      ctx.strokeStyle = "rgba(255, 200, 120, 0.12)";
      ctx.lineWidth = 2;
      for (let y = -40; y < canvasH + 40; y += 20) {
        ctx.beginPath();
        ctx.moveTo(0, y + scroll);
        ctx.lineTo(canvasW, y + scroll + 60);
        ctx.stroke();
      }
      ctx.lineWidth = 1;
      ctx.font = "bold 22px PingFang SC, Arial, sans-serif";
      ctx.fillStyle = "#ffe8c0";
      ctx.textAlign = "center";
      ctx.fillText(active.nextLabel, canvasW / 2, canvasH / 2 - 8);
      ctx.font = "13px PingFang SC, Arial, sans-serif";
      ctx.fillStyle = "#c8b8a0";
      ctx.fillText("魂灯引路中…", canvasW / 2, canvasH / 2 + 22);
      ctx.textAlign = "left";
      if (phase === "land") {
        ctx.globalAlpha = 1 - fade;
        ctx.fillStyle = "#ffe8b0";
        ctx.font = "bold 16px PingFang SC, Arial, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("落点", canvasW / 2, canvasH / 2 + 52);
        ctx.textAlign = "left";
        ctx.globalAlpha = 1;
      }
    }
  }

  window.GameDoorTransition = { start, update, draw, isActive, phaseId };
})();