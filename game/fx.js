/** 命中粒子、飘字、轻震屏 — 纯渲染，不影响 RL */
(function () {
  const fx = {
    particles: [],
    floats: [],
    shakeT: 0,
    shakeMag: 0,
  };

  function burst(x, y, color, n = 6) {
    for (let i = 0; i < n; i += 1) {
      const a = Math.random() * Math.PI * 2;
      const sp = 40 + Math.random() * 90;
      fx.particles.push({
        x, y, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp,
        life: 0.25 + Math.random() * 0.2, ttl: 0.25 + Math.random() * 0.2,
        color, size: 2 + Math.random() * 2,
      });
    }
  }

  function floatText(x, y, text, color = "#ffe4a8") {
    fx.floats.push({ x, y, text, color, life: 0, ttl: 0.7, vy: -42 });
  }

  function shake(mag = 3, dur = 0.12) {
    fx.shakeMag = Math.max(fx.shakeMag, mag);
    fx.shakeT = Math.max(fx.shakeT, dur);
  }

  function update(dt) {
    fx.shakeT = Math.max(0, fx.shakeT - dt);
    fx.particles = fx.particles.filter((p) => {
      p.ttl -= dt;
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.vx *= 0.92;
      p.vy *= 0.92;
      return p.ttl > 0;
    });
    fx.floats = fx.floats.filter((f) => {
      f.life += dt;
      f.y += f.vy * dt;
      return f.life < f.ttl;
    });
  }

  function applyShake(ctx) {
    if (fx.shakeT <= 0) return;
    const m = fx.shakeMag * (fx.shakeT / 0.12);
    ctx.translate((Math.random() - 0.5) * m, (Math.random() - 0.5) * m);
  }

  function draw(ctx) {
    fx.particles.forEach((p) => {
      ctx.globalAlpha = Math.max(0, p.ttl / 0.35);
      ctx.fillStyle = p.color;
      ctx.fillRect(p.x - p.size / 2, p.y - p.size / 2, p.size, p.size);
    });
    ctx.globalAlpha = 1;
    fx.floats.forEach((f) => {
      const a = 1 - f.life / f.ttl;
      ctx.globalAlpha = a;
      ctx.fillStyle = f.color;
      ctx.font = "bold 13px Segoe UI";
      ctx.fillText(f.text, f.x, f.y);
    });
    ctx.globalAlpha = 1;
  }

  window.GameFx = { fx, burst, floatText, shake, update, applyShake, draw };
})();