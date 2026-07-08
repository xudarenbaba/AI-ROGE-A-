/** 子弹射击：射程 + 视线（障碍物遮挡则不开火） */
(function () {
  const RANGES = {
    mob: 380,
    shade: 380,
    elite: 410,
    boss: 440,
    ally_guard: 420,
    ally_assault: 110,
    player: 480,
  };

  const RETRY_CD = 0.25;

  function hasBulletLOS(ax, ay, bx, by, obstacles) {
    const steps = 16;
    for (let i = 1; i < steps; i += 1) {
      const t = i / steps;
      const px = ax + (bx - ax) * t;
      const py = ay + (by - ay) * t;
      for (const o of obstacles || []) {
        if (px >= o.x && px <= o.x + o.w && py >= o.y && py <= o.y + o.h) return false;
      }
    }
    return true;
  }

  function inRange(from, to, range) {
    if (!range || range <= 0) return true;
    return Math.hypot(to.x - from.x, to.y - from.y) <= range;
  }

  function canShoot(from, to, range, obstacles) {
    if (!from || !to) return false;
    if (!inRange(from, to, range)) return false;
    return hasBulletLOS(from.x, from.y, to.x, to.y, obstacles);
  }

  function rangeFor(kind) {
    return RANGES[kind] ?? RANGES.mob;
  }

  window.GameShooting = {
    RANGES,
    RETRY_CD,
    hasBulletLOS,
    inRange,
    canShoot,
    rangeFor,
  };
})();