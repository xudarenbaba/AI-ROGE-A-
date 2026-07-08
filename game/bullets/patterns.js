/**
 * 敌方弹幕模式（几何仍写入 state.enemyBullets，不改 RL obs 结构）
 */
(function () {
  function norm(dx, dy) {
    const len = Math.hypot(dx, dy);
    if (len < 0.0001) return [0, 0];
    return [dx / len, dy / len];
  }

  function pushBullet(bullets, owner, from, vx, vy, speed, damage, extra) {
    bullets.push({
      owner,
      x: from.x,
      y: from.y,
      vx: vx * speed,
      vy: vy * speed,
      radius: extra?.radius ?? 4,
      damage,
      ttl: extra?.ttl ?? 2.2,
      color: extra?.color,
    });
  }

  function aimedShot(bullets, enemy, target, speed, damage) {
    const [nx, ny] = norm(target.x - enemy.x, target.y - enemy.y);
    pushBullet(bullets, "enemy", enemy, nx, ny, speed, damage);
  }

  function fanShot(bullets, enemy, target, speed, damage, spreadDeg, count) {
    const base = Math.atan2(target.y - enemy.y, target.x - enemy.x);
    const half = ((spreadDeg / 2) * Math.PI) / 180;
    const step = count > 1 ? spreadDeg / (count - 1) : 0;
    for (let i = 0; i < count; i += 1) {
      const ang = base - half + ((step * i) * Math.PI) / 180;
      pushBullet(bullets, "enemy", enemy, Math.cos(ang), Math.sin(ang), speed, damage, {
        radius: 3,
        color: "#ff8866",
      });
    }
  }

  function tickWarn(enemy, dt) {
    if (!enemy.warnT) return false;
    enemy.warnT -= dt;
    return enemy.warnT > 0;
  }

  function startWarn(enemy, duration) {
    enemy.warnT = duration;
  }

  window.GameBullets = {
    aimedShot,
    fanShot,
    tickWarn,
    startWarn,
  };
})();