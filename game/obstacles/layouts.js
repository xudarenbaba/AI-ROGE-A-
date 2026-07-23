/**
 * 障碍布局。碰撞盒 x,y,w,h 与 rl/limbs/assault_skirmish/env.py OBSTACLE_LAYOUTS 对齐；kind 仅渲染用。
 */
(function () {
  const LAYOUTS = [
    {
      id: "hall_pillars",
      theme: "pillar",
      boxes: [
        { x: 360, y: 255, w: 180, h: 22, kind: "wall" },
        { x: 240, y: 170, w: 22, h: 130, kind: "pillar" },
        { x: 660, y: 220, w: 22, h: 130, kind: "pillar" },
        { x: 480, y: 140, w: 140, h: 20, kind: "wall" },
        { x: 420, y: 360, w: 140, h: 20, kind: "wall" },
        { x: 300, y: 360, w: 80, h: 20, kind: "rubble" },
      ],
    },
    {
      id: "corridor",
      theme: "corridor",
      boxes: [
        { x: 280, y: 145, w: 200, h: 20, kind: "wall" },
        { x: 560, y: 145, w: 160, h: 20, kind: "wall" },
        { x: 280, y: 375, w: 160, h: 20, kind: "wall" },
        { x: 520, y: 375, w: 200, h: 20, kind: "wall" },
        { x: 235, y: 220, w: 20, h: 110, kind: "pillar" },
        { x: 660, y: 210, w: 20, h: 110, kind: "pillar" },
        { x: 410, y: 245, w: 100, h: 20, kind: "rubble" },
      ],
    },
    {
      id: "scattered",
      theme: "scattered",
      boxes: [
        { x: 270, y: 160, w: 160, h: 20, kind: "wall" },
        { x: 580, y: 200, w: 20, h: 150, kind: "pillar" },
        { x: 340, y: 340, w: 160, h: 20, kind: "wall" },
        { x: 630, y: 330, w: 140, h: 20, kind: "rubble" },
        { x: 240, y: 280, w: 20, h: 100, kind: "pillar" },
        { x: 460, y: 150, w: 20, h: 120, kind: "pillar" },
      ],
    },
    {
      id: "cross",
      theme: "cross",
      boxes: [
        { x: 390, y: 230, w: 120, h: 20, kind: "wall" },
        { x: 445, y: 165, w: 20, h: 140, kind: "pillar" },
        { x: 240, y: 155, w: 130, h: 20, kind: "wall" },
        { x: 620, y: 155, w: 130, h: 20, kind: "wall" },
        { x: 240, y: 365, w: 130, h: 20, kind: "rubble" },
        { x: 620, y: 365, w: 130, h: 20, kind: "rubble" },
      ],
    },
  ];

  function getForFloor(floor) {
    const layout = LAYOUTS[(floor - 1) % LAYOUTS.length];
    return layout.boxes.map((b) => ({ ...b }));
  }

  function draw(ctx, obstacles, meta) {
    const kindStyle = {
      wall: { fill: `${meta.haze}ee`, stroke: `${meta.accent}cc`, cap: `${meta.accent}33` },
      pillar: { fill: `${meta.tone}dd`, stroke: `${meta.accent}ff`, cap: `${meta.accent}44` },
      rubble: { fill: `${meta.haze}aa`, stroke: `${meta.accent}77`, cap: `${meta.accent}22` },
      seal: { fill: `${meta.tone}cc`, stroke: `${meta.accent}55`, cap: `${meta.accent}22` },
      altar: { fill: `${meta.haze}bb`, stroke: `${meta.accent}88`, cap: `${meta.accent}33` },
    };
    obstacles.forEach((o) => {
      const st = kindStyle[o.kind] || kindStyle.wall;
      ctx.fillStyle = st.fill;
      ctx.fillRect(o.x, o.y, o.w, o.h);
      ctx.strokeStyle = st.stroke;
      ctx.lineWidth = 2;
      ctx.strokeRect(o.x, o.y, o.w, o.h);
      ctx.fillStyle = st.cap;
      ctx.fillRect(o.x + 4, o.y + 4, o.w - 8, Math.min(8, o.h - 8));
      if (o.kind === "pillar") {
        ctx.fillStyle = `${meta.accent}55`;
        ctx.fillRect(o.x + o.w / 2 - 2, o.y + 8, 4, o.h - 16);
      }
      ctx.lineWidth = 1;
    });
  }

  window.GameObstacles = { LAYOUTS, getForFloor, draw };
})();