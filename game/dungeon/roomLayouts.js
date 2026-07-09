/** 房间障碍布局（碰撞盒 x,y,w,h） */
(function () {
  const H = 540;
  const WEST_SEAL_W = 72;
  const SPAWN_MIN_X = 98;

  const DEPTH_EXTRAS = {
    entrance: [{ x: 400, y: 240, w: 18, h: 60, kind: "pillar" }],
    corridor: [{ x: 450, y: 200, w: 60, h: 16, kind: "rubble" }],
    combat_a: [{ x: 540, y: 300, w: 18, h: 70, kind: "pillar" }],
    combat_b: [{ x: 420, y: 260, w: 18, h: 55, kind: "pillar" }],
    combat_c: [{ x: 500, y: 220, w: 70, h: 16, kind: "rubble" }],
    combat_d: [
      { x: 400, y: 200, w: 18, h: 140, kind: "pillar" },
      { x: 540, y: 200, w: 18, h: 140, kind: "pillar" },
      { x: 400, y: 268, w: 158, h: 18, kind: "wall" },
    ],
    elite: [{ x: 300, y: 270, w: 18, h: 50, kind: "pillar" }, { x: 640, y: 270, w: 18, h: 50, kind: "pillar" }],
    boss: [
      { x: 360, y: 360, w: 50, h: 16, kind: "rubble" },
      { x: 550, y: 360, w: 50, h: 16, kind: "rubble" },
    ],
  };

  const LAYOUTS = {
    entrance: [
      { x: 320, y: 200, w: 120, h: 18, kind: "wall" },
      { x: 320, y: 322, w: 120, h: 18, kind: "wall" },
      { x: 500, y: 160, w: 18, h: 100, kind: "pillar" },
      { x: 280, y: 260, w: 18, h: 70, kind: "pillar" },
      { x: 420, y: 150, w: 70, h: 16, kind: "rubble" },
      { x: 480, y: 300, w: 80, h: 18, kind: "rubble" },
    ],
    corridor: [
      { x: 200, y: 120, w: 560, h: 18, kind: "wall" },
      { x: 200, y: 402, w: 560, h: 18, kind: "wall" },
      { x: 380, y: 230, w: 18, h: 80, kind: "pillar" },
      { x: 520, y: 280, w: 18, h: 70, kind: "pillar" },
      { x: 300, y: 250, w: 70, h: 16, kind: "rubble" },
      { x: 620, y: 200, w: 18, h: 55, kind: "pillar" },
      { x: 430, y: 165, w: 100, h: 14, kind: "wall" },
    ],
    combat_a: [
      { x: 360, y: 255, w: 180, h: 22, kind: "wall" },
      { x: 240, y: 170, w: 22, h: 130, kind: "pillar" },
      { x: 660, y: 220, w: 22, h: 130, kind: "pillar" },
      { x: 480, y: 140, w: 140, h: 20, kind: "wall" },
      { x: 420, y: 360, w: 140, h: 20, kind: "wall" },
      { x: 320, y: 300, w: 60, h: 16, kind: "rubble" },
    ],
    combat_b: [
      { x: 280, y: 145, w: 200, h: 20, kind: "wall" },
      { x: 560, y: 145, w: 160, h: 20, kind: "wall" },
      { x: 280, y: 375, w: 160, h: 20, kind: "wall" },
      { x: 520, y: 375, w: 200, h: 20, kind: "wall" },
      { x: 235, y: 220, w: 20, h: 110, kind: "pillar" },
      { x: 660, y: 210, w: 20, h: 110, kind: "pillar" },
      { x: 460, y: 260, w: 18, h: 80, kind: "pillar" },
    ],
    combat_c: [
      { x: 270, y: 160, w: 160, h: 20, kind: "wall" },
      { x: 580, y: 200, w: 20, h: 150, kind: "pillar" },
      { x: 340, y: 340, w: 160, h: 20, kind: "wall" },
      { x: 240, y: 280, w: 20, h: 100, kind: "pillar" },
      { x: 460, y: 150, w: 20, h: 120, kind: "pillar" },
      { x: 630, y: 330, w: 140, h: 20, kind: "rubble" },
      { x: 380, y: 220, w: 18, h: 60, kind: "pillar" },
    ],
    combat_d: [
      { x: 300, y: 155, w: 120, h: 18, kind: "wall" },
      { x: 540, y: 155, w: 120, h: 18, kind: "wall" },
      { x: 300, y: 367, w: 120, h: 18, kind: "wall" },
      { x: 540, y: 367, w: 120, h: 18, kind: "wall" },
      { x: 250, y: 230, w: 18, h: 90, kind: "pillar" },
      { x: 692, y: 230, w: 18, h: 90, kind: "pillar" },
    ],
    elite: [
      { x: 390, y: 230, w: 120, h: 20, kind: "wall" },
      { x: 445, y: 165, w: 20, h: 140, kind: "pillar" },
      { x: 240, y: 155, w: 130, h: 20, kind: "wall" },
      { x: 620, y: 155, w: 130, h: 20, kind: "wall" },
      { x: 240, y: 365, w: 130, h: 20, kind: "rubble" },
      { x: 620, y: 365, w: 130, h: 20, kind: "rubble" },
      { x: 360, y: 280, w: 18, h: 45, kind: "pillar" },
      { x: 582, y: 280, w: 18, h: 45, kind: "pillar" },
    ],
    boss: [
      { x: 180, y: 130, w: 600, h: 20, kind: "wall" },
      { x: 180, y: 390, w: 600, h: 20, kind: "wall" },
      { x: 300, y: 200, w: 20, h: 120, kind: "pillar" },
      { x: 640, y: 200, w: 20, h: 120, kind: "pillar" },
      { x: 420, y: 250, w: 120, h: 18, kind: "rubble" },
      { x: 260, y: 330, w: 18, h: 45, kind: "pillar" },
      { x: 682, y: 330, w: 18, h: 45, kind: "pillar" },
    ],
  };

  function pickCombatLayout(floor, roomIndex) {
    const keys = ["combat_a", "combat_b", "combat_c", "combat_d"];
    return keys[(floor + roomIndex) % keys.length];
  }

  function layoutTypeOf(room) {
    const t = room?.type || "";
    if (t === "duo_split" || t === "duo_info" || t === "duo_proxy") return "combat";
    return t;
  }

  function getObstacles(room) {
    let key = room.layoutKey;
    if (!key) {
      const t = layoutTypeOf(room);
      if (t === "combat") key = pickCombatLayout(room.floor || 1, room.index || 0);
      else key = t;
    }
    const boxes = (LAYOUTS[key] || LAYOUTS.combat_a).map((b) => ({ ...b }));
    if ((room.depth || 0) >= 2 && DEPTH_EXTRAS[key]) {
      DEPTH_EXTRAS[key].forEach((e) => boxes.push({ ...e }));
    }
    if ((room.floor || 1) >= 2 && room.type !== "boss" && boxes.length > 0) {
      const rubbleIdx = boxes.findIndex((b) => b.kind === "rubble");
      if (rubbleIdx >= 0) boxes[rubbleIdx] = { ...boxes[rubbleIdx], kind: "pillar", h: Math.max(boxes[rubbleIdx].h, 18) };
    }
    if (room.index > 0) {
      boxes.unshift({ x: 0, y: 0, w: WEST_SEAL_W, h: H, kind: "seal" });
    }
    return boxes;
  }

  function getDoorRect(canvasW, canvasH) {
    return { x: canvasW - 52, y: canvasH * 0.25, w: 48, h: canvasH * 0.5 };
  }

  function getWestSealRect() {
    return { x: 0, y: 0, w: WEST_SEAL_W, h: H };
  }

  function getSpawnPoints(roomType, roomIndex) {
    const minX = roomIndex > 0 ? SPAWN_MIN_X : 100;
    const t = (roomType === "duo_split" || roomType === "duo_info" || roomType === "duo_proxy")
      ? "combat"
      : roomType;
    if (t === "boss") {
      return {
        player: { x: Math.max(minX, 140), y: 270 },
        ally: { x: Math.max(minX + 40, 180), y: 300 },
        guards: [
          { x: 620, y: 190 },
          { x: 620, y: 350 },
        ],
        boss: { x: 720, y: 270 },
      };
    }
    return {
      player: { x: minX, y: 270 },
      ally: { x: minX + 40, y: 300 },
      enemies: "scatter",
    };
  }

  function playerMinX(roomIndex, radius) {
    return (roomIndex > 0 ? SPAWN_MIN_X : 10) + radius;
  }

  window.GameRoomLayouts = {
    LAYOUTS, getObstacles, getDoorRect, getWestSealRect, getSpawnPoints, playerMinX, WEST_SEAL_W, H,
  };
})();