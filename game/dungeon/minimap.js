(function () {
  function draw(ctx, dungeon, roomIndex, canvasW) {
    if (!dungeon?.rooms?.length) return;
    const rooms = dungeon.rooms;
    const n = rooms.length;
    const pad = 10;
    const boxW = 22;
    const boxH = 14;
    const gap = 4;
    const totalW = n * boxW + (n - 1) * gap;
    const ox = canvasW - totalW - pad;
    const oy = pad;

    ctx.fillStyle = "rgba(8, 6, 12, 0.72)";
    ctx.fillRect(ox - 6, oy - 6, totalW + 12, boxH + 20);
    ctx.strokeStyle = "rgba(255, 200, 150, 0.35)";
    ctx.strokeRect(ox - 6, oy - 6, totalW + 12, boxH + 20);

    ctx.font = "10px Segoe UI";
    ctx.fillStyle = "#d8bca6";
    ctx.fillText("层内探索", ox, oy - 10);

    rooms.forEach((r, i) => {
      const x = ox + i * (boxW + gap);
      const y = oy;
      const active = i === roomIndex;
      const done = r.cleared || i < roomIndex;
      const isCoop = r.type === "duo_split" || r.type === "duo_info" || r.type === "duo_proxy"
        || r.coop === "split" || r.coop === "info" || r.coop === "proxy";
      ctx.fillStyle = active ? "#d08050" : done ? "#5a6a58" : isCoop ? "#3a3050" : "#2a2228";
      ctx.fillRect(x, y, boxW, boxH);
      ctx.strokeStyle = active ? "#ffcc99" : isCoop ? "#b090ff" : "#6a5a52";
      ctx.strokeRect(x, y, boxW, boxH);
      if (r.boss) {
        ctx.fillStyle = "#ffaa88";
        ctx.fillRect(x + 8, y + 4, 6, 6);
      } else if (isCoop) {
        ctx.fillStyle = "#c8a0ff";
        ctx.font = "bold 9px Segoe UI";
        ctx.fillText("协", x + 5, y + 11);
      }
    });
  }

  window.GameMinimap = { draw };
})();