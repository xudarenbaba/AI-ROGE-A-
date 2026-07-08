/** 一层地下城：线性 5 房间 */
(function () {
  const ROOM_CHAIN = [
    { type: "entrance", label: "前厅", depth: 0, mobs: 3, elite: 0, boss: false },
    { type: "corridor", label: "阴廊", depth: 1, mobs: 4, elite: 0, boss: false },
    { type: "combat", label: "狱房", depth: 2, mobs: 6, elite: 0, boss: false },
    { type: "elite", label: "精英房", depth: 3, mobs: 3, elite: 1, boss: false },
    { type: "boss", label: "Boss房", depth: 4, mobs: 2, elite: 0, boss: true },
  ];

  function mobCount(tpl, floor) {
    const bonus = floor >= 2 ? 1 : 0;
    return Math.min(9, (tpl.mobs || 0) + bonus);
  }

  function generate(floor) {
    return ROOM_CHAIN.map((tpl, index) => ({
      ...tpl,
      mobs: mobCount(tpl, floor),
      index,
      floor,
      layoutKey: tpl.type === "combat" ? null : tpl.type,
      cleared: false,
    }));
  }

  window.GameDungeonGen = { generate, ROOM_CHAIN, mobCount };
})();