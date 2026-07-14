/** 一层地下城：前厅 → 特殊协同房 → 精英 → Boss */
(function () {
  const ROOM_CHAIN = [
    { type: "entrance", label: "前厅", depth: 0, mobs: 3, elite: 0, boss: false },
    { type: "elite", label: "精英房", depth: 2, mobs: 3, elite: 1, boss: false },
    { type: "boss", label: "Boss房", depth: 3, mobs: 2, elite: 0, boss: true },
  ];

  function mobCount(tpl, floor) {
    const bonus = floor >= 2 ? 1 : 0;
    return Math.min(9, (tpl.mobs || 0) + bonus);
  }

  function pickDuoInsert(floor) {
    // 第 1 层强制裂狱教学；其后轮换判词 / 代行 / 裂狱
    if (floor <= 1) {
      return {
        type: "duo_split",
        label: "裂狱并行",
        depth: 1,
        mobs: 5,
        elite: 0,
        boss: false,
        coop: "split",
      };
    }
    const roll = (floor + 3) % 3;
    if (roll === 0) {
      return {
        type: "duo_info",
        label: "判词分卷",
        depth: 1,
        mobs: 3,
        elite: 0,
        boss: false,
        coop: "info",
      };
    }
    if (roll === 1) {
      return {
        type: "duo_proxy",
        label: "黑签代行",
        depth: 1,
        mobs: 5,
        elite: 0,
        boss: false,
        coop: "proxy",
      };
    }
    return {
      type: "duo_split",
      label: "裂狱并行",
      depth: 1,
      mobs: 5 + Math.min(2, floor - 1),
      elite: 0,
      boss: false,
      coop: "split",
    };
  }

  function generate(floor) {
    const chain = ROOM_CHAIN.map((tpl) => ({ ...tpl }));
    // 前厅之后插入特殊协同房
    const duo = pickDuoInsert(floor);
    chain.splice(1, 0, duo);

    return chain.map((tpl, index) => {
      const layoutType = ["duo_split", "duo_info", "duo_proxy"].includes(tpl.type)
        ? "combat"
        : tpl.type;
      return {
        ...tpl,
        mobs: mobCount(tpl, floor),
        index,
        floor,
        layoutKey: layoutType === "combat" ? null : layoutType,
        cleared: false,
      };
    });
  }

  window.GameDungeonGen = { generate, ROOM_CHAIN, mobCount, pickDuoInsert };
})();
