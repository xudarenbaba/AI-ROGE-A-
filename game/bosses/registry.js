/** Boss 主题：仅三关 demo */
(function () {
  const BOSSES = [
    {
      name: "妄言判官",
      title: "拔舌狱",
      ult: "禁言裂波",
      aoeColor: "#ff8866",
      skillId: "silence_zone",
      skillName: "缄言束",
      archetype: "normal",
    },
    {
      name: "断线剪魔",
      title: "剪刀狱",
      ult: "剪魂落印",
      aoeColor: "#ff6688",
      skillId: "cut_line",
      skillName: "剪线",
      archetype: "normal",
    },
    {
      name: "镜影·乌枭",
      title: "孽镜狱",
      ult: "镜裂天坠",
      aoeColor: "#aa88ff",
      skillId: "mirror_shot",
      skillName: "镜返",
      archetype: "mirror_wuxiao",
    },
  ];

  const MAX_FLOORS = 3;

  function getForFloor(floor) {
    const idx = Math.max(0, Math.min(BOSSES.length - 1, (floor || 1) - 1));
    const meta = BOSSES[idx];
    return {
      ...meta,
      fanChance: Math.min(0.42 + (floor - 1) * 0.02, 0.58),
      ultInterval: Math.max(3.2, 5.0 - (floor - 1) * 0.14),
      hpMul: 1 + (floor - 1) * 0.08,
    };
  }

  window.GameBosses = { getForFloor, BOSSES, MAX_FLOORS };
})();
