/** Boss 主题（按层地狱 meta 轮换） */
(function () {
  const BOSSES = [
    { name: "妄言判官", title: "拔舌狱", ult: "禁言裂波", aoeColor: "#ff8866", skillId: "silence_zone", skillName: "缄言束" },
    { name: "断线剪魔", title: "剪刀狱", ult: "剪魂落印", aoeColor: "#ff6688", skillId: "cut_line", skillName: "剪线" },
    { name: "刺狱树灵", title: "铁树狱", ult: "根须轰坠", aoeColor: "#ccaa66", skillId: "thorn_wall", skillName: "荆棘墙" },
    { name: "镜影婆", title: "孽镜狱", ult: "镜裂天坠", aoeColor: "#aa88ff", skillId: "mirror_shot", skillName: "镜返" },
    { name: "闷盖鬼首", title: "蒸笼狱", ult: "蒸狱烙印", aoeColor: "#ff9944", skillId: "steam_fog", skillName: "蒸汽雾" },
    { name: "铜柱狱首", title: "铜柱狱", ult: "烙狱沉环", aoeColor: "#dd7722", skillId: "magnetic_pull", skillName: "磁引" },
  ];

  function getForFloor(floor) {
    const meta = BOSSES[(floor - 1) % BOSSES.length];
    return {
      ...meta,
      fanChance: Math.min(0.42 + (floor - 1) * 0.02, 0.58),
      ultInterval: Math.max(3.2, 5.0 - (floor - 1) * 0.14),
      hpMul: 1 + (floor - 1) * 0.08,
    };
  }

  window.GameBosses = { getForFloor };
})();