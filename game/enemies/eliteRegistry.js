/** 精英类型（两套轮换） */
(function () {
  const ELITES = {
    chain: {
      id: "chain",
      name: "锁链鬼卒",
      color: "#88aaff",
      skill1: { id: "hook_dash", name: "钩锁冲刺", cd: 2.4 },
      skill2: { id: "ring_burst", name: "锁链环爆", hpPct: 0.5 },
    },
    brand: {
      id: "brand",
      name: "烙印判吏",
      color: "#ffaa66",
      skill1: { id: "ground_brand", name: "地烙圈", cd: 2.2 },
      skill2: { id: "summon_shade", name: "唤影小鬼", cd: 4.5 },
    },
  };

  function pick(floor, roomIndex = 0) {
    const key = (floor + roomIndex) % 2 === 0 ? "chain" : "brand";
    return ELITES[key];
  }

  function get(id) {
    return ELITES[id] || ELITES.chain;
  }

  window.GameEliteRegistry = { ELITES, pick, get };
})();