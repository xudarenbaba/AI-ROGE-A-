/** 狱印成长（仅改玩家/守护数值，不动 RL / ally assault 参数） */
(function () {
  const POOL = [
    {
      id: "swift_soul",
      name: "疾魂步",
      desc: "移速 +15%",
      apply(s) { s.player.speed *= 1.15; },
    },
    {
      id: "iron_veil",
      name: "铁幕印",
      desc: "护盾持续时间 +40%",
      apply(s) {
        s.blessingShieldMul = (s.blessingShieldMul || 1) * 1.4;
      },
    },
    {
      id: "pierce_sigil",
      name: "穿狱矢",
      desc: "玩家弹伤害 +20%",
      apply(s) { s.playerDamageMul = (s.playerDamageMul || 1) * 1.2; },
    },
    {
      id: "ally_guard",
      name: "黑签护幕",
      desc: "守护贴脸时玩家减伤 +15%",
      apply(s) { s.guardDamageReduction = Math.min(0.55, (s.guardDamageReduction || 0.15) + 0.15); },
    },
    {
      id: "dash_ember",
      name: "闪狱纹",
      desc: "闪避冷却 -25%",
      apply(s) { s.dashCdMul = (s.dashCdMul || 1) * 0.75; },
    },
    {
      id: "vital_coal",
      name: "续命炭",
      desc: "最大 HP +25，并回复 25",
      apply(s) {
        s.player.maxHp += 25;
        s.player.hp = Math.min(s.player.maxHp, s.player.hp + 25);
      },
    },
    {
      id: "ally_mend",
      name: "鬼差补衣",
      desc: "乌枭最大 HP +20，并回复 20",
      apply(s) {
        s.ally.maxHp += 20;
        s.ally.hp = Math.min(s.ally.maxHp, s.ally.hp + 20);
      },
    },
    {
      id: "focus_mark",
      name: "集火印",
      desc: "乌枭优先打血量最低敌人（8s）",
      apply(s) {
        s.ally.focusMode = "lowest_hp";
        s.ally.focusUntil = performance.now() / 1000 + 8;
      },
    },
  ];

  function pickThree() {
    const shuffled = [...POOL].sort(() => Math.random() - 0.5);
    return shuffled.slice(0, 3);
  }

  function apply(state, blessing) {
    if (!blessing?.apply) return;
    blessing.apply(state);
    state.blessingsTaken = state.blessingsTaken || [];
    state.blessingsTaken.push(blessing.id);
  }

  window.GameBlessings = { POOL, pickThree, apply };
})();