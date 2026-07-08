/** 狱印：魂体印 / 鬼差印 / 契约印，可重复叠层 */
(function () {
  const FACTION = {
    soul: { id: "soul", label: "魂体印", css: "soul" },
    ally: { id: "ally", label: "鬼差印", css: "ally" },
    pact: { id: "pact", label: "契约印", css: "pact" },
  };

  const KIND = {
    passive:     { label: "常驻", hint: "本局永久" },
    skill:       { label: "技能", hint: "需按键释放" },
    tactical:    { label: "战术", hint: "限时生效" },
    conditional: { label: "条件", hint: "满足场景触发" },
  };

  const CAPS = {
    swift_soul: 3,
    pierce_sigil: 3,
    dash_ember: 4,
    vital_coal: 4,
    ally_mend: 4,
    focus_mark: 99,
    ally_guard: 3,
    iron_veil: 3,
    slayer_vitae: 3,
  };

  const KILL_HEAL_PCT_PER_STACK = 0.05;

  const POOL = [
    {
      id: "swift_soul",
      name: "疾魂步",
      faction: "soul",
      kind: "passive",
      desc: "魂体移速 +15%",
      apply(s) {
        if (stackCount(s, "swift_soul") >= CAPS.swift_soul) return false;
        s.player.speed *= 1.15;
        return true;
      },
    },
    {
      id: "pierce_sigil",
      name: "穿狱矢",
      faction: "soul",
      kind: "passive",
      desc: "魂体弹伤害 +20%",
      apply(s) {
        if (stackCount(s, "pierce_sigil") >= CAPS.pierce_sigil) return false;
        s.playerDamageMul = (s.playerDamageMul || 1) * 1.2;
        return true;
      },
    },
    {
      id: "dash_ember",
      name: "闪狱纹",
      faction: "soul",
      kind: "skill",
      desc: "Shift 闪避冷却 -25%",
      apply(s) {
        if (stackCount(s, "dash_ember") >= CAPS.dash_ember) return false;
        s.dashCdMul = Math.max(0.55, (s.dashCdMul || 1) * 0.75);
        return true;
      },
    },
    {
      id: "vital_coal",
      name: "续命炭",
      faction: "soul",
      kind: "passive",
      desc: "魂体最大 HP +25，并立刻回复",
      apply(s) {
        if (stackCount(s, "vital_coal") >= CAPS.vital_coal) return false;
        s.player.maxHp += 25;
        s.player.hp = Math.min(s.player.maxHp, s.player.hp + 25);
        return true;
      },
    },
    {
      id: "ally_mend",
      name: "鬼差补衣",
      faction: "ally",
      kind: "passive",
      desc: "乌枭最大 HP +20，并立刻回复",
      apply(s) {
        if (stackCount(s, "ally_mend") >= CAPS.ally_mend) return false;
        s.ally.maxHp += 20;
        s.ally.hp = Math.min(s.ally.maxHp, s.ally.hp + 20);
        return true;
      },
    },
    {
      id: "focus_mark",
      name: "集火印",
      faction: "ally",
      kind: "tactical",
      desc: "乌枭 8s 内优先攻击血量最低敌人",
      apply(s) {
        s.ally.focusMode = "lowest_hp";
        s.ally.focusUntil = performance.now() / 1000 + 8;
        return true;
      },
    },
    {
      id: "ally_guard",
      name: "黑签护幕",
      faction: "pact",
      kind: "conditional",
      desc: "守护贴身时魂体减伤 +15%",
      apply(s) {
        if (stackCount(s, "ally_guard") >= CAPS.ally_guard) return false;
        s.guardDamageReduction = Math.min(0.55, (s.guardDamageReduction || 0.15) + 0.15);
        return true;
      },
    },
    {
      id: "iron_veil",
      name: "铁幕印",
      faction: "pact",
      kind: "conditional",
      desc: "乌枭救援时护盾时长 +40%（HP≤45）",
      apply(s) {
        if (stackCount(s, "iron_veil") >= CAPS.iron_veil) return false;
        s.blessingShieldMul = Math.min(2.0, (s.blessingShieldMul || 1) * 1.4);
        return true;
      },
    },
    {
      id: "slayer_vitae",
      name: "噬敌生息",
      faction: "pact",
      kind: "passive",
      desc: "魂体或乌枭击杀敌人时，击杀者回复最大生命 5%",
      apply(s) {
        if (stackCount(s, "slayer_vitae") >= CAPS.slayer_vitae) return false;
        return true;
      },
    },
  ];

  const BY_ID = Object.fromEntries(POOL.map((b) => [b.id, b]));

  function stackCount(state, id) {
    return (state.blessingsTaken || []).filter((x) => x === id).length;
  }

  function killHealAmount(state, entity) {
    const stacks = stackCount(state, "slayer_vitae");
    if (stacks <= 0 || !entity?.maxHp) return 0;
    return Math.max(1, Math.round(entity.maxHp * KILL_HEAL_PCT_PER_STACK * stacks));
  }

  function meta(blessing) {
    const f = FACTION[blessing.faction] || FACTION.soul;
    const k = KIND[blessing.kind] || KIND.passive;
    return { faction: f, kind: k };
  }

  function pickThree(state) {
    const weighted = POOL.map((b) => {
      const stacks = stackCount(state, b.id);
      const cap = CAPS[b.id] ?? 99;
      const w = stacks >= cap ? 0.15 : 1;
      return { b, w };
    });
    const out = [];
    const pool = [...weighted];
    while (out.length < 3 && pool.length) {
      const total = pool.reduce((s, x) => s + x.w, 0);
      let r = Math.random() * total;
      let idx = 0;
      for (let i = 0; i < pool.length; i += 1) {
        r -= pool[i].w;
        if (r <= 0) { idx = i; break; }
      }
      out.push(pool[idx].b);
      pool.splice(idx, 1);
    }
    return out;
  }

  function apply(state, blessing) {
    if (!blessing?.apply) return false;
    const ok = blessing.apply(state);
    if (ok === false) return false;
    state.blessingsTaken = state.blessingsTaken || [];
    state.blessingsTaken.push(blessing.id);
    return true;
  }

  function stackSummary(ids, faction) {
    const counts = {};
    (ids || []).forEach((id) => {
      const b = BY_ID[id];
      if (!b || (faction && b.faction !== faction)) return;
      counts[id] = (counts[id] || 0) + 1;
    });
    return Object.entries(counts).map(([id, n]) => {
      const name = BY_ID[id]?.name || id;
      return n > 1 ? `${name}×${n}` : name;
    });
  }

  function listTaken(ids, faction) {
    return (ids || [])
      .map((id) => BY_ID[id])
      .filter((b) => b && (!faction || b.faction === faction));
  }

  function namesTaken(ids, faction) {
    return stackSummary(ids, faction);
  }

  function buildTags(state) {
    const tags = [];
    if (stackCount(state, "swift_soul")) tags.push("高机动");
    if (stackCount(state, "pierce_sigil")) tags.push("高火力");
    if (stackCount(state, "focus_mark")) tags.push("集火");
    if (stackCount(state, "ally_mend")) tags.push("乌枭耐打");
    if (stackCount(state, "iron_veil")) tags.push("护盾强化");
    if (stackCount(state, "slayer_vitae")) tags.push("击杀回血");
    return tags.slice(0, 4);
  }

  function summarizeForNpc(state) {
    const ids = state.blessingsTaken || [];
    return {
      total: ids.length,
      soul: stackSummary(ids, "soul"),
      ally: stackSummary(ids, "ally"),
      pact: stackSummary(ids, "pact"),
      tags: buildTags(state),
    };
  }

  window.GameBlessings = {
    POOL, BY_ID, FACTION, KIND, CAPS, KILL_HEAL_PCT_PER_STACK,
    stackCount, killHealAmount, meta, pickThree, apply,
    listTaken, namesTaken, stackSummary, buildTags, summarizeForNpc,
  };
})();