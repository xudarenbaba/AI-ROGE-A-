/** 主动技能槽：按键释放，与对话姿态分离 */
(function () {
  const SKILLS = {
    pact_combo: {
      id: "pact_combo",
      name: "契印连携",
      shortName: "连携",
      keyHint: "1",
      desc: "贴乌枭作战攒契印，按1与乌枭齐攻",
      canUse(state) {
        if (!state || state.result) return { ok: false, reason: "战斗外" };
        if (state.floorState !== "playing") return { ok: false, reason: "非战斗" };
        if (window.GameCoop?.isSplit?.(state) && state.coop?.split?.active) {
          return { ok: false, reason: "裂狱中" };
        }
        if (state.ally?.hp <= 0) return { ok: false, reason: "乌枭失联" };
        if ((state.enemies || []).length === 0) return { ok: false, reason: "无敌人" };
        const c = state.coop;
        if (!c?.combo) return { ok: false, reason: "未就绪" };
        if (c.combo.castCd > 0) return { ok: false, reason: "冷却中" };
        if (!c.combo.ready && !c.combo.windowOpen) {
          return { ok: false, reason: "契印未满" };
        }
        return { ok: true };
      },
      cast(state, helpers) {
        const check = this.canUse(state);
        if (!check.ok) return check;
        const r = window.GameCoop?.tryComboBurst?.(state, helpers);
        if (!r?.ok) return r || { ok: false, reason: "释放失败" };
        if (state.coop?.combo) state.coop.combo.castCd = 0.8;
        return { ok: true, skillId: "pact_combo", dmg: r.dmg };
      },
      tick(state, dt) {
        window.GameCoop?.tickCombo?.(state, dt);
        if (state.coop?.combo && state.coop.combo.castCd > 0) {
          state.coop.combo.castCd = Math.max(0, state.coop.combo.castCd - dt);
        }
      },
      chargeRatio(state) {
        const c = state.coop?.combo;
        if (!c) return 0;
        if (c.ready || c.windowOpen) return 1;
        return Math.max(0, Math.min(1, (c.charge || 0) / (c.maxCharge || 100)));
      },
      isReady(state) {
        const c = state.coop?.combo;
        return !!(c && (c.ready || c.windowOpen) && (c.castCd || 0) <= 0);
      },
    },
  };

  function ensureSlot(state) {
    if (!state.skillSlot) {
      state.skillSlot = {
        equippedId: "pact_combo",
      };
    }
    if (!SKILLS[state.skillSlot.equippedId]) {
      state.skillSlot.equippedId = "pact_combo";
    }
    return state.skillSlot;
  }

  function getEquipped(state) {
    const slot = ensureSlot(state);
    return SKILLS[slot.equippedId] || SKILLS.pact_combo;
  }

  function equip(state, skillId) {
    if (!SKILLS[skillId]) return false;
    ensureSlot(state).equippedId = skillId;
    return true;
  }

  function tryCastEquipped(state, helpers) {
    const skill = getEquipped(state);
    return skill.cast(state, helpers || {});
  }

  function tickEquipped(state, dt) {
    const skill = getEquipped(state);
    if (skill.tick) skill.tick(state, dt);
  }

  function drawSlot(ctx, canvas, state) {
    const skill = getEquipped(state);
    const ratio = skill.chargeRatio(state);
    const ready = skill.isReady(state);
    const split = window.GameCoop?.isSplit?.(state) && state.coop?.split?.active;
    const w = 168;
    const h = 36;
    const x = canvas.width / 2 - w / 2;
    const y = canvas.height - 48;

    ctx.fillStyle = "rgba(0,0,0,0.62)";
    ctx.fillRect(x - 6, y - 6, w + 12, h + 12);
    ctx.strokeStyle = ready && !split ? "rgba(255, 220, 160, 0.9)" : "rgba(120, 100, 90, 0.7)";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x - 6, y - 6, w + 12, h + 12);
    ctx.lineWidth = 1;

    ctx.font = "bold 12px PingFang SC, Segoe UI, Arial";
    ctx.fillStyle = split ? "#888" : ready ? "#ffd9bf" : "#d9bca4";
    const title = split ? `[1] ${skill.shortName} · 裂狱禁用` : `[1] ${skill.shortName}`;
    ctx.fillText(title, x, y + 10);

    ctx.fillStyle = "#333";
    ctx.fillRect(x, y + 16, w, 10);
    ctx.fillStyle = ready ? "#ffd9bf" : "#c08050";
    ctx.fillRect(x, y + 16, w * ratio, 10);

    ctx.font = "10px PingFang SC, Segoe UI, Arial";
    ctx.fillStyle = "#c8b8a8";
    if (ready && !split) ctx.fillText("就绪 · 按1释放", x + w - 72, y + 10);
    else if (!split) ctx.fillText(`${Math.round(ratio * 100)}%`, x + w - 28, y + 10);
  }

  window.GameSkills = {
    SKILLS,
    ensureSlot,
    getEquipped,
    equip,
    tryCastEquipped,
    tickEquipped,
    drawSlot,
  };
})();
