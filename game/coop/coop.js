/** 协同战友：状态、口令、分房、判词、连携、结算 */
(function () {
  // 左=乌枭，右=玩家（出口在画布最右）
  const SPLIT_X = 480;
  const CANVAS_W = 960;
  const CANVAS_H = 540;
  // 姿态口令（对话）；主动技能走按键，不在此列
  const CMD_IDS = ["guard", "assault"];

  function defaultCoop() {
    return {
      mode: "none",
      roomTag: "",
      trust: 0.5,
      annoyance: 0.2,
      commands: {
        slotsMax: 3,
        slotsLeft: 3,
        cds: { guard: 0, assault: 0 },
      },
      split: {
        active: false,
        playerKills: 0,
        allyKills: 0,
        playerGoal: 1,
        allyGoal: 1,
        playerDone: false,
        allyDone: false,
        allyEnemies: [],
        allyBullets: [],
        enemyBullets: [],
        allyDownT: 0,
        wave: 1,
      },
      info: {
        active: false,
        pillars: [],
        order: [],
        nextIdx: 0,
        solved: false,
        lastReport: "",
      },
      combo: {
        charge: 0,
        maxCharge: 100,
        windowOpen: false,
        windowT: 0,
        ready: false,
        lastBurstAt: 0,
        castCd: 0,
      },
      proxy: { active: false },
      stats: {
        ordersIssued: 0,
        ordersObeyed: 0,
        allyDowns: 0,
        startedAt: 0,
      },
      flags: {
        splitEnemyMul: 1,
        commandSlotsBonus: 0,
        comboChargeMul: 1,
      },
    };
  }

  function ensure(state) {
    if (!state.coop) state.coop = defaultCoop();
    return state.coop;
  }

  function resetRoom(state) {
    const c = ensure(state);
    const slotsMax = 3 + (c.flags.commandSlotsBonus || 0);
    c.mode = "none";
    c.roomTag = "";
    c.commands.slotsMax = slotsMax;
    c.commands.slotsLeft = slotsMax;
    Object.keys(c.commands.cds).forEach((k) => { c.commands.cds[k] = 0; });
    c.split = {
      active: false,
      playerKills: 0,
      allyKills: 0,
      playerGoal: 1,
      allyGoal: 1,
      playerDone: false,
      allyDone: false,
      allyEnemies: [],
      allyBullets: [],
      enemyBullets: [],
      allyDownT: 0,
      wave: 1,
    };
    c.info = {
      active: false,
      pillars: [],
      order: [],
      nextIdx: 0,
      solved: false,
      lastReport: "",
    };
    c.combo.charge = 0;
    c.combo.windowOpen = false;
    c.combo.windowT = 0;
    c.combo.ready = false;
    c.combo.castCd = 0;
    c.proxy.active = false;
    c.stats = {
      ordersIssued: 0,
      ordersObeyed: 0,
      allyDowns: 0,
      startedAt: performance.now(),
    };
  }

  function isSplit(state) {
    return !!ensure(state).split.active;
  }

  function isInfo(state) {
    return !!ensure(state).info.active;
  }

  function isProxy(state) {
    return !!ensure(state).proxy.active;
  }

  function syncSplitFlags(state) {
    const s = ensure(state).split;
    if (!s.active) return;
    // 以「剩余活敌」为准，避免击杀计数/目标数错位
    if (!s.playerDone && (state.enemies || []).filter((e) => e.hp > 0).length === 0) {
      s.playerDone = true;
    }
    if (!s.allyDone && (s.allyEnemies || []).length === 0) {
      s.allyDone = true;
    }
  }

  function bothSidesClear(state) {
    const s = ensure(state).split;
    if (!s.active) return true;
    syncSplitFlags(state);
    return s.playerDone && s.allyDone;
  }

  function roomClearGate(state) {
    const c = ensure(state);
    // 裂狱房：必须双边完成；即使 active 被误清，也用 _splitCompleted 防误开门
    const room = state.dungeon?.rooms?.[state.roomIndex];
    const isDuoSplit = room && (room.type === "duo_split" || room.coop === "split");
    if (c.split.active || (isDuoSplit && !c._splitCompleted)) {
      if (!c.split.active && isDuoSplit && !c._splitCompleted) {
        // 应处于分房却未 active：禁止清房（防止只清玩家侧就过关）
        return false;
      }
      syncSplitFlags(state);
      const playerAlive = (state.enemies || []).filter((e) => e.hp > 0).length;
      const allyAlive = (c.split.allyEnemies || []).length;
      // 乌枭侧必须曾经有过怪，且清完
      if ((c.split.allyGoal || 0) <= 0) return false;
      return !!(c.split.playerDone && c.split.allyDone && playerAlive === 0 && allyAlive === 0);
    }
    if (c.info.active) return c.info.solved && state.enemies.length === 0;
    if (c.proxy.active) return state.enemies.length === 0;
    return state.enemies.length === 0;
  }

  function endSplit(state, helpers) {
    const c = ensure(state);
    if (!c.split.active && c.mode !== "split") return;
    c.split.active = false;
    c._splitCompleted = true;
    c.split.allyEnemies = [];
    c.split.allyBullets = [];
    c.split.enemyBullets = [];
    // 强制拉离门区（门约 x=908），必须玩家自己再走过去
    state.player.x = 700;
    state.player.y = state.player.y || 270;
    state.ally.x = state.player.x - 40;
    state.ally.y = state.player.y;
    state.ally.stance = "guard";
    state.ally.combatPhase = "approach";
    state.ally.navPath = null;
    state.ally.attackCd = 0;
    state.ally.dead = false;
    (state.enemies || []).forEach((e) => { delete e._splitSide; });
    if (c.mode === "split") c.mode = "none";
    // 需按 → 进门，且冷却 2s
    state._doorGraceT = 2.0;
    state._doorNeedsRightKey = true;
    if (helpers?.setBubble) helpers.setBubble("裂狱合上了。门开了——按→走到最右门进入。");
    if (helpers?.chat) helpers.chat("【门已开】按 D/→ 走到最右侧门，才会进入下一间（不会自动进）。");
  }

  function playerProgress(state) {
    const s = ensure(state).split;
    if (!s.active) return 1;
    if (s.playerDone) return 1;
    const alive = (state.enemies || []).filter((e) => e.hp > 0).length;
    const goal = Math.max(1, s.playerGoal || 1);
    // 用剩余比例：开局 goal=N，剩 k → 进度 (N-k)/N
    return Math.max(0, Math.min(1, (goal - alive) / goal));
  }

  function allyProgress(state) {
    const s = ensure(state).split;
    if (!s.active) return 1;
    if (s.allyDone) return 1;
    const alive = (s.allyEnemies || []).length;
    const goal = Math.max(1, s.allyGoal || 1);
    return Math.max(0, Math.min(1, (goal - alive) / goal));
  }

  // ── 口令 ──────────────────────────────────────────────────────────────────

  const GUARD_HINTS = ["守护我", "跟着我", "贴着我", "别乱跑", "回来", "护着我", "守住", "守护"];
  const ASSAULT_HINTS = ["突击", "冲上去", "开路", "压制", "上去打", "前锋", "进攻", "去突击"];

  function matchHints(text, hints) {
    return hints.some((h) => text.includes(h));
  }

  /** 仅解析姿态口令；技能不走对话 */
  function parseLocalCommand(message) {
    const text = (message || "").trim();
    if (!text || text.length > 28) return null;
    if (matchHints(text, GUARD_HINTS)) return { op: "guard", stance: "guard" };
    if (matchHints(text, ASSAULT_HINTS)) return { op: "assault", stance: "assault" };
    return null;
  }

  function canSpendSlot(state, op) {
    const c = ensure(state);
    if (c.commands.slotsLeft <= 0) return { ok: false, reason: "no_slots" };
    if ((c.commands.cds[op] || 0) > 0) return { ok: false, reason: "cd" };
    return { ok: true };
  }

  function spendSlot(state, op, cdSec) {
    const c = ensure(state);
    c.commands.slotsLeft = Math.max(0, c.commands.slotsLeft - 1);
    c.commands.cds[op] = cdSec || 4;
    c.stats.ordersIssued += 1;
  }

  function tickCds(state, dt) {
    const cds = ensure(state).commands.cds;
    Object.keys(cds).forEach((k) => {
      cds[k] = Math.max(0, cds[k] - dt);
    });
  }

  // ── 分房：生成乌枭侧敌人 ──────────────────────────────────────────────────

  /**
   * 分场约定（与出口对齐）：
   * - 出口门在画布最右侧
   * - 玩家 = 右半场（能走到门）
   * - 乌枭 = 左半场
   * - state.enemies 只含玩家侧；allyEnemies 只含乌枭侧
   */
  function mkAllyMob(state, i, total, hpMul, floor) {
    // 乌枭侧：左半场（随层数变硬）；走 GameEnemyAI，需 initEnemy
    const col = i % 3;
    const row = Math.floor(i / 3);
    const f = floor || state.floor || 1;
    const tough = i === 0;
    const baseHp = tough ? 70 : 48;
    const enemy = {
      kind: tough ? "elite" : "mob",
      x: 90 + col * 110 + (i % 2) * 12,
      y: 110 + row * 95,
      hp: Math.round(baseHp * hpMul * (1 + (f - 1) * 0.12)),
      maxHp: Math.round(baseHp * hpMul * (1 + (f - 1) * 0.12)),
      radius: tough ? 14 : 12,
      speed: tough ? 52 : 48,
      shootCd: 0.25 + Math.random() * 0.35,
      dmgMul: tough ? 1.15 : 1.0,
      shootCdMul: 1,
      _arena: true,
      _splitSide: "ally",
      _tough: tough,
    };
    window.GameEnemyAI?.initEnemy?.(enemy, f, state.roomIndex || 0);
    return enemy;
  }

  function clampAllySide(entity) {
    const r = entity.radius || 12;
    entity.x = Math.max(r + 8, Math.min(SPLIT_X - r - 10, entity.x));
    entity.y = Math.max(r + 8, Math.min(CANVAS_H - r, entity.y));
  }

  function clampPlayerSide(entity) {
    const r = entity.radius || 12;
    // 右半场，且分房战斗中禁止贴到最右门区（门 x≈908）
    const maxX = Math.min(CANVAS_W - r - 8, 860);
    entity.x = Math.max(SPLIT_X + r + 10, Math.min(maxX, entity.x));
    entity.y = Math.max(r + 8, Math.min(CANVAS_H - r, entity.y));
  }

  function collidesObstacle(x, y, radius) {
    const obs = window.GameSpawn
      ? null
      : null;
    // 优先用全局 isClearSpawn / 障碍检测
    if (typeof window.GameSpawn?.isClearSpawn === "function") {
      return !window.GameSpawn.isClearSpawn(x, y, radius, { minX: radius });
    }
    // 兜底：无法检测时当作不碰撞
    return false;
  }

  /** 在指定半场内找安全落点，避免卡进柱子/墙 */
  function safePosInSide(side, preferX, preferY, radius, others) {
    const r = radius || 12;
    const minX = side === "player" ? SPLIT_X + r + 16 : r + 16;
    const maxX = side === "player" ? Math.min(CANVAS_W - r - 16, 850) : SPLIT_X - r - 16;
    const minY = r + 16;
    const maxY = CANVAS_H - r - 16;
    const px = Math.max(minX, Math.min(maxX, preferX));
    const py = Math.max(minY, Math.min(maxY, preferY));

    const spawn = window.GameSpawn;
    if (spawn?.findClearSpawnPos) {
      // 用 prefer 点 + 半场 minX 约束
      const pos = spawn.findClearSpawnPos(px, py, r, {
        minX,
        others: others || [],
        separation: 10,
      });
      if (pos) {
        // 再夹回半场（findClearSpawn 可能仍越界）
        pos.x = Math.max(minX, Math.min(maxX, pos.x));
        pos.y = Math.max(minY, Math.min(maxY, pos.y));
        if (spawn.isClearSpawn?.(pos.x, pos.y, r, { minX, others: others || [] })) {
          return pos;
        }
      }
    }

    // 网格扫描半场
    let best = null;
    let bestD = Infinity;
    for (let y = minY; y <= maxY; y += 22) {
      for (let x = minX; x <= maxX; x += 22) {
        const ok = spawn?.isClearSpawn
          ? spawn.isClearSpawn(x, y, r, { minX, others: others || [] })
          : !collidesObstacle(x, y, r);
        if (!ok) continue;
        const d = Math.hypot(x - px, y - py);
        if (d < bestD) {
          bestD = d;
          best = { x, y };
        }
      }
    }
    if (best) return best;

    // 最后兜底：半场中心附近
    return {
      x: side === "player" ? (SPLIT_X + maxX) * 0.5 : maxX * 0.5,
      y: (minY + maxY) * 0.5,
    };
  }

  function placeEntitySafe(entity, side, preferX, preferY, others) {
    const r = entity.radius || 12;
    const pos = safePosInSide(side, preferX, preferY, r, others);
    entity.x = pos.x;
    entity.y = pos.y;
    entity._splitSide = side;
    entity.navPath = null;
    entity.navGoal = null;
    entity.navReplanCd = 0;
    if (side === "player") clampPlayerSide(entity);
    else clampAllySide(entity);
    return entity;
  }

  function enterSplitMirror(state, room, helpers) {
    const c = ensure(state);
    // 保留 flags，重建本房 split 状态（不要整表 wipe 丢引用）
    const slotsMax = 3 + (c.flags.commandSlotsBonus || 0);
    c.mode = "split";
    c.roomTag = "duo_split_mirror";
    c.commands.slotsMax = slotsMax;
    c.commands.slotsLeft = slotsMax;
    c.commands.cds = { guard: 0, assault: 0 };

    const mul = c.flags.splitEnemyMul || 1;
    const scale = state._combatScale || { hpMul: 1, speedMul: 1, dmgMul: 1 };
    const floor = state.floor || 1;
    // 乌枭侧：5~7 只，随层数增加，保证有一段可看的战斗
    const allyCount = Math.max(5, Math.min(7, Math.round((5 + Math.min(2, floor - 1)) * mul)));

    // 玩家侧：只保留右半场敌人；用安全落点，避免卡墙
    const playerEnemies = [];
    (state.enemies || []).forEach((e, i) => {
      if (!e || e.hp <= 0) return;
      const preferX = SPLIT_X + 100 + (i % 3) * 80;
      const preferY = 140 + Math.floor(i / 3) * 100;
      placeEntitySafe(e, "player", preferX, preferY, playerEnemies);
      playerEnemies.push(e);
    });
    // 若玩家侧被滤空，至少补 3 只
    if (playerEnemies.length === 0) {
      for (let i = 0; i < 3; i += 1) {
        const pe = {
          kind: "mob",
          x: SPLIT_X + 120,
          y: 200,
          hp: Math.round(30 * (scale.hpMul || 1)),
          maxHp: Math.round(30 * (scale.hpMul || 1)),
          radius: 12,
          speed: 42 * (scale.speedMul || 1),
          shootCd: 0.5 + Math.random(),
          dmgMul: (scale.dmgMul || 1) * 0.9,
          shootCdMul: 1,
          _splitSide: "player",
        };
        window.GameEnemyAI?.initEnemy?.(pe, floor, state.roomIndex || 0);
        placeEntitySafe(
          pe,
          "player",
          SPLIT_X + 100 + (i % 3) * 70,
          150 + Math.floor(i / 3) * 100,
          playerEnemies,
        );
        playerEnemies.push(pe);
      }
    }
    // 再扫一遍：仍卡障则重定位
    playerEnemies.forEach((e, i) => {
      const stuck = window.GameSpawn?.isClearSpawn
        ? !window.GameSpawn.isClearSpawn(e.x, e.y, e.radius || 12, {
          minX: SPLIT_X + 20,
          others: playerEnemies.filter((o) => o !== e),
        })
        : false;
      if (stuck) {
        placeEntitySafe(
          e,
          "player",
          SPLIT_X + 120 + (i % 3) * 60,
          180 + Math.floor(i / 3) * 80,
          playerEnemies.filter((o) => o !== e),
        );
      }
    });
    state.enemies = playerEnemies;
    state.bossAlive = false;

    // 乌枭侧：独立生成 + 安全落点
    const allyEnemies = [];
    for (let i = 0; i < allyCount; i += 1) {
      const mob = mkAllyMob(state, i, allyCount, (scale.hpMul || 1) * 0.95, floor);
      if (!mob.hp || mob.hp <= 0) {
        mob.hp = mob.maxHp = Math.round(48 * (scale.hpMul || 1));
      }
      placeEntitySafe(
        mob,
        "ally",
        100 + (i % 3) * 100,
        120 + Math.floor(i / 3) * 90,
        allyEnemies,
      );
      allyEnemies.push(mob);
    }

    c._splitCompleted = false;
    c.split = {
      active: true,
      dividerX: SPLIT_X,
      playerKills: 0,
      allyKills: 0,
      playerGoal: playerEnemies.length,
      allyGoal: allyEnemies.length,
      playerDone: false,
      allyDone: false,
      allyEnemies,
      allyBullets: [],
      enemyBullets: [],
      allyDownT: 0,
      wave: 1,
      _announcedPlayer: false,
      _announcedAlly: false,
    };

    // 玩家 → 右半中部（绝不能在门上）
    state.player.x = 680;
    state.player.y = 270;
    clampPlayerSide(state.player);
    state._doorGraceT = 0;
    state._doorNeedsRightKey = false;

    // 乌枭 → 左半，强制可战斗
    state.ally.dead = false;
    if (state.ally.hp <= 0) state.ally.hp = Math.max(1, Math.floor(state.ally.maxHp * 0.5));
    state.ally.stance = "assault";
    state.ally.combatPhase = "approach";
    state.ally.navPath = null;
    state.ally.attackCd = 0;
    state.ally.x = 180;
    state.ally.y = 270;
    clampAllySide(state.ally);

    console.info(
      "[COOP] enterSplit playerEnemies=", c.split.playerGoal,
      "allyEnemies=", c.split.allyGoal,
      "allyList=", c.split.allyEnemies.map((e) => `${Math.round(e.x)},${Math.round(e.y)},hp${e.hp}`).join("|"),
      "player@", Math.round(state.player.x),
      "ally@", Math.round(state.ally.x),
      "split.active=", c.split.active,
    );

    if (helpers?.setBubble) {
      helpers.setBubble("裂狱了！你清右边（门在最右），我清左边——两边都完才开门。");
    }
    if (helpers?.chat) {
      helpers.chat(
        "【裂狱并行】左=乌枭，右=你。两边都清完后门会解锁，走到最右侧门再进下一间。",
      );
    }
    window.GameFx?.floatText?.(state.player.x, state.player.y - 50, "裂狱·你在右侧", "#ffd9bf");
    window.__npcPushEvent?.("coop_split_enter", {
      tag: "duo_split_mirror",
      ally: c.split.allyGoal,
      player: c.split.playerGoal,
    });
  }

  function onPlayerKill(state, count) {
    const s = ensure(state).split;
    if (!s.active) return;
    s.playerKills += count || 1;
    // 注意：调用时 dead 可能尚未 filter，用 hp>0 判断
    const alive = (state.enemies || []).filter((e) => e.hp > 0).length;
    if (alive === 0 && !s.playerDone) {
      s.playerDone = true;
      if (!s._announcedPlayer) {
        s._announcedPlayer = true;
        window.GameFx?.floatText(state.player.x, state.player.y - 40, "你侧肃清", "#9fe4ff");
        window.__npcPushEvent?.("coop_side_done", { side: "player" });
      }
    }
  }

  /**
   * 裂狱侧战场维护：
   * - 乌枭走位/开火由 game.js updateAlly 的 A*+RL 负责（withSplitAllyCombatContext）
   * - 这里只维护：分区夹紧、左侧敌 AI、左侧敌弹、击杀结算、清完发言
   */
  function tickSplitAllyCombat(state, dt, helpers) {
    const c = ensure(state);
    const s = c.split;
    if (!s.active) return;

    syncSplitFlags(state);

    if (s.playerDone && !s._announcedPlayer) {
      s._announcedPlayer = true;
      window.GameFx?.floatText(state.player.x, state.player.y - 40, "你侧肃清 · 等乌枭", "#9fe4ff");
      if (helpers?.setBubble) helpers.setBubble("你清完了？行，我加快。");
    }

    if (state.ally.hp <= 0) {
      s.allyDownT += dt;
      if (s.allyDownT >= 5) {
        state.ally.hp = Math.max(1, Math.floor(state.ally.maxHp * 0.4));
        state.ally.dead = false;
        s.allyDownT = 0;
        c.stats.allyDowns += 1;
        const scale = state._combatScale || { hpMul: 1 };
        const remain = Math.min(3, Math.max(2, s.allyEnemies.length || 2));
        s.allyEnemies = [];
        for (let i = 0; i < remain; i += 1) {
          s.allyEnemies.push(mkAllyMob(state, i, remain, (scale.hpMul || 1) * 0.75, state.floor));
        }
        s.allyGoal = s.allyKills + remain;
        s.enemyBullets = [];
        state.ally.x = 160;
        state.ally.y = 270;
        state.ally.stance = "assault";
        state.ally.combatPhase = "approach";
        state.ally.navPath = null;
        clampAllySide(state.ally);
        if (helpers?.setBubble) helpers.setBubble("缓过来了，左边收尾。");
      }
      return;
    }

    const ally = state.ally;
    const enemies = s.allyEnemies;

    clampPlayerSide(state.player);
    clampAllySide(ally);
    (state.enemies || []).forEach((e) => {
      if (e.hp <= 0) return;
      e._splitSide = "player";
      if (e.x < SPLIT_X + 20) e.x = SPLIT_X + 40;
      clampPlayerSide(e);
    });

    if (!enemies.length) {
      if (!s.allyDone) {
        s.allyDone = true;
        if (!s._announcedAlly) {
          s._announcedAlly = true;
          window.GameFx?.floatText?.(ally.x, ally.y - 40, "枭侧肃清", "#9af19b");
          window.__npcPushEvent?.("coop_side_done", { side: "ally" });
          const linesWait = [
            "左边收工了。你倒是麻利点。",
            "我这侧清完了——别让我干等。",
            "搞定。门还得你那边清完才开。",
          ];
          const linesBoth = [
            "两边都肃清了。门在你右边，别傻站着。",
            "行，裂狱合上了。靠右门走。",
            "收工。走门，少废话。",
          ];
          const fallback = s.playerDone
            ? linesBoth[Math.floor(Math.random() * linesBoth.length)]
            : linesWait[Math.floor(Math.random() * linesWait.length)];
          if (helpers?.setBubble) helpers.setBubble(fallback);
          if (typeof helpers?.onAllySideClear === "function") {
            try { helpers.onAllySideClear({ playerDone: !!s.playerDone }); }
            catch (_) { /* ignore */ }
          } else if (helpers?.chat) {
            helpers.chat(fallback);
          }
        }
      }
      return;
    }

    // 左侧敌人移动/射击：由 game.js updateEnemies → GameEnemyAI（A*）处理
    // 这里只：半场夹紧、敌弹碰撞乌枭、清理死亡
    enemies.forEach((e) => {
      e._splitSide = "ally";
      clampAllySide(e);
    });

    for (let i = s.enemyBullets.length - 1; i >= 0; i -= 1) {
      const b = s.enemyBullets[i];
      b.x += b.vx * dt;
      b.y += b.vy * dt;
      b.ttl -= dt;
      if (ally.hp > 0 && Math.hypot(b.x - ally.x, b.y - ally.y) <= (b.radius || 4) + ally.radius) {
        ally.hp -= b.damage || 8;
        s.enemyBullets.splice(i, 1);
        continue;
      }
      // 左侧弹不出中线
      if (b.ttl <= 0 || b.x > SPLIT_X + 16 || b.x < -20 || b.y < -20 || b.y > CANVAS_H + 20) {
        s.enemyBullets.splice(i, 1);
      }
    }

    for (let i = enemies.length - 1; i >= 0; i -= 1) {
      if (enemies[i].hp <= 0) {
        s.allyKills += 1;
        window.GameFx?.burst?.(enemies[i].x, enemies[i].y, "#ffb366", 6);
        enemies.splice(i, 1);
      }
    }
    if (!enemies.length && !s.allyDone) {
      s.allyDone = true;
      if (!s._announcedAlly) {
        s._announcedAlly = true;
        window.GameFx?.floatText?.(ally.x, ally.y - 40, "枭侧肃清", "#9af19b");
        window.__npcPushEvent?.("coop_side_done", { side: "ally" });
        const linesWait = [
          "左边收工了。你倒是麻利点。",
          "我这侧清完了——别让我干等。",
          "搞定。门还得你那边清完才开。",
        ];
        const linesBoth = [
          "两边都肃清了。门在你右边，别傻站着。",
          "行，裂狱合上了。靠右门走。",
          "收工。走门，少废话。",
        ];
        const fallback = s.playerDone
          ? linesBoth[Math.floor(Math.random() * linesBoth.length)]
          : linesWait[Math.floor(Math.random() * linesWait.length)];
        if (helpers?.setBubble) helpers.setBubble(fallback);
        if (typeof helpers?.onAllySideClear === "function") {
          try { helpers.onAllySideClear({ playerDone: !!s.playerDone }); }
          catch (_) { /* ignore */ }
        } else if (helpers?.chat) {
          helpers.chat(fallback);
        }
      }
    }
  }

  // ── 判词分卷 ──────────────────────────────────────────────────────────────

  function enterInfoRoom(state, helpers) {
    const c = ensure(state);
    resetRoom(state);
    c.mode = "info";
    c.roomTag = "duo_info";
    c.info.active = true;
    const n = 4;
    const pillars = [];
    const order = [];
    for (let i = 0; i < n; i += 1) {
      order.push(i);
      pillars.push({
        id: i,
        x: 420 + (i % 2) * 160,
        y: 140 + Math.floor(i / 2) * 160,
        r: 22,
        lit: false,
        label: ["甲", "乙", "丙", "丁"][i],
      });
    }
    // shuffle order
    for (let i = order.length - 1; i > 0; i -= 1) {
      const j = Math.floor(Math.random() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    c.info.pillars = pillars;
    c.info.order = order;
    c.info.nextIdx = 0;
    c.info.solved = false;
    c.info.lastReport = formatInfoReport(c.info);
    if (helpers?.setBubble) {
      helpers.setBubble("判词分卷。顺序只有我看得见——问我，或听我报。");
    }
    if (helpers?.chat) {
      helpers.chat(
        "【判词分卷】场上四根柱要按隐藏顺序点亮。"
        + "聊天问我「顺序」或「怎么点」，我报对了你再去碰柱；点错会扣血重置。",
      );
    }
    window.__npcPushEvent?.("coop_info_enter", {});
  }

  function formatInfoReport(info) {
    if (!info?.order?.length) return "";
    const labels = info.order.map((idx) => info.pillars[idx]?.label || "?");
    return `顺序：${labels.join(" → ")}`;
  }

  function reportInfo(state) {
    const info = ensure(state).info;
    if (!info.active) return null;
    info.lastReport = formatInfoReport(info);
    return info.lastReport;
  }

  function tryTouchPillar(state, px, py) {
    const info = ensure(state).info;
    if (!info.active || info.solved) return null;
    for (const p of info.pillars) {
      if (p.lit) continue;
      if (Math.hypot(px - p.x, py - p.y) > p.r + 14) continue;
      const expect = info.order[info.nextIdx];
      if (p.id === expect) {
        p.lit = true;
        info.nextIdx += 1;
        window.GameFx?.floatText(p.x, p.y - 24, "契", "#9fe4ff");
        if (info.nextIdx >= info.order.length) {
          info.solved = true;
          window.GameFx?.floatText(px, py - 40, "判词吻合", "#ffe08a");
          window.__npcPushEvent?.("coop_info_solved", {});
          return "solved";
        }
        return "ok";
      }
      // 错序
      info.pillars.forEach((x) => { x.lit = false; });
      info.nextIdx = 0;
      state.player.hp = Math.max(1, state.player.hp - 12);
      window.GameFx?.floatText(p.x, p.y - 24, "错序", "#ff8866");
      window.GameFx?.shake?.(3, 0.1);
      return "wrong";
    }
    return null;
  }

  // ── 代行 ──────────────────────────────────────────────────────────────────

  function enterProxy(state, helpers) {
    const c = ensure(state);
    resetRoom(state);
    c.mode = "proxy";
    c.roomTag = "duo_proxy";
    c.proxy.active = true;
    c.commands.slotsLeft = c.commands.slotsMax + 1;
    state.player.silenceT = Math.max(state.player.silenceT || 0, 999);
    state.ally.stance = "assault";
    if (helpers?.setBubble) {
      helpers.setBubble("你缄言了。这房我输出，你指挥——别瞎站。");
    }
    if (helpers?.chat) {
      helpers.chat(
        "【黑签代行】你暂时不能射击。清房靠乌枭；聊天改姿态「突击/守护」。技能按1释放。",
      );
    }
    window.__npcPushEvent?.("coop_proxy_enter", {});
  }

  function exitProxySilence(state) {
    if (ensure(state).proxy.active) {
      state.player.silenceT = 0;
    }
  }

  // ── 契印连携（充能/释放；触发由按键技能槽负责）──────────────────────────

  function tickCombo(state, dt) {
    const c = ensure(state);
    if (!c.combo) return;
    if (c.combo.castCd > 0) c.combo.castCd = Math.max(0, c.combo.castCd - dt);
    // 裂狱/代行：暂停充能，缓慢衰减
    if (c.split.active || c.proxy.active) {
      c.combo.charge = Math.max(0, c.combo.charge - 4 * dt);
      if (c.combo.windowOpen) {
        c.combo.windowT -= dt;
        if (c.combo.windowT <= 0) {
          c.combo.windowOpen = false;
          c.combo.ready = false;
        }
      }
      return;
    }
    const ally = state.ally;
    const player = state.player;
    if (ally.hp <= 0 || player.hp <= 0) return;
    const dist = Math.hypot(ally.x - player.x, ally.y - player.y);
    const fighting = (state.enemies || []).length > 0;
    const mul = c.flags.comboChargeMul || 1;
    if (fighting && dist <= 100) {
      const rate = ally.stance === "guard" ? 26 : 18;
      c.combo.charge = Math.min(c.combo.maxCharge, c.combo.charge + rate * dt * mul);
    } else if (fighting && dist <= 160) {
      c.combo.charge = Math.min(c.combo.maxCharge, c.combo.charge + 10 * dt * mul);
    } else if (!fighting) {
      // 无敌人：几乎不掉
      c.combo.charge = Math.max(0, c.combo.charge - 2 * dt);
    } else {
      c.combo.charge = Math.max(0, c.combo.charge - 6 * dt);
    }
    if (c.combo.charge >= c.combo.maxCharge && !c.combo.windowOpen) {
      c.combo.windowOpen = true;
      c.combo.windowT = 10;
      c.combo.ready = true;
      window.GameFx?.floatText(player.x, player.y - 48, "契印就绪 · 按E", "#ffd9bf");
    }
    if (c.combo.windowOpen) {
      c.combo.windowT -= dt;
      if (c.combo.windowT <= 0) {
        c.combo.windowOpen = false;
        c.combo.ready = false;
        c.combo.charge = Math.floor(c.combo.maxCharge * 0.5);
      }
    }
  }

  function tryComboBurst(state, helpers) {
    const c = ensure(state);
    if (c.split.active) return { ok: false, reason: "split" };
    if (!c.combo.ready && !c.combo.windowOpen) {
      return { ok: false, reason: "not_ready" };
    }
    if ((state.enemies || []).length === 0) return { ok: false, reason: "no_enemies" };
    c.combo.charge = 0;
    c.combo.windowOpen = false;
    c.combo.ready = false;
    c.combo.lastBurstAt = performance.now();
    const dmg = 36 + state.floor * 6;
    state.enemies.forEach((e) => {
      e.hp -= dmg;
      e._lastHitBy = "ally";
      window.GameFx?.burst(e.x, e.y, "#ffd9bf", 8);
    });
    window.GameFx?.floatText(state.player.x, state.player.y - 50, "契印连携!", "#ffd9bf");
    if (helpers?.setBubble) helpers.setBubble("契上了！这波输出别停。");
    if (helpers?.chat) helpers.chat("契印连携打出去了。");
    window.__npcPushEvent?.("coop_combo", { dmg });
    return { ok: true, dmg };
  }

  // ── 结算 ──────────────────────────────────────────────────────────────────

  function settleRoom(state, helpers) {
    const c = ensure(state);
    if (c.mode === "none" && !c.split.active && !c.info.active && !c.proxy.active) return null;
    const elapsed = (performance.now() - (c.stats.startedAt || performance.now())) / 1000;
    let score = 70;
    const wasSplit = c.split.active;
    if (wasSplit) {
      score -= c.stats.allyDowns * 12;
      if (c.split.playerDone && c.split.allyDone) score += 20;
    }
    if (c.info.active && c.info.solved) score += 15;
    if (c.stats.ordersObeyed > 0) score += Math.min(15, c.stats.ordersObeyed * 3);
    score = Math.max(0, Math.min(100, Math.round(score)));

    if (score >= 75) c.trust = Math.min(1, c.trust + 0.04);
    if (score < 45) c.annoyance = Math.min(1, c.annoyance + 0.05);
    if (c.stats.allyDowns > 0) c.annoyance = Math.min(1, c.annoyance + 0.03);

    const result = {
      mode: c.mode,
      roomTag: c.roomTag,
      score,
      elapsed: Math.round(elapsed),
      allyDowns: c.stats.allyDowns,
      orders: c.stats.ordersIssued,
      obeyed: c.stats.ordersObeyed,
      trust: c.trust,
    };
    window.__npcPushEvent?.("coop_clear", result);

    // 关键：结束分房，乌枭回到玩家侧，否则门开了人还锁在右边
    if (wasSplit) endSplit(state, helpers);
    exitProxySilence(state);
    if (c.mode === "info") {
      c.info.active = false;
      c.mode = "none";
    }
    if (c.mode === "proxy") {
      c.proxy.active = false;
      c.mode = "none";
    }
    return result;
  }

  // ── scene / HUD ───────────────────────────────────────────────────────────

  function sceneFields(state) {
    const c = ensure(state);
    const s = c.split;
    const pProg = playerProgress(state);
    const aProg = allyProgress(state);
    return {
      coop_mode: c.mode,
      coop_room_tag: c.roomTag,
      split_active: s.active,
      split_player_progress: Math.round(pProg * 100) / 100,
      split_ally_progress: Math.round(aProg * 100) / 100,
      split_player_done: s.playerDone,
      split_ally_done: s.allyDone,
      command_slots_left: c.commands.slotsLeft,
      command_slots_max: c.commands.slotsMax,
      combo_charge: Math.round(c.combo.charge),
      combo_window: c.combo.windowOpen,
      info_active: c.info.active,
      info_solved: c.info.solved,
      info_report: c.info.lastReport || "",
      proxy_active: c.proxy.active,
      coop_trust: Math.round(c.trust * 100) / 100,
      coop_annoyance: Math.round(c.annoyance * 100) / 100,
    };
  }

  function drawOverlay(ctx, canvas, state) {
    const c = ensure(state);
    // 分房：主画面左右分界 + 右侧敌人实体
    if (c.split.active) {
      const s = c.split;
      syncSplitFlags(state);
      const pProg = playerProgress(state);
      const aProg = allyProgress(state);
      const pAlive = (state.enemies || []).filter((e) => e.hp > 0).length;
      const aAlive = (s.allyEnemies || []).length;
      const dx = s.dividerX || SPLIT_X;

      // 左=乌枭(绿)，右=玩家(蓝，含出口)
      ctx.fillStyle = "rgba(20, 40, 30, 0.22)";
      ctx.fillRect(0, 0, dx, canvas.height);
      ctx.fillStyle = "rgba(30, 40, 60, 0.20)";
      ctx.fillRect(dx, 0, canvas.width - dx, canvas.height);
      const pulse = 0.45 + Math.sin(performance.now() / 180) * 0.25;
      ctx.strokeStyle = `rgba(255, 200, 120, ${pulse})`;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(dx, 0);
      ctx.lineTo(dx, canvas.height);
      ctx.stroke();
      ctx.lineWidth = 1;
      ctx.fillStyle = "#ffd9bf";
      ctx.font = "bold 13px PingFang SC, Arial";
      ctx.textAlign = "center";
      ctx.fillText("┃ 裂 狱 ┃", dx, 28);
      ctx.font = "11px PingFang SC, Arial";
      ctx.fillStyle = "#9af19b";
      ctx.fillText("乌枭战场", dx * 0.5, 48);
      ctx.fillStyle = "#6bc8ff";
      ctx.fillText("你的战场 · 门在最右 →", dx + (canvas.width - dx) * 0.5, 48);
      ctx.textAlign = "left";

      // 进度条（顶栏）
      const bx = 16;
      const by = 58;
      ctx.fillStyle = "rgba(0,0,0,0.6)";
      ctx.fillRect(bx - 4, by - 4, 240, 44);
      ctx.fillStyle = "#ffd9bf";
      ctx.font = "bold 12px PingFang SC, Arial";
      ctx.fillText("两边都清完才开门", bx, by + 10);
      ctx.fillStyle = "#444";
      ctx.fillRect(bx, by + 16, 200, 7);
      ctx.fillStyle = s.playerDone ? "#6bc8ff" : "#3a7aaa";
      ctx.fillRect(bx, by + 16, 200 * pProg, 7);
      ctx.fillStyle = "#444";
      ctx.fillRect(bx, by + 28, 200, 7);
      ctx.fillStyle = s.allyDone ? "#9af19b" : "#3a7a4a";
      ctx.fillRect(bx, by + 28, 200 * aProg, 7);
      ctx.fillStyle = "#cde";
      ctx.font = "10px PingFang SC, Arial";
      ctx.fillText(`你 ${Math.round(pProg * 100)}% 剩${pAlive}${s.playerDone ? " ✓" : ""}`, bx + 204, by + 22);
      ctx.fillText(`枭 ${Math.round(aProg * 100)}% 剩${aAlive}${s.allyDone ? " ✓" : ""}`, bx + 204, by + 34);

      // 右侧敌人（主画面可见）
      s.allyEnemies.forEach((e) => {
        const hpR = Math.max(0, e.hp / Math.max(1, e.maxHp));
        ctx.fillStyle = "#4a2030";
        ctx.beginPath();
        ctx.arc(e.x, e.y, e.radius + 2, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#e07070";
        ctx.beginPath();
        ctx.arc(e.x, e.y, e.radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#1a1010";
        ctx.fillRect(e.x - e.radius, e.y - e.radius - 8, e.radius * 2, 3);
        ctx.fillStyle = "#ffaaaa";
        ctx.fillRect(e.x - e.radius, e.y - e.radius - 8, e.radius * 2 * hpR, 3);
        ctx.fillStyle = "#ffccaa";
        ctx.font = "bold 10px PingFang SC, Arial";
        ctx.textAlign = "center";
        ctx.fillText("狱", e.x, e.y + 3);
        ctx.textAlign = "left";
      });

      // 右侧子弹
      s.allyBullets.forEach((b) => {
        ctx.fillStyle = "#9af19b";
        ctx.beginPath();
        ctx.arc(b.x, b.y, b.radius || 3, 0, Math.PI * 2);
        ctx.fill();
      });
      s.enemyBullets.forEach((b) => {
        ctx.fillStyle = "#ff6e58";
        ctx.beginPath();
        ctx.arc(b.x, b.y, b.radius || 3, 0, Math.PI * 2);
        ctx.fill();
      });

      if (s.playerDone && !s.allyDone) {
        ctx.fillStyle = "rgba(255, 220, 120, 0.9)";
        ctx.font = "bold 14px PingFang SC, Arial";
        ctx.fillText("你侧已清 · 等待左边乌枭…", canvas.width * 0.55, canvas.height - 16);
      } else if (!s.playerDone && s.allyDone) {
        ctx.fillStyle = "rgba(154, 241, 155, 0.9)";
        ctx.font = "bold 14px PingFang SC, Arial";
        ctx.fillText("乌枭已清 · 清完右边开门", canvas.width * 0.55, canvas.height - 16);
      } else if (s.playerDone && s.allyDone) {
        ctx.fillStyle = "rgba(255, 230, 160, 0.95)";
        ctx.font = "bold 14px PingFang SC, Arial";
        ctx.fillText("两边肃清 · 靠最右侧门前进 →", canvas.width * 0.45, canvas.height - 16);
      }
    }

    // 判词柱
    if (c.info.active) {
      c.info.pillars.forEach((p) => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = p.lit ? "rgba(120, 200, 255, 0.55)" : "rgba(80, 60, 100, 0.7)";
        ctx.fill();
        ctx.strokeStyle = p.lit ? "#9fe4ff" : "#8866aa";
        ctx.stroke();
        ctx.fillStyle = "#eee";
        ctx.font = "bold 12px PingFang SC, Arial";
        ctx.textAlign = "center";
        ctx.fillText(p.label, p.x, p.y + 4);
        ctx.textAlign = "left";
      });
    }

    // 技能条改由 GameSkills.drawSlot 绘制

    // 姿态口令槽（始终显示）
    const slots = c.commands.slotsLeft;
    const max = c.commands.slotsMax;
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(canvas.width - 118, 8, 108, 36);
    ctx.font = "bold 12px PingFang SC, Arial";
    ctx.fillStyle = slots > 0 ? "#ffd9bf" : "#ff8866";
    ctx.fillText(`姿态 ${slots}/${max}`, canvas.width - 108, 26);
    ctx.font = "9px PingFang SC, Arial";
    ctx.fillStyle = "#a89888";
    ctx.fillText("对话：守护/突击", canvas.width - 108, 38);
  }

  function tick(state, dt, helpers) {
    ensure(state);
    tickCds(state, dt);
    // 技能充能由 GameSkills.tickEquipped 统一调（避免双 tick）
    // tickCombo 仍可被技能层调用
    if (isSplit(state)) tickSplitAllyCombat(state, dt, helpers);
    if (isInfo(state) && state.player) {
      tryTouchPillar(state, state.player.x, state.player.y);
    }
  }

  function onEnterRoom(state, room, helpers) {
    ensure(state);
    resetRoom(state);
    if (!room) return;
    if (room.type === "duo_split" || room.coop === "split") {
      // 等主房敌人生成后再 enter
      setTimeout(() => enterSplitMirror(state, room, helpers), 0);
      return;
    }
    if (room.type === "duo_info" || room.coop === "info") {
      enterInfoRoom(state, helpers);
      return;
    }
    if (room.type === "duo_proxy" || room.coop === "proxy") {
      enterProxy(state, helpers);
    }
  }

  function applyBlessingFlags(state, blessingId) {
    const c = ensure(state);
    if (blessingId === "ally_split_ward") c.flags.splitEnemyMul = Math.max(0.7, c.flags.splitEnemyMul * 0.85);
    if (blessingId === "ally_command_seal") c.flags.commandSlotsBonus = (c.flags.commandSlotsBonus || 0) + 1;
    if (blessingId === "pact_combo_spark") c.flags.comboChargeMul = (c.flags.comboChargeMul || 1) * 1.5;
  }

  window.GameCoop = {
    defaultCoop,
    ensure,
    resetRoom,
    isSplit,
    isInfo,
    isProxy,
    bothSidesClear,
    roomClearGate,
    endSplit,
    syncSplitFlags,
    playerProgress,
    allyProgress,
    parseLocalCommand,
    canSpendSlot,
    spendSlot,
    enterSplitMirror,
    enterInfoRoom,
    enterProxy,
    onPlayerKill,
    tickSplitAllyCombat,
    reportInfo,
    tryTouchPillar,
    tryComboBurst,
    tickCombo,
    settleRoom,
    sceneFields,
    drawOverlay,
    tick,
    onEnterRoom,
    applyBlessingFlags,
    CMD_IDS,
    formatInfoReport,
  };
})();
