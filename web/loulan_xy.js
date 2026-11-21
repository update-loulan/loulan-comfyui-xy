import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const SAMPLER = "LoulanXYSampler";
const SCHEDULER = "LoulanXYScheduler";
const definitions = new Map();
const panels = new Map();
let reviewWindow;
let reviewNode;
let refreshTimer;

const samplerFamilies = [
  ["Euler", /euler/i], ["RES", /(^|[/_])res[_\d]/i], ["DPM++", /dpm(?:pp|\+\+)/i],
  ["RES · 2m/3m/2s/3s/5s/6s", /(^|\/)res_(?:2m|3m|2s|3s|5s|6s)(?:_ode)?$/i],
  ["RES · ODE", /(^|[/_])res_.*_ode($|_)/i],
  ["RES · 非 ODE", /^(?!.*_ode(?:_|$)).*(?:^|[/_])res[_\d]/i],
  ["RES · multistep", /res_multistep/i], ["RES · 2阶", /res_2[ms]/i],
  ["RES · 3阶", /res_3[ms]/i], ["DPM", /dpm_(?!pp)/i], ["Heun", /(^|\/)(?:exp_)?heun(?:pp|_|$)/i],
  ["DEIS", /deis/i], ["RK", /(^|[/_])rk($|_)/i], ["UniPC", /uni_?pc/i],
  ["SDE", /sde/i], ["Ancestral", /ancestral/i], ["CFG++", /cfg_?pp/i],
];
const schedulerFamilies = [
  ["Uniform", /uniform/i], ["Beta", /beta/i], ["Karras", /karras/i],
  ["Exponential", /exp/i], ["Linear", /linear/i], ["Bong", /bong/i],
];

const typeOf = (node) => node?.comfyClass ?? node?.type;
const isAxis = (node) => [SAMPLER, SCHEDULER].includes(typeOf(node));
const widget = (node, name) => node?.widgets?.find((item) => item.name === name);
const value = (node, name) => widget(node, name)?.value;
const unique = (items) => [...new Set(items)];
const readSelection = (node) => Array.isArray(value(node, "selection")) ? unique(value(node, "selection")) : [];

function choiceNames(node) {
  const input = widget(node, "selection");
  const options = input?.inputSpec?.options ?? input?.options?.values ?? definitions.get(typeOf(node))?.options;
  return Array.isArray(options) ? unique(options.filter((name) => typeof name === "string")) : [];
}

function setValue(node, name, next) {
  const item = widget(node, name);
  if (!item) return;
  item.value = next;
  // Component widgets call their callback in the value setter.
  if (!item.component) item.callback?.call(item, next, app.canvas, node);
}

function edit(node, mutation) {
  const graph = node.graph;
  graph?.beforeChange?.();
  try {
    mutation();
  } finally {
    graph?.afterChange?.();
    graph?.change?.();
    node.setDirtyCanvas?.(true, true);
    refreshPanels();
  }
}

function selectNames(node, names) {
  const picked = new Set(names);
  edit(node, () => setValue(node, "selection", choiceNames(node).filter((name) => picked.has(name))));
}

function families(node) {
  const choices = choiceNames(node);
  const groups = typeOf(node) === SAMPLER ? samplerFamilies : schedulerFamilies;
  const result = groups.map(([label, pattern]) => [label, choices.filter((name) => pattern.test(name))])
    .filter(([, names]) => names.length);
  const included = new Set(result.flatMap(([, names]) => names));
  const remaining = choices.filter((name) => !included.has(name));
  if (remaining.length) result.push(["其他", remaining]);
  return result;
}

function otherAxis(node) {
  const index = node.inputs?.findIndex((input) => input.name === "other_axis") ?? -1;
  if (index < 0 || node.inputs[index].link == null) return [value(node, "fixed")];
  let source = node.getInputNode?.(index);
  const seen = new Set();
  while (source && !isAxis(source) && !seen.has(source)) {
    seen.add(source);
    if (!/reroute/i.test(typeOf(source) ?? "") || source.inputs?.length !== 1) return null;
    source = source.getInputNode?.(0);
  }
  return isAxis(source) ? readSelection(source) : null;
}

function validPairs(pairs) {
  return Array.isArray(pairs) && pairs.every((pair) => Array.isArray(pair) && pair.length === 2
    && pair.every((name) => typeof name === "string" && name.length > 0));
}

function validSeed(seed) {
  return typeof seed === "string" && /^\d+$/.test(seed)
    && seed.replace(/^0+/, "").length <= 20 && BigInt(seed) < 2n ** 64n;
}

function preview(node) {
  const selected = readSelection(node);
  const label = typeOf(node) === SAMPLER ? "采样器" : "调度器";
  if (value(node, "mode") === "selection") return `选择模式 · 输出 ${selected.length} 个${label}`;
  let favorites;
  try { favorites = JSON.parse(String(value(node, "favorites") || "").trim() || "[]"); }
  catch { return "收藏配对 JSON 有误"; }
  if (!validPairs(favorites)) return "收藏配对应为 [采样器, 调度器] 的列表";
  const seedText = String(value(node, "seeds") || "").trim();
  const seeds = seedText ? seedText.split(/[,，;；\s]+/) : [];
  if (seeds.some((seed) => !validSeed(seed))) return "复测种子需要是 0 至 2⁶⁴−1 的整数";
  const seedCount = seeds.length ? new Set(seeds.map((seed) => BigInt(seed))).size : 1;
  const baseline = [value(node, "baseline_sampler"), value(node, "baseline_scheduler")];
  let count, containsBaseline, description;
  if (favorites.length) {
    const pairs = new Set(favorites.map((pair) => JSON.stringify(pair)));
    count = pairs.size;
    containsBaseline = pairs.has(JSON.stringify(baseline));
    description = `${count} 个收藏配对`;
  } else {
    const other = otherAxis(node);
    if (!other) return `已选 ${selected.length} 个${label} · 对轴数量在执行时确定`;
    const [samplers, schedulers] = typeOf(node) === SAMPLER ? [selected, other] : [other, selected];
    count = samplers.length * schedulers.length;
    containsBaseline = samplers.includes(baseline[0]) && schedulers.includes(baseline[1]);
    description = `${samplers.length} 采样器 × ${schedulers.length} 调度器`;
  }
  if (count === 0) return "0 张 · 请至少选择一个采样器和一个调度器";
  if (value(node, "use_baseline") && !containsBaseline) {
    count++;
    description += " + 1 基准";
  }
  return `${description} × ${seedCount} 种子 = ${count * seedCount} 张`;
}

function element(tag, content) {
  const el = document.createElement(tag);
  if (content != null) el.textContent = content;
  return el;
}

function button(label, action, title) {
  const b = element("button", label);
  b.type = "button";
  if (title) b.title = title;
  b.addEventListener("click", action);
  return b;
}

function styling() {
  if (document.getElementById("loulan-xy-styles")) return;
  const style = element("style");
  style.id = "loulan-xy-styles";
  style.textContent = `
    .loulan-xy-panel,.loulan-xy-dialog { font:13px/1.4 system-ui,sans-serif; color:var(--input-text,#eee) }
    .loulan-xy-panel { width:100%; box-sizing:border-box; display:flex; flex-direction:column; gap:4px; padding:4px 2px; overflow:auto }
    .loulan-xy-panel output { font-variant-numeric:tabular-nums; font-weight:600; overflow-wrap:anywhere }
    .loulan-xy-tools { display:flex; align-items:center; gap:4px; flex-wrap:wrap }
    .loulan-xy-panel .loulan-xy-families { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); flex-shrink:0 }
    .loulan-xy-families button { min-width:0; overflow:hidden; text-overflow:ellipsis }
    .loulan-xy-panel button,.loulan-xy-dialog button,.loulan-xy-dialog select,.loulan-xy-dialog input[type=search],.loulan-xy-dialog textarea {
        background:var(--comfy-input-bg,#34363a); color:inherit; border:1px solid var(--border-color,#62666d); border-radius:5px; padding:3px 7px; font:inherit }
    .loulan-xy-panel button,.loulan-xy-dialog button { cursor:pointer; white-space:nowrap }
    .loulan-xy-panel button:hover,.loulan-xy-dialog button:hover { border-color:var(--input-text,#ddd) }
    .loulan-xy-dialog { box-sizing:border-box; width:min(620px,92vw); max-height:83vh; overflow:auto; background:var(--comfy-menu-bg,#24262a); border:1px solid var(--border-color,#5e626a); border-radius:10px; padding:16px; margin:auto; box-shadow:0 12px 60px #0008 }
    .loulan-xy-dialog::backdrop { background:#0008 }
    .loulan-xy-dialog[open] { display:grid; gap:10px }
    .loulan-xy-dialog header { display:flex; align-items:center; justify-content:space-between; gap:10px }
    .loulan-xy-dialog header strong { font-size:16px }
    .loulan-xy-dialog input[type=search] { flex:1; min-width:80px }
    .loulan-xy-dialog textarea { min-height:80px; width:100%; box-sizing:border-box }
    .loulan-xy-dialog .loulan-xy-list { overflow:auto; max-height:43vh; min-height:60px; display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,215px),1fr)); align-content:start }
    .loulan-xy-dialog label { display:flex; align-items:center; gap:6px; padding:5px; overflow-wrap:anywhere; cursor:pointer }
    .loulan-xy-dialog label:hover { background:#80808026 }
    .loulan-xy-dialog input[type=checkbox] { accent-color:#6698e3 }
  `;
  document.head.append(style);
}

function picker(node) {
  const choices = choiceNames(node);
  const groups = new Map(families(node));
  const dlg = element("dialog");
  dlg.className = "loulan-xy-dialog";
  const header = element("header");
  header.append(element("strong", typeOf(node) === SAMPLER ? "选择采样器" : "选择调度器"), button("完成", () => dlg.close()));
  const search = element("input");
  search.type = "search";
  search.placeholder = "搜索名称";
  search.setAttribute("aria-label", "搜索名称");
  const filter = element("select");
  filter.setAttribute("aria-label", "筛选分组");
  for (const name of ["全部分组", ...groups.keys()]) filter.append(new Option(name, name));
  const bar = element("div");
  bar.className = "loulan-xy-tools";
  bar.append(search, filter);
  const actions = element("div");
  actions.className = "loulan-xy-tools";
  const list = element("div");
  list.className = "loulan-xy-list";
  const count = element("output");
  const visible = () => (groups.get(filter.value) ?? choices)
    .filter((name) => name.toLowerCase().includes(search.value.trim().toLowerCase()));
  function render() {
    const selected = new Set(readSelection(node));
    const filtered = visible();
    count.textContent = `已选 ${selected.size} / ${choices.length}，当前显示 ${filtered.length}`;
    list.replaceChildren();
    for (const name of filtered) {
      const row = element("label");
      const box = element("input");
      box.type = "checkbox";
      box.checked = selected.has(name);
      box.addEventListener("change", () => {
        const next = new Set(readSelection(node));
        if (box.checked) next.add(name); else next.delete(name);
        selectNames(node, next);
        count.textContent = `已选 ${next.size} / ${choices.length}，当前显示 ${filtered.length}`;
      });
      row.append(box, document.createTextNode(name));
      list.append(row);
    }
    if (!filtered.length) list.append(element("span", "没有匹配项"));
  }
  actions.append(button("勾选显示项", () => { selectNames(node, [...readSelection(node), ...visible()]); render(); }),
    button("取消显示项", () => {
      const remove = new Set(visible());
      selectNames(node, readSelection(node).filter((name) => !remove.has(name))); render();
    }), count);
  search.addEventListener("input", render);
  filter.addEventListener("change", render);
  dlg.append(header, bar, actions, list);
  document.body.append(dlg);
  dlg.addEventListener("close", () => dlg.remove(), { once: true });
  render();
  dlg.showModal();
  search.focus();
}

function moreFamilies(node, groups) {
  const dlg = element("dialog");
  dlg.className = "loulan-xy-dialog";
  const header = element("header");
  header.append(element("strong", "仅选择一个家族"), button("关闭", () => dlg.close()));
  const options = element("div");
  options.className = "loulan-xy-tools";
  for (const [label, names] of groups) {
    options.append(button(`${label} (${names.length})`, () => {
      selectNames(node, names);
      dlg.close();
    }, names.join(", ")));
  }
  dlg.append(header, options);
  document.body.append(dlg);
  dlg.addEventListener("close", () => dlg.remove(), { once: true });
  dlg.showModal();
}

function acceptFavorites(node, data) {
  if (!validPairs(data?.pairs) || !data.pairs.length || !Array.isArray(data.seeds)
    || !data.seeds.every(validSeed)) throw new Error("需要有效的收藏配对与种子列表。");
  edit(node, () => {
    setValue(node, "favorites", JSON.stringify(data.pairs));
    setValue(node, "seeds", unique(data.seeds).join(", "));
    setValue(node, "mode", "test");
  });
}

function importFavorites(node) {
  const dlg = element("dialog");
  dlg.className = "loulan-xy-dialog";
  const box = element("textarea");
  box.placeholder = "粘贴结果查看器复制的复测配置";
  const error = element("output");
  const tools = element("div");
  tools.className = "loulan-xy-tools";
  tools.append(button("应用", () => {
    try {
      acceptFavorites(node, JSON.parse(box.value));
      dlg.close();
    } catch { error.textContent = "请粘贴包含 pairs 和 seeds 的完整复测配置。"; }
  }), button("取消", () => dlg.close()));
  dlg.append(element("strong", "导入收藏复测"), box, error, tools);
  document.body.append(dlg);
  dlg.addEventListener("close", () => dlg.remove(), { once: true });
  dlg.showModal();
  box.focus();
}

function openReview(node) {
  const route = "/loulan-xy/review";
  const url = new URL(api.fileURL?.(route) ?? (api.api_base ?? "") + route, location.origin);
  const run = node.properties?.loulan_xy_run;
  if (run) url.searchParams.set("run", run);
  reviewNode = node;
  reviewWindow = window.open(url.href, "loulan_xy_review");
}

window.addEventListener("message", (event) => {
  if (event.origin !== location.origin || !reviewWindow || event.source !== reviewWindow
    || !["loulan-xy-favorites", "loulan-xy:retest"].includes(event.data?.type)) return;
  const node = reviewNode && panels.has(reviewNode) ? reviewNode
    : Object.values(app.canvas?.selected_nodes ?? {}).find(isAxis);
  if (!node) return;
  try { acceptFavorites(node, event.data); }
  catch (error) { console.warn("[loulan-xy]", error.message); }
});

function refreshPanels() {
  for (const [node, p] of panels) {
    if (!node.graph && !p.root.isConnected) continue;
    const text = p.progress || preview(node);
    if (p.summary.textContent !== text) p.summary.textContent = text;
    const label = `勾选 (${readSelection(node).length}/${choiceNames(node).length})`;
    if (p.pick.textContent !== label) p.pick.textContent = label;
    p.clearFavorites.hidden = !String(value(node, "favorites") || "").trim();
  }
}

function install(node, definition) {
  if (panels.has(node) || !widget(node, "selection")) return;
  const initial = definition?.default;
  if (!readSelection(node).length && Array.isArray(initial)) setValue(node, "selection", [...initial]);
  styling();
  const root = element("div");
  root.className = "loulan-xy-panel";
  const summary = element("output", preview(node));
  summary.title = "dpm_adaptive 和 dpm_fast 主要使用起止 sigma；端点相同的调度器可能得到相同结果。";
  const row = element("div");
  row.className = "loulan-xy-tools";
  const pick = button("勾选", () => picker(node), "搜索并勾选多个项目");
  const clearFavorites = button("退出收藏复测", () => edit(node, () => {
    setValue(node, "favorites", "");
    setValue(node, "seeds", "");
  }));
  row.append(pick, button("全选", () => selectNames(node, choiceNames(node))), button("清空", () => selectNames(node, [])),
    button("查看结果", () => openReview(node)), button("导入复测", () => importFavorites(node)), clearFavorites);
  const presets = element("div");
  presets.className = "loulan-xy-tools loulan-xy-families";
  const groups = families(node);
  for (const [label, names] of groups.slice(0, 3)) {
    presets.append(button(`${label} (${names.length})`, () => selectNames(node, names), `仅选 ${label}，替换当前选择`));
  }
  if (groups.length > 3) presets.append(button("更多家族", () => moreFamilies(node, groups.slice(3))));
  root.append(summary, row, presets);
  for (const event of ["pointerdown", "wheel", "dblclick"]) root.addEventListener(event, (e) => e.stopPropagation());
  node.addDOMWidget("loulan_xy_controls", "loulan_xy_controls", root, {
    serialize: false, getMinHeight: () => 108, getMaxHeight: () => 132,
  });
  const panel = { root, summary, pick, clearFavorites };
  const attach = () => {
    panels.set(node, panel);
    refreshPanels();
    if (!refreshTimer) refreshTimer = window.setInterval(refreshPanels, 450);
  };
  const added = node.onAdded;
  node.onAdded = function (...args) {
    const result = added?.apply(this, args);
    attach();
    return result;
  };
  if (node.graph) attach();
  const remove = node.onRemoved;
  node.onRemoved = function (...args) {
    panels.delete(this);
    if (!panels.size) { clearInterval(refreshTimer); refreshTimer = null; }
    return remove?.apply(this, args);
  };
}

app.registerExtension({
  name: "loulan.xy.selection",
  setup() {
    api.addEventListener("loulan-xy-progress", ({ detail }) => {
      for (const [node, panel] of panels) {
        if (String(node.id) === String(detail.node_id) && node.graph === app.graph) {
          panel.progress = `运行 ${detail.index}/${detail.total} · ${detail.sampler} / ${detail.scheduler}`;
          panel.summary.textContent = panel.progress;
        }
      }
    });
    for (const event of ["execution_start", "execution_error", "execution_interrupted", "execution_success"]) {
      api.addEventListener(event, () => {
        for (const panel of panels.values()) panel.progress = "";
        refreshPanels();
      });
    }
  },
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (![SAMPLER, SCHEDULER].includes(nodeData.name)) return;
    const spec = nodeData.input?.required?.selection;
    definitions.set(nodeData.name, { ...spec?.[1], options: Array.isArray(spec?.[0]) ? spec[0] : spec?.[1]?.options });
    const prev = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (result) {
      const returned = prev?.apply(this, arguments);
      const run = Array.isArray(result?.run_id) ? result.run_id[0] : result?.run_id;
      if (typeof run === "string" && run) {
        this.properties ??= {};
        this.properties.loulan_xy_run = run;
        const panel = panels.get(this);
        if (panel) panel.progress = "";
        refreshPanels();
      }
      return returned;
    };
  },
  nodeCreated(node) {
    if (isAxis(node)) install(node, definitions.get(typeOf(node)));
  },
});
