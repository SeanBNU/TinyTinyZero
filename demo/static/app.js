// TinyTinyZero demo frontend
(function () {
 const $ = (id) => document.getElementById(id);
 let currentProblem = { numbers: [3, 12, 7, 25], target: 42 };

 // --- free-form chat ------------------------------------------------------
 document.querySelectorAll(".preset").forEach(btn => {
 btn.addEventListener("click", () => {
 $("chat-input").value = btn.dataset.msg;
 runChat();
 });
 });
 $("chat-input").addEventListener("keydown", (e) => {
 if ((e.ctrlKey || e.metaKey) && e.key === "Enter") runChat();
 });
 $("chat-send").addEventListener("click", runChat);

 function runChat() {
 const msg = ($("chat-input").value || "").trim();
 if (!msg) return;
 const btn = $("chat-send");
 btn.disabled = true;
 const out = $("chat-text");
 const stats = document.querySelector(".chat-stats");
 out.hidden = false;
 stats.hidden = false;
 out.textContent = "";
 $("chat-rate").textContent = "-- tok/s";
 $("chat-elapsed").textContent = "0.0s";

 const params = new URLSearchParams({
 msg, temperature: "0.8",
 seed: String(Math.floor(Math.random() * 1e6)),
 max_new: "180",
 });
 const es = new EventSource("/api/chat_stream?" + params.toString());
 let totalChars = 0;
 es.addEventListener("token", (ev) => {
 const d = JSON.parse(ev.data);
 out.textContent += d.chunk;
 out.scrollTop = out.scrollHeight;
 totalChars += d.chunk.length;
 const elapsed = d.elapsed || 0;
 $("chat-elapsed").textContent = `${elapsed.toFixed(1)}s`;
 if (elapsed > 0.1) {
 $("chat-rate").textContent = `${((totalChars / 4) / elapsed).toFixed(1)} tok/s`;
 }
 });
 es.addEventListener("done", (ev) => {
 const d = JSON.parse(ev.data);
 $("chat-elapsed").textContent = `${d.elapsed.toFixed(1)}s`;
 es.close();
 btn.disabled = false;
 });
 es.onerror = () => { es.close(); btn.disabled = false; };
 }

 // --- status / model loading ------------------------------------------------
 async function refreshStatus() {
 try {
 const r = await fetch("/api/status");
 const s = await r.json();
 const dot = $("status-dot");
 const txt = $("status-text");
 dot.className = "dot";
 if (s.loading) { dot.classList.add("loading"); txt.textContent = "loading model..."; }
 else if (s.loaded) {
 dot.classList.add("loaded");
 txt.textContent = `ready · ${s.model_id} · ${s.device}`;
 $("load-btn").disabled = true;
 }
 else if (s.error) { dot.classList.add("error"); txt.textContent = `error: ${s.error}`; }
 else { txt.textContent = "idle. click Load to pull Qwen2.5-0.5B"; }
 } catch (e) { /* ignore */ }
 }
 setInterval(refreshStatus, 1500);
 refreshStatus();

 $("load-btn").addEventListener("click", async () => {
 $("load-btn").disabled = true;
 await fetch("/api/load", { method: "POST", headers: {"Content-Type":"application/json"}, body: "{}" });
 });

 // --- problem ---------------------------------------------------------------
 async function newProblem() {
 const r = await fetch("/api/new_problem");
 const p = await r.json();
 currentProblem = { numbers: p.numbers, target: p.target };
 $("p-numbers").textContent = "[" + p.numbers.join(", ") + "]";
 $("p-target").textContent = p.target;
 }
 $("new-problem").addEventListener("click", newProblem);
 // auto-populate a fresh problem on page load so the section isn't empty
 newProblem();

 // --- single stream ---------------------------------------------------------
 $("temp").addEventListener("input", e => { $("temp-val").textContent = e.target.value; });

 $("stream-btn").addEventListener("click", () => {
 const btn = $("stream-btn");
 btn.disabled = true;
 $("stream-text").textContent = "";
 $("big-reward").textContent = "…";
 $("big-reward").className = "big-reward";
 $("reward-detail").textContent = "generating…";

 const params = new URLSearchParams({
 numbers: JSON.stringify(currentProblem.numbers),
 target: String(currentProblem.target),
 temperature: $("temp").value,
 seed: String(Math.floor(Math.random() * 1e6)),
 });
 const es = new EventSource("/api/stream?" + params.toString());
 es.addEventListener("token", (ev) => {
 const data = JSON.parse(ev.data);
 $("stream-text").textContent += data.chunk;
 $("stream-text").scrollTop = $("stream-text").scrollHeight;
 });
 es.addEventListener("done", (ev) => {
 const data = JSON.parse(ev.data);
 const rw = data.reward;
 const br = $("big-reward");
 br.textContent = rw.reward === 1.0 ? "1.0" : "0.0";
 br.className = "big-reward " + (rw.reward === 1.0 ? "good" : "bad");
 let detail;
 if (rw.reward === 1.0) {
 detail = `✔ correct · answer = ${rw.answer} = ${rw.value}`;
 } else {
 detail = `✗ ${rw.reason}${rw.answer ? " · answer = " + rw.answer : ""}${rw.value != null ? " = " + rw.value : ""}`;
 if (rw.value == null && rw.lenient_value != null) {
 detail += ` · lenient eval: ${rw.lenient_value}`;
 }
 }
 $("reward-detail").textContent = detail;
 es.close();
 btn.disabled = false;
 });
 es.onerror = () => { es.close(); btn.disabled = false; };
 });

 // --- GRPO step -------------------------------------------------------------
 $("grpo-btn").addEventListener("click", async () => {
 const btn = $("grpo-btn");
 btn.disabled = true;
 $("grpo-timing").textContent = "sampling…";
 const group = $("grpo-group");
 group.innerHTML = "";

 const G = Number($("g-size").value) || 4;
 const params = new URLSearchParams({
 numbers: JSON.stringify(currentProblem.numbers),
 target: String(currentProblem.target),
 G: String(G),
 temperature: "1.0",
 seed: String(Math.floor(Math.random() * 1e6)),
 });

 try {
 const r = await fetch("/api/grpo_step?" + params.toString());
 const data = await r.json();
 renderGroup(data);
 $("grpo-timing").textContent = `sampled G=${G} in ${data.stats.gen_seconds.toFixed(1)}s on ${data.stats.device}`;
 } catch (e) {
 $("grpo-timing").textContent = "error: " + e.message;
 } finally {
 btn.disabled = false;
 }
 });

 function renderGroup(data) {
 const group = $("grpo-group");
 group.innerHTML = "";
 data.completions.forEach(c => {
 const card = document.createElement("div");
 card.className = "grpo-card " + (c.reward === 1.0 ? "good" : "bad");
 const adv = (c.advantage ?? 0).toFixed(3);
 const tag = c.reward === 1.0 ? '<span class="r1">r=1</span>' : '<span class="r0">r=0</span>';
 const lenient = (c.value == null && c.lenient_value != null) ? ` · lenient: ${c.lenient_value}` : "";
 card.innerHTML = `
 <div class="head"><span>completion #${c.idx}</span><span>seed ${c.seed}</span></div>
 <div class="body">${escapeHtml(c.text).slice(0, 600)}</div>
 <div class="foot">
 <span>${tag} · ${escapeHtml(c.reward_reason)}${c.answer ? " · " + escapeHtml(c.answer) : ""}${lenient}</span>
 <span class="adv">A = ${adv}</span>
 </div>`;
 group.appendChild(card);
 });

 const s = data.stats;
 $("grpo-stats").innerHTML =
 `μ (mean reward) = <span class="mono">${s.mean_reward.toFixed(3)}</span><br>
 σ (std) = <span class="mono">${s.std_reward.toFixed(3)}</span><br>
 solved = <span class="mono">${s.solved} / ${s.total}</span>`;

 $("grpo-adv").innerHTML = data.completions
 .map(c => `A<sub>${c.idx}</sub> = <span class="mono">${(c.advantage ?? 0).toFixed(3)}</span>`)
 .join("<br>");

 const meanAdv = data.completions.reduce((acc, c) => acc + (c.advantage ?? 0), 0) / data.completions.length;
 $("grpo-loss").innerHTML =
 `ε = ${s.epsilon}, β (KL coef) = ${s.kl_coef}<br>
 at ρ=1: L_policy = −mean(A) = <span class="mono">${(-meanAdv).toFixed(3)}</span><br>
 simulated KL(π‖π_ref) ≈ <span class="mono">${s.simulated_kl.toFixed(4)}</span>`;
 }

 function escapeHtml(s) {
 return (s || "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
 }

 // --- TinyZero curve --------------------------------------------------------
 (async () => {
 try {
 const r = await fetch("/api/tinyzero_curves");
 const data = await r.json();
 drawCurve(data);
 } catch (e) { /* ignore */ }
 })();

 function drawCurve(data) {
 const canvas = $("curve");
 const dpr = window.devicePixelRatio || 1;
 canvas.width = canvas.clientWidth * dpr;
 canvas.height = canvas.clientHeight * dpr;
 const ctx = canvas.getContext("2d");
 ctx.scale(dpr, dpr);

 const W = canvas.clientWidth, H = canvas.clientHeight;
 const padL = 50, padR = 20, padT = 20, padB = 34;

 ctx.clearRect(0, 0, W, H);
 ctx.fillStyle = "#8d97a6";
 ctx.font = "11px -apple-system, sans-serif";

 // axes
 ctx.strokeStyle = "#2a3140";
 ctx.lineWidth = 1;
 for (let i = 0; i <= 5; i++) {
 const y = padT + (H - padT - padB) * (i / 5);
 ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
 const v = (5 - i) * 0.1;
 ctx.fillText(v.toFixed(1), 8, y + 3);
 }
 const maxStep = data.points[data.points.length - 1][0];
 for (let i = 0; i <= 5; i++) {
 const x = padL + (W - padL - padR) * (i / 5);
 const step = Math.round((i / 5) * maxStep);
 ctx.fillText(String(step), x - 8, H - 12);
 }
 ctx.fillStyle = "#8d97a6";
 ctx.fillText("train reward", 8, padT - 6);
 ctx.fillText("step", W - padR - 30, H - 12);

 // curve
 ctx.beginPath();
 ctx.strokeStyle = "#7cc4ff";
 ctx.lineWidth = 2;
 data.points.forEach((pt, i) => {
 const [s, r] = pt;
 const x = padL + (W - padL - padR) * (s / maxStep);
 const y = padT + (H - padT - padB) * (1 - Math.min(1, r / 0.5));
 if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
 });
 ctx.stroke();

 // annotation
 const aha = data.annotations && data.annotations.find(a => a.label.includes("aha"));
 if (aha) {
 const x = padL + (W - padL - padR) * (aha.step / maxStep);
 const y = padT + (H - padT - padB) * (1 - Math.min(1, aha.reward / 0.5));
 ctx.beginPath();
 ctx.fillStyle = "#b388ff";
 ctx.arc(x, y, 5, 0, Math.PI * 2);
 ctx.fill();
 ctx.fillStyle = "#e7ecf3";
 ctx.fillText("← \"aha moment\"", x + 10, y - 6);
 }
 }

 // --- Aha gallery -----------------------------------------------------------
 (async () => {
 try {
 const r = await fetch("/api/aha_gallery");
 const data = await r.json();
 const g = $("aha-gallery");
 data.items.forEach(item => {
 const card = document.createElement("div");
 card.className = "aha-card";
 card.innerHTML = `
 <div class="tag">${escapeHtml(item.stage)} · ${escapeHtml(item.size)}</div>
 <h3>${escapeHtml(item.title)}</h3>
 <pre>${escapeHtml(item.trace)}</pre>`;
 g.appendChild(card);
 });
 } catch (e) { /* ignore */ }
 })();
})();
