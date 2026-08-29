const $ = (id) => document.getElementById(id);

const promptEl = $("prompt");
const sendEl = $("send");
const resultEl = $("result");

async function send() {
  const prompt = promptEl.value.trim();
  if (!prompt) return;

  sendEl.disabled = true;
  sendEl.textContent = "...";
  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });
    const data = await res.json();
    renderResult(data);
    refreshMetrics();
  } catch (e) {
    renderError(e);
  } finally {
    sendEl.disabled = false;
    sendEl.textContent = "Send";
  }
}

function renderResult(d) {
  resultEl.classList.remove("hidden");
  $("answer").textContent = d.response || d.detail || "(no response)";

  const parts = [];
  if (d.cache_hit) {
    parts.push(`<span class="pill hit">${d.cache_type} cache HIT</span>`);
    parts.push(`<span class="pill ${d.model}">${d.model}</span> <span>(original answer)</span>`);
  } else {
    parts.push(`<span class="pill ${d.model}">routed &rarr; ${d.model}</span>`);
    if (d.router_confidence != null) {
      parts.push(`confidence ${d.router_confidence.toFixed(2)}`);
    }
    if (d.router && d.router.escalated) {
      parts.push(`<span>escalated from ${d.router.predicted}</span>`);
    }
    if (d.router && d.router.using_fallback) {
      parts.push(`<span title="srmty/meridian not loaded">heuristic fallback</span>`);
    }
  }
  parts.push(`est. cost $${fmtCost(d.estimated_cost)}`);
  parts.push(`baseline $${fmtCost(d.baseline_cost)}`);
  parts.push(`saved $${fmtCost(d.cost_saved)}`);
  $("meta").innerHTML = parts.join("");
}

function renderError(e) {
  resultEl.classList.remove("hidden");
  $("answer").textContent = "Request failed: " + e;
  $("meta").innerHTML = "";
}

function fmtCost(x) {
  if (x == null) return "0";
  return Number(x).toFixed(6);
}

async function refreshMetrics() {
  try {
    const m = await (await fetch("/metrics")).json();

    $("m_total").textContent = m.total_requests;
    $("m_hitrate").textContent = (m.cache_hit_rate * 100).toFixed(0) + "%";
    $("m_exact").textContent = m.exact_cache_hits;
    $("m_semantic").textContent = m.semantic_cache_hits;
    $("m_misses").textContent = m.cache_misses;
    $("m_false").textContent = m.semantic_false_hits;

    const r = m.routing_distribution;
    const total = (r.small + r.medium + r.large) || 1;
    $("c_small").textContent = r.small;
    $("c_medium").textContent = r.medium;
    $("c_large").textContent = r.large;
    $("bar_small").style.width = (r.small / total * 100) + "%";
    $("bar_medium").style.width = (r.medium / total * 100) + "%";
    $("bar_large").style.width = (r.large / total * 100) + "%";

    $("m_cost").textContent = "$" + fmtCost(m.estimated_cost);
    $("m_baseline").textContent = "$" + fmtCost(m.baseline_cost);
    $("m_saved").textContent = "$" + fmtCost(m.cost_saved);
    $("m_savedpct").textContent = m.saving_percentage.toFixed(1) + "%";
    $("m_conf").textContent = m.avg_router_confidence.toFixed(2);
    $("m_latency").textContent = m.avg_latency_ms.toFixed(0) + " ms";
  } catch (e) {
  }
}

async function runAdversarial() {
  const btn = $("run-adv");
  const out = $("adv-out");
  btn.disabled = true;
  try {
    const s = await (await fetch("/adversarial-test")).json();
    out.classList.remove("hidden");
    const lines = s.results.map(
      (r) => `${r.false_hit ? "FALSE HIT" : "ok       "}  [${r.reason}]  "${r.adversarial_query}"` +
             (r.verifier_score != null ? `  score=${r.verifier_score.toFixed(3)}` : "")
    );
    out.textContent =
      `verifier: ${s.verifier_model}\nthreshold: ${s.semantic_threshold}\n` +
      `false hits: ${s.false_hits} / ${s.pairs}\n\n` + lines.join("\n");
    refreshMetrics();
  } catch (e) {
    out.classList.remove("hidden");
    out.textContent = "failed: " + e;
  } finally {
    btn.disabled = false;
  }
}

sendEl.addEventListener("click", send);
promptEl.addEventListener("keydown", (e) => { if (e.key === "Enter") send(); });
$("run-adv").addEventListener("click", runAdversarial);
document.querySelectorAll(".ex").forEach((b) =>
  b.addEventListener("click", () => { promptEl.value = b.dataset.q; send(); })
);

refreshMetrics();
setInterval(refreshMetrics, 3000);
