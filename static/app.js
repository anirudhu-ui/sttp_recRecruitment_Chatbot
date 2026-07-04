const state = {
  threadId: null,
  libraryCount: 0,
  libraryMax: 20,
  stagedFiles: [], // File objects picked, not yet uploaded
};

const el = (id) => document.getElementById(id);

function showError(msg) {
  el("error-banner").textContent = msg || "";
}

function setRailStep(stepName, status) {
  const steps = ["upload", "scan", "score", "questions"];
  const idx = steps.indexOf(stepName);
  document.querySelectorAll(".rail-step").forEach((node) => {
    const nodeIdx = steps.indexOf(node.dataset.step);
    if (nodeIdx < idx) {
      node.classList.add("done");
      node.classList.remove("active");
    } else if (nodeIdx === idx) {
      node.classList.toggle("active", status === "active");
      node.classList.toggle("done", status === "done");
    }
  });
  document.querySelectorAll(".rail-line").forEach((line) => {
    const lineIdx = parseInt(line.dataset.line, 10);
    line.classList.toggle("filled", lineIdx <= idx || (lineIdx === idx + 1 && status === "done"));
  });
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ---------------------------------------------------------------------
// Library status (persistent, shown on load + after every upload)
// ---------------------------------------------------------------------

async function refreshLibrary() {
  try {
    const res = await fetch("/api/library");
    const data = await res.json();
    state.libraryCount = data.count;
    state.libraryMax = data.max;
    el("library-count").textContent = `${data.count} / ${data.max} stored`;
    el("library-list").innerHTML = data.resumes
      .map((r) => `
        <span class="lib-chip">
          ${escapeHtml(r.filename)} · ${r.size_kb}kb
          <button class="lib-remove" data-filename="${escapeHtml(r.filename)}" title="Remove">✕</button>
        </span>
      `)
      .join("") || `<span class="hint" style="margin:0;">No resumes stored yet.</span>`;

    el("library-list").querySelectorAll(".lib-remove").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        const filename = e.target.dataset.filename;
        try {
          const res = await fetch("/api/library/delete", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ filename })
          });
          const delData = await res.json();
          if (!res.ok) throw new Error(delData.error || "Delete failed.");
          await refreshLibrary();
        } catch (err) {
          showError(err.message);
        }
      });
    });

    if (data.count > 0) {
      el("btn-scan").disabled = false;
      el("actions-card").classList.remove("hidden");
    } else {
      el("btn-scan").disabled = true;
    }
  } catch (e) {
    // silent - library status is a nice-to-have, not blocking
  }
}

// ---------------------------------------------------------------------
// Step 1: JD prefill + multi-file staging + upload
// ---------------------------------------------------------------------

async function prefillJD() {
  try {
    const res = await fetch("/api/default-jd");
    const data = await res.json();
    el("jd-text").value = data.jd_text || "";
  } catch (e) {}
}

function wireDropzone() {
  const zone = el("dropzone");
  const input = el("resume-input");

  zone.addEventListener("click", () => input.click());
  ["dragenter", "dragover"].forEach((evt) =>
    zone.addEventListener(evt, (e) => { e.preventDefault(); zone.classList.add("drag-over"); })
  );
  ["dragleave", "drop"].forEach((evt) =>
    zone.addEventListener(evt, (e) => { e.preventDefault(); zone.classList.remove("drag-over"); })
  );
  zone.addEventListener("drop", (e) => addFiles(Array.from(e.dataTransfer.files)));
  input.addEventListener("change", () => addFiles(Array.from(input.files)));
}

function addFiles(files) {
  showError("");
  const valid = files.filter((f) => [".pdf", ".txt"].includes(
    "." + f.name.split(".").pop().toLowerCase()
  ));
  if (valid.length < files.length) {
    showError("Some files were skipped — only .pdf and .txt are supported.");
  }

  const projectedTotal = state.libraryCount + state.stagedFiles.length + valid.length;
  const room = state.libraryMax - state.libraryCount - state.stagedFiles.length;
  const accepted = room > 0 ? valid.slice(0, room) : [];

  if (projectedTotal > state.libraryMax) {
    showError(
      `Library cap is ${state.libraryMax} resumes (${state.libraryCount} already stored, ` +
      `${state.stagedFiles.length} staged). Only added ${accepted.length} of ${valid.length} new file(s).`
    );
  }

  // de-dupe by name+size against already-staged files
  for (const f of accepted) {
    const dup = state.stagedFiles.some((s) => s.name === f.name && s.size === f.size);
    if (!dup) state.stagedFiles.push(f);
  }
  renderFileList();
}

function removeStagedFile(index) {
  state.stagedFiles.splice(index, 1);
  renderFileList();
}

function renderFileList() {
  const list = el("file-list");
  list.innerHTML = state.stagedFiles
    .map(
      (f, i) => `
      <div class="file-row">
        <span class="file-name">${escapeHtml(f.name)}</span>
        <button class="file-remove" data-idx="${i}" title="Remove">✕</button>
      </div>`
    )
    .join("");
  list.querySelectorAll(".file-remove").forEach((btn) =>
    btn.addEventListener("click", () => removeStagedFile(parseInt(btn.dataset.idx, 10)))
  );

  const label = el("dropzone-text");
  if (state.stagedFiles.length > 0) {
    label.textContent = `${state.stagedFiles.length} file(s) ready to upload`;
    label.classList.add("has-file");
  } else {
    label.textContent = "Drop resumes here, or click to browse (multi-select supported)";
    label.classList.remove("has-file");
  }
  el("upload-btn").disabled = state.stagedFiles.length === 0;
}

async function handleUpload() {
  if (state.stagedFiles.length === 0) return;
  showError("");
  const btn = el("upload-btn");
  btn.disabled = true;
  btn.textContent = "Uploading...";

  const form = new FormData();
  state.stagedFiles.forEach((f) => form.append("resumes", f));
  form.append("jd_text", el("jd-text").value || "");
  if (state.threadId) form.append("thread_id", state.threadId);

  try {
    const res = await fetch("/api/upload", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Upload failed.");

    state.threadId = data.thread_id;
    state.stagedFiles = [];
    renderFileList();

    const parts = [];
    if (data.added.length) parts.push(`${data.added.length} added`);
    if (data.renamed.length) parts.push(`${data.renamed.length} renamed (name collision)`);
    if (data.duplicates.length) parts.push(`${data.duplicates.length} already in library, skipped`);
    if (data.rejected.length) parts.push(`${data.rejected.length} rejected (bad type)`);
    el("upload-hint").textContent = parts.join(" · ") || "Nothing new to add.";

    el("actions-card").classList.remove("hidden");
    el("btn-scan").disabled = false;
    setRailStep("upload", "done");
    setRailStep("scan", "active");
    await refreshLibrary();
  } catch (e) {
    showError(e.message);
  }
  btn.disabled = false;
  btn.textContent = "Upload resumes";
}

// ---------------------------------------------------------------------
// Step 2a: Scan
// ---------------------------------------------------------------------

async function handleScan() {
  if (!state.threadId) {
    showError("Upload at least one resume first (needed to attach a JD to this session).");
    return;
  }
  showError("");
  const btn = el("btn-scan");
  btn.classList.add("loading");
  btn.disabled = true;

  try {
    const res = await fetch("/api/scan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ thread_id: state.threadId }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Scan failed.");

    renderScan(data);
    el("scan-result").classList.remove("hidden");
    el("btn-score").disabled = false;
    setRailStep("scan", "done");
    setRailStep("score", "active");
  } catch (e) {
    showError(e.message);
  }
  btn.classList.remove("loading");
  btn.disabled = false;
}

function renderScan(data) {
  const grid = el("scan-kv");
  const rows = [
    ["JD role", data.jd_role || "—"],
    ["JD skills", (data.jd_skills || []).join(", ") || "—"],
    ["Experience", data.jd_experience || "—"],
    ["Resumes loaded", data.resume_count ?? "—"],
  ];
  grid.innerHTML = rows
    .map(([k, v]) => `<div class="k">${k}</div><div class="v">${escapeHtml(String(v))}</div>`)
    .join("");

  el("scan-candidates").innerHTML = (data.candidates || [])
    .map((c) => `<span>${escapeHtml(c.candidate_name)}</span>`)
    .join("") || `<span style="opacity:.5">none</span>`;
}

// ---------------------------------------------------------------------
// Step 2b: Score (whole batch, ranked)
// ---------------------------------------------------------------------

async function handleScore() {
  showError("");
  const btn = el("btn-score");
  btn.classList.add("loading");
  btn.disabled = true;

  try {
    const res = await fetch("/api/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ thread_id: state.threadId }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Scoring failed.");

    renderRankedList(data.ranked_candidates || []);
    el("score-result").classList.remove("hidden");
    setRailStep("score", "done");
    setRailStep("questions", "active");
  } catch (e) {
    showError(e.message);
  }
  btn.classList.remove("loading");
  btn.disabled = false;
}

function scoreBand(score) {
  if (score >= 70) return "good";
  if (score >= 40) return "warn";
  return "bad";
}

function renderRankedList(ranked) {
  const container = el("ranked-list");
  container.innerHTML = ranked
    .map((c, i) => {
      const matched = (c.matched_skills || []).join(", ") || "none";
      const missing = (c.missing_skills || []).join(", ") || "none";
      return `
        <div class="rank-row">
          <div class="rank-index">${i + 1}</div>
          <div class="rank-main">
            <div class="rank-name">${escapeHtml(c.candidate_name || "Unknown")}</div>
            <div class="rank-skills">matched: ${escapeHtml(matched)} · missing: ${escapeHtml(missing)}</div>
          </div>
          <div class="rank-score ${scoreBand(c.match_score)}">${c.match_score}</div>
          <button class="rank-q-btn" data-name="${escapeHtml(c.candidate_name || "")}">Questions</button>
        </div>`;
    })
    .join("");

  container.querySelectorAll(".rank-q-btn").forEach((btn) =>
    btn.addEventListener("click", () => handleQuestions(btn.dataset.name, btn))
  );
}

// ---------------------------------------------------------------------
// Step 2c: Questions (per selected candidate)
// ---------------------------------------------------------------------

async function handleQuestions(candidateName, triggerBtn) {
  showError("");
  if (triggerBtn) {
    triggerBtn.classList.add("loading");
    triggerBtn.disabled = true;
    triggerBtn.textContent = "Working...";
  }

  try {
    const res = await fetch("/api/questions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ thread_id: state.threadId, candidate_name: candidateName }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Question generation failed.");

    el("questions-for").textContent = data.candidate_name || candidateName;
    renderQuestions(data.questions || {});
    el("questions-result").classList.remove("hidden");
    el("questions-result").scrollIntoView({ behavior: "smooth", block: "start" });
    setRailStep("questions", "done");
  } catch (e) {
    showError(e.message);
  }
  if (triggerBtn) {
    triggerBtn.classList.remove("loading");
    triggerBtn.disabled = false;
    triggerBtn.textContent = "Questions";
  }
}

function renderQuestions(questionSet) {
  const body = el("questions-body");
  const labels = {
    technical_questions: "Technical",
    candidate_specific_questions: "Candidate-specific",
    skill_gap_questions: "Skill gap",
    role_specific_questions: "Role-specific",
  };
  const categories = Object.keys(questionSet);
  if (categories.length === 0) {
    body.innerHTML = `<p class="hint">No questions returned.</p>`;
    return;
  }
  body.innerHTML = categories
    .map((cat) => {
      const items = questionSet[cat] || [];
      const label = labels[cat] || cat.replace(/_/g, " ");
      return `
        <div class="q-category">
          <h4>${escapeHtml(label)}</h4>
          <ul>${items.map((q) => `<li>${escapeHtml(q)}</li>`).join("")}</ul>
        </div>`;
    })
    .join("");
}

// ---------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------
window.addEventListener("DOMContentLoaded", () => {
  prefillJD();
  wireDropzone();
  refreshLibrary();
  setRailStep("upload", "active");

  el("upload-btn").addEventListener("click", handleUpload);
  el("btn-scan").addEventListener("click", handleScan);
  el("btn-score").addEventListener("click", handleScore);
});
