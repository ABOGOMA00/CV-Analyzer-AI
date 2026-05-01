// ── Config ────────────────────────────────────────────────────────────────────
const API = window.location.protocol.startsWith('http')
  ? `${window.location.origin}/api`
  : 'http://localhost:8000/api';
const MAX_FILE_BYTES = 5 * 1024 * 1024; // 5 MB

// ── Role → Emoji (all 24 backend categories) ─────────────────────────────────
const ROLE_EMOJI = {
  'INFORMATION-TECHNOLOGY': '💻', 'HR': '👥',           'FINANCE': '💰',
  'DESIGNER': '🎨',               'SALES': '📈',          'BANKING': '🏦',
  'HEALTHCARE': '🏥',             'CHEF': '👨‍🍳',          'ENGINEERING': '⚙️',
  'ACCOUNTANT': '📊',             'TEACHER': '📚',        'DIGITAL-MEDIA': '📱',
  'CONSULTANT': '🤝',             'AVIATION': '✈️',       'AGRICULTURE': '🌱',
  'BUSINESS-DEVELOPMENT': '🚀',   'ARTS': '🎭',           'ADVOCATE': '⚖️',
  'APPAREL': '👗',                'AUTOMOBILE': '🚗',     'BPO': '📞',
  'CONSTRUCTION': '🏗️',           'PUBLIC-RELATIONS': '📣', 'TOURISM': '🌍',
};

// ── Sector → Color (fallback) ─────────────────────────────────────────────────
const SECTOR_COLOR = {
  'Technology':   '#6e7fff',
  'Business':     '#a78bfa',
  'People & Org': '#2dd4bf',
  'Healthcare':   '#34d399',
  'Operations':   '#fbbf24',
};

// ── Career level → style ──────────────────────────────────────────────────────
const CAREER_STYLE = {
  'Junior':           { bg: 'rgba(110,127,255,0.15)', color: '#6e7fff', icon: '🌱' },
  'Mid-level':        { bg: 'rgba(251,191,36,0.15)',  color: '#fbbf24', icon: '🔥' },
  'Senior':           { bg: 'rgba(52,211,153,0.15)',  color: '#34d399', icon: '⭐' },
  'Lead / Executive': { bg: 'rgba(167,139,250,0.15)', color: '#a78bfa', icon: '👑' },
  'Not Specified':    { bg: 'rgba(123,130,168,0.15)', color: '#7b82a8', icon: '❓' },
};

// ── Tip icons ─────────────────────────────────────────────────────────────────
const TIP_ICON = { keywords: '🔑', length: '📏', achievements: '🏆', format: '✨' };

// ── State ─────────────────────────────────────────────────────────────────────
let selectedFile    = null;
let _activeFilter   = 'All';
let _historyData    = [];
// Restored from sessionStorage so the user can rewrite even after a soft reset
let _currentCVText  = (() => { try { return sessionStorage.getItem('cvText') || ''; } catch { return ''; } })();

// ── Navigation ────────────────────────────────────────────────────────────────
function showSection(name) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('section-' + name).classList.add('active');
  document.getElementById('nav-' + name).classList.add('active');
  if (name === 'history') loadHistory();
}

// ── File handling ─────────────────────────────────────────────────────────────
function handleFile(input) {
  const file = input.files[0] || null;
  if (!file) return;
  if (file.size > MAX_FILE_BYTES) {
    showToast(`File too large (max 5 MB). Yours is ${(file.size/1024/1024).toFixed(1)} MB.`, 'error');
    input.value = '';
    return;
  }
  selectedFile = file;
  document.getElementById('file-name').textContent = file.name;
  document.getElementById('file-chosen').classList.add('visible');
}

// ── DOMContentLoaded ──────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const dropZone = document.getElementById('drop-zone');
  if (dropZone) {
    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.classList.remove('drag');
      const file = e.dataTransfer?.files?.[0];
      if (!file) return;
      if (file.size > MAX_FILE_BYTES) { showToast('File too large (max 5 MB).', 'error'); return; }
      selectedFile = file;
      document.getElementById('file-name').textContent = file.name;
      document.getElementById('file-chosen').classList.add('visible');
    });
  }

  // Ctrl+Enter shortcut — preventDefault stops textarea from inserting a newline
  document.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      const btn = document.getElementById('btn-analyze');
      if (btn && !btn.disabled && document.getElementById('section-analyze').classList.contains('active')) {
        e.preventDefault();
        analyzeCV();
      }
    }
  });

  checkBackendHealth();
});

// ── Health check ──────────────────────────────────────────────────────────────
async function checkBackendHealth() {
  const dot  = document.getElementById('status-dot');
  const text = document.getElementById('status-text');
  try {
    const res  = await fetch(`${API.replace('/api','')}/health`, { signal: AbortSignal.timeout(3000) });
    const data = await res.json();
    if (res.ok) { dot.classList.add('online');  text.textContent = `Backend v${data.version ?? '?'} online`; }
    else throw new Error();
  } catch {
    dot.classList.add('offline'); text.textContent = 'Backend offline';
  }
}

// ── Animated loading steps ────────────────────────────────────────────────────
function startLoadingSteps() {
  document.querySelectorAll('.loading-step').forEach(s => s.classList.remove('active','done'));
  [0, 800, 1800, 3200].forEach((delay, i) => {
    setTimeout(() => {
      if (i > 0) document.getElementById(`step-${i}`)?.classList.replace('active','done');
      document.getElementById(`step-${i+1}`)?.classList.add('active');
    }, delay);
  });
}
function stopLoadingSteps() {
  document.querySelectorAll('.loading-step').forEach(s => { s.classList.remove('active'); s.classList.add('done'); });
}

// ── Analyze ───────────────────────────────────────────────────────────────────
async function analyzeCV() {
  const cvText  = document.getElementById('cv-text').value.trim();
  const targetJD = document.getElementById('jd-text').value.trim();
  if (!selectedFile && !cvText) { showToast('Please upload a file or paste CV text.', 'error'); return; }

  const btn = document.getElementById('btn-analyze');
  btn.disabled = true;
  document.getElementById('loading').style.display = 'flex';
  document.getElementById('results').style.display  = 'none';
  startLoadingSteps();

  const fd = new FormData();
  if (selectedFile) fd.append('file', selectedFile);
  else { fd.append('file', new Blob([cvText], {type:'text/plain'}), 'pasted_cv.txt'); }
  if (targetJD) fd.append('target_jd', targetJD);

  try {
    const res = await fetch(`${API}/analyze/upload`, { method:'POST', body: fd });
    if (!res.ok) { const e = await res.json().catch(()=>({detail:'Unknown error'})); throw new Error(e.detail || `HTTP ${res.status}`); }
    const data = await res.json();
    stopLoadingSteps();
    showResults(data);
    showToast('Analysis complete!', 'success');
  } catch(err) {
    console.error(err);
    showToast(`Analysis failed: ${err.message}`, 'error');
  } finally {
    document.getElementById('loading').style.display = 'none';
    btn.disabled = false;
  }
}

// ── Render results ────────────────────────────────────────────────────────────
function showResults(data) {
  _currentCVText = data.cv_text || '';
  // Persist so the user can still rewrite after a soft page state reset
  try { sessionStorage.setItem('cvText', _currentCVText); } catch (_) {}
  const sectorColor  = data.sector_color  || '#6e7fff';
  const roleDisplay  = data.role_display  || data.predicted_role?.replace(/-/g,' ') || '—';
  const careerStyle  = CAREER_STYLE[data.career_level] || CAREER_STYLE['Not Specified'];

  // ── Hero ──────────────────────────────────────────────────────────────────
  document.getElementById('result-emoji').textContent      = ROLE_EMOJI[data.predicted_role] ?? '💼';
  document.getElementById('result-role').textContent       = roleDisplay;
  document.getElementById('result-confidence').textContent = (data.confidence ?? 0).toFixed(1) + '%';

  // Sector badge
  document.getElementById('result-sector').innerHTML = data.sector
    ? `<span class="sector-badge" style="background:${sectorColor}22; color:${sectorColor}; border-color:${sectorColor}44;">
        ${data.sector_icon ?? ''} ${escHtml(data.sector)}
       </span>`
    : '';

  // Career level chip
  document.getElementById('result-career').innerHTML = data.career_level
    ? `<span class="career-chip" style="background:${careerStyle.bg}; color:${careerStyle.color};">
        ${careerStyle.icon} ${escHtml(data.career_level)}
       </span>`
    : '';

  // ── Sub-specialization ────────────────────────────────────────────────────
  renderSubSpec(data.sub_specialization ?? {}, sectorColor);

  // ── Detected skills ───────────────────────────────────────────────────────
  const skillsTags = document.getElementById('hero-skills-tags');
  skillsTags.innerHTML = (data.extracted_skills ?? [])
    .map(s => `<span class="skill-badge">${escHtml(s)}</span>`).join('');

  // ── Missing skills ────────────────────────────────────────────────────────
  const missingSec  = document.getElementById('missing-skills-section');
  const missingTags = document.getElementById('missing-skills-tags');
  if (data.missing_skills?.length) {
    missingTags.innerHTML = data.missing_skills.map(s => `<span class="skill-badge missing">${escHtml(s)}</span>`).join('');
    missingSec.style.display = 'block';
  } else {
    missingSec.style.display = 'none';
  }

  // ── Mismatched Job ────────────────────────────────────────────────────────
  const mismatchSec = document.getElementById('mismatch-warning-section');
  const severeMismatch = Boolean(
    data.is_mismatch &&
    data.ats_score != null &&
    data.ats_score < 25 &&
    Array.isArray(data.missing_skills) &&
    data.missing_skills.length >= 6
  );
  if (severeMismatch) {
    mismatchSec.style.display = 'block';
  } else {
    mismatchSec.style.display = 'none';
  }

  // ── ATS score ─────────────────────────────────────────────────────────────
  const atsContainer = document.getElementById('ats-score-container');
  if (data.ats_score != null) {
    const score = data.ats_score;
    const color = score >= 80 ? 'var(--success)' : score >= 50 ? 'var(--warn)' : 'var(--danger)';
    const label = score >= 80 ? 'Excellent' : score >= 50 ? 'Good' : 'Low';
    atsContainer.innerHTML = `
      <div class="upload-card ats-card" style="border-left:4px solid ${color};">
        <div class="card-label">ATS Compatibility Score</div>
        <div style="display:flex; align-items:center; gap:15px;">
          <div>
            <div class="ats-score-num" style="color:${color};">${score.toFixed(1)}%</div>
            <div class="ats-label" style="color:${color};">${label}</div>
          </div>
          <div class="ats-bar-track">
            <div class="ats-bar-fill" data-width="${score}" style="background:${color};"></div>
          </div>
        </div>
      </div>`;
    requestAnimationFrame(() => {
      const fill = atsContainer.querySelector('.ats-bar-fill');
      if (fill) fill.style.width = fill.dataset.width + '%';
    });
  } else {
    atsContainer.innerHTML = '';
  }

  // ── All scores chart ──────────────────────────────────────────────────────
  renderScoresBreakdown(data.all_scores ?? {});

  // ── Related roles ─────────────────────────────────────────────────────────
  renderRelatedRoles(data.related_roles ?? [], sectorColor, data.sector);

  // ── Tips ──────────────────────────────────────────────────────────────────
  const tipsGrid = document.getElementById('tips-grid');
  tipsGrid.innerHTML = (data.tips?.tips ?? []).map(t => `
    <div class="tip-card ${escHtml(t.type ?? '')}">
      <div class="tip-title"><span class="tip-icon">${TIP_ICON[t.type] ?? '💡'}</span>${escHtml(t.title)}</div>
      <div class="tip-msg">${escHtml(t.message)}</div>
    </div>`).join('');

  document.getElementById('results').style.display      = 'block';
  document.getElementById('upload-form').style.display  = 'none';
  document.getElementById('results').scrollIntoView({ behavior:'smooth', block:'start' });
}

// ── Sub-specialization section ────────────────────────────────────────────────
function renderSubSpec(subSpec, accentColor) {
  const container = document.getElementById('sub-spec-container');
  if (!subSpec || !subSpec.scores?.length) { container.style.display = 'none'; return; }

  const topName = subSpec.top;
  const scores  = subSpec.scores.slice(0, 6);
  const maxScore = scores[0]?.score || 1;

  container.innerHTML = `
    <div class="subspec-top" style="border-color:${accentColor}44;">
      <div class="subspec-top-label">Detected Specialization</div>
      <div class="subspec-top-name" style="color:${accentColor};">${escHtml(topName ?? 'General')}</div>
    </div>
    <div class="subspec-tracks">
      ${scores.map((s, i) => {
        const pct   = (s.score / maxScore) * 100;
        const color = i === 0 ? accentColor : '#3a4060';
        const textC = i === 0 ? accentColor : 'var(--muted)';
        return `
          <div class="subspec-track ${i===0 ? 'subspec-track--top' : ''}">
            <div class="subspec-name" style="color:${textC}">${escHtml(s.name)}</div>
            <div class="subspec-bar-bg">
              <div class="subspec-bar-fill" data-width="${pct}" style="background:${color}; width:0;"></div>
            </div>
            <div class="subspec-pct" style="color:${textC}">${s.score.toFixed(0)}%</div>
          </div>`;
      }).join('')}
    </div>`;

  container.style.display = 'block';
  requestAnimationFrame(() => requestAnimationFrame(() => {
    container.querySelectorAll('.subspec-bar-fill').forEach(b => b.style.width = b.dataset.width + '%');
  }));
}

// ── All scores breakdown ──────────────────────────────────────────────────────
function renderScoresBreakdown(allScores) {
  const container = document.getElementById('scores-breakdown');
  const list      = document.getElementById('scores-list');
  const entries   = Object.entries(allScores).sort((a,b) => b[1]-a[1]).slice(0, 8);
  if (!entries.length) { container.style.display = 'none'; return; }

  const maxScore = entries[0][1];
  list.innerHTML = entries.map(([role, score], idx) => {
    const pct      = maxScore > 0 ? (score / maxScore) * 100 : 0;
    const sector   = _getSector(role);
    const barColor = idx === 0 ? (SECTOR_COLOR[sector] || 'var(--accent)') : '#2a3050';
    const textColor= idx === 0 ? (SECTOR_COLOR[sector] || 'var(--accent)') : 'var(--muted)';
    return `
      <div class="score-row ${idx===0 ? 'score-row--top' : ''}">
        <span class="score-role" style="color:${textColor}">
          ${ROLE_EMOJI[role] ?? '💼'} ${escHtml(role.replace(/-/g,' '))}
        </span>
        <div class="score-bar-wrap">
          <div class="score-bar-bg">
            <div class="score-bar-val" data-width="${pct}" style="background:${barColor}; width:0;"></div>
          </div>
        </div>
        <span class="score-pct" style="color:${textColor}">${score.toFixed(1)}%</span>
      </div>`;
  }).join('');

  container.style.display = 'block';
  requestAnimationFrame(() => requestAnimationFrame(() => {
    list.querySelectorAll('.score-bar-val').forEach(b => b.style.width = b.dataset.width + '%');
  }));
}

// ── Related roles ─────────────────────────────────────────────────────────────
function renderRelatedRoles(roles, color, sector) {
  const container = document.getElementById('related-roles-container');
  if (!roles.length) { container.style.display = 'none'; return; }
  container.innerHTML = `
    <div class="card-label" style="margin-bottom:1rem;">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
      Other roles in ${escHtml(sector ?? 'this field')}
    </div>
    <div class="related-grid">
      ${roles.slice(0,4).map(r => `
        <div class="related-card" style="border-color:${color}33;">
          <div class="related-emoji">${escHtml(r.emoji)}</div>
          <div class="related-name">${escHtml(r.display)}</div>
        </div>`).join('')}
    </div>`;
  container.style.display = 'block';
}

// ── History filter ────────────────────────────────────────────────────────────
function filterHistory(sector) {
  _activeFilter = sector;
  document.querySelectorAll('.filter-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.filter === sector);
  });
  renderHistoryList(_activeFilter === 'All'
    ? _historyData
    : _historyData.filter(i => _getSector(i.predicted_role) === sector)
  );
}

function renderHistoryList(items) {
  const list = document.getElementById('history-list');
  if (!items.length) {
    list.innerHTML = `<div class="empty-state">
      <svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
      <p>No analyses found${_activeFilter !== 'All' ? ' for this sector' : ''}.</p>
    </div>`;
    return;
  }
  list.innerHTML = items.map(item => {
    const sColor = item.sector_color || SECTOR_COLOR[_getSector(item.predicted_role)] || '#7b82a8';
    const disp   = item.predicted_role?.replace(/-/g, ' ') ?? '—';
    return `
      <div class="history-item" id="hist-${item.id}">
        <span class="hist-emoji">${ROLE_EMOJI[item.predicted_role] ?? '💼'}</span>
        <div class="history-role-wrap">
          <div class="history-role">
            <span class="hist-sector-dot" style="background:${sColor};"></span>
            ${escHtml(disp)}
          </div>
          <div class="history-file">${escHtml(item.cv_filename)}</div>
        </div>
        <div class="hist-right">
          <div class="history-conf">${item.confidence.toFixed(1)}%</div>
          ${item.ats_score != null ? `<div class="history-ats">ATS ${item.ats_score.toFixed(1)}%</div>` : ''}
          <div class="history-date">${formatDate(item.created_at)}</div>
        </div>
        <button class="delete-btn" onclick="deleteAnalysis(${item.id})" aria-label="Delete">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/>
            <path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4h6v2"/>
          </svg>
        </button>
      </div>`;
  }).join('');
}

async function loadHistory() {
  const list = document.getElementById('history-list');
  list.innerHTML = '<div class="empty-state"><p>Loading…</p></div>';
  try {
    const res  = await fetch(`${API}/history/`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    _historyData = await res.json();

    if (!_historyData.length) {
      list.innerHTML = `<div class="empty-state">
        <svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
        <p>No analyses yet. Upload your first CV!</p>
      </div>`;
      return;
    }
    // Build filter tabs using data-filter + event delegation.
    // Avoids XSS: escHtml converts ' → &#39; which is safe in HTML attributes
    // but NOT safe inside an inline onclick="filterHistory('...')".
    const sectors = ['All', ...new Set(_historyData.map(i => _getSector(i.predicted_role)).filter(Boolean))];
    const tabsEl  = document.getElementById('history-filter-tabs');
    if (tabsEl) {
      tabsEl.innerHTML = sectors.map(s =>
        `<button class="filter-tab ${s === _activeFilter ? 'active' : ''}" data-filter="${escHtml(s)}">${escHtml(s)}</button>`
      ).join('');
      tabsEl.onclick = e => {
        const btn = e.target.closest('.filter-tab');
        if (btn) filterHistory(btn.dataset.filter);
      };
    }

    filterHistory(_activeFilter);
  } catch(err) {
    list.innerHTML = `<div class="empty-state"><p>Failed to load: ${escHtml(err.message)}</p></div>`;
  }
}

async function deleteAnalysis(id) {
  const el = document.getElementById(`hist-${id}`);
  // Guard against double-click race condition
  if (!el || el.dataset.deleting) return;
  el.dataset.deleting = 'true';

  try {
    const res = await fetch(`${API}/history/${id}`, { method:'DELETE' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    el.style.cssText += 'opacity:0; transform:translateX(30px); transition:all 0.3s ease;';
    setTimeout(() => {
      _historyData = _historyData.filter(i => i.id !== id);
      el.remove();
      if (!document.getElementById('history-list').children.length) loadHistory();
    }, 300);
    showToast('Deleted.', 'success');
  } catch(err) {
    delete el.dataset.deleting; // Allow retry on failure
    showToast(`Delete failed: ${friendlyError(err.message)}`, 'error');
  }
}

// ── Reset ─────────────────────────────────────────────────────────────────────
function resetForm() {
  selectedFile   = null;
  _currentCVText = '';
  try { sessionStorage.removeItem('cvText'); } catch (_) {}

  document.getElementById('file-input').value = '';
  document.getElementById('file-name').textContent = '';
  document.getElementById('file-chosen').classList.remove('visible');
  document.getElementById('cv-text').value = '';
  document.getElementById('jd-text').value = '';
  document.getElementById('results').style.display     = 'none';
  document.getElementById('upload-form').style.display = 'block';
  document.getElementById('ats-score-container').innerHTML = '';
  document.getElementById('scores-breakdown').style.display = 'none';
  document.getElementById('sub-spec-container').style.display = 'none';
  document.getElementById('related-roles-container').style.display = 'none';

  // Clear skill tags and tips to avoid stale data on edge-case re-opens
  document.getElementById('tips-grid').innerHTML          = '';
  document.getElementById('hero-skills-tags').innerHTML   = '';
  document.getElementById('missing-skills-tags').innerHTML = '';

  const mismatchWarning = document.getElementById('mismatch-warning-section');
  if (mismatchWarning) mismatchWarning.style.display = 'none';

  // Clear rewrite section
  document.getElementById('rewrite-section').style.display = 'none';
  document.getElementById('rewrite-original-text').textContent = '';
  document.getElementById('rewrite-new-text').innerHTML = '';
  document.getElementById('rewrite-ats-scores').style.display = 'none';

  document.getElementById('upload-form').scrollIntoView({ behavior:'smooth', block:'start' });
}

// ── Rewrite CV ────────────────────────────────────────────────────────────────
async function rewriteCV() {
  const jd = document.getElementById('jd-text').value.trim();
  if (!jd) {
    showToast('Please provide a target job description to rewrite your CV.', 'error');
    return;
  }
  if (!_currentCVText) {
    showToast('CV text not available. Please re-analyze.', 'error');
    return;
  }

  const btn = document.getElementById('btn-rewrite');
  btn.disabled = true;
  document.getElementById('rewrite-section').style.display = 'block';
  document.getElementById('rewrite-loading').style.display = 'block';
  document.getElementById('rewrite-content').style.display = 'none';

  try {
    const res = await fetch(`${API}/rewrite/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cv_text: _currentCVText, job_description: jd })
    });
    
    if (!res.ok) {
      const e = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(e.detail || `HTTP ${res.status}`);
    }
    
    const data = await res.json();
    
    document.getElementById('rewrite-original-text').textContent = _currentCVText;
    document.getElementById('rewrite-new-text').innerHTML = highlightDifferences(_currentCVText, data.rewritten_cv);
    
    if (data.old_ats_score != null && data.new_ats_score != null) {
      document.getElementById('rewrite-old-score').textContent = data.old_ats_score.toFixed(1) + '%';
      document.getElementById('rewrite-new-score').textContent = data.new_ats_score.toFixed(1) + '%';
      document.getElementById('rewrite-ats-scores').style.display = 'flex';
    } else {
      document.getElementById('rewrite-ats-scores').style.display = 'none';
    }
    
    document.getElementById('rewrite-content').style.display = 'block';
    
    // Store rewritten text for download/copy
    window._rewrittenCVText = data.rewritten_cv;
    window._rewrittenCVCleanText = extractCleanRewrittenCV(data.rewritten_cv);
    
    showToast('CV rewritten successfully!', 'success');
  } catch (err) {
    console.error(err);
    showToast(`Rewrite failed: ${err.message}`, 'error');
    document.getElementById('rewrite-content').style.display = 'none';
  } finally {
    document.getElementById('rewrite-loading').style.display = 'none';
    btn.disabled = false;
  }
}

/**
 * Highlights lines in newText that don't appear in oldText.
 * NOTE: This is an approximate line-level diff. Lines with minor whitespace
 * differences or reordered content may be incorrectly flagged as new/unchanged.
 * A dedicated diff library (e.g. diff-match-patch) would give precise results.
 */
function highlightDifferences(oldText, newText) {
  const oldLines = new Set(oldText.split('\n').map(l => l.trim().toLowerCase()).filter(Boolean));
  const newLines = newText.split('\n');

  return newLines.map(line => {
    const trimmed = line.trim();
    if (!trimmed) return line;
    if (!oldLines.has(trimmed.toLowerCase())) {
      return `<span style="background: rgba(52,211,153,0.15); border-left: 2px solid var(--success); padding-left: 6px; display: block; margin: 2px 0;">${escHtml(line)}</span>`;
    }
    return escHtml(line);
  }).join('\n');
}

function copyRewrittenCV() {
  const textToCopy = window._rewrittenCVCleanText || window._rewrittenCVText;
  if (!textToCopy) return;
  navigator.clipboard.writeText(textToCopy).then(() => {
    showToast('Copied to clipboard!', 'success');
  }).catch(() => {
    showToast('Failed to copy', 'error');
  });
}

async function downloadRewrittenCV() {
  const cleanText = window._rewrittenCVCleanText || window._rewrittenCVText;
  if (!cleanText) return;

  const originalName = selectedFile?.name ? selectedFile.name.replace(/\.[^.]+$/, '') : 'CV';
  const fileName = `${originalName}_Optimized.docx`;

  try {
    showToast('Generating document...', 'success');
    const res = await fetch(`${API}/rewrite/download`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rewritten_cv: cleanText })
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showToast('Downloaded as Word Doc!', 'success');
  } catch (err) {
    console.error(err);
    // Fallback if document generation fails
    const blob = new Blob([cleanText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${originalName}_Optimized.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast('Docx export unavailable, downloaded as TXT instead.', 'error');
  }
}

function extractCleanRewrittenCV(rawText) {
  let text = String(rawText || '').trim();
  if (!text) return '';

  const improvedHeader = /===\s*IMPROVED CV\s*===/i;
  const compareHeader = /===\s*BEFORE vs AFTER\s*===/i;

  if (improvedHeader.test(text)) {
    text = text.split(improvedHeader)[1] || text;
  }
  if (compareHeader.test(text)) {
    text = text.split(compareHeader)[0] || text;
  }

  return text
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/\*   /g, '- ')
    .replace(/�/g, '-')
    .trim();
}

// ── Utilities ─────────────────────────────────────────────────────────────────
const _SECTOR_LOOKUP = {
  'INFORMATION-TECHNOLOGY': 'Technology', 'ENGINEERING':    'Technology',
  'DESIGNER':               'Technology', 'DIGITAL-MEDIA':  'Technology',
  'FINANCE':    'Business',  'BANKING':   'Business', 'ACCOUNTANT': 'Business',
  'SALES':      'Business',  'CONSULTANT':'Business', 'BUSINESS-DEVELOPMENT': 'Business',
  'PUBLIC-RELATIONS': 'Business',
  'HR':         'People & Org', 'TEACHER':  'People & Org', 'ADVOCATE':  'People & Org',
  'HEALTHCARE': 'Healthcare',   'AGRICULTURE': 'Healthcare',
  'CHEF':       'Operations',   'AVIATION':    'Operations', 'AUTOMOBILE': 'Operations',
  'CONSTRUCTION':'Operations',  'BPO':         'Operations', 'APPAREL':   'Operations',
  'ARTS':       'Creative',     'TOURISM':     'Creative',
};
function _getSector(role) { return _SECTOR_LOOKUP[role] ?? 'General'; }

// ── Friendly HTTP error messages ──────────────────────────────────────────────
function friendlyError(raw) {
  if (/422/.test(raw)) return 'File format not supported or missing fields.';
  if (/413/.test(raw)) return 'File is too large for the server.';
  if (/503|502|504/.test(raw)) return 'Server is busy — please try again shortly.';
  if (/401|403/.test(raw)) return 'Access denied.';
  if (/404/.test(raw)) return 'Resource not found.';
  if (/500/.test(raw)) return 'Server error — check backend logs.';
  return raw;
}

function showToast(message, type = 'success') {
  const toast = document.getElementById('toast');
  toast.textContent = message;
  toast.className   = `toast show ${type}`;
  clearTimeout(toast._tmr);
  toast._tmr = setTimeout(() => toast.classList.remove('show'), 3500);
}

function escHtml(str) {
  return String(str ?? '')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function formatDate(iso) {
  try { return new Date(iso).toLocaleDateString(undefined,{year:'numeric',month:'short',day:'numeric'}); }
  catch { return iso; }
}
