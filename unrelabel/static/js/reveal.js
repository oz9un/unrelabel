// unrelabel/static/js/reveal.js
// Narrative reveal sequence + inline report for Step 3
(function() {
  'use strict';

  function severityLabel(score) {
    if (score <= 5) return 'Clean';
    if (score <= 40) return 'Low';
    if (score <= 65) return 'Medium';
    if (score <= 85) return 'High';
    return 'Critical';
  }

  var skipReveal = false;
  var revealTimeouts = [];

  function clearReveals() {
    revealTimeouts.forEach(clearTimeout);
    revealTimeouts = [];
  }

  function schedule(fn, delay) {
    if (skipReveal) { fn(); return; }
    revealTimeouts.push(setTimeout(fn, delay));
  }

  window.startReveal = function(result, datasetInfo) {
    skipReveal = false;
    clearReveals();

    var scoreEl = document.getElementById('reveal-score');
    var severityEl = document.getElementById('reveal-severity');
    var metricsEl = document.getElementById('reveal-metrics');
    var summaryEl = document.getElementById('reveal-summary');
    var reportInline = document.getElementById('report-inline');

    // Reset
    [scoreEl, severityEl, metricsEl, summaryEl].forEach(function(el) {
      el.innerHTML = '';
      el.style.opacity = '0';
    });
    reportInline.innerHTML = '';
    reportInline.style.opacity = '0';

    var sweep = result.sweep_results || [];
    var worst = sweep.length > 0
      ? sweep.reduce(function(a, b) { return a.vulnerability_score > b.vulnerability_score ? a : b; })
      : null;
    var vscore = worst ? worst.vulnerability_score : result.vulnerability_score;
    var severity = severityLabel(vscore);

    // Skip handler
    var skipHandler = function() {
      skipReveal = true;
      clearReveals();
      [scoreEl, severityEl, metricsEl, summaryEl].forEach(function(el) { el.style.opacity = '1'; });
      reportInline.style.opacity = '1';
      // Instantly reveal all report sections
      reportInline.querySelectorAll('.report-section').forEach(function(sec) {
        sec.classList.add('visible');
      });
      document.removeEventListener('click', skipHandler);
    };
    document.addEventListener('click', skipHandler);

    // T+0: Score typewriter
    schedule(function() {
      var scoreStr = vscore.toFixed(1);
      scoreEl.style.opacity = '1';
      var idx = 0;
      var typeInterval = setInterval(function() {
        if (idx <= scoreStr.length || skipReveal) {
          scoreEl.innerHTML = '<span class="score-digits">' + scoreStr.slice(0, idx) + '</span>';
          if (idx === scoreStr.length) {
            scoreEl.innerHTML += '<span class="score-suffix">/100</span>';
            clearInterval(typeInterval);
          }
          idx++;
        }
      }, skipReveal ? 0 : 120);
    }, 0);

    // T+800: Severity badge
    schedule(function() {
      severityEl.style.opacity = '1';
      severityEl.innerHTML = '<span class="severity-badge">' + severity.toUpperCase() + '</span>';
    }, 800);

    // T+1400: Metric cards
    schedule(function() {
      var cleanAcc = (result.clean_accuracy * 100).toFixed(1) + '%';
      var poisonAcc = ((worst ? worst.poisoned_accuracy : result.poisoned_accuracy) * 100).toFixed(1) + '%';
      var drop = ((worst ? worst.accuracy_drop : result.accuracy_drop) * 100).toFixed(1) + '%';

      var cards;
      if (result.attack_type === 'label_flip') {
        var nf = worst ? worst.n_flipped : (result.flipped_indices ? result.flipped_indices.length : 0);
        cards = [
          { label: 'CLEAN ACC', value: cleanAcc, red: false },
          { label: 'POISONED ACC', value: poisonAcc, red: true },
          { label: 'ACC DROP', value: drop, red: true },
          { label: 'LABELS FLIPPED', value: nf, red: false },
        ];
      } else if (result.attack_type === 'targeted_label') {
        var tmr = worst ? worst.targeted_misclassification_rate : result.config.targeted_misclassification_rate;
        cards = [
          { label: 'CLEAN ACC', value: cleanAcc, red: false },
          { label: 'POISONED ACC', value: poisonAcc, red: true },
          { label: 'ACC DROP', value: drop, red: true },
          { label: 'TMR', value: (tmr * 100).toFixed(1) + '%', red: false },
        ];
      } else {
        // Clean label: different metric layout
        var nPerturbed = result.flipped_indices ? result.flipped_indices.length : 0;
        var success = result.config.attack_success;
        cards = [
          { label: 'STEALTH', value: drop === '0.0%' ? 'PERFECT' : drop, red: false },
          { label: 'PERTURBED', value: nPerturbed + ' samples', red: false },
          { label: 'TARGET MISCLASS', value: success ? 'YES' : 'NO', red: success },
          { label: 'DETECTABLE', value: drop === '0.0%' ? 'NO' : 'MAYBE', red: false },
        ];
      }

      metricsEl.style.opacity = '1';
      metricsEl.innerHTML = '<div class="metric-grid">' +
        cards.map(function(c, i) {
          return '<div class="metric-card" style="animation-delay:' + (i * 100) + 'ms">' +
            '<span class="metric-label">' + c.label + '</span>' +
            '<span class="metric-value' + (c.red ? ' red' : '') + '">' + c.value + '</span>' +
            '</div>';
        }).join('') +
        '</div>';
    }, 1400);

    // T+2200: Summary
    schedule(function() {
      summaryEl.style.opacity = '1';
      if (result.attack_type === 'clean_label') {
        var nPerturbed = result.flipped_indices ? result.flipped_indices.length : 0;
        var classNames = datasetInfo ? datasetInfo.class_names : [];
        var srcName = classNames[result.config.source_class] || ('class ' + result.config.source_class);
        var tgtName = classNames[result.config.target_class] || ('class ' + result.config.target_class);
        var successStr = result.config.attack_success
          ? 'The target <span class="hl-white">' + tgtName + '</span> sample (index #' + result.config.target_index + ') is now misclassified as <span class="hl-red">' + srcName + '</span>.'
          : 'The target sample was <span class="hl-white">not misclassified</span>, the model resisted the attack.';
        summaryEl.innerHTML = '<p class="summary-text">Perturbed <span class="hl-red">' + nPerturbed +
          ' ' + srcName + '</span> training samples to poison the model. Overall accuracy is <span class="hl-white">unchanged</span>, the attack is invisible to standard metrics. ' +
          successStr + ' Severity: <span class="hl-red">' + severity + '</span>.</p>';
      } else {
        var pctDrop = ((worst ? worst.accuracy_drop : result.accuracy_drop) * 100).toFixed(1);
        var rate = worst ? (worst.poison_rate * 100).toFixed(0) + '%' : '';
        summaryEl.innerHTML = '<p class="summary-text">Model accuracy dropped <span class="hl-red">' + pctDrop +
          '%</span> under attack' + (rate ? ' at <span class="hl-white">' + rate + '</span> poison rate' : '') +
          '. Severity: <span class="hl-red">' + severity + '</span>.</p>';
      }
    }, 2200);

    // T+2800: Inline report
    schedule(function() {
      reportInline.style.opacity = '1';
      renderInlineReport(result, datasetInfo, sweep, worst);
      document.removeEventListener('click', skipHandler);
    }, 2800);
  };

  function renderInlineReport(result, datasetInfo, sweep, worst) {
    var el = document.getElementById('report-inline');
    var classNames = datasetInfo ? datasetInfo.class_names : [];
    var html = '';

    // Clean Label explainer section
    if (result.attack_type === 'clean_label') {
      var srcName = classNames[result.config.source_class] || ('class ' + result.config.source_class);
      var tgtName = classNames[result.config.target_class] || ('class ' + result.config.target_class);
      var nPerturbed = result.flipped_indices ? result.flipped_indices.length : 0;
      var success = result.config.attack_success;

      html += '<div class="report-section">';
      html += '<div class="report-section-title">WHAT CHANGED</div>';
      html += '<div class="report-section-subtitle">Clean label attacks are stealthy. Labels stay the same, only features are perturbed.</div>';
      html += '<div style="display:grid; grid-template-columns:1fr auto 1fr; gap:20px; align-items:center; padding:24px; background:var(--surface); border:1px solid var(--border);">';

      // Left: perturbed samples
      html += '<div style="text-align:center;">';
      html += '<div style="font-family:var(--font-display); font-size:32px; color:var(--red);">' + nPerturbed + '</div>';
      html += '<div style="font-size:13px; color:var(--muted); margin-top:4px;">' + srcName.toUpperCase() + ' SAMPLES PERTURBED</div>';
      html += '<div style="font-size:12px; color:var(--tertiary); margin-top:8px;">Training indices: [' + (result.flipped_indices || []).join(', ') + ']</div>';
      html += '</div>';

      // Arrow
      html += '<div style="font-size:28px; color:var(--red);">&#10132;</div>';

      // Right: target result
      html += '<div style="text-align:center;">';
      if (success) {
        html += '<div style="font-family:var(--font-display); font-size:18px; color:var(--red);">MISCLASSIFIED</div>';
        html += '<div style="font-size:13px; color:var(--muted); margin-top:4px;">' + tgtName.toUpperCase() + ' SAMPLE #' + result.config.target_index + '</div>';
        html += '<div style="font-size:13px; color:var(--muted); margin-top:4px;">Now predicted as: <span style="color:var(--red); font-weight:700;">' + srcName + '</span></div>';
      } else {
        html += '<div style="font-family:var(--font-display); font-size:18px; color:var(--green);">RESISTED</div>';
        html += '<div style="font-size:13px; color:var(--muted); margin-top:4px;">' + tgtName.toUpperCase() + ' SAMPLE #' + result.config.target_index + '</div>';
        html += '<div style="font-size:13px; color:var(--muted); margin-top:4px;">Still correctly classified</div>';
      }
      html += '</div>';

      html += '</div>';

      // Stealth note
      html += '<div style="margin-top:16px; padding:14px 20px; background:var(--red-dim); border-left:3px solid var(--red); font-size:13px; color:var(--muted); line-height:1.6;">';
      html += '<span style="color:var(--red); font-weight:700;">WHY IS THIS DANGEROUS?</span> ';
      html += 'The confusion matrices below look identical. Standard accuracy metrics detect nothing. ';
      html += 'An attacker can plant targeted misclassifications that bypass all conventional model validation.';
      html += '</div>';

      html += '</div>';
    }

    // Confusion Matrices
    html += '<div class="report-section">';
    html += '<div class="report-section-title">CONFUSION MATRICES</div>';
    html += '<div class="report-section-subtitle">How the model classifies samples: diagonal = correct, off-diagonal = errors. <span style="color:var(--tertiary)">Click to zoom.</span></div>';
    html += '<div class="chart-container" id="cm-pair-container"><span class="zoom-hint">CLICK TO ZOOM</span>';
    html += '<div class="cm-pair">';
    html += '<div><div class="cm-title clean">CLEAN MODEL <span class="cm-acc">' + (result.clean_accuracy * 100).toFixed(1) + '%</span></div><div id="cm-clean"></div></div>';
    var pAcc = worst ? worst.poisoned_accuracy : result.poisoned_accuracy;
    html += '<div><div class="cm-title poisoned">POISONED MODEL <span class="cm-acc">' + (pAcc * 100).toFixed(1) + '%</span></div><div id="cm-poisoned"></div></div>';
    html += '</div>';
    html += '<div class="cm-legend"><div class="cm-legend-item"><div class="cm-legend-swatch" style="background:#1a1a1a;border:1px solid #333"></div>Correct</div>';
    html += '<div class="cm-legend-item"><div class="cm-legend-swatch" style="background:#440000"></div>Errors (red = severity)</div></div>';
    html += '</div></div>';

    // Sweep bar chart
    if (sweep.length > 1) {
      html += '<div class="report-section">';
      html += '<div class="report-section-title">SWEEP RESULTS</div>';
      html += '<div class="report-section-subtitle">Accuracy drop across poison rates. <span style="color:var(--tertiary)">Click to zoom.</span></div>';
      html += '<div class="chart-container" id="sweep-bar-container"><span class="zoom-hint">CLICK TO ZOOM</span><div id="sweep-bar"></div></div>';
      html += '</div>';
    }

    // Per-rate detail
    if (sweep.length > 1) {
      html += '<div class="report-section">';
      html += '<div class="report-section-title">PER-RATE DETAIL</div>';
      html += '<div class="sweep-tabs" id="sweep-tabs-v2"></div>';
      html += '<div id="sweep-detail-v2"></div>';
      html += '</div>';
    }

    // Attack config
    html += '<div class="report-section">';
    html += '<div class="report-section-title">ATTACK CONFIG</div>';
    html += '<div class="config-grid">';
    var attackName = result.attack_type === 'label_flip' ? 'Label Flipping'
                   : result.attack_type === 'targeted_label' ? 'Targeted Label'
                   : 'Clean Label';
    html += '<div class="config-item"><div class="config-key">TYPE</div><div class="config-val">' + attackName + '</div></div>';
    html += '<div class="config-item"><div class="config-key">DATASET</div><div class="config-val">' + (datasetInfo ? (datasetInfo.class_names.length + ' classes') : 'N/A') + '</div></div>';
    html += '<div class="config-item"><div class="config-key">SEED</div><div class="config-val">' + (result.config.seed || 42) + '</div></div>';
    if (sweep.length > 0) {
      html += '<div class="config-item"><div class="config-key">POISON RATES</div><div class="config-val">' + sweep.map(function(s) { return (s.poison_rate * 100).toFixed(0) + '%'; }).join(', ') + '</div></div>';
    }
    if (datasetInfo) {
      html += '<div class="config-item"><div class="config-key">SAMPLES</div><div class="config-val">' + datasetInfo.n_train + ' train / ' + datasetInfo.n_test + ' test</div></div>';
      html += '<div class="config-item"><div class="config-key">FEATURES</div><div class="config-val">' + datasetInfo.n_features + '</div></div>';
    }
    html += '</div></div>';

    el.innerHTML = html;

    // Render charts after DOM is ready
    requestAnimationFrame(function() {
      // Confusion matrices
      if (result.confusion_matrices) {
        drawConfusionMatrix('cm-clean', result.confusion_matrices.clean, classNames, '', false);
        drawConfusionMatrix('cm-poisoned', result.confusion_matrices.poisoned, classNames, '', true);
      }

      // CM zoom click
      var cmContainer = document.getElementById('cm-pair-container');
      if (cmContainer && typeof openZoom === 'function') {
        cmContainer.onclick = function() {
          openZoom(function(targetId) {
            var target = document.getElementById(targetId);
            target.innerHTML = '<div class="cm-pair"><div><div class="cm-title clean">CLEAN MODEL <span class="cm-acc">' +
              (result.clean_accuracy * 100).toFixed(1) + '%</span></div><div id="zoom-cm-clean"></div></div>' +
              '<div><div class="cm-title poisoned">POISONED MODEL <span class="cm-acc">' +
              (pAcc * 100).toFixed(1) + '%</span></div><div id="zoom-cm-poisoned"></div></div></div>';
            if (result.confusion_matrices) {
              drawConfusionMatrix('zoom-cm-clean', result.confusion_matrices.clean, classNames, '', false, { large: true });
              drawConfusionMatrix('zoom-cm-poisoned', result.confusion_matrices.poisoned, classNames, '', false, { large: true });
            }
          });
        };
      }

      // Sweep bar chart
      if (sweep.length > 1) {
        var vals = sweep.map(function(s) { return s.accuracy_drop; });
        var lbls = sweep.map(function(s) { return (s.poison_rate * 100).toFixed(0) + '%'; });
        drawBarChart('sweep-bar', vals, lbls, 'Accuracy Drop', true);
      }

      // Per-rate tabs
      if (sweep.length > 1) {
        var tabsEl = document.getElementById('sweep-tabs-v2');
        tabsEl.innerHTML = sweep.map(function(s, i) {
          return '<button class="sweep-tab-btn' + (i === 0 ? ' active' : '') + '" data-idx="' + i + '">' +
            (s.poison_rate * 100).toFixed(0) + '%</button>';
        }).join('');

        tabsEl.querySelectorAll('.sweep-tab-btn').forEach(function(tab) {
          tab.addEventListener('click', function() {
            tabsEl.querySelectorAll('.sweep-tab-btn').forEach(function(t) { t.classList.remove('active'); });
            tab.classList.add('active');
            renderSweepDetail(sweep[parseInt(tab.dataset.idx)], classNames);
          });
        });
        renderSweepDetail(sweep[0], classNames);
      }

      // Staggered section reveal
      var sections = el.querySelectorAll('.report-section');
      sections.forEach(function(sec, i) {
        setTimeout(function() {
          sec.classList.add('visible');
        }, i * 300);
      });
    });
  }

  function renderSweepDetail(entry, classNames) {
    var el = document.getElementById('sweep-detail-v2');
    var sev = severityLabel(entry.vulnerability_score);
    var sevColor = sev === 'Low' ? 'amber' : (sev === 'Clean' ? '' : 'red');

    el.innerHTML = '<div class="sweep-detail-metrics">' +
      '<div><div class="sdm-label">RATE</div><div class="sdm-value">' + (entry.poison_rate * 100).toFixed(0) + '%</div></div>' +
      '<div><div class="sdm-label">ACC DROP</div><div class="sdm-value red">' + (entry.accuracy_drop * 100).toFixed(1) + '%</div></div>' +
      '<div><div class="sdm-label">VULN SCORE</div><div class="sdm-value">' + entry.vulnerability_score.toFixed(1) + '</div></div>' +
      '<div><div class="sdm-label">SEVERITY</div><div class="sdm-value ' + sevColor + '">' + sev.toUpperCase() + '</div></div>' +
      '</div>' +
      '<div class="chart-container" id="sweep-cm-detail"><span class="zoom-hint">CLICK TO ZOOM</span></div>';

    if (entry.confusion_matrix) {
      drawConfusionMatrix('sweep-cm-detail', entry.confusion_matrix, classNames,
        'POISONED @ ' + (entry.poison_rate * 100).toFixed(0) + '%', false);

      // Re-add zoom hint after drawConfusionMatrix clears innerHTML
      var cmEl = document.getElementById('sweep-cm-detail');
      var hint = document.createElement('span');
      hint.className = 'zoom-hint';
      hint.textContent = 'CLICK TO ZOOM';
      cmEl.appendChild(hint);
      if (cmEl && typeof openZoom === 'function') {
        cmEl.onclick = function() {
          openZoom(function(targetId) {
            drawConfusionMatrix(targetId, entry.confusion_matrix, classNames,
              'POISONED @ ' + (entry.poison_rate * 100).toFixed(0) + '%', false, { large: true });
          });
        };
      }
    }
  }

  // ---- Download ----
  window.downloadJSON = function() {
    fetch('/api/report/json')
      .then(function(resp) { return resp.blob(); })
      .then(function(blob) {
        var a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'attack_result.json';
        a.click();
      });
  };

  window.downloadHTML = function() {
    fetch('/api/report/html')
      .then(function(resp) { return resp.blob(); })
      .then(function(blob) {
        var a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'attack_report.html';
        a.click();
      });
  };

})();
