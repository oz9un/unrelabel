// unrelabel/static/js/app.js
(function() {
  'use strict';

  // ---- State ----
  var datasetInfo = null;
  var currentStep = 1;
  var currentSource = 'sklearn';
  var currentAttack = 'label_flipping';
  var previewDebounce = null;

  // ---- DOM refs ----
  var steps = document.querySelectorAll('.step');
  var stepTabs = document.querySelectorAll('.step-tab');
  var sourceTabs = document.querySelectorAll('.source-tab');
  var sourcePanels = document.querySelectorAll('.source-panel');
  var attackCards = document.querySelectorAll('.attack-card');
  var attackForms = document.querySelectorAll('.attack-form');
  var errorBanner = document.getElementById('error-banner');
  var loadingOverlay = document.getElementById('loading-overlay');
  var headerDownloads = document.getElementById('header-downloads');

  // ---- Step navigation ----
  window.goToStep = function(n) {
    steps.forEach(function(s) { s.classList.remove('active'); });
    stepTabs.forEach(function(t) { t.classList.remove('active'); });
    document.getElementById('step' + n).classList.add('active');
    document.querySelector('[data-step="' + n + '"]').classList.add('active');
    currentStep = n;
    // Show/hide header downloads
    if (headerDownloads) {
      if (n === 3 && window._attackResult) {
        headerDownloads.classList.remove('hidden');
      } else {
        headerDownloads.classList.add('hidden');
      }
    }
  };

  stepTabs.forEach(function(tab) {
    tab.addEventListener('click', function() {
      var n = parseInt(tab.dataset.step);
      if (n < currentStep || (n === 2 && datasetInfo) || (n === 3 && window._attackResult)) {
        window.goToStep(n);
      }
    });
  });

  // ---- Source tabs ----
  sourceTabs.forEach(function(tab) {
    tab.addEventListener('click', function() {
      sourceTabs.forEach(function(t) { t.classList.remove('active'); });
      sourcePanels.forEach(function(p) { p.classList.remove('active'); });
      tab.classList.add('active');
      currentSource = tab.dataset.source;
      document.getElementById('source-' + currentSource).classList.add('active');
    });
  });

  // ---- Attack cards ----
  attackCards.forEach(function(card) {
    card.addEventListener('click', function() {
      attackCards.forEach(function(c) { c.classList.remove('active'); });
      attackForms.forEach(function(f) { f.classList.remove('active'); });
      card.classList.add('active');
      currentAttack = card.dataset.attack;
      document.getElementById('form-' + currentAttack).classList.add('active');
      schedulePreviewUpdate();
    });
  });

  // ---- Rate toggles ----
  document.querySelectorAll('.rate-toggle').forEach(function(btn) {
    btn.addEventListener('click', function() {
      btn.classList.toggle('active');
      schedulePreviewUpdate();
    });
  });

  // ---- Class dropdown changes trigger preview ----
  ['lf-source-class', 'lf-target-class', 'tl-source-class', 'tl-target-class', 'cl-source-class', 'cl-target-class'].forEach(function(id) {
    var el = document.getElementById(id);
    if (el) el.addEventListener('change', function() { schedulePreviewUpdate(); });
  });

  // ---- Error handling ----
  function showError(msg) {
    errorBanner.textContent = msg;
    errorBanner.classList.remove('hidden');
    setTimeout(function() { errorBanner.classList.add('hidden'); }, 5000);
    errorBanner.onclick = function() { errorBanner.classList.add('hidden'); };
  }

  function showLoading() { loadingOverlay.classList.remove('hidden'); }
  function hideLoading() { loadingOverlay.classList.add('hidden'); }

  // ---- API helpers ----
  function api(method, url, body, quiet) {
    if (!quiet) showLoading();
    var opts = { method: method, headers: {} };
    if (body && !(body instanceof FormData)) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    } else if (body) {
      opts.body = body;
    }
    return fetch(url, opts)
      .then(function(resp) {
        if (!resp.ok) {
          return resp.json().catch(function() { return { detail: resp.statusText }; })
            .then(function(err) { throw new Error(err.detail || JSON.stringify(err)); });
        }
        return resp.json();
      })
      .catch(function(e) {
        showError(e.message);
        throw e;
      })
      .finally(function() {
        if (!quiet) hideLoading();
      });
  }

  // ---- Sliders ----
  var testSlider = document.getElementById('test-size');
  var testVal = document.getElementById('test-size-val');
  testSlider.addEventListener('input', function() {
    testVal.textContent = Math.round(testSlider.value * 100) + '%';
  });

  var epsSlider = document.getElementById('cl-epsilon');
  var epsVal = document.getElementById('cl-epsilon-val');
  epsSlider.addEventListener('input', function() {
    epsVal.textContent = parseFloat(epsSlider.value).toFixed(2);
  });

  // ---- File drop zone ----
  var dropZone = document.getElementById('drop-zone');
  var fileInput = document.getElementById('file-input');
  var uploadedFile = null;
  dropZone.addEventListener('click', function() { fileInput.click(); });
  dropZone.addEventListener('dragover', function(e) { e.preventDefault(); dropZone.classList.add('dragover'); });
  dropZone.addEventListener('dragleave', function() { dropZone.classList.remove('dragover'); });
  dropZone.addEventListener('drop', function(e) {
    e.preventDefault(); dropZone.classList.remove('dragover');
    uploadedFile = e.dataTransfer.files[0];
    dropZone.querySelector('p').textContent = uploadedFile.name;
  });
  fileInput.addEventListener('change', function() {
    uploadedFile = fileInput.files[0];
    dropZone.querySelector('p').textContent = uploadedFile.name;
  });

  // ---- Load & Fit ----
  document.getElementById('btn-load').addEventListener('click', function() {
    var p;
    if (currentSource === 'sklearn') {
      p = api('POST', '/api/dataset/load', {
        source: 'sklearn',
        name: document.getElementById('sklearn-name').value,
        test_size: parseFloat(testSlider.value),
        seed: 42,
        model: document.getElementById('model-select').value,
      });
    } else if (currentSource === 'upload') {
      if (!uploadedFile) { showError('Select a file first.'); return; }
      var fd = new FormData();
      fd.append('file', uploadedFile);
      fd.append('label_col', document.getElementById('upload-label-col').value);
      fd.append('test_size', testSlider.value);
      fd.append('seed', '42');
      fd.append('model', document.getElementById('model-select').value);
      p = api('POST', '/api/dataset/upload', fd);
    } else if (currentSource === 'huggingface') {
      p = api('POST', '/api/dataset/load', {
        source: 'huggingface',
        dataset_id: document.getElementById('hf-dataset-id').value,
        label_col: document.getElementById('hf-label-col').value,
        test_size: parseFloat(testSlider.value),
        seed: 42,
        model: document.getElementById('model-select').value,
      });
    }
    if (p) {
      p.then(function(data) {
        datasetInfo = data;
        renderDatasetStats(data);
        loadScatterPlot();
        populateClassDropdowns(data.class_names);
        document.getElementById('btn-continue-1').classList.remove('hidden');
      }).catch(function() { /* error already shown */ });
    }
  });

  function renderDatasetStats(d) {
    var el = document.getElementById('dataset-stats');
    var balanceHtml = d.balance
      ? Object.entries(d.balance).map(function(entry) {
          return '<div class="stat"><span class="stat-label">' + entry[0].toUpperCase() + '</span><span class="stat-value">' + entry[1] + '</span></div>';
        }).join('')
      : '';
    el.innerHTML =
      '<div class="stat"><span class="stat-label">SAMPLES</span><span class="stat-value">' + (d.n_train + d.n_test) + '</span></div>' +
      '<div class="stat"><span class="stat-label">TRAIN</span><span class="stat-value">' + d.n_train + '</span></div>' +
      '<div class="stat"><span class="stat-label">TEST</span><span class="stat-value">' + d.n_test + '</span></div>' +
      '<div class="stat"><span class="stat-label">FEATURES</span><span class="stat-value">' + d.n_features + '</span></div>' +
      '<div class="stat"><span class="stat-label">CLASSES</span><span class="stat-value">' + d.n_classes + '</span></div>' +
      '<div class="stat"><span class="stat-label">BASELINE ACC</span><span class="stat-value">' + (d.baseline_accuracy * 100).toFixed(1) + '%</span></div>' +
      balanceHtml;
  }

  function loadScatterPlot() {
    var container = document.getElementById('scatter-container');
    api('GET', '/api/dataset/scatter')
      .then(function(data) {
        container.style.display = '';
        if (typeof drawScatter === 'function') {
          drawScatter('scatter-container', data);
        }
      })
      .catch(function() { /* optional */ });
  }

  function populateClassDropdowns(classNames) {
    var selectors = ['lf-source-class', 'lf-target-class', 'tl-source-class',
                      'tl-target-class', 'cl-source-class', 'cl-target-class'];
    selectors.forEach(function(id) {
      var sel = document.getElementById(id);
      var keepFirst = id.startsWith('lf-');
      while (sel.options.length > (keepFirst ? 1 : 0)) sel.remove(keepFirst ? 1 : 0);
      classNames.forEach(function(name, idx) {
        var opt = document.createElement('option');
        opt.value = idx;
        opt.textContent = name + ' (' + idx + ')';
        sel.appendChild(opt);
      });
    });
  }

  // ---- Continue to Step 2 ----
  document.getElementById('btn-continue-1').addEventListener('click', function() {
    window.goToStep(2);
    schedulePreviewUpdate();
  });

  // ---- Run Another ----
  document.getElementById('btn-run-another').addEventListener('click', function() {
    window.goToStep(2);
  });

  // ---- Debounced preview update ----
  function schedulePreviewUpdate() {
    if (previewDebounce) clearTimeout(previewDebounce);
    previewDebounce = setTimeout(function() {
      loadPreviewScatter();
    }, 300);
  }

  function getSelectedRates() {
    var formId = 'form-' + currentAttack;
    var form = document.getElementById(formId);
    if (!form) return [];
    var rates = [];
    form.querySelectorAll('.rate-toggle.active').forEach(function(btn) {
      rates.push(parseFloat(btn.dataset.rate));
    });
    return rates;
  }

  function loadPreviewScatter() {
    var url = '/api/dataset/scatter?attack_type=' + currentAttack;
    if (currentAttack === 'label_flipping' || currentAttack === 'targeted_label') {
      var rates = getSelectedRates();
      var maxRate = rates.length ? Math.max.apply(null, rates) : 0.1;
      url += '&poison_rate=' + maxRate;
    }

    var srcId = currentAttack === 'label_flipping' ? 'lf-source-class'
              : currentAttack === 'targeted_label' ? 'tl-source-class'
              : 'cl-source-class';
    var src = document.getElementById(srcId);
    if (src && src.value) url += '&source_class=' + src.value;

    api('GET', url, null, true)
      .then(function(data) {
        if (typeof drawScatter === 'function') {
          drawScatter('preview-scatter', data);
        }
        // Update count
        var countEl = document.getElementById('preview-count');
        if (countEl && data.highlight_indices) {
          var n = data.highlight_indices.length;
          var rates = getSelectedRates();
          var maxRate = rates.length ? Math.max.apply(null, rates) : 0;
          if (currentAttack === 'clean_label') {
            countEl.innerHTML = '<span class="count-value">' + n + ' samples</span> in source class';
          } else {
            countEl.innerHTML = '<span class="count-value">' + n + ' samples</span> selected for poisoning at ' + (maxRate * 100).toFixed(0) + '% rate';
          }
        } else if (countEl) {
          countEl.innerHTML = '';
        }
      })
      .catch(function() { /* optional */ });
  }

  // ---- Run Attack ----
  document.getElementById('btn-run-attack').addEventListener('click', function() {
    var body = { attack_type: currentAttack, seed: 42 };

    if (currentAttack === 'label_flipping') {
      var rates = getSelectedRates();
      if (rates.length === 0) { showError('Select at least one poison rate.'); return; }
      body.poison_rates = rates;
      var src = document.getElementById('lf-source-class').value;
      var tgt = document.getElementById('lf-target-class').value;
      if (src) body.source_class = parseInt(src);
      if (tgt) body.target_class = parseInt(tgt);
    } else if (currentAttack === 'targeted_label') {
      var tRates = getSelectedRates();
      if (tRates.length === 0) { showError('Select at least one poison rate.'); return; }
      body.poison_rates = tRates;
      body.source_class = parseInt(document.getElementById('tl-source-class').value);
      body.target_class = parseInt(document.getElementById('tl-target-class').value);
    } else if (currentAttack === 'clean_label') {
      body.source_class = parseInt(document.getElementById('cl-source-class').value);
      body.target_class = parseInt(document.getElementById('cl-target-class').value);
      body.n_neighbors = parseInt(document.getElementById('cl-neighbors').value);
      body.epsilon = parseFloat(epsSlider.value);
    }

    api('POST', '/api/attack/run', body)
      .then(function(result) {
        window._attackResult = result;
        window.goToStep(3);
        if (typeof startReveal === 'function') {
          startReveal(result, datasetInfo);
        }
      })
      .catch(function() { /* error already shown */ });
  });

})();
