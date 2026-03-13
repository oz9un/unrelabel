// unrelabel/static/js/zoom.js
// Full-screen zoom modal for charts
(function() {
  'use strict';

  let isOpen = false;

  window.openZoom = function(renderFn) {
    const modal = document.getElementById('zoom-modal');
    const target = document.getElementById('zoom-target');
    if (!modal || !target) return;

    target.innerHTML = '';
    modal.classList.remove('hidden');
    isOpen = true;

    // Render after modal is visible so container has dimensions
    requestAnimationFrame(function() {
      renderFn('zoom-target');
    });
  };

  window.closeZoom = function() {
    const modal = document.getElementById('zoom-modal');
    const target = document.getElementById('zoom-target');
    if (!modal) return;

    modal.classList.add('hidden');
    if (target) target.innerHTML = '';
    isOpen = false;
  };

  // Escape key
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && isOpen) {
      closeZoom();
    }
  });
})();
